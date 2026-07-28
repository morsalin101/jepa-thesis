"""Self-supervised pretraining engine, shared by I-JEPA, MAE, SimCLR and MoCo v3.

One engine for all four methods is what makes the comparison a comparison: the
optimiser, schedules, precision policy, data pipeline, checkpointing and logging are
literally the same code path, so the only thing that differs is the objective.

Kaggle shapes several decisions here:

* A run is chopped into 2-3 sessions by the ~9h cap, so the loop is resumable and the
  session guard exits *cleanly* before the platform kills us (a killed notebook is
  marked failed and its output — the free checkpoint mirror — is discarded).
* T4 is sm_75: fp16 tensor cores, no bf16 hardware. Precision comes from the capability
  gate in `src.config`, never from an assumption.
* Global batch is pinned; per-GPU batch is derived. Otherwise a session that lands on a
  1-GPU P100 instead of T4 x2 would silently halve the global batch and shift every
  schedule.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.config import (
    PretrainCfg,
    amp_config,
    build_pretrain_cfg,
    config_dict,
    config_hash,
    default_output_dir,
    derive_batch,
    on_kaggle,
    resolve_dataset_dir,
)
from src.data.hyperkvasir import HyperKvasirUnlabeled
from src.data.transforms import GPUAugment, TwoViewTransform, make_pretrain_transform
from src.masks.multiblock import MaskCollator
from src.model import METHOD_SPECS, build_pretrain_model
from src.utils import ddp
from src.utils.checkpoint import (
    ResumeMismatch,
    RunIdMismatch,
    build_checkpoint,
    export_encoder,
    find_checkpoint,
    load_checkpoint,
    save_checkpoint,
    session_record,
)
from src.utils.kaggle_io import (
    CheckpointPusher,
    SessionGuard,
    append_metrics,
    resume_candidates,
    stage_file,
)
from src.utils.schedulers import ScheduleSet, cosine_momentum, param_groups_with_wd_exclusion

CKPT_NAME = "latest.pt"


# ------------------------------------------------------------------ data


def build_dataloader(cfg: PretrainCfg, per_gpu_batch: int, data_root: Path | None = None):
    spec = METHOD_SPECS[cfg.method]
    a, m = cfg.aug, cfg.model

    if spec["needs_two_views"]:
        # CPU produces uint8 crops; colour jitter / blur / normalise happen on the GPU.
        view = make_pretrain_transform(
            m.img_size, a.crop_scale, horizontal_flip=False, to_uint8=True
        )
        transform = TwoViewTransform(view)
    else:
        transform = make_pretrain_transform(
            m.img_size, a.crop_scale, a.horizontal_flip, to_uint8=False, mean=a.mean, std=a.std
        )

    exclude_file = Path(__file__).resolve().parents[2] / "splits" / "pretrain_excluded.txt"
    dataset = HyperKvasirUnlabeled(
        root=data_root,
        transform=transform,
        exclude_file=exclude_file if exclude_file.is_file() else None,
        cache_dir=default_output_dir(),
    )
    if ddp.is_main():
        print(f"[data] {dataset.describe()}")

    collate = None
    if spec["needs_masks"]:
        k = cfg.mask
        collate = MaskCollator(
            input_size=m.img_size,
            patch_size=m.patch_size,
            enc_mask_scale=tuple(k.enc_mask_scale),
            pred_mask_scale=tuple(k.pred_mask_scale),
            aspect_ratio=tuple(k.aspect_ratio),
            num_enc_masks=k.num_enc_masks,
            num_pred_masks=k.num_pred_masks,
            min_keep=k.min_keep,
            allow_overlap=k.allow_overlap,
        )

    sampler = (
        DistributedSampler(dataset, shuffle=True, drop_last=True, seed=cfg.runtime.seed)
        if ddp.is_dist()
        else None
    )
    r = cfg.runtime
    loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=r.num_workers,
        pin_memory=r.pin_memory,
        drop_last=True,
        collate_fn=collate,
        persistent_workers=r.persistent_workers and r.num_workers > 0,
        prefetch_factor=r.prefetch_factor if r.num_workers > 0 else None,
    )
    return dataset, loader, sampler, collate


def build_gpu_augment(cfg: PretrainCfg, device: str) -> GPUAugment | None:
    if not METHOD_SPECS[cfg.method]["needs_two_views"]:
        return None
    a = cfg.aug
    return GPUAugment(
        color_jitter_strength=a.color_jitter_strength,
        color_distortion=a.color_distortion,
        gaussian_blur_p=0.5 if a.gaussian_blur else 0.0,
        solarize_p=0.2 if a.solarize else 0.0,
        horizontal_flip=a.horizontal_flip,
        mean=a.mean,
        std=a.std,
    ).to(device)


# ------------------------------------------------------------------ one step


def forward_loss(cfg: PretrainCfg, model: nn.Module, batch, device: str, gpu_aug) -> torch.Tensor:
    """Dispatch to the method's objective. The only method-specific code in the loop."""
    core = ddp.unwrap(model)
    method = cfg.method

    if method == "ijepa":
        imgs, masks_enc, masks_pred = batch
        imgs = imgs.to(device, non_blocking=True)
        masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
        masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]
        z, h = core(imgs, masks_enc, masks_pred)
        return core.loss(z, h)

    if method == "mae":
        imgs = batch.to(device, non_blocking=True)
        loss, _, _ = core(imgs)
        return loss

    # simclr / mocov3
    v1, v2 = batch
    v1 = gpu_aug(v1.to(device, non_blocking=True))
    v2 = gpu_aug(v2.to(device, non_blocking=True))
    return core((v1, v2))


# ------------------------------------------------------------------ main loop


def pretrain(cfg: PretrainCfg, rank: int = 0, world_size: int = 1, data_root: Path | None = None) -> Path:
    ddp.setup(rank, world_size)
    amp = amp_config()
    device = f"cuda:{rank}" if amp.device == "cuda" else amp.device
    if amp.device == "cuda":
        torch.cuda.set_device(rank)
    ddp.seed_everything(cfg.runtime.seed + rank)

    spec = METHOD_SPECS[cfg.method]
    o = cfg.optim

    # Contrastive losses are batch-coupled: gradient accumulation does NOT reproduce a
    # larger batch for InfoNCE the way it does for a per-sample loss. So the global batch
    # can only be preserved by having the right world size.
    if spec["needs_two_views"] and o.accum_steps != 1:
        raise ValueError(
            f"{cfg.method}: accum_steps must be 1. InfoNCE is computed over the batch, so "
            "accumulation would silently shrink the negative set rather than emulate a "
            "larger batch. Run on the world size that divides global_batch instead."
        )
    per_gpu = derive_batch(o.global_batch, world_size, o.accum_steps)

    if ddp.is_main():
        print(f"[pretrain] {cfg.run_id}")
        print(f"[pretrain] device={amp.name} sm_{amp.sm} autocast={amp.dtype} scaler={amp.use_scaler}")
        print(
            f"[pretrain] global_batch={o.global_batch} = per_gpu {per_gpu} "
            f"x world {world_size} x accum {o.accum_steps}"
        )

    dataset, loader, sampler, collate = build_dataloader(cfg, per_gpu, data_root)
    gpu_aug = build_gpu_augment(cfg, device)

    model = build_pretrain_model(cfg).to(device)
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank] if amp.device == "cuda" else None,
            # MoCo v3 freezes the patch embedding and MAE's decoder is only partly used
            # on some steps; DDP would otherwise error on unused parameters.
            find_unused_parameters=spec["has_ema"] or cfg.method == "mocov3",
        )
    core = ddp.unwrap(model)

    iters_per_epoch = len(loader) // o.accum_steps
    sched = ScheduleSet(
        iters_per_epoch=iters_per_epoch,
        epochs=o.epochs,
        warmup_epochs=o.warmup_epochs,
        start_lr=o.start_lr,
        ref_lr=o.ref_lr,
        final_lr=o.final_lr,
        weight_decay=o.weight_decay,
        final_weight_decay=o.final_weight_decay,
        ema=tuple(o.ema),
        ipe_scale=o.ipe_scale,
    )
    optimizer = torch.optim.AdamW(
        param_groups_with_wd_exclusion(core, base_lr=o.ref_lr), betas=(0.9, 0.95)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp.use_scaler)

    out_dir = default_output_dir()
    # One checkpoint directory per run, so concurrent or sequential runs of different
    # methods never contend for the same file. Metrics stay in one shared file (each
    # record carries its method) so the figures can read every run from one place.
    ckpt_dir = out_dir / "ckpt" / cfg.run_id
    ckpt_path = ckpt_dir / CKPT_NAME
    metrics_path = out_dir / "ckpt" / "metrics.jsonl"
    cfg_hash = config_hash(cfg)
    resolved = config_dict(cfg)

    # ------------------------------------------------------------ resume
    start_epoch, global_step, loss_history, wall_clock, sessions = 0, 0, [], 0.0, []
    found = find_checkpoint(
        resume_candidates(CKPT_NAME, cfg.runtime.ckpt_dataset_slug, working_dir=str(ckpt_dir))
    )
    if found is not None:
        try:
            ck = load_checkpoint(
                found,
                expect_run_id=cfg.run_id,
                expect_config_hash=cfg_hash,
                modules=core.checkpoint_modules(),
                optimizer=optimizer,
                scaler=scaler,
                map_location=device,
            )
            start_epoch = ck["epoch"]
            global_step = ck["global_step"]
            loss_history = ck["loss_history"]
            wall_clock = ck["wall_clock_s"]
            sessions = ck["sessions"]
            if ddp.is_main():
                print(
                    f"[resume] {found} -> epoch {start_epoch}/{o.epochs}, step {global_step}, "
                    f"{wall_clock / 3600:.2f} GPU-h already spent"
                )
        except RunIdMismatch as e:
            # Someone else's checkpoint (e.g. a different method sharing the mounted
            # Kaggle dataset). Not an error — just not ours.
            if ddp.is_main():
                print(f"[resume] ignoring foreign checkpoint: {e}")
        except ResumeMismatch as e:
            # Same run, different config: the schedules would silently shift. Stop.
            if ddp.is_main():
                print(f"[resume] refusing to resume: {e}")
            raise

    if start_epoch >= o.epochs:
        if ddp.is_main():
            print(f"[pretrain] already complete ({start_epoch}/{o.epochs} epochs)")
            _finalise(cfg, ckpt_path, out_dir)
        ddp.cleanup()
        return ckpt_path

    pusher = CheckpointPusher(
        cfg.runtime.ckpt_dataset_slug,
        ckpt_dir,
        push_minutes=cfg.runtime.ckpt_push_minutes,
        enabled=bool(cfg.runtime.ckpt_dataset_slug),
    )
    guard = SessionGuard(cfg.runtime.session_guard_hours)
    session_start = time.time()

    def write_ckpt(epoch: int, message: str, push: bool = False) -> None:
        ck = build_checkpoint(
            run_id=cfg.run_id,
            config_hash=cfg_hash,
            resolved_cfg=resolved,
            modules=core.checkpoint_modules(),
            optimizer=optimizer,
            scaler=scaler,
            global_step=global_step,
            epoch=epoch,
            total_steps=sched.total_steps,
            base_seed=cfg.runtime.seed,
            loss_history=loss_history,
            wall_clock_s=wall_clock + (time.time() - session_start),
            sessions=sessions + [session_record(session_start, epoch - start_epoch, amp.name)],
        )
        save_checkpoint(ck, ckpt_path)
        if push and pusher.enabled:
            stage_file(ckpt_path, ckpt_dir)
            pusher.push(message, force=True)

    # ------------------------------------------------------------ train
    nan_streak = 0
    epochs_done = 0

    for epoch in range(start_epoch, o.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        if collate is not None:
            collate.set_epoch(epoch, iters_per_epoch)
        # Data order is a pure function of (base_seed, epoch), so resuming at epoch k
        # reproduces exactly the stream an uninterrupted run would have seen.
        ddp.seed_everything(ddp.epoch_seed(cfg.runtime.seed, epoch) + rank)

        t_epoch = time.time()
        running, n_batches = 0.0, 0
        pbar = (
            tqdm(loader, desc=f"{cfg.method} ep {epoch + 1}/{o.epochs}", disable=not ddp.is_main())
            if ddp.is_main()
            else loader
        )

        optimizer.zero_grad(set_to_none=True)
        for it, batch in enumerate(pbar):
            lr, wd = sched.apply(optimizer, global_step)

            with torch.autocast(
                device_type="cuda" if amp.device == "cuda" else "cpu",
                dtype=amp.dtype or torch.float32,
                enabled=amp.enabled,
            ):
                loss = forward_loss(cfg, model, batch, device, gpu_aug)
                loss = loss / o.accum_steps

            if not torch.isfinite(loss):
                nan_streak += 1
                optimizer.zero_grad(set_to_none=True)
                if nan_streak >= 20:
                    raise RuntimeError(
                        f"loss non-finite for {nan_streak} consecutive steps at epoch "
                        f"{epoch + 1}. Reload the last checkpoint and halve the LR "
                        f"(currently {lr:.2e})."
                    )
                continue
            nan_streak = 0

            scaler.scale(loss).backward()

            if (it + 1) % o.accum_steps == 0:
                if o.grad_clip > 0:
                    scaler.unscale_(optimizer)  # required before clipping
                    torch.nn.utils.clip_grad_norm_(core.parameters(), o.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # EMA / momentum update, fp32 and outside autocast. See model docstrings:
                # in fp16 the (1-m) term rounds away and the target silently stops tracking.
                if cfg.method == "ijepa":
                    core.update_target_encoder(sched.momentum(global_step))
                elif cfg.method == "mocov3":
                    # MoCo v3 ramps its momentum on a *cosine*, not linearly like i-jepa.
                    core.momentum_update(
                        cosine_momentum(
                            global_step, sched.total_steps, tuple(cfg.contrastive.moco_momentum)
                        )
                    )
                global_step += 1

            running += float(loss.detach()) * o.accum_steps
            n_batches += 1
            if ddp.is_main() and isinstance(pbar, tqdm) and it % 10 == 0:
                pbar.set_postfix(loss=f"{running / max(1, n_batches):.4f}", lr=f"{lr:.2e}")

        epoch_loss = ddp.all_reduce_mean(running / max(1, n_batches))
        epoch_time = time.time() - t_epoch
        loss_history.append(epoch_loss)
        epochs_done += 1

        if ddp.is_main():
            print(
                f"[{cfg.method} ep {epoch + 1}/{o.epochs}] loss={epoch_loss:.4f} "
                f"lr={sched.lr(global_step):.2e} wd={sched.wd(global_step):.3f} "
                f"{epoch_time / 60:.1f} min/epoch | {guard.summary()}"
            )
            append_metrics(
                metrics_path,
                {
                    "run_id": cfg.run_id,
                    "method": cfg.method,
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "lr": sched.lr(global_step),
                    "wd": sched.wd(global_step),
                    "momentum": (
                        sched.momentum(global_step)
                        if cfg.method == "ijepa"
                        else cosine_momentum(
                            global_step, sched.total_steps, tuple(cfg.contrastive.moco_momentum)
                        )
                        if cfg.method == "mocov3"
                        else None
                    ),
                    "global_step": global_step,
                    "samples_seen": global_step * o.global_batch,
                    "epoch_time_s": round(epoch_time, 1),
                    "accelerator": amp.name,
                    "world_size": world_size,
                },
            )
            write_ckpt(epoch + 1, f"{cfg.run_id} ep={epoch + 1} loss={epoch_loss:.4f}", push=pusher.due())

        # Stop before Kaggle stops us, so the commit succeeds and its output survives.
        if guard.would_exceed(epoch_time):
            if ddp.is_main():
                print(
                    f"[guard] {guard.summary()}; next epoch (~{epoch_time / 60:.1f} min) would "
                    f"overrun. Stopping cleanly at epoch {epoch + 1}/{o.epochs}."
                )
                write_ckpt(epoch + 1, f"{cfg.run_id} guard-exit ep={epoch + 1}", push=True)
            break

    ddp.barrier()
    if ddp.is_main():
        done_epochs = min(start_epoch + epochs_done, o.epochs)
        write_ckpt(done_epochs, f"{cfg.run_id} session end", push=True)
        # Completion is decided by epochs finished, not by how the loop exited: the
        # guard can legitimately fire on the very last epoch, and that is still a
        # finished run that deserves its exported encoder.
        if done_epochs >= o.epochs:
            _finalise(cfg, ckpt_path, out_dir)
        else:
            print(
                f"[pretrain] {o.epochs - done_epochs} epochs remaining — "
                "re-run this notebook to continue."
            )
    ddp.cleanup()
    return ckpt_path


def _finalise(cfg: PretrainCfg, ckpt_path: Path, out_dir: Path) -> None:
    """Export publishable encoder weights once a run completes."""
    weights = out_dir / "weights" / f"{cfg.method}_{cfg.model.arch}_{cfg.model.img_size}.pt"
    export_encoder(ckpt_path, weights)
    print(f"[pretrain] run complete. Encoder exported to {weights} ({weights.stat().st_size / 1e6:.0f} MB)")


# ------------------------------------------------------------------ launcher


def _worker(rank: int, world_size: int, cfg: PretrainCfg, data_root: str | None) -> None:
    pretrain(cfg, rank, world_size, Path(data_root) if data_root else None)


def run(cfg: PretrainCfg, data_root: Path | None = None, world_size: int | None = None) -> None:
    """Launch training, spawning one process per GPU when there is more than one."""
    if world_size is None:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    spec = METHOD_SPECS[cfg.method]
    if spec["needs_two_views"] and on_kaggle() and world_size != 2:
        raise RuntimeError(
            f"{cfg.method} needs world_size 2 to keep global_batch={cfg.optim.global_batch} "
            f"with accum_steps=1, but found {world_size} GPU(s). Kaggle assigned a single-GPU "
            "accelerator (usually P100). Set Session options -> Accelerator -> GPU T4 x2 and "
            "re-run; the checkpoint is safe. (I-JEPA and MAE can absorb this with accum_steps, "
            "contrastive losses cannot.)"
        )

    if world_size > 1:
        torch.multiprocessing.spawn(
            _worker, args=(world_size, cfg, str(data_root) if data_root else None), nprocs=world_size
        )
    else:
        pretrain(cfg, 0, 1, data_root)


def main() -> None:
    ap = argparse.ArgumentParser(description="SSL pretraining")
    ap.add_argument("--method", required=True, choices=sorted(METHOD_SPECS))
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--global-batch", type=int, default=None)
    ap.add_argument("--accum-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--patch-size", type=int, default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt-slug", default=None, help="Kaggle dataset slug for cross-session resume")
    ap.add_argument("--guard-hours", type=float, default=None)
    ap.add_argument("--world-size", type=int, default=None)
    args = ap.parse_args()

    overrides: dict = {"optim": {}, "model": {}, "runtime": {}}
    if args.epochs is not None:
        overrides["optim"]["epochs"] = args.epochs
    if args.global_batch is not None:
        overrides["optim"]["global_batch"] = args.global_batch
    if args.accum_steps is not None:
        overrides["optim"]["accum_steps"] = args.accum_steps
    if args.lr is not None:
        overrides["optim"]["ref_lr"] = args.lr
    if args.img_size is not None:
        overrides["model"]["img_size"] = args.img_size
    if args.patch_size is not None:
        overrides["model"]["patch_size"] = args.patch_size
    if args.arch is not None:
        overrides["model"]["arch"] = args.arch
    if args.seed is not None:
        overrides["runtime"]["seed"] = args.seed
    if args.num_workers is not None:
        overrides["runtime"]["num_workers"] = args.num_workers
    if args.ckpt_slug is not None:
        overrides["runtime"]["ckpt_dataset_slug"] = args.ckpt_slug
    if args.guard_hours is not None:
        overrides["runtime"]["session_guard_hours"] = args.guard_hours
    overrides = {k: v for k, v in overrides.items() if v}

    cfg = build_pretrain_cfg(args.method, args.config, overrides)
    root = Path(args.data_root) if args.data_root else None
    if root is None and not on_kaggle():
        try:
            root = resolve_dataset_dir("hyperkvasir")
        except FileNotFoundError:
            pass
    run(cfg, root, args.world_size)


if __name__ == "__main__":
    main()

"""Segmentation fine-tuning on Kvasir-SEG.

Runs identically for all five encoders (I-JEPA, MAE, SimCLR, MoCo v3, random init) —
same decoder, same initialisation, same schedule, same augmentation, same splits. Only
the encoder weights differ. That is the entire experiment.

Protocol decisions enforced here rather than left to discipline, because each is a place
where segmentation papers routinely leak:

* **Model selection uses val, never test.** The checkpoint is chosen by best val Dice
  with early stopping. `evaluate_test` refuses to run unless `training_complete` is set,
  so the common error of reporting `max over epochs of test Dice` is not reachable by
  accident.
* **Val and test transforms have no augmentation.** They are separate dataset objects
  with their own eval transform, not a `Subset` of the augmented training set.
* **Test metrics are computed at native resolution** via `src.eval.metrics`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    SegCfg,
    amp_config,
    build_seg_cfg,
    config_dict,
    config_hash,
    default_output_dir,
    resolve_dataset_dir,
)
from src.data.kvasir_seg import KvasirSegDataset, read_split
from src.data.transforms import make_seg_transforms
from src.eval.metrics import aggregate, dice_from_logits, evaluate_dataset
from src.model.segformer_head import ViTSegFormer
from src.utils import ddp
from src.utils.kaggle_io import append_metrics
from src.utils.schedulers import ScheduleSet


# ------------------------------------------------------------------ loss


class DiceBCELoss(nn.Module):
    """Dice + BCE. Dice handles the class imbalance (polyps are a small minority of
    pixels), BCE keeps the gradient well-behaved where Dice saturates."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # fp32: BCE-with-logits is numerically delicate under fp16 autocast.
        logits, target = logits.float(), target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2, 3))
        denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1 - ((2 * inter + self.eps) / (denom + self.eps)).mean()
        return self.bce_weight * bce + self.dice_weight * dice


# ------------------------------------------------------------------ model


def build_seg_model(cfg: SegCfg) -> nn.Module:
    m = cfg.model
    if cfg.decoder == "segformer":
        return ViTSegFormer(
            arch=m.arch,
            img_size=m.img_size,
            patch_size=m.patch_size,
            fpn_layers=tuple(cfg.fpn_layers),
            decoder_embed_dim=cfg.decoder_embed_dim,
            drop_path_rate=m.drop_path_rate,
        )
    if cfg.decoder == "unet":
        from src.model.unet import ViTUNet

        return ViTUNet(
            arch=m.arch,
            img_size=m.img_size,
            patch_size=m.patch_size,
            skip_layers=tuple(cfg.fpn_layers[:3]),
        )
    raise KeyError(f"unknown decoder {cfg.decoder!r}")


def resolve_encoder_ckpt(cfg: SegCfg, weights_dir: Path) -> Path | None:
    """Find the exported encoder for this arm. `random` deliberately returns None.

    Searches the local working directory *and* any mounted Kaggle dataset. Relying on a
    notebook cell to copy the weights into place first was fragile: Kaggle changed its
    mount layout to /kaggle/input/datasets/<owner>/<slug>/, the fixed-path copy silently
    found nothing, and the failure surfaced here as a confusing "run pretraining first"
    for a run that had already finished.
    """
    if cfg.encoder == "random":
        return None
    if cfg.pretrained_ckpt:
        p = Path(cfg.pretrained_ckpt)
        if not p.is_file():
            raise FileNotFoundError(f"pretrained_ckpt not found: {p}")
        return p

    pattern = f"{cfg.encoder}_{cfg.model.arch}_*.pt"
    matches = sorted(weights_dir.glob(pattern))

    if not matches and Path("/kaggle/input").is_dir():
        matches = sorted(Path("/kaggle/input").glob(f"**/{pattern}"))
        if matches:
            print(f"[segment] found encoder in a mounted dataset: {matches[-1]}")

    if not matches:
        searched = [str(weights_dir / pattern)]
        if Path("/kaggle/input").is_dir():
            searched.append(f"/kaggle/input/**/{pattern}")
            available = sorted(p.name for p in Path("/kaggle/input").glob("**/*.pt"))
            hint = f"\n.pt files that ARE mounted: {available or 'none'}"
        else:
            hint = ""
        raise FileNotFoundError(
            f"no exported encoder matching {pattern}.\nSearched: {searched}{hint}\n"
            f"Either run pretraining for '{cfg.encoder}', attach the "
            f"jepa-thesis-weights dataset, or pass --pretrained-ckpt."
        )
    return matches[-1]


# ------------------------------------------------------------------ data


def build_seg_loaders(cfg: SegCfg, root: Path | None = None):
    train_tf, eval_tf = make_seg_transforms(cfg.model.img_size)
    root = root or resolve_dataset_dir("kvasir_seg")

    ds_train = KvasirSegDataset(
        root,
        stems=read_split(cfg.split, "train"),
        transform=train_tf,
        label_fraction=cfg.label_fraction,
        seed=cfg.runtime.seed,
    )
    # Separate dataset objects with the eval transform — never a Subset of the augmented
    # training set, which would randomly flip validation images.
    ds_val = KvasirSegDataset(root, stems=read_split(cfg.split, "val"), transform=eval_tf)
    ds_test = KvasirSegDataset(root, stems=read_split(cfg.split, "test"), transform=eval_tf)

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.runtime.num_workers,
        pin_memory=True,
        drop_last=len(ds_train) > cfg.batch_size,
        persistent_workers=cfg.runtime.num_workers > 0,
    )
    dl_val = DataLoader(
        ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.runtime.num_workers
    )
    return (ds_train, ds_val, ds_test), (dl_train, dl_val)


# ------------------------------------------------------------------ train


def segment(cfg: SegCfg, data_root: Path | None = None) -> dict:
    amp = amp_config()
    device = "cuda:0" if amp.device == "cuda" else amp.device
    ddp.seed_everything(cfg.runtime.seed)

    out_dir = default_output_dir()
    run_dir = out_dir / "seg" / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (ds_train, ds_val, ds_test), (dl_train, dl_val) = build_seg_loaders(cfg, data_root)
    print(f"[segment] {cfg.run_id}")
    print(f"[segment] train={len(ds_train)} val={len(ds_val)} test={len(ds_test)} split={cfg.split}")

    model = build_seg_model(cfg).to(device)
    weights_dir = out_dir / "weights"
    ckpt = resolve_encoder_ckpt(cfg, weights_dir)
    if ckpt is not None:
        report = model.load_pretrained_encoder(str(ckpt))
        print(f"[segment] encoder <- {ckpt.name} (from {report['source_module']}, run {report['run_id']})")
    else:
        print("[segment] encoder: random initialisation (control arm)")

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.parameters()) - n_enc
    print(f"[segment] params: encoder {n_enc / 1e6:.2f}M  decoder {n_dec / 1e6:.2f}M")

    criterion = DiceBCELoss(cfg.bce_weight, cfg.dice_weight)
    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.enc_lr, cfg.dec_lr, cfg.layer_decay), betas=(0.9, 0.999)
    )
    iters = max(1, len(dl_train))
    sched = ScheduleSet(
        iters_per_epoch=iters,
        epochs=cfg.epochs,
        warmup_epochs=cfg.warmup_epochs,
        start_lr=cfg.enc_lr * 0.1,
        ref_lr=cfg.enc_lr,
        final_lr=cfg.enc_lr * 0.01,
        weight_decay=cfg.weight_decay,
        final_weight_decay=cfg.weight_decay,
        ema=(0.0, 0.0),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp.use_scaler)

    best_dice, best_epoch, patience = -1.0, -1, 0
    best_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.jsonl"
    step = 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        running, n = 0.0, 0
        for imgs, masks in tqdm(dl_train, desc=f"seg ep {epoch + 1}/{cfg.epochs}", leave=False):
            sched.apply(optimizer, step)
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda" if amp.device == "cuda" else "cpu",
                dtype=amp.dtype or torch.float32,
                enabled=amp.enabled,
            ):
                loss = criterion(model(imgs), masks)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            n += 1
            step += 1

        # -- validation: no augmentation, deterministic
        model.eval()
        val_dice, val_loss, vb = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, masks in dl_val:
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.autocast(
                    device_type="cuda" if amp.device == "cuda" else "cpu",
                    dtype=amp.dtype or torch.float32,
                    enabled=amp.enabled,
                ):
                    logits = model(imgs)
                val_loss += float(criterion(logits, masks))
                val_dice += dice_from_logits(logits, masks)
                vb += 1
        val_dice /= max(1, vb)
        val_loss /= max(1, vb)
        train_loss = running / max(1, n)
        improved = val_dice > best_dice

        print(
            f"[seg ep {epoch + 1}/{cfg.epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}" + ("  <- best" if improved else "")
        )
        append_metrics(
            metrics_path,
            {
                "run_id": cfg.run_id,
                "encoder": cfg.encoder,
                "decoder": cfg.decoder,
                "seed": cfg.runtime.seed,
                "label_fraction": cfg.label_fraction,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
            },
        )

        if improved:
            best_dice, best_epoch, patience = val_dice, epoch + 1, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_dice": val_dice,
                    "config_hash": config_hash(cfg),
                    "resolved_cfg": config_dict(cfg),
                },
                best_path,
            )
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"[segment] early stop at epoch {epoch + 1} (best {best_dice:.4f} @ {best_epoch})")
                break

    train_minutes = (time.time() - t0) / 60
    print(
        f"[segment] training done in {train_minutes:.1f} min; "
        f"best val Dice {best_dice:.4f} @ epoch {best_epoch}"
    )

    summary = evaluate_test(cfg, model, ds_test, best_path, device, amp, run_dir, training_complete=True)
    summary.update(
        {
            "best_val_dice": best_dice,
            "best_epoch": best_epoch,
            "train_minutes": round(train_minutes, 1),
            "encoder_params": n_enc,
            "decoder_params": n_dec,
        }
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def evaluate_test(
    cfg: SegCfg,
    model: nn.Module,
    ds_test,
    best_path: Path,
    device: str,
    amp,
    run_dir: Path,
    training_complete: bool = False,
) -> dict:
    """Score the held-out test set. Refuses to run mid-training.

    The guard is the point: it makes 'peek at test each epoch and report the best' an
    error rather than a temptation.
    """
    if not training_complete:
        raise RuntimeError(
            "evaluate_test called before training finished. The test set is scored exactly "
            "once, on the best-val checkpoint. Use the validation loader for any "
            "during-training signal."
        )
    if best_path.is_file():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])

    results = evaluate_dataset(
        model,
        ds_test,
        device=device,
        amp_dtype=amp.dtype,
        save_predictions=run_dir / "test_predictions.npz",
    )
    (run_dir / "test_per_image.json").write_text(
        json.dumps([r.as_dict() for r in results], indent=2) + "\n"
    )
    summary = aggregate(results)
    summary.update(
        {
            "run_id": cfg.run_id,
            "encoder": cfg.encoder,
            "decoder": cfg.decoder,
            "seed": cfg.runtime.seed,
            "label_fraction": cfg.label_fraction,
            "split": cfg.split,
        }
    )
    print(
        f"[test] Dice={summary['dice']:.4f} IoU={summary['iou']:.4f} "
        f"HD95={summary['hd95']:.1f}px failure_rate={summary['failure_rate']:.2%} "
        f"(n={summary['n']}, hd95 substituted {summary['hd95_substituted']}x)"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Segmentation fine-tuning on Kvasir-SEG")
    ap.add_argument("--encoder", default=None, help="ijepa|mae|simclr|mocov3|random")
    ap.add_argument("--decoder", default=None, choices=["segformer", "unet"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--patch-size", type=int, default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--label-fraction", type=float, default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--pretrained-ckpt", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    ov: dict = {"model": {}, "runtime": {}}
    for key, val in [
        ("encoder", args.encoder),
        ("decoder", args.decoder),
        ("epochs", args.epochs),
        ("batch_size", args.batch_size),
        ("label_fraction", args.label_fraction),
        ("split", args.split),
        ("pretrained_ckpt", args.pretrained_ckpt),
    ]:
        if val is not None:
            ov[key] = val
    if args.img_size is not None:
        ov["model"]["img_size"] = args.img_size
    if args.patch_size is not None:
        ov["model"]["patch_size"] = args.patch_size
    if args.arch is not None:
        ov["model"]["arch"] = args.arch
    if args.seed is not None:
        ov["runtime"]["seed"] = args.seed
    if args.num_workers is not None:
        ov["runtime"]["num_workers"] = args.num_workers
    ov = {k: v for k, v in ov.items() if v != {}}

    cfg = build_seg_cfg(args.config, ov)
    segment(cfg, Path(args.data_root) if args.data_root else None)


if __name__ == "__main__":
    main()

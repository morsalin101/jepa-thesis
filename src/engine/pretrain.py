from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CONFIG
from src.data import ISICImageDataset
from src.model.jepa import IJEPA
from src.utils.masking import sample_target_block


def make_collate(n_h: int, n_w: int, scale_range, aspect_range):
    def collate(batch):
        images = torch.stack(list(batch), dim=0)
        B = images.shape[0]

        ctx_list = []
        tgt_list = []
        for _ in range(B):
            m = sample_target_block(n_h, n_w, scale_range, aspect_range)
            ctx_list.append(m["ctx_indices"])
            tgt_list.append(m["tgt_indices"])

        n_ctx_max = max(t.shape[0] for t in ctx_list)
        n_tgt_max = max(t.shape[0] for t in tgt_list)

        ctx_indices = torch.zeros(B, n_ctx_max, dtype=torch.long)
        ctx_valid = torch.zeros(B, n_ctx_max, dtype=torch.bool)
        tgt_indices = torch.zeros(B, n_tgt_max, dtype=torch.long)
        tgt_valid = torch.zeros(B, n_tgt_max, dtype=torch.bool)

        for i in range(B):
            n_c = ctx_list[i].shape[0]
            ctx_indices[i, :n_c] = ctx_list[i]
            ctx_valid[i, :n_c] = True
            n_t = tgt_list[i].shape[0]
            tgt_indices[i, :n_t] = tgt_list[i]
            tgt_valid[i, :n_t] = True

        return {
            "images": images,
            "ctx_indices": ctx_indices,
            "ctx_valid": ctx_valid,
            "tgt_indices": tgt_indices,
            "tgt_valid": tgt_valid,
        }

    return collate


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int = 100) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    import math
    return base_lr * 0.5 * (1.0 + math.cos(progress * math.pi))


def pretrain(cfg) -> Path:
    from src.config import on_kaggle
    if on_kaggle() and cfg.device == "cpu":
        raise RuntimeError(
            "Aborting: Kaggle assigned a GPU that the installed PyTorch cannot use "
            "(sm_60 P100 or older). On the Kaggle web UI, open this notebook, "
            "Session options -> Accelerator -> GPU T4 x2, then re-run."
        )
    print(f"[pretrain] device={cfg.device}  epochs={cfg.jepa.epochs}  bs={cfg.jepa.batch_size}")
    print(f"[pretrain] data_dir={cfg.data_dir}")
    print(f"[pretrain] img={cfg.jepa.img_size} patch={cfg.jepa.patch_size} "
          f"({cfg.jepa.n_h}x{cfg.jepa.n_w}={cfg.jepa.n_h * cfg.jepa.n_w} tokens)")

    ds = ISICImageDataset(cfg.data_dir, img_size=cfg.jepa.img_size, augment=False)
    print(f"[pretrain] dataset size: {len(ds)}")

    collate = make_collate(
        cfg.jepa.n_h,
        cfg.jepa.n_w,
        tuple(cfg.jepa.mask_scale),
        tuple(cfg.jepa.mask_aspect),
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.jepa.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )

    model = IJEPA(
        img_size=cfg.jepa.img_size,
        patch_size=cfg.jepa.patch_size,
        enc_dim=cfg.jepa.enc_dim,
        enc_depth=cfg.jepa.enc_depth,
        enc_heads=cfg.jepa.enc_heads,
        pred_dim=cfg.jepa.pred_dim,
        pred_depth=cfg.jepa.pred_depth,
        pred_heads=cfg.jepa.pred_heads,
        ema_momentum=cfg.jepa.ema_momentum,
    ).to(cfg.device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] params: trainable={n_trainable/1e6:.2f}M  total={n_total/1e6:.2f}M")

    optim = torch.optim.AdamW(
        [
            {"params": model.context_enc.parameters(), "lr": cfg.jepa.enc_lr},
            {"params": model.predictor.parameters(), "lr": cfg.jepa.pred_lr},
        ],
        weight_decay=cfg.jepa.weight_decay,
    )

    steps_per_epoch = max(1, len(dl))
    total_steps = max(1, steps_per_epoch * cfg.jepa.epochs)
    use_bf16 = cfg.device == "cuda"

    losses: list[float] = []
    step = 0
    t0 = time.time()

    for epoch in range(cfg.jepa.epochs):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{cfg.jepa.epochs}", total=steps_per_epoch)
        for batch in pbar:
            images = batch["images"].to(cfg.device, non_blocking=True)
            ctx_idx = batch["ctx_indices"].to(cfg.device, non_blocking=True)
            ctx_valid = batch["ctx_valid"].to(cfg.device, non_blocking=True)
            tgt_idx = batch["tgt_indices"].to(cfg.device, non_blocking=True)
            tgt_valid = batch["tgt_valid"].to(cfg.device, non_blocking=True)

            lr = cosine_lr(step, total_steps, cfg.jepa.enc_lr, warmup_steps=min(100, total_steps // 10))
            for pg in optim.param_groups:
                pg["lr"] = lr

            optim.zero_grad(set_to_none=True)

            if use_bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(images, ctx_idx, ctx_valid, tgt_idx, tgt_valid)
            else:
                loss = model(images, ctx_idx, ctx_valid, tgt_idx, tgt_valid)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            model.update_target_encoder()

            losses.append(loss.item())
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        avg = sum(losses[-steps_per_epoch:]) / steps_per_epoch
        print(f"[epoch {epoch+1}] avg_loss={avg:.4f}")

    elapsed = time.time() - t0
    last_n = min(50, len(losses))
    print(
        f"[pretrain] done in {elapsed:.1f}s  "
        f"first_loss={losses[0]:.4f}  last{last_n}_avg={sum(losses[-last_n:])/last_n:.4f}"
    )

    ckpt_path = cfg.output_dir / "jepa_vit_tiny_smoke.pt"
    torch.save(
        {
            "context_enc": model.context_enc.state_dict(),
            "config": {
                "img_size": cfg.jepa.img_size,
                "patch_size": cfg.jepa.patch_size,
                "enc_dim": cfg.jepa.enc_dim,
                "enc_depth": cfg.jepa.enc_depth,
                "enc_heads": cfg.jepa.enc_heads,
            },
            "step": step,
            "loss_history": losses[:: max(1, len(losses) // 200)],
        },
        ckpt_path,
    )
    print(f"[pretrain] saved {ckpt_path}")
    return ckpt_path

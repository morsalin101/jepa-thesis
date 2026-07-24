from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CONFIG
from src.data_seg import ISICSegDataset
from src.model.unet import ViTUNet


class DiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        inter = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * inter + eps) / (union + eps)
        return 1 - dice.mean()


def iou_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean()


def save_visualization(
    images: torch.Tensor,
    masks_gt: torch.Tensor,
    masks_pred: torch.Tensor,
    out_path: Path,
    n: int = 8,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n, images.shape[0])
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    if n == 1:
        axes = axes.reshape(3, 1)
    for i in range(n):
        img = images[i].detach().cpu() * 0.5 + 0.5
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[0, i].set_title("image")
        axes[0, i].axis("off")
        axes[1, i].imshow(masks_gt[i, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title("GT mask")
        axes[1, i].axis("off")
        pred = (torch.sigmoid(masks_pred[i, 0]) > 0.5).float().cpu().numpy()
        axes[2, i].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[2, i].set_title("Pred mask")
        axes[2, i].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def find_mask_dir(image_dir: Path) -> Path:
    from src.config import on_kaggle
    if on_kaggle():
        return image_dir.parent.parent / "ISIC2018_Task1_Training_GroundTruth"
    return Path("data/ISIC2018_Task1_Training_GroundTruth")


def segment(cfg) -> Path:
    from src.config import on_kaggle
    if on_kaggle() and cfg.device == "cpu":
        raise RuntimeError("Aborting: Kaggle CPU fallback — pick T4 in sidebar")

    print(f"[segment] device={cfg.device}  epochs={cfg.seg.epochs}  bs={cfg.seg.batch_size}")
    print(f"[segment] img_dir={cfg.data_dir}")
    mask_dir = find_mask_dir(cfg.data_dir)
    print(f"[segment] mask_dir={mask_dir}")

    ds_train = ISICSegDataset(cfg.data_dir, mask_dir, img_size=cfg.seg.img_size, augment=True)
    print(f"[segment] dataset size: {len(ds_train)}")

    val_n = min(64, max(8, len(ds_train) // 20))
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(ds_train), generator=g).tolist()
    val_idx = sorted(perm[:val_n])
    train_idx = sorted(perm[val_n:])
    ds_val = torch.utils.data.Subset(ds_train, val_idx)
    ds_train = torch.utils.data.Subset(ds_train, train_idx)
    print(f"[segment] split: train={len(ds_train)}  val={len(ds_val)}")

    dl_train = DataLoader(
        ds_train, batch_size=cfg.seg.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    dl_val = DataLoader(
        ds_val, batch_size=cfg.seg.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    model = ViTUNet(
        img_size=cfg.seg.img_size,
        patch_size=cfg.jepa.patch_size,
        dim=cfg.jepa.enc_dim,
        depth=cfg.jepa.enc_depth,
        heads=cfg.jepa.enc_heads,
    ).to(cfg.device)

    if cfg.seg.pretrained_ckpt and Path(cfg.seg.pretrained_ckpt).exists():
        print(f"[segment] loading pretrained encoder from {cfg.seg.pretrained_ckpt}")
        model.load_encoder_from_jepa(cfg.seg.pretrained_ckpt)
    else:
        print(f"[segment] no pretrained ckpt found — training from scratch")

    if cfg.seg.freeze_encoder_epochs > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print(f"[segment] encoder frozen for first {cfg.seg.freeze_encoder_epochs} epochs (linear-probe phase)")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[segment] params: trainable={n_train/1e6:.2f}M  total={n_total/1e6:.2f}M")

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    optim = torch.optim.AdamW(
        [
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": cfg.seg.enc_lr},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": cfg.seg.lr},
        ],
        weight_decay=cfg.seg.weight_decay,
    )

    use_bf16 = cfg.device == "cuda"
    best_iou = -1.0
    best_ckpt = cfg.output_dir / "unet_best.pt"
    final_ckpt = cfg.output_dir / "unet_final.pt"

    t0 = time.time()
    for epoch in range(cfg.seg.epochs):
        if epoch == cfg.seg.freeze_encoder_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = True
            print(f"[segment] encoder unfrozen at epoch {epoch+1}")

        model.train()
        train_losses: list[float] = []
        pbar = tqdm(dl_train, desc=f"epoch {epoch+1}/{cfg.seg.epochs} train")
        for imgs, masks in pbar:
            imgs = imgs.to(cfg.device, non_blocking=True)
            masks = masks.to(cfg.device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(imgs)
                    loss = bce(logits, masks) + dice(logits, masks)
            else:
                logits = model(imgs)
                loss = bce(logits, masks) + dice(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_losses: list[float] = []
        val_ious: list[float] = []
        with torch.no_grad():
            for imgs, masks in dl_val:
                imgs = imgs.to(cfg.device, non_blocking=True)
                masks = masks.to(cfg.device, non_blocking=True)
                if use_bf16:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(imgs)
                        loss = bce(logits, masks) + dice(logits, masks)
                else:
                    logits = model(imgs)
                    loss = bce(logits, masks) + dice(logits, masks)
                val_losses.append(loss.item())
                val_ious.append(iou_score(logits, masks).item())

        train_loss = sum(train_losses) / max(1, len(train_losses))
        val_loss = sum(val_losses) / max(1, len(val_losses))
        val_iou = sum(val_ious) / max(1, len(val_ious))
        print(
            f"[epoch {epoch+1}] train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_iou={val_iou:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_iou": val_iou,
                    "config": {"img_size": cfg.seg.img_size, "patch_size": cfg.jepa.patch_size},
                },
                best_ckpt,
            )
            print(f"[epoch {epoch+1}] saved best -> {best_ckpt}  (iou={val_iou:.4f})")

        if (epoch + 1) % max(1, cfg.seg.vis_every_epochs) == 0 or epoch == cfg.seg.epochs - 1:
            with torch.no_grad():
                sample_imgs, sample_masks = next(iter(dl_val))
                sample_imgs = sample_imgs.to(cfg.device)
                sample_masks = sample_masks.to(cfg.device)
                if use_bf16:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        sample_logits = model(sample_imgs)
                else:
                    sample_logits = model(sample_imgs)
                vis_path = cfg.output_dir / f"seg_epoch{epoch+1:03d}.png"
                save_visualization(sample_imgs, sample_masks, sample_logits, vis_path, n=8)
                print(f"[epoch {epoch+1}] saved visualization -> {vis_path}")

    torch.save(
        {"model": model.state_dict(), "epoch": cfg.seg.epochs, "config": {"img_size": cfg.seg.img_size}},
        final_ckpt,
    )
    elapsed = time.time() - t0
    print(
        f"[segment] done in {elapsed:.1f}s  "
        f"best_val_iou={best_iou:.4f}  saved_best={best_ckpt}  saved_final={final_ckpt}"
    )
    return best_ckpt

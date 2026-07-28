"""Model factory.

Every SSL method exposes the same surface so `src/engine/pretrain.py` can drive all four
without branching on the method name anywhere except loss computation:

    model.encoder            the ViT to export and later fine-tune
    model.target_encoder     EMA branch, or None
    model.checkpoint_modules()  -> {"model": self}
"""
from __future__ import annotations

import torch.nn as nn

from src.config import PretrainCfg

# `needs_two_views` drives the transform; `needs_masks` drives the collator.
METHOD_SPECS: dict[str, dict[str, bool]] = {
    "ijepa": {"needs_two_views": False, "needs_masks": True, "has_ema": True},
    "mae": {"needs_two_views": False, "needs_masks": False, "has_ema": False},
    "simclr": {"needs_two_views": True, "needs_masks": False, "has_ema": False},
    "mocov3": {"needs_two_views": True, "needs_masks": False, "has_ema": True},
}


def build_pretrain_model(cfg: PretrainCfg) -> nn.Module:
    m, method = cfg.model, cfg.method
    common = dict(
        arch=m.arch,
        img_size=m.img_size,
        patch_size=m.patch_size,
        drop_path_rate=m.drop_path_rate,
    )

    if method == "ijepa":
        from src.model.jepa import IJEPA

        return IJEPA(pred_emb_dim=cfg.jepa.pred_emb_dim, pred_depth=cfg.jepa.pred_depth, **common)

    if method == "mae":
        from src.model.mae import MAE

        return MAE(
            mask_ratio=cfg.mae.mask_ratio,
            decoder_embed_dim=cfg.mae.decoder_embed_dim,
            decoder_depth=cfg.mae.decoder_depth,
            decoder_num_heads=cfg.mae.decoder_num_heads,
            norm_pix_loss=cfg.mae.norm_pix_loss,
            **common,
        )

    if method == "simclr":
        from src.model.simclr import SimCLR

        c = cfg.contrastive
        return SimCLR(
            proj_hidden_dim=c.proj_hidden_dim,
            proj_out_dim=c.proj_out_dim,
            proj_num_layers=c.proj_num_layers,
            temperature=c.temperature,
            **common,
        )

    if method == "mocov3":
        from src.model.mocov3 import MoCoV3

        c = cfg.contrastive
        return MoCoV3(
            proj_hidden_dim=c.proj_hidden_dim,
            proj_out_dim=c.proj_out_dim,
            proj_num_layers=c.proj_num_layers,
            pred_hidden_dim=c.pred_hidden_dim,
            temperature=c.temperature,
            freeze_patch_embed=c.freeze_patch_embed,
            symmetric_loss=c.symmetric_loss,
            **common,
        )

    raise KeyError(f"unknown method {method!r}; known: {sorted(METHOD_SPECS)}")


__all__ = ["build_pretrain_model", "METHOD_SPECS"]

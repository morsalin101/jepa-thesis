from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictor import Predictor
from .vit import ViT


class IJEPA(nn.Module):
    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 8,
        enc_dim: int = 192,
        enc_depth: int = 12,
        enc_heads: int = 3,
        enc_mlp_ratio: float = 4.0,
        pred_dim: int = 192,
        pred_depth: int = 6,
        pred_heads: int = 4,
        pred_mlp_ratio: float = 4.0,
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        if enc_dim != pred_dim:
            raise ValueError(
                f"enc_dim ({enc_dim}) must equal pred_dim ({pred_dim}); "
                "the scaffold assumes shared embedding space."
            )
        self.ema_momentum = ema_momentum

        self.context_enc = ViT(
            img_size=img_size,
            patch_size=patch_size,
            dim=enc_dim,
            depth=enc_depth,
            heads=enc_heads,
            mlp_ratio=enc_mlp_ratio,
        )
        self.target_enc = copy.deepcopy(self.context_enc)
        for p in self.target_enc.parameters():
            p.requires_grad = False

        self.predictor = Predictor(
            n_patches=self.context_enc.n_patches,
            dim=pred_dim,
            depth=pred_depth,
            heads=pred_heads,
            mlp_ratio=pred_mlp_ratio,
        )

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        m = self.ema_momentum
        for ctx_p, tgt_p in zip(self.context_enc.parameters(), self.target_enc.parameters()):
            tgt_p.data.mul_(m).add_(ctx_p.data, alpha=1.0 - m)

    def forward(
        self,
        images: torch.Tensor,
        ctx_indices: torch.Tensor,
        ctx_valid: torch.Tensor,
        tgt_indices: torch.Tensor,
        tgt_valid: torch.Tensor,
    ) -> torch.Tensor:
        ctx_emb = self.context_enc.forward_context(images, ctx_indices, ctx_valid)

        with torch.no_grad():
            tgt_emb_full = self.target_enc(images)
            tgt_emb = torch.gather(
                tgt_emb_full,
                1,
                tgt_indices.unsqueeze(-1).expand(-1, -1, tgt_emb_full.shape[-1]),
            )

        pred_emb = self.predictor(ctx_emb, ctx_valid, tgt_indices, tgt_valid)

        per = F.smooth_l1_loss(pred_emb, tgt_emb, reduction="none").mean(dim=-1)
        mask = tgt_valid.float()
        denom = mask.sum().clamp(min=1)
        return (per * mask).sum() / denom

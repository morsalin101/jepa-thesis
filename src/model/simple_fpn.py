"""ViTDet-style simple feature pyramid.

SegFormer's all-MLP head expects a 4-scale feature pyramid, but a plain ViT is
single-scale: every block outputs tokens at stride 16. The ViTDet paper's answer, which
we use here, is to build the pyramid from *one* backbone by resampling features taken
from several blocks:

    block  3  -> ConvTranspose x4 -> stride 4
    block  6  -> ConvTranspose x2 -> stride 8
    block  9  -> identity         -> stride 16
    block 12  -> MaxPool x2       -> stride 32

This is worth being precise about in the write-up: calling the result "SegFormer" without
qualification is inaccurate, because SegFormer's own encoder (MiT) is genuinely
hierarchical. The honest description is "SegFormer-style all-MLP decoder on a ViT simple
feature pyramid".
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleFeaturePyramid(nn.Module):
    """Turn a list of [B, N, D] token tensors into feature maps at strides {4,8,16,32}."""

    def __init__(self, embed_dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        out_dim = out_dim or embed_dim
        d = embed_dim

        # LayerNorm over channels for the 2-D maps. nn.LayerNorm on NCHW would normalise
        # the wrong axes, so we use GroupNorm(1, C), which is the same statistic.
        def norm(c: int) -> nn.Module:
            return nn.GroupNorm(1, c)

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(d, d // 2, 2, stride=2),
            norm(d // 2),
            nn.GELU(),
            nn.ConvTranspose2d(d // 2, d // 4, 2, stride=2),
        )
        self.up2 = nn.Sequential(nn.ConvTranspose2d(d, d // 2, 2, stride=2))
        self.id1 = nn.Identity()
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.out_channels = [d // 4, d // 2, d, d]
        self.out_dim = out_dim

    @staticmethod
    def tokens_to_map(x: torch.Tensor) -> torch.Tensor:
        """[B, N, D] -> [B, D, H, W] for a square token grid.

        `.contiguous()` is not decorative: `transpose` leaves a non-contiguous view, and
        autograd's backward for the following reshape then tries a `view` on a gradient
        whose strides straddle two subspaces, which raises. Forcing the copy here makes
        the backward well-defined.
        """
        b, n, d = x.shape
        g = int(round(n**0.5))
        if g * g != n:
            raise ValueError(f"expected a square token grid, got {n} tokens")
        return x.transpose(1, 2).contiguous().reshape(b, d, g, g)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != 4:
            raise ValueError(f"expected 4 intermediate features, got {len(features)}")
        f1, f2, f3, f4 = (self.tokens_to_map(f) for f in features)
        return [self.up4(f1), self.up2(f2), self.id1(f3), self.down2(f4)]

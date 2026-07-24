from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import ViT


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.fuse = ConvBNReLU(in_ch // 2 + skip_ch, out_ch)
        self.refine = ConvBNReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.refine(x)
        return x


class ViTUNet(nn.Module):
    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 8,
        in_chans: int = 3,
        dim: int = 192,
        depth: int = 12,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        skip_layer_indices: List[int] | None = None,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size={img_size} not divisible by patch_size={patch_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_h = img_size // patch_size
        self.n_w = img_size // patch_size
        self.dim = dim

        self.encoder = ViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
        )

        if skip_layer_indices is None:
            skip_layer_indices = [depth // 3, (2 * depth) // 3, depth - 1]
        self.skip_layer_indices = sorted(set(skip_layer_indices))

        self.decoder_dims = [dim, dim // 2, dim // 4, dim // 8]
        up_blocks = []
        for i in range(len(self.decoder_dims) - 1):
            in_ch = self.decoder_dims[i]
            out_ch = self.decoder_dims[i + 1]
            skip_ch = dim if (i < len(self.skip_layer_indices)) else 0
            up_blocks.append(UpBlock(in_ch, skip_ch, out_ch))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.head = nn.Conv2d(self.decoder_dims[-1], 1, 1)

    def encode_with_skips(self, x: torch.Tensor):
        B = x.shape[0]
        feats = self.encoder.patch_embed(x) + self.encoder.pos_embed
        skip_feats: dict[int, torch.Tensor] = {}
        for i, block in enumerate(self.encoder.blocks):
            feats = block(feats)
            if i in self.skip_layer_indices:
                H = W = int(feats.shape[1] ** 0.5)
                sf = feats.transpose(1, 2).reshape(B, self.dim, H, W)
                skip_feats[i] = sf
        feats = self.encoder.norm(feats)
        H = W = int(feats.shape[1] ** 0.5)
        bottleneck = feats.transpose(1, 2).reshape(B, self.dim, H, W)
        return bottleneck, [skip_feats[i] for i in sorted(skip_feats)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skips = self.encode_with_skips(x)
        feat = bottleneck
        skip_iter = iter(skips)
        for i, up in enumerate(self.up_blocks):
            skip = None
            if i < len(skips):
                skip = next(skip_iter)
            feat = up(feat, skip)
        return self.head(feat)

    def load_encoder_from_jepa(self, ckpt_path: str | bytes) -> None:
        sd = torch.load(ckpt_path, map_location="cpu")
        if "context_enc" in sd:
            sd = sd["context_enc"]
        missing, unexpected = self.encoder.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"[unet] warning: unexpected keys ignored: {unexpected}")
        if missing:
            print(f"[unet] warning: missing keys: {missing}")

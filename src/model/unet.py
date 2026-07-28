"""ViT-encoder UNet decoder — kept purely as a decoder ablation.

The primary decoder is the SegFormer all-MLP head in `segformer_head.py`. This one exists
so the thesis can report a decoder-ablation row: does the encoder ranking (I-JEPA vs MAE
vs SimCLR vs MoCo v3) survive a change of decoder? If it does, the result is about the
*representations*, which is the claim being made. That is worth one extra table row.

It exposes the same interface as `ViTSegFormer` (`load_pretrained_encoder`,
`param_groups`) so `segment.py` can swap decoders without branching.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.vit import VisionTransformer, build_vit


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
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
        return self.refine(self.fuse(x))


class ViTUNet(nn.Module):
    def __init__(
        self,
        arch: str = "vit_small",
        img_size: int = 352,
        patch_size: int = 16,
        skip_layers: tuple[int, ...] = (2, 5, 8),
        num_classes: int = 1,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder: VisionTransformer = build_vit(
            arch, img_size=img_size, patch_size=patch_size, drop_path_rate=drop_path_rate
        )
        d = self.encoder.embed_dim
        self.skip_layers = list(skip_layers)

        # Skips are consumed deepest-first, so reverse the (ascending) block order.
        dims = [d, d // 2, d // 4, d // 8]
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(dims[i], d if i < len(self.skip_layers) else 0, dims[i + 1])
                for i in range(len(dims) - 1)
            ]
        )
        self.head = nn.Conv2d(dims[-1], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        bottleneck, skips = self.encoder(x, return_intermediates=self.skip_layers)
        b, n, d = bottleneck.shape
        g = int(round(n**0.5))
        y = bottleneck.transpose(1, 2).reshape(b, d, g, g)
        maps = [s.transpose(1, 2).reshape(b, d, g, g) for s in skips][::-1]

        for i, blk in enumerate(self.up_blocks):
            y = blk(y, maps[i] if i < len(maps) else None)
        return F.interpolate(self.head(y), size=size, mode="bilinear", align_corners=False)

    def load_pretrained_encoder(self, ckpt_path: str, strict_report: bool = True) -> dict:
        from src.model.segformer_head import ViTSegFormer

        return ViTSegFormer.load_pretrained_encoder(self, ckpt_path, strict_report)

    def decoder_modules(self) -> list[nn.Module]:
        return [self.up_blocks, self.head]

    def param_groups(self, enc_lr: float, dec_lr: float, layer_decay: float = 0.75) -> list[dict]:
        from src.model.segformer_head import layerwise_param_groups

        return layerwise_param_groups(
            self.encoder, self.decoder_modules(), enc_lr, dec_lr, layer_decay
        )

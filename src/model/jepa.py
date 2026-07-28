"""I-JEPA: encoder + predictor + EMA target encoder.

The objective, in one line: encode a large *context* block, predict the representations
of four disjoint *target* blocks, and compare against an exponential-moving-average copy
of the same encoder run on the full image.

Two details here are load-bearing and both were missing from the earlier scaffold:

1. **`F.layer_norm` on the target encoder output, before masking.** Without it the model
   can trivially minimise the loss by shrinking the target representations toward a
   constant — representation collapse. The LayerNorm removes the scale degree of freedom
   the collapse would exploit. This is the single most important line in the file.
2. **The EMA update runs in fp32, outside autocast.** With momentum 0.9995 the update
   term is `(1-m) = 5e-4` times a parameter of order 1e-2. In fp16 that rounds to zero,
   so the target encoder silently stops tracking and the loss flatlines at a plausible
   value. Nothing errors; the run is simply meaningless. Hence the explicit no_grad +
   float path.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.masks.utils import apply_masks, repeat_interleave_batch
from src.model.predictor import VisionTransformerPredictor
from src.model.vit import VisionTransformer, build_vit


class IJEPA(nn.Module):
    def __init__(
        self,
        arch: str = "vit_small",
        img_size: int = 224,
        patch_size: int = 16,
        pred_emb_dim: int = 192,
        pred_depth: int = 6,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder: VisionTransformer = build_vit(
            arch, img_size=img_size, patch_size=patch_size, drop_path_rate=drop_path_rate
        )
        self.predictor = VisionTransformerPredictor(
            num_patches=self.encoder.patch_embed.num_patches,
            embed_dim=self.encoder.embed_dim,
            predictor_embed_dim=pred_emb_dim,
            depth=pred_depth,
            num_heads=self.encoder.num_heads,
        )
        # Target encoder starts as an exact copy and is only ever updated by EMA.
        self.target_encoder: VisionTransformer = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, m: float) -> None:
        """EMA: target <- m * target + (1 - m) * online.

        Runs in fp32 regardless of the surrounding autocast context — see module
        docstring for why that is not optional.
        """
        for q, k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            k.data.mul_(m).add_(q.detach().data.to(k.dtype), alpha=1.0 - m)

    def forward_target(
        self, imgs: torch.Tensor, masks_pred: list[torch.Tensor], n_enc: int
    ) -> torch.Tensor:
        with torch.no_grad():
            h = self.target_encoder(imgs)
            h = F.layer_norm(h, (h.size(-1),))  # anti-collapse; see module docstring
            B = len(h)
            h = apply_masks(h, masks_pred)
            return repeat_interleave_batch(h, B, repeat=n_enc)

    def forward_context(
        self,
        imgs: torch.Tensor,
        masks_enc: list[torch.Tensor],
        masks_pred: list[torch.Tensor],
    ) -> torch.Tensor:
        z = self.encoder(imgs, masks_enc)
        return self.predictor(z, masks_enc, masks_pred)

    def forward(
        self,
        imgs: torch.Tensor,
        masks_enc: list[torch.Tensor],
        masks_pred: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.forward_target(imgs, masks_pred, n_enc=len(masks_enc))
        z = self.forward_context(imgs, masks_enc, masks_pred)
        return z, h

    @staticmethod
    def loss(z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Smooth L1 in fp32.

        The loss is order 1e-2 and fp16 has ~3 decimal digits there, so accumulating it
        in half precision throws away most of the gradient signal.
        """
        return F.smooth_l1_loss(z.float(), h.float())

    def checkpoint_modules(self) -> dict[str, nn.Module]:
        return {"model": self}

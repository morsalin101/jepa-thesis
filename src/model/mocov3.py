"""MoCo v3 (Chen et al., 2021) with a ViT backbone.

The fairest contrastive comparison for I-JEPA, because it also uses an EMA target
encoder — so any difference between the two is attributable to the *objective*
(contrastive instance discrimination vs latent block prediction) rather than to the
presence or absence of a momentum branch.

Also the most expensive method in the study: two full views through the query encoder
*with* gradients, plus two more through the momentum encoder without, plus a symmetrised
loss. About 2.4x I-JEPA and 4.8x MAE per image.

Three details that matter:

* **Frozen random patch embedding.** The paper's own fix for ViT training instability:
  the patch-embed conv is initialised randomly and never updated. Without it MoCo v3 on
  a ViT is prone to loss spikes and partial collapse, and fp16 makes that worse.
* **The momentum update runs in fp32, outside autocast.** With m -> 1.0 the update term
  `(1-m)` falls below fp16 resolution relative to the parameter magnitude, so the key
  encoder silently stops tracking the query encoder and the loss plateaus at a plausible
  value. Nothing errors. The `momentum_update` method below is deliberately not
  autocast-wrapped.
* **InfoNCE in fp32**, for the same overflow reason as SimCLR.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.heads import PooledBackbone, build_mlp
from src.model.vit import build_vit
from src.utils.ddp import gather_with_grad


class MoCoV3(nn.Module):
    def __init__(
        self,
        arch: str = "vit_small",
        img_size: int = 224,
        patch_size: int = 16,
        proj_hidden_dim: int = 2048,
        proj_out_dim: int = 256,
        proj_num_layers: int = 3,
        pred_hidden_dim: int = 2048,
        temperature: float = 0.2,
        freeze_patch_embed: bool = True,
        symmetric_loss: bool = True,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = build_vit(
            arch, img_size=img_size, patch_size=patch_size, drop_path_rate=drop_path_rate
        )
        embed_dim = self.encoder.embed_dim
        self.backbone = PooledBackbone(self.encoder)
        self.projector = build_mlp(
            embed_dim, proj_hidden_dim, proj_out_dim, proj_num_layers, last_bn=True
        )
        # Asymmetric predictor on the query branch only — this asymmetry is what stops
        # the two branches collapsing onto a constant.
        self.predictor = build_mlp(proj_out_dim, pred_hidden_dim, proj_out_dim, 2, last_bn=False)

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)
        for p in list(self.target_encoder.parameters()) + list(self.target_projector.parameters()):
            p.requires_grad = False
        self.target_backbone = PooledBackbone(self.target_encoder)

        self.temperature = temperature
        self.symmetric_loss = symmetric_loss

        if freeze_patch_embed:
            for p in self.encoder.patch_embed.parameters():
                p.requires_grad = False
            for p in self.target_encoder.patch_embed.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def momentum_update(self, m: float) -> None:
        """key <- m * key + (1 - m) * query, in fp32. See module docstring."""
        for q, k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            k.data.mul_(m).add_(q.detach().data.to(k.dtype), alpha=1.0 - m)
        for q, k in zip(self.projector.parameters(), self.target_projector.parameters()):
            k.data.mul_(m).add_(q.detach().data.to(k.dtype), alpha=1.0 - m)
        for q, k in zip(self.projector.buffers(), self.target_projector.buffers()):
            k.data.copy_(q.data)  # BatchNorm running stats are copied, not averaged

    def _query(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.predictor(self.projector(self.backbone(x))), dim=-1)

    @torch.no_grad()
    def _key(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.target_projector(self.target_backbone(x)), dim=-1)

    def infonce(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Contrast each query against all gathered keys, in fp32."""
        with torch.autocast(device_type=q.device.type, enabled=False):
            q = q.float()
            k = gather_with_grad(k.float())
            logits = (q @ k.T) / self.temperature
            # Positives sit on the diagonal of this rank's slice of the gathered keys.
            from src.utils.ddp import get_rank

            offset = get_rank() * q.shape[0]
            labels = torch.arange(q.shape[0], device=q.device) + offset
            # 2*tau matches the paper's scaling, which keeps the loss magnitude
            # comparable across temperature choices.
            return F.cross_entropy(logits, labels) * (2 * self.temperature)

    def forward(self, views: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        v1, v2 = views
        q1, q2 = self._query(v1), self._query(v2)
        k1, k2 = self._key(v1), self._key(v2)
        if self.symmetric_loss:
            return self.infonce(q1, k2) + self.infonce(q2, k1)
        return self.infonce(q1, k2)

    def checkpoint_modules(self) -> dict[str, nn.Module]:
        return {"model": self}

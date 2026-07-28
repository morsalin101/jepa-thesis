"""SimCLR (Chen et al., 2020) with a ViT backbone.

Two augmented views per image, NT-Xent over the batch. Everything expensive about this
method is structural: both views run through the encoder *with gradients*, so a step
costs roughly twice a supervised step — about 1.8x I-JEPA and 3.6x MAE.

Two implementation points that are easy to get wrong and silent when you do:

* **Embeddings are all-gathered across ranks with gradient flow** (`gather_with_grad`).
  Plain `dist.all_gather` detaches, which would quietly reduce the negatives each rank
  sees from `global_batch - 1` to `per_gpu_batch - 1` — at global batch 512 on 2 GPUs,
  255 instead of 511. The loss still trains; it is just a weaker method than reported.
* **The logits are computed in fp32.** Under fp16 autocast, `exp()` on the similarity
  matrix overflows to inf and the loss becomes NaN or silently degenerate. This is the
  single most common half-precision bug in contrastive code.

Budget caveat for the thesis: SimCLR benefits substantially from batches far larger than
512. Holding global batch equal across all four methods is the right *fairness* call
(equal optimisation budget) but it does disadvantage SimCLR, and that should be
disclosed rather than buried.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.heads import PooledBackbone, build_mlp
from src.model.vit import build_vit
from src.utils.ddp import gather_with_grad, get_rank, get_world_size


class SimCLR(nn.Module):
    def __init__(
        self,
        arch: str = "vit_small",
        img_size: int = 224,
        patch_size: int = 16,
        proj_hidden_dim: int = 2048,
        proj_out_dim: int = 256,
        proj_num_layers: int = 3,
        temperature: float = 0.1,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = build_vit(
            arch, img_size=img_size, patch_size=patch_size, drop_path_rate=drop_path_rate
        )
        self.backbone = PooledBackbone(self.encoder)
        self.projector = build_mlp(
            self.encoder.embed_dim, proj_hidden_dim, proj_out_dim, proj_num_layers
        )
        self.temperature = temperature

    target_encoder = None

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.backbone(x)), dim=-1)

    def nt_xent(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xent over the *global* batch, computed in fp32."""
        with torch.autocast(device_type=z1.device.type, enabled=False):
            z1, z2 = z1.float(), z2.float()
            local_n = z1.shape[0]

            g1, g2 = gather_with_grad(z1), gather_with_grad(z2)
            z = torch.cat([g1, g2], dim=0)  # [2N, D]
            n = g1.shape[0]

            sim = (z @ z.T) / self.temperature
            sim.fill_diagonal_(float("-inf"))  # never contrast a sample with itself

            # Positive for row i in [0,N) is row i+N, and vice versa.
            targets = torch.cat(
                [torch.arange(n, 2 * n, device=z.device), torch.arange(0, n, device=z.device)]
            )

            # Only this rank's rows contribute to the loss; the rest are negatives whose
            # gradients arrive through the gather's backward. Averaging every row on every
            # rank would count each sample world_size times.
            rank, world = get_rank(), get_world_size()
            if world > 1:
                idx = torch.cat(
                    [
                        torch.arange(rank * local_n, (rank + 1) * local_n, device=z.device),
                        torch.arange(n + rank * local_n, n + (rank + 1) * local_n, device=z.device),
                    ]
                )
                sim, targets = sim[idx], targets[idx]

            return F.cross_entropy(sim, targets)

    def forward(self, views: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        v1, v2 = views
        return self.nt_xent(self.embed(v1), self.embed(v2))

    def checkpoint_modules(self) -> dict[str, nn.Module]:
        return {"model": self}

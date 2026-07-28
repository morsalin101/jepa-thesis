"""Projection and prediction heads shared by the contrastive baselines."""
from __future__ import annotations

import torch
import torch.nn as nn


def build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 3,
    use_bn: bool = True,
    last_bn: bool = False,
) -> nn.Sequential:
    """MLP head with BatchNorm between layers.

    BatchNorm in the projector is not incidental — SimCLR and MoCo v3 both report clear
    drops without it. `last_bn` (an affine-free BN on the output) is MoCo v3's variant.
    """
    layers: list[nn.Module] = []
    d = in_dim
    for i in range(num_layers - 1):
        layers.append(nn.Linear(d, hidden_dim, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim, bias=not last_bn))
    if last_bn:
        layers.append(nn.BatchNorm1d(out_dim, affine=False))
    return nn.Sequential(*layers)


class PooledBackbone(nn.Module):
    """ViT + mean pooling over patch tokens.

    These models have no class token, so the sequence has to be reduced some other way.
    Mean pooling is the standard choice and is what the segmentation decoder's features
    are consistent with.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    @property
    def embed_dim(self) -> int:
        return self.encoder.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).mean(dim=1)

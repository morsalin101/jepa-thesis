from __future__ import annotations

import torch
import torch.nn as nn

from .vit import Block


class Predictor(nn.Module):
    def __init__(
        self,
        n_patches: int,
        dim: int = 192,
        depth: int = 6,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_patches = n_patches
        self.dim = dim

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [Block(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        context_emb: torch.Tensor,
        ctx_valid: torch.Tensor,
        tgt_indices: torch.Tensor,
        tgt_valid: torch.Tensor,
    ) -> torch.Tensor:
        B = context_emb.shape[0]
        n_tgt = tgt_indices.shape[1]

        tgt_pos = self.pos_embed.expand(B, -1, -1).gather(
            1, tgt_indices.unsqueeze(-1).expand(-1, -1, self.dim)
        )
        mask_tokens = self.mask_token.expand(B, n_tgt, -1) + tgt_pos

        valid = torch.cat([ctx_valid, tgt_valid], dim=1)
        seq = torch.cat([context_emb, mask_tokens], dim=1)
        kpm = ~valid

        for block in self.blocks:
            seq = block(seq, key_padding_mask=kpm)
        seq = self.norm(seq)

        n_ctx = context_emb.shape[1]
        return seq[:, n_ctx:]

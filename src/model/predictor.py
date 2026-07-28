"""I-JEPA predictor.

Takes the context tokens the encoder produced and predicts the *representations* of the
target blocks — never pixels. That is the whole point of JEPA versus MAE.

Ported from `VisionTransformerPredictor` in
reference/ijepa/src/models/vision_transformer.py. The pieces the previous version of
this project was missing, and why each matters:

* **`predictor_embed` / `predictor_proj`.** The predictor runs in its own narrower width
  (192 here vs the encoder's 384) with a Linear in and out. Forcing the two widths equal
  instead, as the old code did, makes the predictor as expensive as the encoder and
  removes the bottleneck that stops it from learning an identity shortcut.
* **Predictor position embeddings added to the context tokens** (`x += apply_masks(...)`).
  Without them the predictor cannot tell *where* the context patches came from, so it
  cannot know the spatial relationship between context and target — which is the only
  signal it has.
* **Its own sin-cos position table**, separate from the encoder's.
"""
from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn

from src.masks.utils import apply_masks, repeat_interleave_batch
from src.model.vit import Block, get_2d_sincos_pos_embed, trunc_normal_


class VisionTransformerPredictor(nn.Module):
    def __init__(
        self,
        num_patches: int,
        embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.init_std = init_std
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
        )
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(
                get_2d_sincos_pos_embed(predictor_embed_dim, int(round(num_patches**0.5)))
            )
            .float()
            .unsqueeze(0)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    predictor_embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self) -> None:
        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        masks_x: list[torch.Tensor],
        masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        :param x: encoder output for the context tokens, [n_enc*B, K_ctx, embed_dim]
        :param masks_x: the context mask indices (to look up their positions)
        :param masks: the target mask indices (what to predict)
        :return: predicted target representations, [n_pred*n_enc*B, K_tgt, embed_dim]
        """
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks = [masks]

        B = len(x) // len(masks_x)

        x = self.predictor_embed(x)

        # Tell the predictor where the context patches live in the image.
        x_pos = self.predictor_pos_embed.repeat(B, 1, 1)
        x = x + apply_masks(x_pos, masks_x)
        _, n_ctxt, _ = x.shape

        # Query tokens: a shared learned token plus the position of what to predict.
        pos = self.predictor_pos_embed.repeat(B, 1, 1)
        pos = apply_masks(pos, masks)
        pos = repeat_interleave_batch(pos, B, repeat=len(masks_x))
        pred_tokens = self.mask_token.repeat(pos.size(0), pos.size(1), 1) + pos

        # One copy of the context per target block, so each target attends to all of it.
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        return self.predictor_proj(x[:, n_ctxt:])


def vit_predictor(**kwargs) -> VisionTransformerPredictor:
    return VisionTransformerPredictor(mlp_ratio=4, qkv_bias=True, **kwargs)

"""Vision Transformer backbone.

Follows reference/ijepa/src/models/vision_transformer.py: no class token, fixed 2-D
sin-cos position embeddings, `fix_init_weight` residual rescaling, and a `masks`
argument that drops tokens *before* the blocks run (so the context encoder really does
cost less than a full forward pass).

Three deliberate changes from the reference, each for a concrete reason:

* **`F.scaled_dot_product_attention`** instead of an explicit softmax matmul. Same math,
  but it picks a memory-efficient kernel and cuts activation memory noticeably on a
  16 GB T4. The explicit path is kept behind `return_attention` for the attention-map
  figure, since SDPA does not expose the weights.
* **A correct `interpolate_pos_encoding`.** The reference version assumes a class token
  (`npatch = x.shape[1] - 1`) that these models do not have, so it mis-indexes. We need
  this working for real: segmentation fine-tunes at 352px, i.e. a 14x14 -> 22x22
  position-embedding stretch.
* **`return_intermediates`**, so the segmentation decoder can tap blocks {3,6,9,12}
  without a forward hook.
"""
from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.masks.utils import apply_masks

# --------------------------------------------------------------- position embeds


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """[grid_size**2, embed_dim] 2-D sin-cos table. Width varies fastest."""
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


# ------------------------------------------------------------------ layers


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rand = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep) * rand.floor_()


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if return_attention:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            return attn.softmax(dim=-1)

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if return_attention:
            return self.attn(self.norm1(x), return_attention=True)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


def trunc_normal_(tensor: torch.Tensor, mean=0.0, std=1.0, a=-2.0, b=2.0) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


# ------------------------------------------------------------------ backbone


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
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
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.patch_size = patch_size
        self.init_std = init_std

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Fixed, not learned: sin-cos generalises to unseen grid sizes, which is what
        # makes the 224 -> 352 transfer at fine-tuning time well-posed.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        self.pos_embed.data.copy_(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, self.patch_embed.grid_size))
            .float()
            .unsqueeze(0)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
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
        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self) -> None:
        """Scale down deeper residual branches by 1/sqrt(2*layer_id).

        Keeps the residual stream's variance from growing with depth, which is what lets
        a 12-block ViT train without extra warmup tricks. Same as the reference.
        """

        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
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
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def interpolate_pos_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Bicubically resize the position table to match x's token count.

        No class token to skip, unlike the reference implementation this is ported from.
        """
        npatch = x.shape[1]
        n = self.pos_embed.shape[1]
        if npatch == n:
            return self.pos_embed
        dim = self.pos_embed.shape[-1]
        src = int(round(math.sqrt(n)))
        dst = int(round(math.sqrt(npatch)))
        if src * src != n or dst * dst != npatch:
            raise ValueError(f"non-square token grids not supported: {n} -> {npatch}")
        pos = self.pos_embed.reshape(1, src, src, dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(dst, dst), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, dst * dst, dim)

    def forward(
        self,
        x: torch.Tensor,
        masks: list[torch.Tensor] | torch.Tensor | None = None,
        return_intermediates: list[int] | None = None,
    ):
        """
        :param masks: index tensors of patches to keep. Applied after the position
            embedding, so masked-out tokens never enter the blocks at all.
        :param return_intermediates: 0-indexed block outputs to also return, for the
            segmentation feature pyramid.
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        x = self.patch_embed(x)
        x = x + self.interpolate_pos_encoding(x)
        if masks is not None:
            x = apply_masks(x, masks)

        wanted = set(return_intermediates or [])
        intermediates: list[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in wanted:
                intermediates.append(x)

        x = self.norm(x)
        if return_intermediates:
            return x, intermediates
        return x

    def get_last_selfattention(self, x: torch.Tensor) -> torch.Tensor:
        """Attention weights of the final block — used by the attention-map figure."""
        x = self.patch_embed(x)
        x = x + self.interpolate_pos_encoding(x)
        for blk in self.blocks[:-1]:
            x = blk(x)
        return self.blocks[-1](x, return_attention=True)


# ------------------------------------------------------------------ factories

def vit_tiny(patch_size=16, **kw):
    return VisionTransformer(patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, **kw)


def vit_small(patch_size=16, **kw):
    return VisionTransformer(patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, **kw)


def vit_base(patch_size=16, **kw):
    return VisionTransformer(patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, **kw)


def vit_large(patch_size=16, **kw):
    return VisionTransformer(patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, **kw)


VIT_FACTORY = {
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
}


def build_vit(arch: str, img_size: int, patch_size: int, **kw) -> VisionTransformer:
    if arch not in VIT_FACTORY:
        raise KeyError(f"unknown arch {arch!r}; known: {sorted(VIT_FACTORY)}")
    return VIT_FACTORY[arch](patch_size=patch_size, img_size=img_size, **kw)

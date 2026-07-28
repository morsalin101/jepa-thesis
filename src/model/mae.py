"""Masked Autoencoder (He et al., 2021) — the most important baseline here.

I-JEPA's central claim is that predicting in *latent* space beats predicting in *pixel*
space, because pixel reconstruction forces the encoder to keep high-frequency detail that
carries no semantics. MAE is the pixel-space arm of exactly that comparison, so if only
one baseline survives a budget cut, this is the one to keep.

Architecture matches the paper: asymmetric, with the encoder seeing only the 25% visible
patches and a shallow narrow decoder reconstructing all of them. Decoder width is 256 =
0.67 x 384, preserving MAE's own 512/768 decoder/encoder ratio for ViT-B rather than
copying the absolute 512.

Budget caveat to state in the thesis: MAE is known to keep improving out to 800-1600
epochs, while contrastive methods saturate earlier. A fixed 100-epoch budget therefore
*structurally disadvantages* MAE. That is a limitation to disclose, not to hide.
"""
from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from src.model.vit import (
    Block,
    VisionTransformer,
    build_vit,
    get_2d_sincos_pos_embed,
    trunc_normal_,
)


class MAE(nn.Module):
    def __init__(
        self,
        arch: str = "vit_small",
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 8,
        decoder_num_heads: int = 8,
        norm_pix_loss: bool = True,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder: VisionTransformer = build_vit(
            arch, img_size=img_size, patch_size=patch_size, drop_path_rate=drop_path_rate
        )
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        embed_dim = self.encoder.embed_dim
        num_patches = self.encoder.patch_embed.num_patches
        grid = self.encoder.patch_embed.grid_size

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(get_2d_sincos_pos_embed(decoder_embed_dim, grid)).float().unsqueeze(0)
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.decoder_blocks = nn.ModuleList(
            [
                Block(decoder_embed_dim, decoder_num_heads, 4.0, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        trunc_normal_(self.mask_token, std=0.02)
        self.decoder_blocks.apply(self._init_weights)
        self._init_weights(self.decoder_embed)
        self._init_weights(self.decoder_pred)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------ patch helpers

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """[B,C,H,W] -> [B, L, p*p*C]"""
        p, c = self.patch_size, self.in_chans
        b, _, h, w = imgs.shape
        gh, gw = h // p, w // p
        x = imgs.reshape(b, c, gh, p, gw, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(b, gh * gw, p * p * c)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, p*p*C] -> [B,C,H,W]"""
        p, c = self.patch_size, self.in_chans
        b, ln, _ = x.shape
        g = int(round(ln**0.5))
        x = x.reshape(b, g, g, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(b, c, g * p, g * p)

    # ---------------------------------------------------------------- masking

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """Per-sample random shuffle, keep the first (1-mask_ratio) fraction.

        Returns (kept tokens, binary mask with 1 = removed, ids_restore) — the restore
        indices are what let the decoder put mask tokens back in the right places.
        """
        b, ln, d = x.shape
        keep = int(ln * (1 - mask_ratio))
        noise = torch.rand(b, ln, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))

        mask = torch.ones(b, ln, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    # ---------------------------------------------------------------- forward

    def forward_encoder(self, imgs: torch.Tensor, mask_ratio: float):
        enc = self.encoder
        x = enc.patch_embed(imgs)
        x = x + enc.interpolate_pos_encoding(x)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        for blk in enc.blocks:
            x = blk(x)
        return enc.norm(x), mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)
        b, _, d = x.shape
        n_missing = ids_restore.shape[1] - x.shape[1]
        mask_tokens = self.mask_token.expand(b, n_missing, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, d))  # unshuffle
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        return self.decoder_pred(self.decoder_norm(x))

    def loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """MSE on removed patches only, in fp32.

        `norm_pix_loss` normalises each target patch by its own mean/std, which the paper
        found materially improves representation quality — it stops the loss being
        dominated by locally bright or high-contrast patches.
        """
        target = self.patchify(imgs).float()
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred.float() - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum().clamp(min=1)

    def forward(self, imgs: torch.Tensor, mask_ratio: float | None = None):
        mr = self.mask_ratio if mask_ratio is None else mask_ratio
        latent, mask, ids_restore = self.forward_encoder(imgs, mr)
        pred = self.forward_decoder(latent, ids_restore)
        return self.loss(imgs, pred, mask), pred, mask

    # MAE has no EMA target encoder — the exported weights are the online encoder.
    target_encoder = None

    def checkpoint_modules(self) -> dict[str, nn.Module]:
        return {"model": self}

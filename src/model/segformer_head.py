"""SegFormer all-MLP decoder (Xie et al., NeurIPS 2021).

Deliberately minimal: unify every pyramid level to one width with a 1x1 Linear, upsample
all of them to the finest stride, concatenate, fuse with one more MLP, and predict. No
attention, no heavy conv stack — about 2.4M parameters on top of a ViT-S encoder, which
is what makes "lightweight transformer-based decoder" an accurate description.

The same head, with the same initialisation, is used for all five encoders (I-JEPA, MAE,
SimCLR, MoCo v3, random init). That identity is the entire basis of the comparison, so it
is enforced in code rather than left to discipline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.simple_fpn import SimpleFeaturePyramid
from src.model.vit import VisionTransformer, build_vit


class MLPProject(nn.Module):
    """1x1 projection implemented as a Linear over flattened spatial positions."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        # See SimpleFeaturePyramid.tokens_to_map: reshape-after-transpose needs an
        # explicit copy or the backward pass hits a non-contiguous view.
        return self.proj(x).transpose(1, 2).contiguous().reshape(b, -1, h, w)


class SegFormerHead(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        embed_dim: int = 256,
        num_classes: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList([MLPProject(c, embed_dim) for c in in_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[2:]  # finest level, stride 4
        outs = []
        for proj, f in zip(self.projections, features):
            y = proj(f)
            if y.shape[2:] != target_size:
                y = F.interpolate(y, size=target_size, mode="bilinear", align_corners=False)
            outs.append(y)
        x = self.fuse(torch.cat(outs, dim=1))
        return self.classifier(self.dropout(x))


class ViTSegFormer(nn.Module):
    """ViT encoder + simple feature pyramid + SegFormer all-MLP head."""

    def __init__(
        self,
        arch: str = "vit_small",
        img_size: int = 352,
        patch_size: int = 16,
        fpn_layers: tuple[int, int, int, int] = (2, 5, 8, 11),
        decoder_embed_dim: int = 256,
        num_classes: int = 1,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder: VisionTransformer = build_vit(
            arch, img_size=img_size, patch_size=patch_size, drop_path_rate=drop_path_rate
        )
        self.fpn_layers = list(fpn_layers)
        depth = self.encoder.depth
        bad = [i for i in self.fpn_layers if not 0 <= i < depth]
        if bad:
            raise ValueError(f"fpn_layers {bad} out of range for a {depth}-block encoder")
        self.pyramid = SimpleFeaturePyramid(self.encoder.embed_dim)
        self.head = SegFormerHead(self.pyramid.out_channels, decoder_embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        _, intermediates = self.encoder(x, return_intermediates=self.fpn_layers)
        logits = self.head(self.pyramid(intermediates))
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)

    # ------------------------------------------------------------- pretrained

    def load_pretrained_encoder(self, ckpt_path: str, strict_report: bool = True) -> dict:
        """Load exported SSL encoder weights.

        Position embeddings are a fixed sin-cos buffer regenerated at construction for
        the current image size, so a 224-pretrained checkpoint transfers to a 352 model
        without touching them — `interpolate_pos_encoding` handles any residual mismatch
        at forward time.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("encoder", ckpt)
        state = {k: v for k, v in state.items() if not k.endswith("pos_embed")}

        # `strict=False` forgives missing/unexpected keys but still hard-errors on a
        # shape mismatch, and PyTorch's message names a tensor rather than the cause.
        # Catch it here and say what actually differs.
        own = self.encoder.state_dict()
        shape_mismatch = {
            k: (tuple(v.shape), tuple(own[k].shape))
            for k, v in state.items()
            if k in own and v.shape != own[k].shape
        }
        if shape_mismatch:
            hint = ""
            if any("patch_embed" in k for k in shape_mismatch):
                got, want = next(v for k, v in shape_mismatch.items() if "patch_embed" in k)
                hint = (
                    f" The patch embedding differs ({got[-1]}px vs {want[-1]}px), so the "
                    f"checkpoint was pretrained with patch_size={got[-1]} but this model "
                    f"uses patch_size={want[-1]}."
                )
            raise RuntimeError(
                f"encoder weights are incompatible with this model: "
                f"{len(shape_mismatch)} tensor(s) differ in shape, e.g. "
                f"{list(shape_mismatch.items())[:2]}.{hint} "
                f"Rebuild the segmentation model with the pretraining arch/patch_size, "
                f"or point --pretrained-ckpt at a matching encoder. ({ckpt_path})"
            )

        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        real_missing = [k for k in missing if not k.endswith("pos_embed")]
        report = {
            "source": ckpt_path,
            "source_module": ckpt.get("source_module"),
            "run_id": ckpt.get("run_id"),
            "loaded": len(state),
            "missing": real_missing,
            "unexpected": list(unexpected),
        }
        if strict_report and (real_missing or unexpected):
            raise RuntimeError(
                f"encoder weights do not match the model: missing={real_missing[:5]} "
                f"unexpected={list(unexpected)[:5]}. Check that arch/patch_size agree "
                f"with the pretraining config. ({ckpt_path})"
            )
        return report

    def decoder_modules(self) -> list[nn.Module]:
        return [self.pyramid, self.head]

    def param_groups(self, enc_lr: float, dec_lr: float, layer_decay: float = 0.75) -> list[dict]:
        return layerwise_param_groups(
            self.encoder, self.decoder_modules(), enc_lr, dec_lr, layer_decay
        )


def layerwise_param_groups(
    encoder: VisionTransformer,
    decoder_modules: list[nn.Module],
    enc_lr: float,
    dec_lr: float,
    layer_decay: float = 0.75,
) -> list[dict]:
    """Layer-wise LR decay for the encoder, flat LR for the decoder.

    Earlier blocks encode generic structure and are worth preserving; later blocks need
    to adapt to the task. `lr_scale` is what `ScheduleSet.apply` multiplies the scheduled
    LR by, so the decay survives the cosine schedule. Because the schedule's reference LR
    is `enc_lr`, the decoder's scale is `dec_lr / enc_lr` — every group's live LR is
    therefore `scheduled_lr * lr_scale`, and the `lr` field below is just its value at
    step 0.

    Groups are built from *all* parameters, not from `p.requires_grad`. Building them
    from `requires_grad` while a module is frozen produces an empty group, and unfreezing
    later then does nothing because those tensors were never registered with the
    optimizer — a silent no-op that is easy to ship and hard to notice.
    """
    depth = encoder.depth
    groups: list[dict] = []
    seen: set[int] = set()

    def split(named):
        decay, no_decay = [], []
        for n, p in named:
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            (no_decay if (p.ndim <= 1 or n.endswith(".bias")) else decay).append(p)
        return decay, no_decay

    def add(params, scale, wd_exclude):
        if params:
            groups.append(
                {
                    "params": params,
                    "lr": enc_lr * scale,
                    "lr_scale": scale,
                    **({"WD_exclude": True, "weight_decay": 0.0} if wd_exclude else {}),
                }
            )

    def emit(named, scale):
        d, nd = split(named)
        add(d, scale, False)
        add(nd, scale, True)

    # Patch embedding behaves as layer 0 — the most generic, so the most decayed.
    emit(list(encoder.patch_embed.named_parameters()), layer_decay**depth)
    for i, blk in enumerate(encoder.blocks):
        emit(list(blk.named_parameters()), layer_decay ** (depth - i - 1))
    emit(list(encoder.norm.named_parameters()), 1.0)

    dec_scale = (dec_lr / enc_lr) if enc_lr else 1.0
    dec_named: list = []
    for m in decoder_modules:
        dec_named += list(m.named_parameters())
    emit(dec_named, dec_scale)
    return groups

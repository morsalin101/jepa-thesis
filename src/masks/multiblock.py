"""I-JEPA multi-block masking.

This is the part of I-JEPA that actually carries the method, and it is where the
previous version of this project diverged most: it sampled **one** target block and used
its exact set-complement as context. The paper samples **four** target blocks and a
**separately sampled, large** context block (85-100% of the image) from which the union
of the targets is then removed. Predicting four scattered blocks from a large,
independently-chosen context is what forces semantic rather than local-texture features.

Ported from reference/ijepa/src/masks/multiblock.py, preserving its behaviour including
two quirks worth knowing about:

* `_sample_block_size` draws **one** random number and uses it for *both* the area scale
  and the aspect ratio, so the two are perfectly correlated within a call. That looks
  like a bug, but it is what the released models were trained with, so we keep it.
* Block *sizes* are drawn once per batch from a seeded generator (shared across ranks
  via a step counter, so DDP ranks agree), while block *positions* use the global RNG
  and therefore differ per image.

One **deliberate deviation** from the reference: it draws positions with
`torch.randint(0, height - h)`, whose exclusive upper bound means a block can never
touch the bottom or right edge of the grid. We use `height - h + 1`, so every valid
position is reachable. On ImageNet that bias is cosmetic; on endoscopy it is not, since
polyps sit against the frame edge often enough to matter.

The batch-shared size plus truncation to the batch-wide `min_keep` is what lets masks
collate into rectangular tensors with no padding — the reason this implementation needs
no `key_padding_mask` anywhere.
"""
from __future__ import annotations

import math
from multiprocessing import Value

import torch


class MaskCollator:
    """Collate function producing (images, enc_masks, pred_masks) per batch.

    Returned masks are lists of index tensors:
      * `enc_masks`:  `num_enc_masks` tensors of shape [B, K_enc]
      * `pred_masks`: `num_pred_masks` tensors of shape [B, K_pred]
    """

    def __init__(
        self,
        input_size: int | tuple[int, int] = 224,
        patch_size: int = 16,
        enc_mask_scale: tuple[float, float] = (0.85, 1.0),
        pred_mask_scale: tuple[float, float] = (0.15, 0.2),
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        num_enc_masks: int = 1,
        num_pred_masks: int = 4,
        min_keep: int = 10,
        allow_overlap: bool = False,
    ) -> None:
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.patch_size = patch_size
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = num_enc_masks
        self.npred = num_pred_masks
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        # Shared across dataloader workers so every worker advances the same counter;
        # the seed derived from it keeps block sizes consistent within a batch.
        self._itr_counter = Value("i", -1)
        self._validate_min_keep()

    def _block_area(self, scale_frac: float, aspect: float) -> int:
        """Patch count of the block the sampler would produce for these parameters."""
        max_keep = int(self.height * self.width * scale_frac)
        h = int(round(math.sqrt(max(1, max_keep) * aspect)))
        w = int(round(math.sqrt(max(1, max_keep) / aspect)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1
        return max(1, h) * max(1, w)

    def _min_block_area(self, scale: tuple[float, float], aspect: tuple[float, float]) -> int:
        """Smallest block this sampler can produce.

        `_sample_block_size` draws a single random number and uses it for both the area
        scale and the aspect ratio, so the two are perfectly correlated and the minimum
        is not necessarily at either endpoint. Sweeping the draw is exact and cheap.
        """
        return min(
            self._block_area(
                scale[0] + r * (scale[1] - scale[0]), aspect[0] + r * (aspect[1] - aspect[0])
            )
            for r in (i / 200 for i in range(201))
        )

    def _validate_min_keep(self) -> None:
        """Fail loudly at construction if `min_keep` is unsatisfiable.

        `_sample_block_mask` loops until it finds a mask with more than `min_keep`
        patches. For target blocks there are no `acceptable_regions` to relax, so if the
        block size itself can never exceed `min_keep` the loop spins forever with no
        output — the worst possible failure on a metered GPU session. This turns that
        into an error at startup.
        """
        grid = f"{self.height}x{self.width}={self.height * self.width} patches"
        for name, scale, aspect in (
            ("pred_mask_scale", self.pred_mask_scale, self.aspect_ratio),
            ("enc_mask_scale", self.enc_mask_scale, (1.0, 1.0)),
        ):
            smallest = self._min_block_area(scale, aspect)
            if smallest <= self.min_keep:
                raise ValueError(
                    f"min_keep={self.min_keep} is unsatisfiable: on a {grid} grid the "
                    f"smallest block from {name}={tuple(scale)} is {smallest} patches, so "
                    f"mask sampling would loop forever. Either lower min_keep below "
                    f"{smallest}, raise the input resolution, or lower the patch size."
                )

    def step(self) -> int:
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            return i.value

    def set_epoch(self, epoch: int, iters_per_epoch: int) -> None:
        """Align the internal counter after a resume.

        Without this, session 2 would restart mask sampling from counter 0 and repeat
        session 1's exact mask sequence. Harmless for correctness, but it wastes the
        stochasticity the method depends on.
        """
        with self._itr_counter.get_lock():
            self._itr_counter.value = epoch * iters_per_epoch - 1

    def _sample_block_size(
        self,
        generator: torch.Generator,
        scale: tuple[float, float],
        aspect_ratio_scale: tuple[float, float],
    ) -> tuple[int, int]:
        rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        min_ar, max_ar = aspect_ratio_scale
        aspect = min_ar + rand * (max_ar - min_ar)

        h = int(round(math.sqrt(max_keep * aspect)))
        w = int(round(math.sqrt(max_keep / aspect)))
        # Strictly smaller than the grid: `torch.randint(0, height - h)` below needs a
        # non-empty range, and a full-height block would leave no room to translate.
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1
        return max(1, h), max(1, w)

    def _sample_block_mask(
        self,
        b_size: tuple[int, int],
        acceptable_regions: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one rectangular block, optionally constrained away from other blocks.

        `acceptable_regions` are 0/1 grids (the complements of already-sampled target
        blocks). Multiplying the candidate by them removes overlap. If that leaves fewer
        than `min_keep` patches after 20 tries, the constraint is relaxed by dropping one
        region — otherwise a large context block against four targets can loop forever.
        """
        h, w = b_size

        def constrain(mask: torch.Tensor, tries: int) -> None:
            assert acceptable_regions is not None
            n = max(int(len(acceptable_regions) - tries), 0)
            for k in range(n):
                mask *= acceptable_regions[k]

        tries = 0
        timeout = og_timeout = 20
        attempts = 0
        # Belt-and-braces bound. `_validate_min_keep` already rules out the case where no
        # mask can ever satisfy min_keep; this catches anything that slips through with a
        # diagnostic instead of an unbounded spin.
        max_attempts = og_timeout * (len(acceptable_regions or ()) + 2) + 200
        top = left = 0
        mask_idx = torch.empty(0, dtype=torch.long)
        valid = False
        while not valid:
            top = int(torch.randint(0, self.height - h + 1, (1,)).item())
            left = int(torch.randint(0, self.width - w + 1, (1,)).item())
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1
            if acceptable_regions is not None:
                constrain(mask, tries)
            mask_idx = torch.nonzero(mask.flatten()).squeeze(-1)
            valid = len(mask_idx) > self.min_keep
            if not valid:
                attempts += 1
                if attempts > max_attempts:
                    raise RuntimeError(
                        f"could not sample a {h}x{w} block with more than min_keep="
                        f"{self.min_keep} patches on a {self.height}x{self.width} grid "
                        f"after {attempts} attempts (constrained by "
                        f"{len(acceptable_regions or ())} region(s)). Lower min_keep or "
                        "raise the input resolution."
                    )
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout

        complement = torch.ones((self.height, self.width), dtype=torch.int32)
        complement[top : top + h, left : left + w] = 0
        return mask_idx, complement

    def __call__(self, batch):
        """Collate a batch and attach one mask set to it."""
        collated = torch.utils.data.default_collate(batch)
        B = len(batch)

        g = torch.Generator()
        g.manual_seed(self.step())
        p_size = self._sample_block_size(g, self.pred_mask_scale, self.aspect_ratio)
        # Context blocks are square by construction: aspect (1, 1).
        e_size = self._sample_block_size(g, self.enc_mask_scale, (1.0, 1.0))

        collated_pred: list[list[torch.Tensor]] = []
        collated_enc: list[list[torch.Tensor]] = []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width

        for _ in range(B):
            masks_p, complements = [], []
            for _ in range(self.npred):
                mask, comp = self._sample_block_mask(p_size)
                masks_p.append(mask)
                complements.append(comp)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_pred.append(masks_p)

            acceptable = None if self.allow_overlap else complements
            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_enc.append(masks_e)

        # Truncate every mask to the batch-wide minimum so they stack into rectangles.
        # This is how the reference avoids padding entirely.
        collated_pred = [[m[:min_keep_pred] for m in ms] for ms in collated_pred]
        collated_enc = [[m[:min_keep_enc] for m in ms] for ms in collated_enc]

        return (
            collated,
            torch.utils.data.default_collate(collated_enc),
            torch.utils.data.default_collate(collated_pred),
        )


def masks_to_grid(mask_idx: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Scatter a flat index tensor back to an [height, width] bool grid (for figures)."""
    grid = torch.zeros(height * width, dtype=torch.bool)
    grid[mask_idx.flatten()] = True
    return grid.view(height, width)

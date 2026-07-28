"""Mask application helpers.

Ported verbatim in behaviour from reference/ijepa/src/masks/utils.py and
reference/ijepa/src/utils/tensors.py. Both functions stack along the *batch* dimension
rather than adding a mask axis, which is what makes the rest of the model mask-agnostic:
the encoder just sees a bigger batch of shorter sequences.
"""
from __future__ import annotations

import torch


def apply_masks(x: torch.Tensor, masks: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """Keep only the indexed patches, concatenating masks along the batch dim.

    :param x: [B, N, D]
    :param masks: list of M index tensors, each [B, K]
    :return: [M*B, K, D]
    """
    if not isinstance(masks, list):
        masks = [masks]
    out = []
    for m in masks:
        idx = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        out.append(torch.gather(x, dim=1, index=idx))
    return torch.cat(out, dim=0)


def repeat_interleave_batch(x: torch.Tensor, B: int, repeat: int) -> torch.Tensor:
    """Repeat each length-B block of the batch `repeat` times, keeping blocks contiguous.

    With 4 target masks and 1 context mask, `apply_masks` gives targets ordered
    [mask0(B), mask1(B), mask2(B), mask3(B)]. The predictor output is ordered the same
    way but with the context repeated per target. This aligns the two so `smooth_l1`
    compares the right pairs — get it wrong and the loss still decreases, just against
    the wrong targets.
    """
    N = len(x) // B
    return torch.cat(
        [torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0) for i in range(N)],
        dim=0,
    )

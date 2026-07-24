from __future__ import annotations

from typing import Tuple

import torch


def sample_target_block(
    n_h: int,
    n_w: int,
    scale_range: Tuple[float, float] = (0.15, 0.20),
    aspect_range: Tuple[float, float] = (0.75, 1.5),
    generator: torch.Generator | None = None,
) -> dict:
    n_total = n_h * n_w
    s = torch.empty(1).uniform_(*scale_range, generator=generator).item()
    r = torch.empty(1).uniform_(*aspect_range, generator=generator).item()

    area = max(1.0, s * n_total)
    h_t = int(round((area * r) ** 0.5))
    w_t = int(round((area / max(r, 1e-6)) ** 0.5))
    h_t = max(1, min(h_t, n_h))
    w_t = max(1, min(w_t, n_w))

    top = int(torch.randint(0, n_h - h_t + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, n_w - w_t + 1, (1,), generator=generator).item())

    mask_hw = torch.zeros(n_h, n_w, dtype=torch.bool)
    mask_hw[top : top + h_t, left : left + w_t] = True
    mask_flat = mask_hw.view(-1)

    tgt_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(-1)
    ctx_indices = torch.nonzero(~mask_flat, as_tuple=False).squeeze(-1)

    return {
        "tgt_mask_hw": mask_hw,
        "tgt_indices": tgt_indices,
        "ctx_indices": ctx_indices,
        "n_ctx": int(ctx_indices.shape[0]),
        "n_tgt": int(tgt_indices.shape[0]),
    }

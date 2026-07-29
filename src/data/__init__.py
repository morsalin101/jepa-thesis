"""Dataset package.

Imports are lazy (PEP 562 `__getattr__`) so that modules which genuinely need no deep
learning stack can run without one. `src.data.splits` and `scripts/dedup_phash.py` use
only numpy and Pillow; eagerly importing `transforms` here would pull in torchvision and
make split generation impossible on a machine without it — which is exactly the machine
you want to generate splits on, since they are pure CPU work.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers only; never executed at runtime
    from src.data.hyperkvasir import HyperKvasirLabeled, HyperKvasirUnlabeled
    from src.data.kvasir_seg import KvasirSegDataset
    from src.data.transforms import (
        GPUAugment,
        TwoViewTransform,
        make_pretrain_transform,
        make_seg_transforms,
    )

_EXPORTS = {
    "HyperKvasirUnlabeled": "src.data.hyperkvasir",
    "HyperKvasirLabeled": "src.data.hyperkvasir",
    "KvasirSegDataset": "src.data.kvasir_seg",
    "GPUAugment": "src.data.transforms",
    "TwoViewTransform": "src.data.transforms",
    "make_pretrain_transform": "src.data.transforms",
    "make_seg_transforms": "src.data.transforms",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(_EXPORTS[name]), name)


def __dir__() -> list[str]:
    return sorted(__all__)

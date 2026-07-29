"""HyperKvasir datasets — the SSL pretraining corpus.

The unlabeled split is 99,417 GI endoscopy frames with no annotations of any kind, which
is exactly what self-supervised pretraining wants and exactly what Kvasir-SEG's 1000
images cannot provide. The labeled split (10,662 images, 23 classes) is never trained
on; it is used only for k-NN and linear probing, which is how we get a representation
quality number before spending GPU hours on segmentation.

The file list is cached to disk on first scan. Walking ~100k files on Kaggle's network
filesystem takes ~40 s, and paying that once per worker per epoch is a real cost.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

from src.config import IMAGE_EXTS, resolve_dataset_dir

# Endoscopy frames are frequently large and occasionally slightly truncated; without
# this a single bad file aborts an 8-hour run at a random epoch.
Image.MAX_IMAGE_PIXELS = None
try:
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    pass


def scan_images(root: Path, cache: Path | None = None, exclude: set[str] | None = None) -> list[Path]:
    """Recursively list image files under `root`, sorted, with an optional path cache."""
    if cache is not None and cache.is_file():
        try:
            names = json.loads(cache.read_text())
            paths = [root / n for n in names]
            if paths and paths[0].exists():
                if exclude:
                    paths = [p for p in paths if p.name not in exclude]
                return paths
        except (json.JSONDecodeError, OSError):
            pass  # stale or corrupt cache; fall through to a fresh scan

    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps([str(p.relative_to(root)) for p in paths]))
        except OSError:
            pass  # read-only /kaggle/input mount; caching is best-effort

    if exclude:
        paths = [p for p in paths if p.name not in exclude]
    return paths


class HyperKvasirUnlabeled(Dataset):
    """Unlabeled pretraining corpus. Returns a transformed image (or a view pair)."""

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        transform: Callable | None = None,
        exclude_file: str | os.PathLike[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        max_images: int = 0,
        seed: int = 0,
    ) -> None:
        self.root = Path(root) if root is not None else resolve_dataset_dir("hyperkvasir")
        self.transform = transform

        # Filenames removed by scripts/dedup_phash.py because they near-duplicate a
        # Kvasir-SEG image. Leaving them in would leak the downstream test set into
        # pretraining — the single most damaging objection this thesis can face.
        exclude: set[str] = set()
        if exclude_file is not None and Path(exclude_file).is_file():
            exclude = {
                line.strip()
                for line in Path(exclude_file).read_text().splitlines()
                if line.strip() and not line.startswith("#")
            }

        cache = Path(cache_dir) / "hyperkvasir_files.json" if cache_dir else None
        self.samples = scan_images(self.root, cache=cache, exclude=exclude)
        self.n_excluded = len(exclude)
        if not self.samples:
            raise FileNotFoundError(f"no images found under {self.root}")

        self.n_total = len(self.samples)
        self.max_images = max_images
        if max_images and max_images < len(self.samples):
            # Seeded random subset, not the first N: the files are sorted by hash-like
            # filename, but any positional slice risks correlating with capture order.
            # Random keeps the subset representative of the whole corpus.
            import random

            self.samples = sorted(random.Random(seed).sample(self.samples, max_images))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img = Image.open(self.samples[idx]).convert("RGB")
        return self.transform(img) if self.transform else img

    def describe(self) -> str:
        s = f"HyperKvasirUnlabeled: {len(self.samples)} images from {self.root}"
        if self.n_excluded:
            s += f" ({self.n_excluded} excluded as Kvasir-SEG near-duplicates)"
        if self.max_images and self.max_images < self.n_total:
            s += (
                f"\n  *** SUBSET RUN: {len(self.samples)} of {self.n_total} images "
                f"({len(self.samples) / self.n_total:.1%}). For pipeline validation only "
                "— not a thesis result. ***"
            )
        return s


class HyperKvasirLabeled(Dataset):
    """Labeled split, used only for k-NN / linear probing.

    Class label is the parent directory name, which is how HyperKvasir ships its
    `labeled-images/<anatomical-landmarks|pathological-findings>/<class>/` tree.
    """

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        transform: Callable | None = None,
        classes: Sequence[str] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else resolve_dataset_dir("hyperkvasir_labeled")
        self.transform = transform
        paths = scan_images(self.root)
        if not paths:
            raise FileNotFoundError(f"no images found under {self.root}")

        names = sorted({p.parent.name for p in paths}) if classes is None else list(classes)
        self.classes = names
        self.class_to_idx = {c: i for i, c in enumerate(names)}
        self.samples = [(p, self.class_to_idx[p.parent.name]) for p in paths if p.parent.name in self.class_to_idx]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return (self.transform(img) if self.transform else img), label

"""Kvasir-SEG: 1000 polyp images with binary masks.

Kaggle layout is `<root>/Kvasir-SEG/{images,masks}/*.jpg`, with the mask carrying the
same filename as its image.

Two things this dataset does that the metric code depends on:

* It can return the **native (H, W)** of each image. Every metric in the thesis is
  computed at native resolution — you upsample the *logits* back to the original size
  and threshold there. Computing Dice at 352x352 against a downsampled ground truth
  inflates it by roughly 1-2 points and is not comparable to published numbers.
* Train and eval use *different transform objects*, never a `Subset` of one augmented
  dataset, so validation is deterministic.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

from src.config import IMAGE_EXTS, REPO_ROOT, resolve_dataset_dir

try:
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    pass


def find_image_and_mask_dirs(root: Path) -> tuple[Path, Path]:
    """Locate the images/ and masks/ pair under a Kvasir-SEG root."""
    for base in (root, root / "Kvasir-SEG", root / "kvasir-seg"):
        img_d, msk_d = base / "images", base / "masks"
        if img_d.is_dir() and msk_d.is_dir():
            return img_d, msk_d
    # Last resort: any descendant pair named images/ + masks/
    for cand in root.rglob("images"):
        if cand.is_dir() and (cand.parent / "masks").is_dir():
            return cand, cand.parent / "masks"
    raise FileNotFoundError(f"could not find images/ and masks/ under {root}")


def read_split(name: str, split: str, splits_dir: Path | None = None) -> list[str]:
    """Read `splits/<name>/<split>.txt` -> list of stems.

    Split files are committed to the repo so any run, on any machine, uses byte-identical
    splits. Regenerate them with `python -m src.data.splits`.
    """
    d = (splits_dir or REPO_ROOT / "splits") / name
    p = d / f"{split}.txt"
    if not p.is_file():
        raise FileNotFoundError(
            f"split file {p} not found. Generate it with:  python -m src.data.splits"
        )
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


class KvasirSegDataset(Dataset):
    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        stems: Sequence[str] | None = None,
        transform: Callable | None = None,
        return_native_size: bool = False,
        label_fraction: float = 1.0,
        seed: int = 0,
    ) -> None:
        """
        :param stems: filename stems to include (from a split file). None = all.
        :param label_fraction: keep only this fraction, for the low-label ablation.
            Subsampling is seeded and *nested* — the 10% set is a subset of the 25% set —
            so the ablation curve is monotone in data, not confounded by which images
            happened to be drawn.
        """
        # required_subdirs matters on Kaggle: without it the recursive fallback picks
        # whichever directory holds the most images — i.e. `images/` itself — and then
        # find_image_and_mask_dirs cannot locate its sibling `masks/`.
        self.root = (
            Path(root)
            if root is not None
            else resolve_dataset_dir("kvasir_seg", required_subdirs=("images", "masks"))
        )
        self.image_dir, self.mask_dir = find_image_and_mask_dirs(self.root)
        self.transform = transform
        self.return_native_size = return_native_size

        by_stem = {p.stem: p for p in sorted(self.image_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS}
        if stems is not None:
            missing = [s for s in stems if s not in by_stem]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} stems from the split are not in {self.image_dir} "
                    f"(first few: {missing[:5]})"
                )
            selected = [by_stem[s] for s in stems]
        else:
            selected = list(by_stem.values())

        if label_fraction < 1.0:
            import random

            rng = random.Random(seed)
            order = list(range(len(selected)))
            rng.shuffle(order)
            keep = max(1, int(round(len(selected) * label_fraction)))
            selected = [selected[i] for i in sorted(order[:keep])]

        self.samples: list[tuple[Path, Path]] = []
        for img_p in selected:
            mask_p = self._find_mask(img_p.stem)
            if mask_p is None:
                raise FileNotFoundError(f"no mask for {img_p.name} in {self.mask_dir}")
            self.samples.append((img_p, mask_p))

        if not self.samples:
            raise RuntimeError(f"no image/mask pairs found under {self.root}")

    def _find_mask(self, stem: str) -> Path | None:
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            p = self.mask_dir / f"{stem}{ext}"
            if p.is_file():
                return p
        for suffix in ("_mask", "_segmentation"):
            for ext in (".jpg", ".png"):
                p = self.mask_dir / f"{stem}{suffix}{ext}"
                if p.is_file():
                    return p
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_p, mask_p = self.samples[idx]
        img = Image.open(img_p).convert("RGB")
        mask = Image.open(mask_p).convert("L")
        native = (img.height, img.width)

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        if self.return_native_size:
            return img, mask, native[0], native[1], idx
        return img, mask

    def stems(self) -> list[str]:
        return [p.stem for p, _ in self.samples]

    def native_mask_path(self, idx: int) -> Path:
        """Full-resolution mask path, for metrics computed at native resolution."""
        return self.samples[idx][1]

    def describe(self) -> str:
        return f"KvasirSeg: {len(self.samples)} pairs from {self.image_dir}"

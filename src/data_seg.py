from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def _find_mask(mask_dir: Path, stem: str) -> Path | None:
    candidates = [
        mask_dir / f"{stem}_segmentation.png",
        mask_dir / f"{stem}_segmentation_lesion.png",
        mask_dir / f"{stem}_Segmentation.png",
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}_mask.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


class ISICSegDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        img_size: int = 96,
        augment: bool = True,
        exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".JPG"),
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"mask_dir not found: {self.mask_dir}")

        exts_lower = {e.lower() for e in exts}
        self.records: list[tuple[Path, Path]] = []
        for p in sorted(self.image_dir.iterdir()):
            if p.suffix.lower() not in exts_lower:
                continue
            mask = _find_mask(self.mask_dir, p.stem)
            if mask is not None:
                self.records.append((p, mask))
        if not self.records:
            raise RuntimeError(f"No image/mask pairs found under {self.image_dir}")

        self.img_size = img_size
        self.augment = augment

        self.resize_img = T.Resize(
            (img_size, img_size), interpolation=T.InterpolationMode.BILINEAR
        )
        self.resize_mask = T.Resize(
            (img_size, img_size), interpolation=T.InterpolationMode.NEAREST
        )
        self.normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.records[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.resize_img(img)
        mask = self.resize_mask(mask)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

        img_t = self.normalize(TF.to_tensor(img))
        mask_t = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0).unsqueeze(0)
        mask_t = (mask_t > 0.5).float()
        return img_t, mask_t

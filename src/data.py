from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class ISICImageDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        img_size: int = 96,
        augment: bool = False,
        exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp"),
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"ISIC image dir not found: {self.root}")
        exts_lower = {e.lower() for e in exts}
        self.paths = sorted(p for p in self.root.iterdir() if p.suffix.lower() in exts_lower)
        if not self.paths:
            raise RuntimeError(
                f"No images with ext {sorted(exts_lower)} found under {self.root}"
            )
        self.img_size = img_size
        self.augment = augment

        tfms: list = [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        ]
        if augment:
            tfms.append(T.RandomHorizontalFlip())
        tfms += [
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        self.transform = T.Compose(tfms)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)

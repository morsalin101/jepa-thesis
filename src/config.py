"""Central config. Detects whether we're running on Kaggle vs locally
so paths and device selection just work in both places."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


def on_kaggle() -> bool:
    return os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            try:
                major, _ = torch.cuda.get_device_capability(0)
                supported = torch.cuda.get_arch_list()
                if any(int(a.split("_")[1]) // 10 <= major for a in supported if a.startswith("sm_")):
                    torch.zeros(1, device="cuda")
                    return "cuda"
                print(f"[warn] GPU sm_{major}0 not supported by this torch "
                      f"({supported}); falling back to CPU. Pick a T4 GPU on Kaggle.")
            except Exception as e:
                print(f"[warn] CUDA present but unusable ({e}); falling back to CPU.")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def find_isic_image_dir() -> Path:
    if on_kaggle():
        candidates = [
            Path("/kaggle/input/isic2018-challenge-task1-data-segmentation/ISIC2018_Task1-2_Training_Input"),
            Path("/kaggle/input/isic2018-challenge-task1-data-segmentation"),
            Path("/kaggle/input"),
        ]
        for c in candidates:
            if c.exists() and c.is_dir() and any(c.glob("*.jpg")):
                return c
        for top in Path("/kaggle/input").iterdir() if Path("/kaggle/input").exists() else []:
            sub = top / "ISIC2018_Task1-2_Training_Input"
            if sub.exists() and any(sub.glob("*.jpg")):
                return sub
        return candidates[0]
    return Path("data/ISIC2018_Task1-2_Training_Input")


@dataclass
class JEPACfg:
    img_size: int = 96
    patch_size: int = 8
    enc_dim: int = 192
    enc_depth: int = 12
    enc_heads: int = 3
    enc_mlp_ratio: float = 4.0
    pred_dim: int = 192
    pred_depth: int = 6
    pred_heads: int = 4
    pred_mlp_ratio: float = 4.0
    mask_scale: Tuple[float, float] = (0.15, 0.20)
    mask_aspect: Tuple[float, float] = (0.75, 1.5)
    ema_momentum: float = 0.996
    enc_lr: float = 1e-3
    pred_lr: float = 1e-3
    weight_decay: float = 0.05
    batch_size: int = 64
    epochs: int = 1

    @property
    def n_h(self) -> int:
        return self.img_size // self.patch_size

    @property
    def n_w(self) -> int:
        return self.img_size // self.patch_size


def _load_yaml_into_jepa(jepa: JEPACfg) -> None:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "pretrain_jepa.yaml"
    if not cfg_path.exists():
        return
    try:
        import yaml
    except ImportError:
        print("[warn] pyyaml not installed; skipping configs/pretrain_jepa.yaml")
        return
    with open(cfg_path) as f:
        data = yaml.safe_load(f) or {}
    for k, v in data.items():
        if hasattr(jepa, k):
            setattr(jepa, k, v)


@dataclass
class Config:
    output_dir: Path = field(
        default_factory=lambda: Path("/kaggle/working") if on_kaggle() else Path("outputs")
    )
    data_dir: Path = field(default_factory=find_isic_image_dir)
    device: str = field(default_factory=pick_device)
    seed: int = 42
    jepa: JEPACfg = field(default_factory=JEPACfg)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _load_yaml_into_jepa(self.jepa)


CONFIG = Config()

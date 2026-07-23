"""Central config. Detects whether we're running on Kaggle vs locally
so paths and device selection just work in both places."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def on_kaggle() -> bool:
    return os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon, for local smoke tests
    except ImportError:
        pass
    return "cpu"


@dataclass
class Config:
    # Where outputs (checkpoints, logs) go. Kaggle only persists /kaggle/working.
    output_dir: Path = Path("/kaggle/working") if on_kaggle() else Path("outputs")
    # Where input data lives. On Kaggle, attach datasets under /kaggle/input.
    data_dir: Path = Path("/kaggle/input") if on_kaggle() else Path("data")

    device: str = pick_device()
    seed: int = 42

    # --- training hyperparameters (edit freely) ---
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


CONFIG = Config()

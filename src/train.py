"""Training entrypoint.

Run locally (CPU/MPS smoke test):   python -m src.train --mode pretrain --epochs 1
Run on Kaggle (GPU):                same command inside the notebook cell.

Keep this file as the single entrypoint the Kaggle notebook calls, so the
"run" surface never changes even as the model code grows.
"""
from __future__ import annotations

import argparse

from src.config import CONFIG, on_kaggle
from src.engine.pretrain import pretrain


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JEPA training")
    p.add_argument("--mode", choices=["pretrain", "segment"], default="pretrain")
    p.add_argument("--epochs", type=int, default=CONFIG.jepa.epochs)
    p.add_argument("--batch-size", type=int, default=CONFIG.jepa.batch_size)
    p.add_argument("--lr", type=float, default=CONFIG.jepa.enc_lr)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[env]    running on Kaggle: {on_kaggle()}")
    print(f"[device] {CONFIG.device}")
    print(f"[paths]  data={CONFIG.data_dir}  output={CONFIG.output_dir}")
    print(f"[hparams] mode={args.mode} epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}")

    if args.mode == "pretrain":
        CONFIG.jepa.epochs = args.epochs
        CONFIG.jepa.batch_size = args.batch_size
        CONFIG.jepa.enc_lr = args.lr
        CONFIG.jepa.pred_lr = args.lr
        pretrain(CONFIG)
    else:
        raise NotImplementedError("segment mode is scaffolded in the next pass")


if __name__ == "__main__":
    main()

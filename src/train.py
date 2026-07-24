"""Training entrypoint.

Run locally (CPU/MPS smoke test):   python -m src.train --mode pretrain --epochs 1
                                    python -m src.train --mode segment --epochs 30
Run on Kaggle (GPU):                same commands in the notebook cells.

Keep this file as the single entrypoint the Kaggle notebook calls, so the
"run" surface never changes even as the model code grows.
"""
from __future__ import annotations

import argparse

from src.config import CONFIG, on_kaggle
from src.engine.pretrain import pretrain
from src.engine.segment import segment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JEPA training")
    p.add_argument("--mode", choices=["pretrain", "segment"], default="pretrain")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[env]    running on Kaggle: {on_kaggle()}")
    print(f"[device] {CONFIG.device}")
    print(f"[paths]  data={CONFIG.data_dir}  output={CONFIG.output_dir}")
    print(f"[hparams] mode={args.mode} epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}")

    if args.mode == "pretrain":
        if args.epochs is not None:
            CONFIG.jepa.epochs = args.epochs
        if args.batch_size is not None:
            CONFIG.jepa.batch_size = args.batch_size
        if args.lr is not None:
            CONFIG.jepa.enc_lr = args.lr
            CONFIG.jepa.pred_lr = args.lr
        pretrain(CONFIG)
    elif args.mode == "segment":
        if args.epochs is not None:
            CONFIG.seg.epochs = args.epochs
        if args.batch_size is not None:
            CONFIG.seg.batch_size = args.batch_size
        if args.lr is not None:
            CONFIG.seg.lr = args.lr
        segment(CONFIG)
    else:
        raise NotImplementedError(f"mode={args.mode}")


if __name__ == "__main__":
    main()

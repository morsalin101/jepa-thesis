"""Training entrypoint.

Run locally (CPU/MPS smoke test):   python -m src.train --epochs 1
Run on Kaggle (GPU):                same command inside the notebook cell.

Keep this file as the single entrypoint the Kaggle notebook calls, so the
"run" surface never changes even as the model code grows.
"""
from __future__ import annotations

import argparse

from src.config import CONFIG, on_kaggle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JEPA training")
    p.add_argument("--epochs", type=int, default=CONFIG.epochs)
    p.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    p.add_argument("--lr", type=float, default=CONFIG.lr)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[env]    running on Kaggle: {on_kaggle()}")
    print(f"[device] {CONFIG.device}")
    print(f"[paths]  data={CONFIG.data_dir}  output={CONFIG.output_dir}")
    print(f"[hparams] epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}")

    # ------------------------------------------------------------------
    # TODO: build your JEPA model, dataloaders, optimizer and train loop.
    # This stub proves the local<->Kaggle plumbing works end to end.
    # ------------------------------------------------------------------
    for epoch in range(args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs} ... (replace with real training)")

    ckpt = CONFIG.output_dir / "checkpoint.txt"
    ckpt.write_text("placeholder checkpoint\n")
    print(f"[done] wrote {ckpt}")


if __name__ == "__main__":
    main()

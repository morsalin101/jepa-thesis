"""Regenerate every thesis figure.

Runs with no GPU and no checkpoints — everything reads from the JSON/JSONL artefacts
written during training, so figures can be iterated on a laptop while Kaggle trains.
Figures whose inputs do not exist yet are skipped with a message rather than crashing,
so this is safe to run at any point in the project.

    python -m src.viz.make_all                 # everything available
    python -m src.viz.make_all --only masking schedules
    python -m src.viz.make_all --list
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from src.config import REPO_ROOT, default_output_dir
from src.viz.figures import FIGURES
from src.viz.style import setup


def build(names: list[str], out_dir: Path, base: Path, kvasir_root: Path | None = None) -> None:
    setup()
    seg_dir = base / "seg"
    ckpt_metrics = base / "ckpt" / "metrics.jsonl"
    results = base / "results"

    args_for = {
        "masking": {},
        "schedules": {},
        "pretrain_curves": {"metrics_path": ckpt_metrics},
        "seg_curves": {"seg_dir": seg_dir},
        "dice_distribution": {"seg_dir": seg_dir},
        "paired_forest": {"comparisons_path": results / "comparisons.json"},
        "low_label": {"seg_dir": seg_dir},
        "efficiency": {"seg_dir": seg_dir},
        "qualitative": {"seg_dir": seg_dir, "kvasir_root": kvasir_root},
        "knn_probe": {"probe_path": results / "probe.json"},
        "embedding_space": {"embed_path": results / "embeddings.npz"},
    }

    ok, skipped, failed = 0, 0, 0
    for name in names:
        fn = FIGURES[name]
        try:
            result = fn(out_dir, **args_for.get(name, {}))
            if result is None:
                skipped += 1
            else:
                ok += 1
        except FileNotFoundError as e:
            print(f"[viz] skipping {name}: {e}")
            skipped += 1
        except Exception:  # noqa: BLE001 - one broken figure must not block the rest
            print(f"[viz] FAILED {name}:")
            traceback.print_exc(limit=3)
            failed += 1

    print(f"\n[viz] {ok} written, {skipped} skipped (inputs not ready), {failed} failed -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate thesis figures")
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES), default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "figures"))
    ap.add_argument("--base", default=None, help="outputs dir holding ckpt/, seg/, results/")
    ap.add_argument("--kvasir-root", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k in sorted(FIGURES):
            print(f"  {k}")
        return

    build(
        args.only or list(FIGURES),
        Path(args.out),
        Path(args.base) if args.base else default_output_dir(),
        Path(args.kvasir_root) if args.kvasir_root else None,
    )


if __name__ == "__main__":
    main()

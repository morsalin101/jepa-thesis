"""One-time creation of the two private Kaggle Datasets the pipeline needs.

    jepa-thesis-ckpt      hot resume state, re-versioned every ~45 min mid-run
    jepa-thesis-weights   the finished encoders (~88 MB each), consumed by segmentation

Both have to exist *before* the notebooks reference them in `dataset_sources`, otherwise
`kaggle kernels push` rejects the metadata. They start with a placeholder file.

Run this once from your laptop:

    python scripts/setup_kaggle_datasets.py --user <kaggle-username>
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def dataset_exists(slug: str) -> bool:
    """True if the dataset already exists on Kaggle.

    `kaggle datasets status` exits 0 even when the API returns 403/404, so the return
    code is useless here — the answer is in stdout ("ready" vs a "... Client Error"
    line). Getting this wrong the other way would make `create` be skipped and every
    later push fail with a missing-dataset error.
    """
    r = subprocess.run(
        ["kaggle", "datasets", "status", slug], capture_output=True, text=True
    )
    out = (r.stdout or "") + (r.stderr or "")
    if "error" in out.lower() or "not found" in out.lower():
        return False
    return bool(out.strip())


def create(slug: str, title: str, note: str) -> None:
    if dataset_exists(slug):
        print(f"[setup] {slug} already exists — leaving it alone")
        return

    d = Path(tempfile.mkdtemp())
    try:
        (d / "README.md").write_text(f"# {title}\n\n{note}\n")
        (d / "dataset-metadata.json").write_text(
            json.dumps(
                {"title": title, "id": slug, "licenses": [{"name": "CC0-1.0"}]}, indent=2
            )
        )
        print(f"[setup] creating {slug} ...")
        # No --private flag: datasets are private by default in the Kaggle CLI, and
        # --public is the opt-in. Passing --private is an error on 1.7.x.
        r = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(d), "--dir-mode", "zip"],
            capture_output=True,
            text=True,
        )
        out = (r.stdout or "") + (r.stderr or "")
        print("  " + out.strip().splitlines()[-1] if out.strip() else "  (no output)")
        if r.returncode != 0 or "error" in out.lower():
            raise SystemExit(
                f"failed to create {slug}.\n{out.strip()}\n"
                "Check that `kaggle datasets list --mine` works (valid, unexpired token)."
            )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create the pipeline's Kaggle Datasets")
    ap.add_argument("--user", required=True)
    args = ap.parse_args()

    create(
        f"{args.user}/jepa-thesis-ckpt",
        "jepa-thesis-ckpt",
        "Resumable pretraining state. Overwritten every ~45 min during a run. "
        "Safe to delete and recreate between methods to reclaim version storage.",
    )
    create(
        f"{args.user}/jepa-thesis-weights",
        "jepa-thesis-weights",
        "Exported SSL encoders (ViT-S/16, ~88 MB each), consumed by the segmentation "
        "notebook. One file per method.",
    )

    print(
        "\n[setup] done. Both datasets exist and are private.\n"
        "        Next: push notebooks/data-prep and run it (CPU session, no GPU quota)."
    )


if __name__ == "__main__":
    main()

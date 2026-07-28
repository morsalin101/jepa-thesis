"""Build the pre-resized HyperKvasir pretraining corpus, then publish it as a Kaggle Dataset.

Run this in a Kaggle **CPU** notebook — CPU sessions do not consume the GPU quota, so
this whole step is free.

Why pre-resize at all: Kaggle gives 4 vCPUs. HyperKvasir's frames are frequently
1280x1024, and decoding those at the ~400 views/s SimCLR needs is simply not possible on
4 cores. You would run at roughly a third of GPU speed and burn quota producing nothing.
Resizing the short side to 256 px once cuts decode cost by 6-8x and shrinks the corpus
from ~24 GB to ~3 GB.

Disk strategy: the download goes to /kaggle/tmp (~60 GiB scratch, not counted against the
20 GiB output cap) and is streamed member-by-member out of the zip, so the full archive is
never extracted. Only the resized JPEGs land in /kaggle/working.

    python scripts/build_hyperkvasir_dataset.py --publish morsalin101/hyperkvasir-unlabeled-256
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import IMAGE_EXTS, on_kaggle, scratch_dir  # noqa: E402

try:
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    pass
Image.MAX_IMAGE_PIXELS = None

# Official Simula mirrors. The unlabeled archive is ~24 GB.
UNLABELED_URL = "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-unlabeled-images.zip"
LABELED_URL = "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip"


def download(url: str, dest: Path) -> Path:
    """Resumable download via curl (present on Kaggle; far more robust than urllib here)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"[build] reusing existing {dest} ({dest.stat().st_size / 1e9:.1f} GB)")
        return dest
    print(f"[build] downloading {url}\n[build]   -> {dest}  (this is ~24 GB; expect 10-30 min)")
    r = subprocess.run(["curl", "-L", "-C", "-", "--fail", "-o", str(dest), url])
    if r.returncode != 0 or not dest.is_file():
        raise RuntimeError(
            f"download failed (curl exit {r.returncode}). If the Simula mirror is "
            "unreachable from Kaggle, run this script locally and upload the resulting "
            "~3 GB directory with `kaggle datasets create`."
        )
    return dest


def resize_one(args: tuple[bytes, Path, int, int]) -> bool:
    data, out_path, short_side, quality = args
    try:
        import io

        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        scale = short_side / min(w, h)
        if scale < 1.0:  # only ever downscale
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
        img.save(out_path, "JPEG", quality=quality, optimize=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[build] skipping {out_path.name}: {e}")
        return False


def stream_resize_zip(
    zip_path: Path, out_dir: Path, short_side: int = 256, quality: int = 90, workers: int = 4
) -> int:
    """Read image members one at a time and write resized JPEGs. Never fully extracts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    t0 = time.time()

    with zipfile.ZipFile(zip_path) as zf:
        members = [
            m
            for m in zf.namelist()
            if Path(m).suffix.lower() in IMAGE_EXTS and not Path(m).name.startswith(".")
        ]
        print(f"[build] {len(members)} images in archive")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            batch: list[tuple[bytes, Path, int, int]] = []
            for i, name in enumerate(members):
                out_path = out_dir / f"{Path(name).stem}.jpg"
                if out_path.is_file():
                    skipped += 1
                    continue
                batch.append((zf.read(name), out_path, short_side, quality))

                if len(batch) >= 256:
                    written += sum(pool.map(resize_one, batch))
                    batch.clear()
                    rate = (i + 1) / max(1e-6, time.time() - t0)
                    eta = (len(members) - i - 1) / max(1e-6, rate) / 60
                    print(f"[build] {i + 1}/{len(members)}  {rate:.0f} img/s  ETA {eta:.0f} min")
            if batch:
                written += sum(pool.map(resize_one, batch))

    size_gb = sum(p.stat().st_size for p in out_dir.glob("*.jpg")) / 1e9
    print(f"[build] wrote {written} (+{skipped} already present) -> {out_dir}  ({size_gb:.2f} GB)")
    return written + skipped


def publish(out_dir: Path, slug: str, title: str | None = None) -> None:
    """Create or version a Kaggle Dataset from the resized corpus."""
    meta = {
        "title": title or slug.split("/")[-1],
        "id": slug,
        "licenses": [{"name": "CC-BY-4.0"}],  # HyperKvasir is CC BY 4.0
    }
    (out_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    # `kaggle datasets status` exits 0 even on a 403/404, so read stdout, not the code.
    probe = subprocess.run(
        ["kaggle", "datasets", "status", slug], capture_output=True, text=True
    )
    probe_out = (probe.stdout or "") + (probe.stderr or "")
    exists = bool(probe_out.strip()) and "error" not in probe_out.lower()

    # Datasets are private by default; --public is the opt-in. There is no --private.
    cmd = (
        ["kaggle", "datasets", "version", "-p", str(out_dir), "-m", "update", "--dir-mode", "zip"]
        if exists
        else ["kaggle", "datasets", "create", "-p", str(out_dir), "--dir-mode", "zip"]
    )
    print(f"[build] {'versioning' if exists else 'creating'} {slug} (upload takes a while)")
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out)
    if r.returncode != 0 or "error" in out.lower():
        raise RuntimeError(f"kaggle datasets command failed:\n{out.strip()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the 256px HyperKvasir pretraining corpus")
    ap.add_argument("--split", default="unlabeled", choices=["unlabeled", "labeled"])
    ap.add_argument("--url", default=None)
    ap.add_argument("--zip", default=None, help="use an already-downloaded archive")
    ap.add_argument("--out", default=None)
    ap.add_argument("--short-side", type=int, default=256)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--publish", default=None, help="Kaggle dataset slug, e.g. user/hyperkvasir-unlabeled-256")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    out_dir = Path(
        args.out or (Path("/kaggle/working") if on_kaggle() else Path("data"))
    ) / ("hk256" if args.split == "unlabeled" else "hk_labeled256")

    if args.zip:
        zip_path = Path(args.zip)
    else:
        url = args.url or (UNLABELED_URL if args.split == "unlabeled" else LABELED_URL)
        zip_path = download(url, scratch_dir() / Path(url).name)

    n = stream_resize_zip(zip_path, out_dir, args.short_side, args.quality, args.workers)

    if not args.keep_zip and not args.zip and zip_path.is_file():
        zip_path.unlink()
        print(f"[build] removed {zip_path} to free scratch space")

    print(f"[build] corpus ready: {n} images in {out_dir}")
    if args.publish:
        publish(out_dir, args.publish)
        print(
            f"[build] done. Add '{args.publish}' to dataset_sources in your "
            "notebooks/*/kernel-metadata.json."
        )


if __name__ == "__main__":
    main()

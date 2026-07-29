"""Generate Kvasir-SEG train/val/test splits.

Three properties the split must have, in decreasing order of how badly getting them
wrong would hurt:

1. **Group-aware.** Kvasir-SEG frames are extracted from colonoscopy *videos*, so the
   dataset contains near-duplicate views of the same polyp. If two frames of one polyp
   land in train and test, test Dice is measuring memorisation. We perceptual-hash every
   image, take connected components under a Hamming threshold, and keep whole components
   in one split.
2. **Stratified.** Polyp masks span roughly 0.8% to 62% of image area, and native
   resolutions range from 332x487 to 1920x1072. An unstratified 100-image test set can
   easily over-represent large, easy polyps and flatter every method equally.
3. **Committed to the repo.** The output `.txt` files are version-controlled so every
   run on every machine uses byte-identical splits.

We emit two split sets:
  * `800_100_100` — the primary. The test set is used exactly once, at the very end.
  * `880_120`     — the literature-standard split, so our numbers can be placed against
                    published Kvasir-SEG results. Note that the common practice of
                    selecting on those 120 *and* reporting on them is precisely the
                    leakage the primary split avoids.

Run:  python -m src.data.splits
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import REPO_ROOT, resolve_dataset_dir
from src.data.kvasir_seg import find_image_and_mask_dirs

HASH_SIZE = 8


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    m[0] /= np.sqrt(2)
    return m * np.sqrt(2 / n)


def phash(img: Image.Image, hash_size: int = HASH_SIZE, highfreq_factor: int = 4) -> int:
    """Perceptual hash (DCT-based), returned as a 64-bit int.

    More robust than a difference hash to the brightness and scale changes that separate
    two frames of the same polyp, which is exactly the case we need to catch.
    """
    size = hash_size * highfreq_factor
    px = np.asarray(img.convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float64)
    d = _dct_matrix(size)
    coeffs = d @ px @ d.T
    low = coeffs[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])  # skip DC, which only encodes mean brightness
    bits = (low > med).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def group_near_duplicates(hashes: list[int], threshold: int = 6) -> list[int]:
    """Connected components under Hamming <= threshold. Returns a group id per item.

    O(n^2) at n=1000 is 500k integer XORs — under a second, so no need for an LSH index.
    """
    uf = UnionFind(len(hashes))
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hamming(hashes[i], hashes[j]) <= threshold:
                uf.union(i, j)
    roots = {}
    out = []
    for i in range(len(hashes)):
        r = uf.find(i)
        if r not in roots:
            roots[r] = len(roots)
        out.append(roots[r])
    return out


def scan_dataset(root: Path) -> list[dict]:
    """Per-image record: stem, phash, mask area fraction, native resolution."""
    image_dir, mask_dir = find_image_and_mask_dirs(root)
    recs = []
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for n, img_p in enumerate(images):
        if n % 200 == 0:
            print(f"[splits] hashing {n}/{len(images)} ...")
        img = Image.open(img_p)
        mask_p = next(
            (mask_dir / f"{img_p.stem}{e}" for e in (".jpg", ".png", ".jpeg") if (mask_dir / f"{img_p.stem}{e}").is_file()),
            None,
        )
        if mask_p is None:
            raise FileNotFoundError(f"no mask for {img_p.name}")
        m = np.asarray(Image.open(mask_p).convert("L"), dtype=np.uint8) > 127
        recs.append(
            {
                "stem": img_p.stem,
                "phash": phash(img),
                "area": float(m.mean()),
                "h": img.height,
                "w": img.width,
            }
        )
    return recs


def _strata(recs: list[dict]) -> list[str]:
    """Stratum label = mask-area quartile x resolution tercile."""
    areas = np.array([r["area"] for r in recs])
    pixels = np.array([r["h"] * r["w"] for r in recs])
    aq = np.digitize(areas, np.quantile(areas, [0.25, 0.5, 0.75]))
    rq = np.digitize(pixels, np.quantile(pixels, [1 / 3, 2 / 3]))
    return [f"a{a}r{r}" for a, r in zip(aq, rq)]


def make_splits(
    recs: list[dict],
    sizes: dict[str, int],
    seed: int = 0,
    dup_threshold: int = 6,
) -> tuple[dict[str, list[str]], dict]:
    """Assign whole duplicate-groups to splits, filling stratum quotas proportionally."""
    groups = group_near_duplicates([r["phash"] for r in recs], dup_threshold)
    strata = _strata(recs)
    for r, g, s in zip(recs, groups, strata):
        r["group"] = g
        r["stratum"] = s

    members: dict[int, list[dict]] = defaultdict(list)
    for r in recs:
        members[r["group"]].append(r)

    # A group's stratum is its most common member stratum.
    def group_stratum(gid: int) -> str:
        counts: dict[str, int] = defaultdict(int)
        for r in members[gid]:
            counts[r["stratum"]] += 1
        return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    rng = np.random.RandomState(seed)
    gids = sorted(members)
    rng.shuffle(gids)

    n_total = len(recs)
    # Per-stratum quota for each split, proportional to that stratum's overall share.
    stratum_counts: dict[str, int] = defaultdict(int)
    for r in recs:
        stratum_counts[r["stratum"]] += 1
    quota = {
        split: {s: stratum_counts[s] * n / n_total for s in stratum_counts}
        for split, n in sizes.items()
    }

    assigned: dict[str, list[str]] = {k: [] for k in sizes}
    filled: dict[str, dict[str, float]] = {k: defaultdict(float) for k in sizes}
    counts = {k: 0 for k in sizes}

    # Largest groups first — they are the least flexible to place.
    for gid in sorted(gids, key=lambda g: -len(members[g])):
        s = group_stratum(gid)
        size = len(members[gid])
        # Prefer the split furthest below its quota for this stratum, then overall.
        best = min(
            sizes,
            key=lambda k: (
                counts[k] + size > sizes[k],  # hard cap first
                filled[k][s] - quota[k][s],
                counts[k] - sizes[k],
            ),
        )
        assigned[best] += [r["stem"] for r in members[gid]]
        filled[best][s] += size
        counts[best] += size

    stats = {
        "seed": seed,
        "dup_threshold": dup_threshold,
        "n_images": n_total,
        "n_groups": len(members),
        "largest_group": max(len(v) for v in members.values()),
        "multi_image_groups": sum(1 for v in members.values() if len(v) > 1),
        "counts": {k: len(v) for k, v in assigned.items()},
        "stratum_distribution": {
            k: dict(sorted(((s, int(c)) for s, c in filled[k].items()))) for k in assigned
        },
    }
    return {k: sorted(v) for k, v in assigned.items()}, stats


def write_splits(splits: dict[str, list[str]], stats: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, stems in splits.items():
        (out_dir / f"{name}.txt").write_text("\n".join(stems) + "\n")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[splits] wrote {out_dir}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Kvasir-SEG splits")
    ap.add_argument("--root", default=None, help="Kvasir-SEG root (auto-detected if omitted)")
    ap.add_argument("--out", default=str(REPO_ROOT / "splits"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dup-threshold", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.root) if args.root else resolve_dataset_dir("kvasir_seg")
    print(f"[splits] scanning {root}")
    recs = scan_dataset(root)
    print(f"[splits] {len(recs)} images hashed")

    out = Path(args.out)
    primary, stats = make_splits(
        recs, {"train": 800, "val": 100, "test": 100}, args.seed, args.dup_threshold
    )
    write_splits(primary, stats, out / "800_100_100")
    print(
        f"[splits] {stats['n_groups']} duplicate-groups, "
        f"{stats['multi_image_groups']} with >1 image, largest={stats['largest_group']}"
    )

    secondary, stats2 = make_splits(
        recs, {"train": 880, "val": 120}, args.seed, args.dup_threshold
    )
    secondary["test"] = list(secondary["val"])  # literature convention: val == test here
    write_splits(secondary, stats2, out / "880_120")


if __name__ == "__main__":
    main()

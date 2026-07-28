"""Statistical comparison of the encoder arms.

The design choices here are the ones a thesis committee will ask about, so each is
justified in place:

* **Unit of analysis is the test image (n=100), paired across methods** — not the seed.
  Seeds give n=5, which is hopelessly underpowered: with 5 samples per arm you cannot
  detect anything smaller than a very large effect. Per-image pairing also removes
  image difficulty as a nuisance variable, which is the dominant source of variance in
  polyp segmentation.

* **Wilcoxon signed-rank, not a paired t-test.** Per-image Dice is bounded on [0,1],
  strongly left-skewed, and has a point mass near 0 for completely-missed polyps. The
  t-test's normality assumption is visibly violated; the sign-rank test only needs
  symmetry of the *differences*, which is far weaker.

* **Holm-Bonferroni**, not raw p-values, because we make one planned comparison per
  baseline. Holm is uniformly more powerful than Bonferroni at the same family-wise
  error rate, so there is no reason to use the latter.

* **A BCa bootstrap CI on the effect size, always.** A significant p-value on a +0.004
  Dice difference is not a result. The interval is what makes that visible.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Comparison:
    method_a: str
    method_b: str
    n: int
    median_diff: float
    mean_diff: float
    ci_low: float
    ci_high: float
    statistic: float
    p_raw: float
    p_adjusted: float
    significant: bool
    n_better: int
    n_worse: int
    n_tied: int

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ bootstrap


def bca_ci(
    x: np.ndarray,
    statistic=np.median,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap confidence interval.

    BCa rather than the percentile bootstrap because the sampling distribution of a
    median of skewed paired differences is itself skewed, and the percentile interval is
    then noticeably off-centre. BCa corrects for both the bias and the skew.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))

    rng = np.random.RandomState(seed)
    theta_hat = float(statistic(x))
    boot = np.array([statistic(x[rng.randint(0, n, n)]) for _ in range(n_boot)])

    # z0: bias correction, from the fraction of bootstrap replicates below the estimate.
    prop = float(np.mean(boot < theta_hat))
    if prop <= 0 or prop >= 1:  # degenerate; fall back to percentile
        return float(np.percentile(boot, 100 * alpha / 2)), float(
            np.percentile(boot, 100 * (1 - alpha / 2))
        )
    z0 = _norm_ppf(prop)

    # a: acceleration, from the jackknife skewness.
    jack = np.array([statistic(np.delete(x, i)) for i in range(n)])
    jack_mean = jack.mean()
    num = float(((jack_mean - jack) ** 3).sum())
    den = float(6.0 * (((jack_mean - jack) ** 2).sum() ** 1.5))
    a = num / den if den != 0 else 0.0

    def adjust(p: float) -> float:
        z = _norm_ppf(p)
        return _norm_cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    lo = float(np.percentile(boot, 100 * adjust(alpha / 2)))
    hi = float(np.percentile(boot, 100 * adjust(1 - alpha / 2)))
    return lo, hi


def _norm_cdf(z: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    try:
        from scipy.stats import norm

        return float(norm.ppf(p))
    except ImportError:
        # Acklam's rational approximation; accurate to ~1e-9, plenty for CI endpoints.
        if not 0 < p < 1:
            return float("inf") if p >= 1 else float("-inf")
        a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
             1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
        b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
             6.680131188771972e01, -1.328068155288572e01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
             -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
             3.754408661907416e00]
        pl, ph = 0.02425, 1 - 0.02425
        if p < pl:
            q = (-2 * np.log(p)) ** 0.5
            return float((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                         ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
        if p > ph:
            q = (-2 * np.log(1 - p)) ** 0.5
            return float(-(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                         ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
        q = p - 0.5
        r = q * q
        return float((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
                     (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1))


# ------------------------------------------------------------------ tests


def wilcoxon(diff: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test. Returns (statistic, p-value)."""
    try:
        from scipy.stats import wilcoxon as _w

        nz = diff[diff != 0]
        if len(nz) == 0:
            return 0.0, 1.0
        stat, p = _w(diff, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except ImportError:
        # Normal approximation with tie correction — adequate at n=100.
        nz = diff[diff != 0]
        n = len(nz)
        if n == 0:
            return 0.0, 1.0
        order = np.argsort(np.abs(nz))
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        w_plus = ranks[nz > 0].sum()
        mu = n * (n + 1) / 4
        sigma = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
        z = (w_plus - mu) / sigma if sigma > 0 else 0.0
        return float(w_plus), float(2 * (1 - _norm_cdf(abs(z))))


def holm_bonferroni(pvals: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Holm step-down adjustment. Returns (adjusted p-values, reject flags)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist(), [bool(a <= alpha) for a in adjusted]


# ------------------------------------------------------------------ pipeline


def load_per_image_dice(seg_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Read every run's per-image test results.

    Returns {encoder: {seed: {stem: dice}}} from `outputs/seg/*/test_per_image.json`,
    keyed by the encoder and seed recorded in each run's summary.json.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for run in sorted(Path(seg_dir).glob("*/test_per_image.json")):
        summary_p = run.parent / "summary.json"
        if not summary_p.is_file():
            continue
        summary = json.loads(summary_p.read_text())
        # The low-label arms are a separate experiment; keep them out of the main table.
        if summary.get("label_fraction", 1.0) != 1.0:
            continue
        enc = summary["encoder"]
        seed = str(summary.get("seed", 0))
        records = json.loads(run.read_text())
        out.setdefault(enc, {})[seed] = {r["stem"]: r["dice"] for r in records}
    return out


def aggregate_over_seeds(per_seed: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean per-image Dice across seeds -> one value per image.

    This is the aggregation step that turns 5 noisy runs into one paired observation per
    image, which is what the n=100 test then operates on.
    """
    stems = set.intersection(*(set(d) for d in per_seed.values())) if per_seed else set()
    return {s: float(np.mean([d[s] for d in per_seed.values()])) for s in sorted(stems)}


def compare_all(
    dice_by_encoder: dict[str, dict[str, float]],
    reference: str = "ijepa",
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> list[Comparison]:
    """Paired comparison of `reference` against every other arm."""
    if reference not in dice_by_encoder:
        raise KeyError(f"reference {reference!r} not among {sorted(dice_by_encoder)}")

    others = [k for k in sorted(dice_by_encoder) if k != reference]
    raw: list[Comparison] = []
    for other in others:
        stems = sorted(set(dice_by_encoder[reference]) & set(dice_by_encoder[other]))
        a = np.array([dice_by_encoder[reference][s] for s in stems])
        b = np.array([dice_by_encoder[other][s] for s in stems])
        diff = a - b
        stat, p = wilcoxon(diff)
        lo, hi = bca_ci(diff, np.median, alpha, n_boot, seed)
        raw.append(
            Comparison(
                method_a=reference,
                method_b=other,
                n=len(stems),
                median_diff=float(np.median(diff)),
                mean_diff=float(diff.mean()),
                ci_low=lo,
                ci_high=hi,
                statistic=stat,
                p_raw=p,
                p_adjusted=p,
                significant=False,
                n_better=int((diff > 0).sum()),
                n_worse=int((diff < 0).sum()),
                n_tied=int((diff == 0).sum()),
            )
        )

    adj, reject = holm_bonferroni([c.p_raw for c in raw], alpha)
    for c, a_, r in zip(raw, adj, reject):
        c.p_adjusted = a_
        c.significant = r
    return raw


def run(seg_dir: Path, out_path: Path, reference: str = "ijepa") -> list[Comparison]:
    by_encoder = load_per_image_dice(seg_dir)
    if not by_encoder:
        raise FileNotFoundError(f"no completed segmentation runs under {seg_dir}")

    dice = {enc: aggregate_over_seeds(per_seed) for enc, per_seed in by_encoder.items()}
    for enc, per_seed in by_encoder.items():
        print(f"[stats] {enc}: {len(per_seed)} seed(s), {len(dice[enc])} test images")

    comparisons = compare_all(dice, reference=reference)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([c.as_dict() for c in comparisons], indent=2) + "\n")

    print(f"\n{'comparison':28s} {'median Δ':>10s} {'95% BCa CI':>22s} {'p_adj':>10s}")
    for c in comparisons:
        star = " *" if c.significant else "  "
        print(
            f"{c.method_a + ' vs ' + c.method_b:28s} {c.median_diff:+10.4f} "
            f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}] {c.p_adjusted:10.4g}{star}"
        )
    print("\n* = significant at alpha=0.05 after Holm-Bonferroni")
    return comparisons


def main() -> None:
    import argparse

    from src.config import default_output_dir

    ap = argparse.ArgumentParser(description="Paired statistical comparison of encoders")
    ap.add_argument("--seg-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reference", default="ijepa")
    args = ap.parse_args()

    base = default_output_dir()
    run(
        Path(args.seg_dir) if args.seg_dir else base / "seg",
        Path(args.out) if args.out else base / "results" / "comparisons.json",
        args.reference,
    )


if __name__ == "__main__":
    main()

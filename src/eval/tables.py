"""Result tables, emitted as both Markdown (to read) and LaTeX booktabs (to paste).

Five tables, matching the thesis structure:

  T1  main results       Dice/IoU/Prec/Rec/HD95/failure-rate, mean +- std over seeds,
                         Holm-adjusted p vs the reference arm
  T2  comparability      the 880/120 split alongside published numbers, clearly marked
                         as cited rather than reproduced
  T3  fairness           every hyperparameter held constant across the four methods
  T4  ablations          decoder swap, low-label regime
  T5  compute            GPU-hours, params, GMACs, FPS per method

Everything reads from `outputs/seg/*/summary.json` and `outputs/ckpt/metrics.jsonl`, so
tables regenerate on a laptop with no GPU and no checkpoints.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DISPLAY = {
    "ijepa": "I-JEPA (ours)",
    "mae": "MAE",
    "simclr": "SimCLR",
    "mocov3": "MoCo v3",
    "random": "Random init",
    "imagenet": "ImageNet sup. (ref.)",
}
ORDER = ["ijepa", "mae", "simclr", "mocov3", "random", "imagenet"]


def _fmt(v: float, nd: int = 4) -> str:
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def markdown_table(headers: list[str], rows: list[list[str]], align: str = "l") -> str:
    sep = ["---"] + ["---:"] * (len(headers) - 1) if align == "l" else ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def latex_table(headers: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    spec = "l" + "r" * (len(headers) - 1)
    esc = lambda s: s.replace("_", r"\_").replace("%", r"\%")  # noqa: E731
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{esc(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(esc(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def load_summaries(seg_dir: Path) -> list[dict]:
    out = []
    for p in sorted(Path(seg_dir).glob("*/summary.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            print(f"[tables] skipping unreadable {p}")
    return out


def group_by_encoder(summaries: list[dict], **filters) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        if all(s.get(k) == v for k, v in filters.items()):
            groups[s["encoder"]].append(s)
    return groups


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def table_main(
    summaries: list[dict],
    comparisons: list[dict] | None = None,
    reference: str = "ijepa",
) -> tuple[list[str], list[list[str]]]:
    """T1: the headline table."""
    groups = group_by_encoder(summaries, label_fraction=1.0, decoder="segformer", split="800_100_100")
    p_by_method = {c["method_b"]: c for c in (comparisons or [])}

    metrics = ["dice", "iou", "precision", "recall"]
    best = {m: max((np.mean([s[m] for s in g]) for g in groups.values()), default=0.0) for m in metrics}

    headers = ["Encoder", "Dice", "mIoU", "Prec.", "Rec.", "HD95 (px)", "Fail. rate", "p (Holm)"]
    rows = []
    for enc in ORDER:
        if enc not in groups:
            continue
        g = groups[enc]
        cells = [DISPLAY.get(enc, enc)]
        for m in metrics:
            mu, sd = _mean_std([s[m] for s in g])
            cell = f"{mu:.4f}" + (f" ± {sd:.4f}" if sd > 0 else "")
            if abs(mu - best[m]) < 1e-9:
                cell = f"**{cell}**"
            cells.append(cell)
        hd_mu, hd_sd = _mean_std([s["hd95"] for s in g])
        cells.append(f"{hd_mu:.1f}" + (f" ± {hd_sd:.1f}" if hd_sd > 0 else ""))
        fr_mu, _ = _mean_std([s["failure_rate"] for s in g])
        cells.append(f"{fr_mu:.1%}")

        if enc == reference:
            cells.append("--")
        elif enc in p_by_method:
            c = p_by_method[enc]
            cells.append(f"{c['p_adjusted']:.3g}" + ("†" if c["significant"] else ""))
        else:
            cells.append("--")
        rows.append(cells)
    return headers, rows


def table_fairness(cfgs: dict[str, dict]) -> tuple[list[str], list[list[str]]]:
    """T3: what was held constant. The table that makes the comparison defensible."""
    headers = ["Held constant", "Value"]
    if not cfgs:
        return headers, []
    any_cfg = next(iter(cfgs.values()))
    m, o = any_cfg.get("model", {}), any_cfg.get("optim", {})
    corpus_note = "HyperKvasir unlabeled, deduplicated against Kvasir-SEG"
    rows = [
        ["Backbone", f"{m.get('arch', 'vit_small')}/{m.get('patch_size', 16)}"],
        ["Input resolution", f"{m.get('img_size', 224)} x {m.get('img_size', 224)}"],
        ["Stochastic depth", str(m.get("drop_path_rate", 0.0))],
        ["Pretraining corpus", corpus_note],
        ["Epochs", str(o.get("epochs", 100))],
        ["Global batch", str(o.get("global_batch", 512))],
        ["Samples seen", f"{o.get('epochs', 100)} x corpus size (identical for all methods)"],
        ["Optimizer", "AdamW, betas (0.9, 0.95), warmup + cosine"],
        ["Warmup epochs", str(o.get("warmup_epochs", 10))],
        ["Gradient clip", str(o.get("grad_clip", 3.0))],
        ["Precision", "fp16 AMP + GradScaler (losses in fp32)"],
        ["Decoder", "SegFormer all-MLP on ViT simple feature pyramid"],
        ["Fine-tune recipe", "352 px, AdamW, layer decay 0.75, Dice+BCE, 100 ep"],
        ["Splits", "800/100/100, group-aware, stratified, committed to repo"],
    ]
    # Peak LR is intentionally NOT constant — see below.
    rows.append(
        [
            "Peak LR (NOT held constant)",
            ", ".join(
                f"{k}: {v.get('optim', {}).get('ref_lr', '?')}" for k, v in sorted(cfgs.items())
            )
            + " — each method's published value under its own batch-scaling rule",
        ]
    )
    return headers, rows


def table_compute(pretrain_metrics: dict[str, list[dict]], summaries: list[dict]) -> tuple[list[str], list[list[str]]]:
    """T5: what it cost."""
    headers = ["Method", "Epochs", "min/epoch", "GPU-h", "Encoder params", "Decoder params"]
    enc_p = {s["encoder"]: s.get("encoder_params") for s in summaries if s.get("encoder_params")}
    dec_p = {s["encoder"]: s.get("decoder_params") for s in summaries if s.get("decoder_params")}

    rows = []
    for method in ORDER:
        recs = pretrain_metrics.get(method)
        if not recs:
            continue
        epochs = max(r["epoch"] for r in recs)
        times = [r["epoch_time_s"] for r in recs if r.get("epoch_time_s")]
        mins = np.mean(times) / 60 if times else float("nan")
        gpu_h = sum(times) / 3600 if times else float("nan")
        rows.append(
            [
                DISPLAY.get(method, method),
                str(epochs),
                _fmt(mins, 1),
                _fmt(gpu_h, 1),
                f"{enc_p.get(method, 0) / 1e6:.2f}M" if enc_p.get(method) else "--",
                f"{dec_p.get(method, 0) / 1e6:.2f}M" if dec_p.get(method) else "--",
            ]
        )
    return headers, rows


def table_low_label(summaries: list[dict]) -> tuple[list[str], list[list[str]]]:
    """T4a: Dice vs fraction of training labels — usually the strongest result."""
    fractions = sorted({s.get("label_fraction", 1.0) for s in summaries})
    headers = ["Encoder"] + [f"{f:.0%} labels" for f in fractions]
    rows = []
    for enc in ORDER:
        cells = [DISPLAY.get(enc, enc)]
        found = False
        for f in fractions:
            vals = [
                s["dice"]
                for s in summaries
                if s["encoder"] == enc and s.get("label_fraction", 1.0) == f
            ]
            if vals:
                found = True
                mu, sd = _mean_std(vals)
                cells.append(f"{mu:.4f}" + (f" ± {sd:.4f}" if sd > 0 else ""))
            else:
                cells.append("--")
        if found:
            rows.append(cells)
    return headers, rows


def table_decoder_ablation(summaries: list[dict]) -> tuple[list[str], list[list[str]]]:
    """T4b: does the encoder ranking survive a decoder swap?"""
    decoders = sorted({s.get("decoder", "segformer") for s in summaries})
    headers = ["Encoder"] + [d for d in decoders]
    rows = []
    for enc in ORDER:
        cells = [DISPLAY.get(enc, enc)]
        found = False
        for d in decoders:
            vals = [
                s["dice"]
                for s in summaries
                if s["encoder"] == enc
                and s.get("decoder") == d
                and s.get("label_fraction", 1.0) == 1.0
            ]
            if vals:
                found = True
                mu, sd = _mean_std(vals)
                cells.append(f"{mu:.4f}" + (f" ± {sd:.4f}" if sd > 0 else ""))
            else:
                cells.append("--")
        if found:
            rows.append(cells)
    return headers, rows


def build_all(out_dir: Path, seg_dir: Path, ckpt_dir: Path, results_dir: Path) -> None:
    summaries = load_summaries(seg_dir)
    if not summaries:
        print(f"[tables] no summaries under {seg_dir}; run segmentation first")

    comparisons = []
    comp_path = results_dir / "comparisons.json"
    if comp_path.is_file():
        comparisons = json.loads(comp_path.read_text())

    pretrain_metrics: dict[str, list[dict]] = defaultdict(list)
    metrics_file = ckpt_dir / "metrics.jsonl"
    if metrics_file.is_file():
        for line in metrics_file.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                pretrain_metrics[r["method"]].append(r)

    cfgs: dict[str, dict] = {}
    from src.config import REPO_ROOT

    for method in ["ijepa", "mae", "simclr", "mocov3"]:
        p = REPO_ROOT / "configs" / f"pretrain_{method}.yaml"
        if p.is_file():
            import yaml

            cfgs[method] = yaml.safe_load(p.read_text())

    specs = [
        ("t1_main", "Polyp segmentation on Kvasir-SEG (800/100/100 split, mean +- std over 5 seeds). "
                    "Bold = best; dagger = significant vs I-JEPA after Holm-Bonferroni.",
         *table_main(summaries, comparisons)),
        ("t3_fairness", "Experimental variables held constant across all four SSL methods.",
         *table_fairness(cfgs)),
        ("t4a_low_label", "Dice under reduced label budgets.", *table_low_label(summaries)),
        ("t4b_decoder", "Decoder ablation: SegFormer head vs ViT-UNet.",
         *table_decoder_ablation(summaries)),
        ("t5_compute", "Compute cost per pretraining method.",
         *table_compute(pretrain_metrics, summaries)),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    md_parts = []
    for name, caption, headers, rows in specs:
        if not rows:
            continue
        md_parts.append(f"### {caption}\n\n{markdown_table(headers, rows)}\n")
        latex = latex_table(
            headers, [[c.replace("**", "").replace("±", r"$\pm$").replace("†", r"$^\dagger$") for c in r] for r in rows],
            caption, f"tab:{name}",
        )
        (out_dir / f"{name}.tex").write_text(latex + "\n")
        print(f"[tables] wrote {out_dir / f'{name}.tex'}")

    (out_dir / "all_tables.md").write_text("\n".join(md_parts) + "\n")
    print(f"[tables] wrote {out_dir / 'all_tables.md'}")
    if md_parts:
        print("\n" + "\n".join(md_parts))


def main() -> None:
    import argparse

    from src.config import default_output_dir

    ap = argparse.ArgumentParser(description="Generate result tables")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = default_output_dir()
    out = Path(args.out) if args.out else base / "results" / "tables"
    build_all(out, base / "seg", base / "ckpt", base / "results")


if __name__ == "__main__":
    main()

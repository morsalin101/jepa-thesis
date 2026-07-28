"""All thesis figures.

Every function takes a data source and an output directory, and each is independently
runnable — `python -m src.viz.make_all --only masking`. Nothing here needs a GPU or a
checkpoint: figures read `metrics.jsonl`, `summary.json`, `test_per_image.json` and
`comparisons.json`, so they can be iterated locally while Kaggle trains.

Chart-form choices, briefly:
  * change-over-time (loss, Dice vs epoch, Dice vs label fraction) -> line
  * identity comparison across a few arms (k-NN accuracy) -> horizontal bar, sorted
  * distribution of a paired measure (per-image Dice) -> violin + box, not a bar of means
  * effect size with uncertainty -> forest plot with CIs, which is the honest form for
    "how much better, and how sure are we" — a bar chart of means hides both
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.viz.style import (
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    despine,
    encoder_style,
    present_encoders,
    save,
)


# ------------------------------------------------------------------ loading


def load_jsonl(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def load_seg_summaries(seg_dir: Path) -> list[dict]:
    out = []
    for p in sorted(Path(seg_dir).glob("*/summary.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            pass
    return out


def _no_data(name: str, what: str) -> None:
    print(f"[viz] skipping {name}: no {what} yet")


# ------------------------------------------------- 1. masking (method figure)


def fig_masking(out_dir: Path, image_path: Path | None = None, n_examples: int = 3, seed: int = 0):
    """The I-JEPA mask: one large context block and four disjoint target blocks.

    This is the figure that demonstrates the method was implemented as published — a
    single target block whose complement is the context (the common shortcut) looks
    obviously different here.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import torch

    from src.masks.multiblock import MaskCollator, masks_to_grid

    torch.manual_seed(seed)
    collator = MaskCollator(input_size=224, patch_size=16, num_enc_masks=1, num_pred_masks=4, min_keep=10)

    if image_path is not None and Path(image_path).is_file():
        from PIL import Image

        img = np.asarray(Image.open(image_path).convert("RGB").resize((224, 224))) / 255.0
        base = [torch.zeros(3, 224, 224) for _ in range(n_examples)]
    else:
        img = None
        base = [torch.zeros(3, 224, 224) for _ in range(n_examples)]

    _, enc_masks, pred_masks = collator(base)
    g = collator.height

    fig, axes = plt.subplots(1, n_examples, figsize=(2.3 * n_examples, 2.7))
    axes = np.atleast_1d(axes)
    ctx_color, tgt_color, unused_color = SERIES[0], SERIES[1], "#e8e7e2"

    def rgb(hex_str: str) -> tuple[float, float, float]:
        h = hex_str.lstrip("#")
        return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))

    for i, ax in enumerate(axes):
        ctx = masks_to_grid(enc_masks[0][i], g, g).numpy()
        # Union of the four target blocks, flattened to a single boolean mask. Drawing
        # them as four translucent layers would darken where blocks overlap and read as
        # a third category; they are one category.
        tgt = np.zeros((g, g), dtype=bool)
        for j in range(len(pred_masks)):
            tgt |= masks_to_grid(pred_masks[j][i], g, g).numpy()

        canvas = np.zeros((g, g, 3))
        canvas[:] = rgb(unused_color)
        canvas[ctx] = rgb(ctx_color)
        canvas[tgt] = rgb(tgt_color)

        if img is not None:
            ax.imshow(img, extent=(0, g, g, 0))
            ax.imshow(canvas, extent=(0, g, g, 0), alpha=0.72, interpolation="nearest")
        else:
            ax.imshow(canvas, extent=(0, g, g, 0), interpolation="nearest")

        # Patch grid, so the 14x14 tokenisation is legible.
        for k in range(g + 1):
            ax.axhline(k, color="white", lw=0.4)
            ax.axvline(k, color="white", lw=0.4)

        ax.set_xlim(0, g)
        ax.set_ylim(g, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title(
            f"context {int(ctx.sum())}  ·  targets {int(tgt.sum())}", fontsize=8, color=INK_SECONDARY
        )

    handles = [
        mpatches.Patch(facecolor=ctx_color, label="context block (encoder sees)"),
        mpatches.Patch(facecolor=tgt_color, label="4 target blocks (predicted)"),
        mpatches.Patch(facecolor=unused_color, label="unused"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("I-JEPA multi-block masking (14×14 patch grid, 224px input)", fontsize=10)
    fig.text(
        0.5, -0.19,
        "Context is a separately sampled 85–100% block with the target union removed — not the "
        "complement of the targets.\nContext counts are equal across panels because masks are "
        "truncated to the batch-wide minimum so they collate without padding.",
        ha="center", fontsize=6.5, color=INK_MUTED,
    )
    return save(fig, out_dir, "fig01_masking")


# ------------------------------------------------------------ 2. schedules


def fig_schedules(out_dir: Path, epochs: int = 100, iters_per_epoch: int = 194):
    """LR, weight decay and EMA momentum against training step.

    Worth a figure because two of the three are counter-intuitive: weight decay
    *increases* over training (0.04 -> 0.4) and momentum ramps to exactly 1.0, freezing
    the target encoder at the end.
    """
    import matplotlib.pyplot as plt

    from src.utils.schedulers import ScheduleSet, cosine_momentum

    s = ScheduleSet(iters_per_epoch, epochs, 10, 2.0e-4, 2.5e-4, 1.0e-6, 0.04, 0.4, (0.996, 1.0))
    steps = np.arange(0, s.total_steps + 1, max(1, s.total_steps // 500))
    ep = steps / iters_per_epoch

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.3))
    for ax, (vals, title, ylab, color) in zip(
        axes,
        [
            ([s.lr(int(t)) for t in steps], "Learning rate", "LR", SERIES[0]),
            ([s.wd(int(t)) for t in steps], "Weight decay", "WD", SERIES[1]),
            ([s.momentum(int(t)) for t in steps], "EMA momentum", "m", SERIES[2]),
        ],
    ):
        ax.plot(ep, vals, color=color)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
        despine(ax)

    axes[0].axvline(10, color=INK_MUTED, ls=":", lw=1)
    axes[0].annotate(
        "warmup ends", (10, 2.5e-4), xytext=(6, -10), textcoords="offset points",
        fontsize=7, color=INK_SECONDARY,
    )
    axes[2].plot(
        ep,
        [cosine_momentum(int(t), s.total_steps, (0.99, 1.0)) for t in steps],
        color=SERIES[3],
        ls="--",
    )
    # Offset the labels off their own curves so neither sits on a line.
    axes[2].annotate(
        "I-JEPA (linear)", (epochs * 0.42, s.momentum(int(s.total_steps * 0.42))),
        xytext=(0, 7), textcoords="offset points", fontsize=7, color=SERIES[2],
    )
    axes[2].annotate(
        "MoCo v3 (cosine)",
        (epochs * 0.30, cosine_momentum(int(s.total_steps * 0.30), s.total_steps, (0.99, 1.0))),
        xytext=(0, -13), textcoords="offset points", fontsize=7, color=SERIES[3],
    )
    axes[1].annotate(
        "weight decay increases\nover training (0.04 → 0.4)", (55, 0.16),
        fontsize=6.5, color=INK_MUTED,
    )
    return save(fig, out_dir, "fig02_schedules")


# ------------------------------------------------- 3. pretraining loss curves


def fig_pretrain_curves(out_dir: Path, metrics_path: Path):
    """Per-method pretraining loss against samples seen.

    Separate panels, not shared axes: the four objectives produce losses on completely
    different scales (smooth-L1 on normalised features vs MSE on pixels vs InfoNCE), so
    overlaying them on one axis would invite a comparison that is not meaningful. The
    shared x-axis (samples seen) is the controlled budget and *is* comparable.
    """
    import matplotlib.pyplot as plt

    recs = load_jsonl(metrics_path)
    if not recs:
        return _no_data("fig_pretrain_curves", "pretraining metrics")

    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_method[r["method"]].append(r)
    methods = present_encoders(by_method)

    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(2.1 * (len(methods) + 1), 2.4))
    axes = np.atleast_1d(axes)

    for ax, method in zip(axes, methods):
        rs = sorted(by_method[method], key=lambda r: r["epoch"])
        st = encoder_style(method)
        x = np.array([r.get("samples_seen", r["epoch"]) for r in rs]) / 1e6
        ax.plot(x, [r["loss"] for r in rs], color=st["color"])
        ax.set_title(st["label"], fontsize=9)
        ax.set_xlabel("samples seen (M)")
        despine(ax)
    axes[0].set_ylabel("training loss")

    # Wall-clock panel: the cost side of the same runs.
    ax = axes[-1]
    for i, method in enumerate(methods):
        rs = sorted(by_method[method], key=lambda r: r["epoch"])
        st = encoder_style(method)
        cum_h = np.cumsum([r.get("epoch_time_s", 0) for r in rs]) / 3600
        ax.plot([r["epoch"] for r in rs], cum_h, color=st["color"], label=st["label"])
    ax.set_title("Cumulative cost", fontsize=9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("GPU-hours")
    ax.legend(fontsize=6.5, loc="upper left")
    despine(ax)
    return save(fig, out_dir, "fig03_pretrain_curves")


# ----------------------------------------------------- 4. segmentation curves


def fig_seg_curves(out_dir: Path, seg_dir: Path):
    """Validation Dice against epoch, mean ± std over seeds, one line per encoder."""
    import matplotlib.pyplot as plt

    per_enc: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_enc_loss: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for p in sorted(Path(seg_dir).glob("*/metrics.jsonl")):
        for r in load_jsonl(p):
            if r.get("label_fraction", 1.0) != 1.0:
                continue
            per_enc[r["encoder"]][r["epoch"]].append(r["val_dice"])
            per_enc_loss[r["encoder"]][r["epoch"]].append(r["train_loss"])
    if not per_enc:
        return _no_data("fig_seg_curves", "segmentation metrics")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for enc in present_encoders(per_enc):
        st = encoder_style(enc)
        for ax, src, ylab in ((ax1, per_enc_loss[enc], "train loss"), (ax2, per_enc[enc], "val Dice")):
            eps = sorted(src)
            mu = np.array([np.mean(src[e]) for e in eps])
            sd = np.array([np.std(src[e], ddof=1) if len(src[e]) > 1 else 0.0 for e in eps])
            ax.plot(eps, mu, color=st["color"], label=st["label"])
            if sd.any():
                ax.fill_between(eps, mu - sd, mu + sd, color=st["color"], alpha=0.15, lw=0)
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylab)
            despine(ax)
    ax2.legend(loc="lower right", fontsize=7)
    fig.suptitle("Segmentation fine-tuning on Kvasir-SEG (mean ± s.d. over seeds)", fontsize=10)
    return save(fig, out_dir, "fig04_seg_curves")


# ------------------------------------------------- 5. per-image Dice spread


def fig_dice_distribution(out_dir: Path, seg_dir: Path):
    """Violin + box of per-image test Dice.

    A bar chart of mean Dice would hide the thing that matters clinically: the lower
    tail, where the model misses a polyp entirely. The violin shows it.
    """
    import matplotlib.pyplot as plt

    from src.eval.stats import aggregate_over_seeds, load_per_image_dice

    by_enc = load_per_image_dice(Path(seg_dir))
    if not by_enc:
        return _no_data("fig_dice_distribution", "per-image test results")
    dice = {e: aggregate_over_seeds(v) for e, v in by_enc.items()}
    encs = present_encoders(dice)

    fig, ax = plt.subplots(figsize=(1.15 * len(encs) + 2.0, 3.2))
    data = [list(dice[e].values()) for e in encs]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, enc in zip(parts["bodies"], encs):
        body.set_facecolor(encoder_style(enc)["color"])
        body.set_alpha(0.28)
        body.set_edgecolor("none")
    bp = ax.boxplot(data, widths=0.16, patch_artist=True, showfliers=False, medianprops={"color": INK})
    for patch, enc in zip(bp["boxes"], encs):
        patch.set_facecolor("white")
        patch.set_edgecolor(encoder_style(enc)["color"])
        patch.set_linewidth(1.4)

    ax.axhline(0.5, color=INK_MUTED, ls=":", lw=1)
    ax.annotate("Dice < 0.5: polyp effectively missed", (0.55, 0.505), fontsize=7, color=INK_SECONDARY)
    ax.set_xticks(range(1, len(encs) + 1))
    ax.set_xticklabels([encoder_style(e)["label"] for e in encs], rotation=12, ha="right")
    ax.set_ylabel("per-image Dice (test)")
    ax.set_ylim(0, 1.02)
    despine(ax)
    fig.suptitle("Distribution of per-image Dice on the held-out test set", fontsize=10)
    return save(fig, out_dir, "fig05_dice_distribution")


# ---------------------------------------------------------- 6. forest plot


def fig_paired_forest(out_dir: Path, comparisons_path: Path):
    """Median paired Dice difference vs I-JEPA, with 95% BCa CIs.

    The statistical headline. A CI that crosses zero says "no detectable difference"
    far more honestly than a p-value alone, and the effect size stays visible so a
    significant-but-tiny difference cannot masquerade as a result.
    """
    import matplotlib.pyplot as plt

    if not Path(comparisons_path).is_file():
        return _no_data("fig_paired_forest", "comparisons.json (run src.eval.stats)")
    comps = json.loads(Path(comparisons_path).read_text())
    if not comps:
        return _no_data("fig_paired_forest", "comparisons")

    # Largest difference at the top, the usual forest-plot reading order.
    comps = sorted(comps, key=lambda c: c["median_diff"])
    fig, ax = plt.subplots(figsize=(6.2, 0.55 * len(comps) + 1.6))
    ys = np.arange(len(comps))

    for y, c in zip(ys, comps):
        st = encoder_style(c["method_b"])
        ax.plot([c["ci_low"], c["ci_high"]], [y, y], color=st["color"], lw=2, solid_capstyle="round")
        ax.plot(c["median_diff"], y, marker=st["marker"], color=st["color"], ms=7,
                markeredgecolor="white", markeredgewidth=1.2)
        # Annotations live in a fixed right-hand column (axes fraction on x, data on y)
        # rather than trailing each interval — otherwise a negative CI pushes its label
        # across the zero line and the two collide.
        ax.annotate(
            f"{c['median_diff']:+.3f}   p={c['p_adjusted']:.2g}{'*' if c['significant'] else ''}",
            xy=(1.02, y), xycoords=("axes fraction", "data"),
            va="center", ha="left", fontsize=7.5, color=INK_SECONDARY, annotation_clip=False,
        )

    ax.axvline(0, color=INK_MUTED, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"vs {encoder_style(c['method_b'])['label']}" for c in comps])
    ax.set_xlabel("median paired Dice difference (I-JEPA − baseline)")
    ax.set_ylim(-0.6, len(comps) - 0.4)
    ax.margins(x=0.10)
    ax.grid(axis="y", visible=False)
    despine(ax, left=True)
    ax.annotate("← baseline better", xy=(0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(-6, 6), textcoords="offset points", ha="right", fontsize=7, color=INK_MUTED)
    ax.annotate("I-JEPA better →", xy=(0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(6, 6), textcoords="offset points", ha="left", fontsize=7, color=INK_MUTED)
    fig.suptitle("Paired per-image comparison with 95% BCa bootstrap CIs", fontsize=10)
    fig.text(0.01, -0.02, "* significant at α=0.05 after Holm–Bonferroni; n = test images",
             fontsize=7, color=INK_MUTED)
    return save(fig, out_dir, "fig06_paired_forest")


# --------------------------------------------------------- 7. low-label regime


def fig_low_label(out_dir: Path, seg_dir: Path):
    """Dice against the fraction of training labels used.

    Usually the strongest result in a study like this: pretraining differences are
    largest exactly where labels are scarce, which is also the regime that matters for
    a clinical dataset.
    """
    import matplotlib.pyplot as plt

    summaries = load_seg_summaries(Path(seg_dir))
    by: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in summaries:
        if s.get("decoder", "segformer") != "segformer":
            continue
        by[s["encoder"]][s.get("label_fraction", 1.0)].append(s["dice"])
    fractions = sorted({f for v in by.values() for f in v})
    if len(fractions) < 2:
        return _no_data("fig_low_label", "runs at more than one label fraction")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for enc in present_encoders(by):
        st = encoder_style(enc)
        fr = [f for f in fractions if f in by[enc]]
        mu = np.array([np.mean(by[enc][f]) for f in fr])
        sd = np.array([np.std(by[enc][f], ddof=1) if len(by[enc][f]) > 1 else 0.0 for f in fr])
        ax.plot([f * 100 for f in fr], mu, color=st["color"], marker=st["marker"], label=st["label"])
        if sd.any():
            ax.fill_between([f * 100 for f in fr], mu - sd, mu + sd, color=st["color"], alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set_xticks([f * 100 for f in fractions])
    ax.set_xticklabels([f"{f:.0%}" for f in fractions])
    ax.set_xlabel("fraction of training labels")
    ax.set_ylabel("test Dice")
    ax.legend(fontsize=7, loc="lower right")
    despine(ax)
    fig.suptitle("Label efficiency of the pretrained encoders", fontsize=10)
    return save(fig, out_dir, "fig07_low_label")


# ---------------------------------------------------------- 8. efficiency


def fig_efficiency(out_dir: Path, seg_dir: Path):
    """Dice against pretraining cost.

    Every point is direct-labelled and carries its own marker shape, because a scatter
    puts all pairs of colours in play at once and this palette is only validated for
    adjacent pairs. Colour here is reinforcement, not the identity channel.
    """
    import matplotlib.pyplot as plt

    summaries = [
        s for s in load_seg_summaries(Path(seg_dir))
        if s.get("label_fraction", 1.0) == 1.0 and s.get("decoder", "segformer") == "segformer"
    ]
    if not summaries:
        return _no_data("fig_efficiency", "segmentation summaries")

    by_enc: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        by_enc[s["encoder"]].append(s)

    hours = {}
    from src.config import default_output_dir

    for r in load_jsonl(default_output_dir() / "ckpt" / "metrics.jsonl"):
        hours[r["method"]] = hours.get(r["method"], 0.0) + r.get("epoch_time_s", 0) / 3600

    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for enc in present_encoders(by_enc):
        st = encoder_style(enc)
        dice = float(np.mean([s["dice"] for s in by_enc[enc]]))
        cost = hours.get(enc, 0.0)
        ax.scatter([cost], [dice], s=90, color=st["color"], marker=st["marker"],
                   edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(st["label"], (cost, dice), xytext=(8, 4), textcoords="offset points",
                    fontsize=8, color=INK)
    ax.set_xlabel("pretraining cost (GPU-hours)")
    ax.set_ylabel("test Dice")
    ax.margins(0.22)
    despine(ax)
    fig.suptitle("Segmentation quality against pretraining cost", fontsize=10)
    return save(fig, out_dir, "fig08_efficiency")


# ---------------------------------------------------------- 9. qualitative


def fig_qualitative(out_dir: Path, seg_dir: Path, kvasir_root: Path | None = None, n_rows: int = 5):
    """Input | ground truth | one column per encoder.

    Rows span the difficulty range by I-JEPA's own Dice quartiles and deliberately
    include the two worst cases. Showing failures is not a weakness in a thesis — a
    qualitative figure with only successes is the one a committee distrusts.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    from src.config import resolve_dataset_dir
    from src.data.kvasir_seg import find_image_and_mask_dirs

    runs: dict[str, Path] = {}
    for p in sorted(Path(seg_dir).glob("*/summary.json")):
        s = json.loads(p.read_text())
        if s.get("label_fraction", 1.0) != 1.0 or s.get("decoder", "segformer") != "segformer":
            continue
        if s.get("seed", 0) == 0 and (p.parent / "test_predictions.npz").is_file():
            runs[s["encoder"]] = p.parent / "test_predictions.npz"
    if not runs:
        return _no_data("fig_qualitative", "saved test predictions")

    encs = present_encoders(runs)
    preds = {e: np.load(runs[e], allow_pickle=False) for e in encs}
    ref = encs[0]
    stems = [str(s) for s in preds[ref]["stems"]]
    shapes = preds[ref]["shapes"]

    per_image = json.loads((runs[ref].parent / "test_per_image.json").read_text())
    dice_by_stem = {r["stem"]: r["dice"] for r in per_image}
    ranked = sorted(stems, key=lambda s: -dice_by_stem.get(s, 0))
    picks = [ranked[0], ranked[len(ranked) // 4], ranked[len(ranked) // 2]][: max(1, n_rows - 2)]
    picks += ranked[-2:]  # the two worst — failure cases

    root = Path(kvasir_root) if kvasir_root else resolve_dataset_dir("kvasir_seg")
    img_dir, mask_dir = find_image_and_mask_dirs(root)

    ncols = 2 + len(encs)
    fig, axes = plt.subplots(len(picks), ncols, figsize=(1.55 * ncols, 1.55 * len(picks)))
    axes = np.atleast_2d(axes)

    for r, stem in enumerate(picks):
        idx = stems.index(stem)
        h, w = int(shapes[idx][0]), int(shapes[idx][1])
        img_p = next((img_dir / f"{stem}{e}" for e in (".jpg", ".png") if (img_dir / f"{stem}{e}").is_file()), None)
        msk_p = next((mask_dir / f"{stem}{e}" for e in (".jpg", ".png") if (mask_dir / f"{stem}{e}").is_file()), None)
        img = np.asarray(Image.open(img_p).convert("RGB")) if img_p else np.zeros((h, w, 3), np.uint8)
        gt = (np.asarray(Image.open(msk_p).convert("L")) > 127) if msk_p else np.zeros((h, w), bool)

        axes[r, 0].imshow(img)
        axes[r, 1].imshow(gt, cmap="gray")
        for c, enc in enumerate(encs):
            pred = np.unpackbits(preds[enc][stem])[: h * w].reshape(h, w).astype(bool)
            ax = axes[r, 2 + c]
            ax.imshow(img)
            overlay = np.zeros((h, w, 4))
            rgb = tuple(int(encoder_style(enc)["color"].lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
            overlay[pred] = [*rgb, 0.55]
            ax.imshow(overlay)
            d = 2 * (pred & gt).sum() / max(1, pred.sum() + gt.sum())
            ax.set_xlabel(f"{d:.3f}", fontsize=7, color=INK_SECONDARY, labelpad=1)

        for c in range(ncols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].grid(False)

    titles = ["Input", "Ground truth"] + [encoder_style(e)["label"] for e in encs]
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=8)
    fig.suptitle("Qualitative results (Dice below each panel; last two rows are failure cases)", fontsize=10)
    return save(fig, out_dir, "fig09_qualitative")


# ------------------------------------------------------- 10. k-NN / probe


def fig_knn_probe(out_dir: Path, probe_path: Path):
    """k-NN and linear-probe accuracy on HyperKvasir-labelled.

    Representation quality with no fine-tuning at all — the cheapest signal available,
    and it can be read before any segmentation run finishes.
    """
    import matplotlib.pyplot as plt

    if not Path(probe_path).is_file():
        return _no_data("fig_knn_probe", "probe results (run src.eval.probe)")
    res = json.loads(Path(probe_path).read_text())
    encs = present_encoders(res)

    fig, ax = plt.subplots(figsize=(5.0, 0.42 * len(encs) + 1.8))
    ys = np.arange(len(encs))
    h = 0.36
    for i, enc in enumerate(encs):
        st = encoder_style(enc)
        knn = res[enc].get("knn_top1", 0) * 100
        lin = res[enc].get("linear_top1", 0) * 100
        ax.barh(ys[i] + h / 2, knn, height=h, color=st["color"], alpha=0.95)
        ax.barh(ys[i] - h / 2, lin, height=h, color=st["color"], alpha=0.45)
        ax.annotate(f"{knn:.1f}", (knn, ys[i] + h / 2), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK_SECONDARY)
        ax.annotate(f"{lin:.1f}", (lin, ys[i] - h / 2), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK_SECONDARY)
    ax.set_yticks(ys)
    ax.set_yticklabels([encoder_style(e)["label"] for e in encs])
    ax.set_xlabel("top-1 accuracy (%)")
    ax.grid(axis="y", visible=False)
    ax.annotate("solid = k-NN (k=20)   ·   faded = linear probe", (0.0, len(encs) - 0.35),
                fontsize=7.5, color=INK_MUTED)
    despine(ax, left=True)
    fig.suptitle("Frozen-feature evaluation on HyperKvasir-labelled (23 classes)", fontsize=10)
    return save(fig, out_dir, "fig10_knn_probe")


# ---------------------------------------------------- 11. embedding space


def fig_embedding_space(out_dir: Path, embed_path: Path):
    """2-D projection of frozen features, coloured by HyperKvasir class.

    Class count exceeds the categorical palette by a wide margin, so this uses a
    perceptually uniform continuous map for class index — identity is carried by the
    per-panel legend and the structure, not by memorising 23 hues.
    """
    import matplotlib.pyplot as plt

    if not Path(embed_path).is_file():
        return _no_data("fig_embedding_space", "embeddings (run src.eval.probe --save-embeddings)")
    data = np.load(embed_path, allow_pickle=True)
    methods = [k for k in data.files if k.endswith("_xy")]
    if not methods:
        return _no_data("fig_embedding_space", "projected embeddings")

    labels = data["labels"]
    fig, axes = plt.subplots(1, len(methods), figsize=(2.4 * len(methods), 2.6))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, methods):
        xy = data[key]
        ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="viridis", s=3, alpha=0.6, linewidths=0)
        ax.set_title(encoder_style(key[:-3])["label"], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("UMAP of frozen encoder features, coloured by class", fontsize=10)
    return save(fig, out_dir, "fig11_embedding_space")


FIGURES = {
    "masking": fig_masking,
    "schedules": fig_schedules,
    "pretrain_curves": fig_pretrain_curves,
    "seg_curves": fig_seg_curves,
    "dice_distribution": fig_dice_distribution,
    "paired_forest": fig_paired_forest,
    "low_label": fig_low_label,
    "efficiency": fig_efficiency,
    "qualitative": fig_qualitative,
    "knn_probe": fig_knn_probe,
    "embedding_space": fig_embedding_space,
}

"""Shared figure style for the thesis.

Every figure is vector PDF (for LaTeX) plus a PNG (to eyeball quickly), and every one
regenerates from JSON/JSONL on disk with no GPU and no checkpoints — so figures can be
iterated on a laptop while Kaggle is busy training.

**Colour policy.** The categorical palette is a validated 6-slot set: it clears the
lightness band, chroma floor, adjacent-pair CVD separation (worst ΔE 9.1, protan) and the
normal-vision floor (worst ΔE 19.6) on a light surface. Three slots sit below 3:1
contrast against white, which triggers the relief rule — so every figure here carries
either direct labels or a legend plus a companion table in `src/eval/tables.py`, and
identity is never carried by colour alone.

The same palette does **not** clear the all-pairs test (needed when marks are scattered
rather than ordered). The one scatter figure therefore direct-labels every point and
varies marker shape, so colour is reinforcement rather than the identity channel.

Colour is assigned to the *encoder*, fixed, and never re-assigned when a figure plots a
subset — a filter that repaints the survivors makes two figures in the same chapter
disagree about what blue means.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (light surface). Order is fixed; never cycle it.
COLORS = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
SERIES = [COLORS["blue"], COLORS["orange"], COLORS["aqua"], COLORS["yellow"], COLORS["magenta"], COLORS["green"]]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e3e2dd"
SURFACE = "#ffffff"

# Fixed identity per encoder. Slot order follows the order arms appear in the thesis.
ENCODER_COLOR = {
    "ijepa": COLORS["blue"],
    "mae": COLORS["orange"],
    "simclr": COLORS["aqua"],
    "mocov3": COLORS["yellow"],
    "random": COLORS["magenta"],
    "imagenet": COLORS["green"],
}
ENCODER_LABEL = {
    "ijepa": "I-JEPA (ours)",
    "mae": "MAE",
    "simclr": "SimCLR",
    "mocov3": "MoCo v3",
    "random": "Random init",
    "imagenet": "ImageNet sup.",
}
ENCODER_MARKER = {
    "ijepa": "o",
    "mae": "s",
    "simclr": "^",
    "mocov3": "D",
    "random": "v",
    "imagenet": "P",
}
ENCODER_ORDER = ["ijepa", "mae", "simclr", "mocov3", "random", "imagenet"]


def setup(base_size: int = 9) -> None:
    """Apply the thesis figure style. Recessive grid and axes, thin marks, no chartjunk."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": base_size,
            "axes.titlesize": base_size + 1,
            "axes.labelsize": base_size,
            "xtick.labelsize": base_size - 1,
            "ytick.labelsize": base_size - 1,
            "legend.fontsize": base_size - 1,
            "axes.edgecolor": INK_SECONDARY,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
            # Keep text as text in the PDF so LaTeX search/copy works.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def despine(ax, left: bool = False, bottom: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)


def save(fig, out_dir: Path, name: str, formats: tuple[str, ...] = ("pdf", "png")) -> list[Path]:
    """Write a figure in every requested format and report the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        written.append(p)
    plt.close(fig)
    print(f"[viz] {name}: " + ", ".join(str(p) for p in written))
    return written


def encoder_style(encoder: str) -> dict:
    """Colour + marker for an encoder, stable across every figure."""
    return {
        "color": ENCODER_COLOR.get(encoder, INK_SECONDARY),
        "marker": ENCODER_MARKER.get(encoder, "o"),
        "label": ENCODER_LABEL.get(encoder, encoder),
    }


def present_encoders(keys) -> list[str]:
    """Filter to known encoders in the canonical order, so colours never shift."""
    keys = set(keys)
    return [e for e in ENCODER_ORDER if e in keys] + sorted(keys - set(ENCODER_ORDER))

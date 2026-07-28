"""Segmentation metrics, computed at native resolution.

Three conventions are fixed here on purpose, because each is a place where published
polyp-segmentation numbers quietly disagree with each other:

1. **Native resolution.** Predictions are made at 352x352, then the *logits* are
   bilinearly upsampled to the image's original H x W and thresholded there, and compared
   against the original mask. Thresholding first and upsampling the binary mask, or
   comparing at 352 against a downsampled ground truth, inflates Dice by roughly 1-2
   points and is not comparable to the literature.

2. **Per-image mean, not dataset-aggregated.** Dice computed by pooling all pixels across
   the test set differs substantially from the mean of per-image Dice, because large
   polyps dominate the pooled version. The per-image mean is the clinically meaningful
   one and is this thesis's primary endpoint. Both are reported so the difference is
   visible.

3. **An explicit empty-prediction convention for HD95.** If the prediction is empty and
   the ground truth is not, the Hausdorff distance is undefined. Silently dropping those
   cases as NaN biases the metric toward methods that fail *completely* rather than
   partially — exactly backwards. We substitute the image diagonal and report how many
   times that happened.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-7


@dataclass
class ImageResult:
    stem: str
    dice: float
    iou: float
    precision: float
    recall: float
    hd95: float
    hd95_substituted: bool
    gt_area: float
    pred_area: float
    height: int
    width: int
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Dice / IoU / precision / recall for one boolean image pair.

    When both masks are empty the prediction is perfect, so Dice and IoU are 1.0. That
    case does not arise in Kvasir-SEG (every image contains a polyp) but the convention
    matters for the external CVC-ClinicDB evaluation.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())

    if tp + fp + fn == 0:
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}

    return {
        "dice": 2 * tp / (2 * tp + fp + fn + EPS),
        "iou": tp / (tp + fp + fn + EPS),
        "precision": tp / (tp + fp + EPS) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn + EPS) if (tp + fn) > 0 else 0.0,
    }


def _surface_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distances from every boundary pixel of `a` to the nearest boundary pixel of `b`."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    a_border = a ^ binary_erosion(a)
    b_border = b ^ binary_erosion(b)
    if not a_border.any() or not b_border.any():
        return np.array([])
    dt = distance_transform_edt(~b_border)
    return dt[a_border]


def hd95(pred: np.ndarray, gt: np.ndarray, diagonal: float) -> tuple[float, bool]:
    """95th-percentile symmetric Hausdorff distance in pixels.

    Returns (value, substituted). `substituted` is True when one mask was empty and the
    image diagonal was used instead — see module docstring.
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0, False
    if not pred.any() or not gt.any():
        return float(diagonal), True

    try:
        d1 = _surface_distances(pred, gt)
        d2 = _surface_distances(gt, pred)
    except ImportError:
        return float("nan"), False
    if d1.size == 0 or d2.size == 0:
        return float(diagonal), True
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95))), False


def logits_to_native_mask(
    logits: torch.Tensor, height: int, width: int, threshold: float = 0.5
) -> np.ndarray:
    """Upsample logits to native resolution, then threshold. Order matters — see module docstring."""
    if logits.ndim == 3:
        logits = logits.unsqueeze(0)
    up = F.interpolate(logits.float(), size=(height, width), mode="bilinear", align_corners=False)
    return (torch.sigmoid(up)[0, 0] > threshold).cpu().numpy()


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    dataset,
    device: str = "cuda",
    amp_dtype: torch.dtype | None = None,
    threshold: float = 0.5,
    compute_hd95: bool = True,
    save_predictions: Path | None = None,
) -> list[ImageResult]:
    """Run the model over a dataset and score every image at its native resolution.

    Batch size is 1 because each image has a different native size; the cost is
    negligible (100 test images) and it keeps the resolution handling unambiguous.
    """
    from PIL import Image

    model.eval()
    results: list[ImageResult] = []
    preds_to_save: dict[str, np.ndarray] = {}

    for idx in range(len(dataset)):
        img, _mask = dataset[idx][:2]
        img_t = img.unsqueeze(0).to(device)

        with torch.autocast(
            device_type="cuda" if str(device).startswith("cuda") else "cpu",
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None,
        ):
            logits = model(img_t)

        # Ground truth is read from disk at full resolution, never from the resized
        # tensor the model saw.
        gt_path = dataset.native_mask_path(idx)
        gt = np.asarray(Image.open(gt_path).convert("L"), dtype=np.uint8) > 127
        h, w = gt.shape

        pred = logits_to_native_mask(logits, h, w, threshold)
        m = binary_metrics(pred, gt)
        diag = float(np.hypot(h, w))
        hd, sub = hd95(pred, gt, diag) if compute_hd95 else (float("nan"), False)

        stem = dataset.samples[idx][0].stem
        results.append(
            ImageResult(
                stem=stem,
                dice=m["dice"],
                iou=m["iou"],
                precision=m["precision"],
                recall=m["recall"],
                hd95=hd,
                hd95_substituted=sub,
                gt_area=float(gt.mean()),
                pred_area=float(pred.mean()),
                height=h,
                width=w,
            )
        )
        if save_predictions is not None:
            preds_to_save[stem] = np.packbits(pred)  # 8x smaller than bool

    if save_predictions is not None and preds_to_save:
        save_predictions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_predictions,
            shapes=np.array([[r.height, r.width] for r in results]),
            stems=np.array([r.stem for r in results]),
            **preds_to_save,
        )
    return results


def aggregate(results: list[ImageResult]) -> dict[str, float]:
    """Summary statistics over a set of per-image results."""
    if not results:
        return {}
    d = np.array([r.dice for r in results])
    i = np.array([r.iou for r in results])
    p = np.array([r.precision for r in results])
    rc = np.array([r.recall for r in results])
    hd = np.array([r.hd95 for r in results], dtype=float)

    return {
        "n": len(results),
        "dice": float(d.mean()),
        "dice_std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "dice_median": float(np.median(d)),
        "iou": float(i.mean()),
        "precision": float(p.mean()),
        "recall": float(rc.mean()),
        "hd95": float(np.nanmean(hd)),
        "hd95_substituted": int(sum(r.hd95_substituted for r in results)),
        # The most clinically interpretable single number: how often the model
        # essentially missed the polyp.
        "failure_rate": float((d < 0.5).mean()),
    }


def dice_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Fast in-training Dice at model resolution, for the val-selection signal only.

    Deliberately separate from the native-resolution path above: this one runs every
    epoch on the GPU and only needs to rank checkpoints, not to be reported.
    """
    pred = (torch.sigmoid(logits) > threshold).float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return float(((2 * inter + EPS) / (denom + EPS)).mean())

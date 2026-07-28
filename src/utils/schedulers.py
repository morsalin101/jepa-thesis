"""Closed-form LR / weight-decay / EMA-momentum schedules.

The official i-jepa implementation uses stateful scheduler *objects* whose `.step()`
advances an internal counter, and on resume it replays them with

    for _ in range(start_epoch * ipe):
        scheduler.step(); wd_scheduler.step(); next(momentum_scheduler)

That works, but it makes the schedule position implicit state that must be replayed
exactly. On Kaggle, where a run is chopped into 2-3 sessions, we want the schedule to be
a pure function of `global_step` so a resume cannot drift. These functions produce
numerically identical values to the reference implementation when called with
`step = scheduler._step`.

Reference: reference/ijepa/src/utils/schedulers.py
"""
from __future__ import annotations

import math


def warmup_cosine_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    start_lr: float,
    ref_lr: float,
    final_lr: float = 0.0,
) -> float:
    """Linear warmup start_lr -> ref_lr, then cosine ref_lr -> final_lr.

    Mirrors `WarmupCosineSchedule`, whose `T_max` is `total_steps - warmup_steps`.
    """
    if step < warmup_steps:
        progress = step / max(1, warmup_steps)
        return start_lr + progress * (ref_lr - start_lr)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    lr = final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(final_lr, lr)


def cosine_wd(step: int, total_steps: int, ref_wd: float, final_wd: float = 0.0) -> float:
    """Cosine weight decay, ref_wd -> final_wd.

    Note the direction: i-jepa *increases* WD over training (0.04 -> 0.4), so the
    clamp has to work in both directions. Mirrors `CosineWDSchedule`.
    """
    progress = min(1.0, step / max(1, total_steps))
    wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(final_wd, wd) if final_wd <= ref_wd else min(final_wd, wd)


def linear_momentum(step: int, total_steps: int, ema: tuple[float, float]) -> float:
    """EMA momentum ramp, ema[0] -> ema[1] linearly.

    i-jepa builds this as a generator over `int(ipe * num_epochs * ipe_scale) + 1`
    values; this is the same sequence indexed directly. The multiply-then-divide order
    is kept as-is so the values are bitwise identical to the reference generator.
    """
    lo, hi = ema
    step = min(step, total_steps)
    return lo + step * (hi - lo) / max(1, total_steps)


def cosine_momentum(step: int, total_steps: int, m: tuple[float, float]) -> float:
    """Cosine momentum ramp, used by MoCo v3 (0.99 -> 1.0)."""
    lo, hi = m
    progress = min(1.0, step / max(1, total_steps))
    return hi - (hi - lo) * 0.5 * (1.0 + math.cos(math.pi * progress))


class ScheduleSet:
    """Bundles the three schedules for one run and applies them to an optimizer.

    Holds no mutable step state of its own — `step` is always passed in — so a
    checkpoint only needs `global_step` to restore the schedule exactly.
    """

    def __init__(
        self,
        iters_per_epoch: int,
        epochs: int,
        warmup_epochs: int,
        start_lr: float,
        ref_lr: float,
        final_lr: float,
        weight_decay: float,
        final_weight_decay: float,
        ema: tuple[float, float],
        ipe_scale: float = 1.0,
    ) -> None:
        self.iters_per_epoch = iters_per_epoch
        self.total_steps = max(1, int(ipe_scale * epochs * iters_per_epoch))
        self.warmup_steps = int(warmup_epochs * iters_per_epoch)
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.ref_wd = weight_decay
        self.final_wd = final_weight_decay
        self.ema = tuple(ema)

    def lr(self, step: int) -> float:
        return warmup_cosine_lr(
            step, self.warmup_steps, self.total_steps, self.start_lr, self.ref_lr, self.final_lr
        )

    def wd(self, step: int) -> float:
        return cosine_wd(step, self.total_steps, self.ref_wd, self.final_wd)

    def momentum(self, step: int) -> float:
        return linear_momentum(step, self.total_steps, self.ema)

    def apply(self, optimizer, step: int) -> tuple[float, float]:
        """Write lr/wd into the optimizer's param groups. Returns (lr, wd).

        Groups flagged `WD_exclude` (biases and 1-D params such as LayerNorm weights)
        keep weight_decay=0 — decaying them measurably hurts ViT training and the
        reference implementation excludes them too.
        """
        lr, wd = self.lr(step), self.wd(step)
        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
            if not group.get("WD_exclude", False):
                group["weight_decay"] = wd
        return lr, wd


def param_groups_with_wd_exclusion(*modules, base_lr: float = 0.0) -> list[dict]:
    """Split parameters into decayed / not-decayed groups.

    Biases and any 1-D parameter (LayerNorm weights, the mask token, class tokens) are
    excluded from weight decay, matching `init_opt` in reference/ijepa/src/helper.py.
    """
    decay, no_decay = [], []
    for module in modules:
        if module is None:
            continue
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if (p.ndim == 1 or name.endswith(".bias")) else decay).append(p)
    groups: list[dict] = []
    if decay:
        groups.append({"params": decay, "lr": base_lr})
    if no_decay:
        groups.append({"params": no_decay, "lr": base_lr, "WD_exclude": True, "weight_decay": 0.0})
    return groups

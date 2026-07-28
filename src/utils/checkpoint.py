"""Resumable checkpointing.

A 100-epoch pretraining run does not fit in one Kaggle session, so every run is chopped
into 2-3 sessions and *must* resume exactly. The previous version of this project saved
only encoder weights, which is not resumable: you lose AdamW's second-moment estimates
(thousands of steps to re-warm), the GradScaler's loss scale (~2000 steps), and the
position in the LR/WD/EMA schedules.

Two invariants worth stating because violating either is silent, not loud:

* **Atomic writes.** A session killed mid-`torch.save` leaves a truncated file. Since
  the next session loads exactly that file, one badly-timed kill would otherwise cost
  the whole run. We write to `.tmp` and `os.replace`, which is atomic on POSIX.
* **Refuse, don't warn, on config drift.** Resuming a checkpoint whose config differs
  changes `total_steps`, which shifts every schedule. That produces a run that trains
  fine and means nothing. The config hash mismatch raises.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch

from src.utils.ddp import get_world_size, is_main, unwrap

FORMAT_VERSION = 2


def _state(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, torch.nn.Module):
        return unwrap(obj).state_dict()
    return obj.state_dict()


def build_checkpoint(
    *,
    run_id: str,
    config_hash: str,
    resolved_cfg: dict[str, Any],
    modules: dict[str, Any],
    optimizer: Any,
    scaler: Any,
    global_step: int,
    epoch: int,
    total_steps: int,
    base_seed: int,
    loss_history: list[float],
    wall_clock_s: float,
    sessions: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the full resume state.

    `epoch` is the *next* epoch to run, so a checkpoint written after finishing epoch 40
    carries epoch=41 and resuming needs no off-by-one reasoning at the call site.
    """
    ckpt: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "run_id": run_id,
        "config_hash": config_hash,
        "resolved_cfg": resolved_cfg,
        "modules": {k: _state(v) for k, v in modules.items()},
        "optimizer": _state(optimizer),
        "scaler": _state(scaler),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "total_steps": int(total_steps),
        "base_seed": int(base_seed),
        "world_size": get_world_size(),
        "loss_history": list(loss_history),
        "wall_clock_s": float(wall_clock_s),
        "sessions": list(sessions),
    }
    if extra:
        ckpt.update(extra)
    return ckpt


def save_checkpoint(ckpt: dict[str, Any], path: str | os.PathLike[str]) -> Path:
    """Atomically write a checkpoint. Rank 0 only; other ranks return the path."""
    path = Path(path)
    if not is_main():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    return path


def find_checkpoint(candidates: list[str | os.PathLike[str]]) -> Path | None:
    """First readable checkpoint from an ordered candidate list.

    Order matters: the mid-run Kaggle Dataset push comes first, the previous kernel
    version's committed output second (it is at most one session stale), the local
    working directory last (only valid within the current session).
    """
    for c in candidates:
        p = Path(c)
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


class ResumeMismatch(RuntimeError):
    """The checkpoint cannot be resumed from. Fatal — see `RunIdMismatch` for the
    benign case."""


class RunIdMismatch(ResumeMismatch):
    """The checkpoint belongs to a different run entirely.

    Distinct from config drift on purpose. A foreign checkpoint just means "not mine" —
    the caller should ignore it and start fresh. Config drift *within* the same run_id
    means the experiment definition changed under a resume, which silently shifts every
    schedule, and that must stop the run.
    """


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    expect_run_id: str,
    expect_config_hash: str,
    modules: dict[str, Any],
    optimizer: Any = None,
    scaler: Any = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Restore state in place and return the checkpoint's bookkeeping fields.

    Raises `ResumeMismatch` if the checkpoint belongs to a different run or config.
    That is deliberate — the caller should either start fresh or fix the config, never
    silently continue from an incompatible state.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    if ckpt.get("format_version") != FORMAT_VERSION:
        raise ResumeMismatch(
            f"checkpoint format v{ckpt.get('format_version')} != v{FORMAT_VERSION} ({path})"
        )
    if ckpt.get("run_id") != expect_run_id:
        raise RunIdMismatch(
            f"checkpoint is for run {ckpt.get('run_id')!r}, not {expect_run_id!r} ({path})"
        )
    if ckpt.get("config_hash") != expect_config_hash:
        raise ResumeMismatch(
            f"config drift: checkpoint hash {ckpt.get('config_hash')} != {expect_config_hash}. "
            "Something in the experiment config changed, which would shift the LR/WD/EMA "
            f"schedules. Start a new run_id or restore the original config. ({path})"
        )

    saved = ckpt.get("modules", {})
    for name, module in modules.items():
        if module is None:
            continue
        if name not in saved or saved[name] is None:
            if strict:
                raise ResumeMismatch(f"checkpoint has no state for module {name!r} ({path})")
            continue
        target = unwrap(module) if isinstance(module, torch.nn.Module) else module
        target.load_state_dict(saved[name])

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    return ckpt


def session_record(start_time: float, epochs_done: int, accelerator: str) -> dict[str, Any]:
    """One row of the `sessions` audit trail.

    Worth keeping: it is where the thesis's compute table comes from, and it makes a
    mixed-hardware run (some sessions T4 x2, some P100) visible instead of invisible.
    """
    return {
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
        "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(time.time() - start_time, 1),
        "epochs": epochs_done,
        "accelerator": accelerator,
        "world_size": get_world_size(),
    }


def _extract_prefix(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Pull out `prefix.*` keys with the prefix stripped."""
    n = len(prefix)
    return {k[n:]: v for k, v in state.items() if k.startswith(prefix)}


def export_encoder(
    ckpt_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    prefer_target: bool = True,
) -> Path:
    """Strip a training checkpoint down to publishable encoder weights (~88 MB).

    Every SSL model here saves its whole state under `modules["model"]`, so the encoder
    is recovered by prefix rather than by a separate key. We prefer the *target* (EMA)
    encoder where one exists (I-JEPA, MoCo v3): the EMA weights are what those papers
    evaluate and they transfer better than the online encoder. MAE and SimCLR have no
    target branch and fall through to `encoder.`.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("modules", {}).get("model")
    if state is None:
        raise KeyError(f"checkpoint has no modules['model'] state: {ckpt_path}")

    source = ""
    enc: dict[str, Any] = {}
    if prefer_target:
        enc = _extract_prefix(state, "target_encoder.")
        source = "target_encoder"
    if not enc:
        enc = _extract_prefix(state, "encoder.")
        source = "encoder"
    if not enc:
        raise KeyError(f"no 'encoder.' or 'target_encoder.' keys in checkpoint {ckpt_path}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": enc,
            "source_module": source,
            "run_id": ckpt.get("run_id"),
            "config_hash": ckpt.get("config_hash"),
            "resolved_cfg": ckpt.get("resolved_cfg"),
            "epoch": ckpt.get("epoch"),
            "global_step": ckpt.get("global_step"),
            "wall_clock_s": ckpt.get("wall_clock_s"),
        },
        out,
    )
    return out

"""Minimal DDP helpers for Kaggle.

Kaggle gives you T4 x2 (world_size 2) or a single P100. There is no torchrun and no
srun, so a run is launched with `torch.multiprocessing.spawn` from inside the notebook
and the rendezvous env vars are set by hand.

Single-GPU and CPU are first-class: every helper degrades to a no-op so the same engine
code runs unchanged on a Mac for smoke tests.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return get_rank() == 0


def setup(rank: int, world_size: int, backend: str | None = None, port: str = "29517") -> None:
    """Initialise the process group. Safe to call with world_size=1 (does nothing)."""
    if world_size <= 1:
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", port)
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if is_dist():
        dist.barrier()


def all_reduce_mean(value: torch.Tensor | float) -> float:
    """Average a scalar across ranks. Used for logging, not for gradients."""
    if not is_dist():
        return float(value)
    t = value.detach().clone() if isinstance(value, torch.Tensor) else torch.tensor(float(value))
    t = t.to(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / dist.get_world_size())


class GatherWithGrad(torch.autograd.Function):
    """all_gather that propagates gradients back to the local shard.

    Plain `dist.all_gather` detaches — using it for contrastive embeddings would give
    each rank gradients only from its own slice, quietly reducing SimCLR's effective
    negative count from (global_batch - 1) back to (per_gpu_batch - 1). This is the
    standard fix and it is why SimCLR here really does see 511 negatives at
    global_batch 512, not 255.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if not is_dist():
            return x
        ctx.rank = dist.get_rank()
        ctx.batch = x.shape[0]
        out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x.contiguous())
        return torch.cat(out, dim=0)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):  # type: ignore[override]
        if not is_dist():
            return grad
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        start = ctx.rank * ctx.batch
        return grad[start : start + ctx.batch]


def gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    return GatherWithGrad.apply(x)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DistributedDataParallel so state_dict keys have no `module.` prefix."""
    return model.module if hasattr(model, "module") else model


@contextmanager
def main_process_first():
    """Let rank 0 run a block (e.g. a download) before the others enter it."""
    if not is_dist():
        yield
        return
    if not is_main():
        dist.barrier()
    yield
    if is_main():
        dist.barrier()


def epoch_seed(base_seed: int, epoch: int) -> int:
    """Deterministic per-epoch seed.

    We derive the data order from (base_seed, epoch) rather than checkpointing and
    replaying RNG state. Bitwise RNG restoration across a DDP restart is fragile —
    worker count, cuDNN autotune and kernel selection all perturb it — whereas this
    reproduces the exact shuffle and augmentation stream for epoch k regardless of
    which session runs it.
    """
    return (base_seed * 100_003 + epoch) % (2**31 - 1)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Shapes are fixed for the whole run (masks are truncated to min_keep), so
        # autotuning pays off and never re-triggers.
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int, base: int = 0) -> None:
    import random

    import numpy as np

    s = (base + worker_id) % (2**31 - 1)
    random.seed(s)
    np.random.seed(s % (2**32))
    torch.manual_seed(s)

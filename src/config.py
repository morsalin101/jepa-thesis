"""Central configuration.

Two things this module owns that the rest of the codebase must not duplicate:

1. **The AMP capability gate** (`amp_config`). Kaggle hands out T4 (sm_75) or, less
   happily, P100 (sm_60). Turing has fp16 tensor cores but *no bf16 hardware* — a
   bfloat16 autocast there is emulated and runs slower than fp32. P100 has no tensor
   cores at all but does support fp16 AMP fine. So we pick the dtype from the compute
   capability rather than assuming, and we never abort on old hardware: a slow session
   beats a dead one.

2. **Global-batch pinning** (`derive_batch`). T4 x2 gives world_size 2, P100 gives 1.
   If per-GPU batch stayed constant the *global* batch would silently halve mid-run and
   the LR schedule, contrastive negative count and EMA schedule would all shift —
   invalidating the experiment. We pin the global batch and derive per-GPU from it.

Configs are dataclasses overlaid with a YAML file from `configs/`. There is deliberately
no module-level singleton: dataset resolution touches the filesystem and must not run at
import time (the previous version raised on `import src.config` when data was absent).
Call `build_config(...)` explicitly.
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ViT variants, matching facebookresearch/ijepa's model zoo exactly.
VIT_ARCHS: dict[str, dict[str, int]] = {
    "vit_tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
    "vit_small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "vit_base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "vit_large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
}


# ------------------------------------------------------------------ environment

def on_kaggle() -> bool:
    return os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def default_output_dir() -> Path:
    """Persisted-within-session output. Capped at 20 GiB on Kaggle."""
    return Path("/kaggle/working") if on_kaggle() else REPO_ROOT / "outputs"


def scratch_dir() -> Path:
    """Large scratch space that does NOT count against the 20 GiB output cap.

    Kaggle gives ~60 GiB here; it is wiped when the session ends. Use it for
    downloads and intermediate archives, never for anything you need to keep.
    """
    if on_kaggle():
        for c in (Path("/kaggle/tmp"), Path("/kaggle/temp")):
            if c.parent.exists():
                c.mkdir(parents=True, exist_ok=True)
                return c
    return Path("/tmp")


# ------------------------------------------------------- AMP / device capability

@dataclass(frozen=True)
class AmpConfig:
    """Resolved autocast settings for whatever accelerator we actually landed on."""

    device: str  # "cuda" | "mps" | "cpu"
    dtype: Any  # torch.dtype or None (None => run in fp32)
    use_scaler: bool
    sm: int  # compute capability as major*10+minor, 0 if not CUDA
    name: str
    speed_factor: float  # rough throughput vs a single T4, for time estimates

    @property
    def enabled(self) -> bool:
        return self.dtype is not None


def amp_config(force_fp32: bool = False) -> AmpConfig:
    """Pick device + autocast dtype from the actual hardware.

    sm_80+ (A100/L4/A10) -> bfloat16, no GradScaler needed (bf16 has fp32 range).
    sm_70/75 (V100/T4)   -> float16 + GradScaler. Tensor cores; bf16 would be emulated.
    sm_60/61 (P100)      -> float16 + GradScaler. No tensor cores, ~2x slower, but works.
    older / non-CUDA     -> fp32.
    """
    try:
        import torch
    except ImportError:  # figure scripts run without torch
        return AmpConfig("cpu", None, False, 0, "cpu (no torch)", 0.02)

    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability(0)
            sm = major * 10 + minor
            name = torch.cuda.get_device_name(0)
            if force_fp32:
                return AmpConfig("cuda", None, False, sm, name, 1.0)
            if sm >= 80:
                return AmpConfig("cuda", torch.bfloat16, False, sm, name, 3.0)
            if sm >= 70:
                return AmpConfig("cuda", torch.float16, True, sm, name, 1.0)
            if sm >= 60:
                warnings.warn(
                    f"{name} (sm_{sm}) has no tensor cores; expect roughly 2x slower "
                    "training than a T4. Continuing in fp16. For a faster run, set "
                    "Accelerator -> GPU T4 x2 in the Kaggle session options.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return AmpConfig("cuda", torch.float16, True, sm, name, 0.45)
            warnings.warn(f"{name} (sm_{sm}) is too old for AMP; running fp32.", RuntimeWarning)
            return AmpConfig("cuda", None, False, sm, name, 0.2)
        except Exception as e:  # noqa: BLE001 - CUDA present but unusable
            warnings.warn(f"CUDA present but unusable ({e}); falling back to CPU.", RuntimeWarning)

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return AmpConfig("mps", None, False, 0, "apple-mps", 0.1)
    return AmpConfig("cpu", None, False, 0, "cpu", 0.02)


def derive_batch(global_batch: int, world_size: int, accum_steps: int = 1) -> int:
    """Per-GPU batch size that exactly preserves `global_batch`.

    Raises rather than rounding: a global batch that silently drifts between sessions
    is the kind of bug that invalidates a whole experiment without ever erroring.
    """
    denom = world_size * accum_steps
    if global_batch % denom != 0:
        raise ValueError(
            f"global_batch={global_batch} is not divisible by world_size({world_size}) "
            f"* accum_steps({accum_steps})={denom}. Adjust accum_steps so the global "
            "batch is preserved exactly."
        )
    return global_batch // denom


# ----------------------------------------------------------- dataset resolution

def _count_images(d: Path, recursive: bool = False) -> int:
    it = d.rglob("*") if recursive else d.iterdir()
    n = 0
    for p in it:
        if p.suffix.lower() in IMAGE_EXTS:
            n += 1
            if n > 4:  # cheap existence probe; callers only need "has images"
                if not recursive:
                    return n
    return n


def _has_images(d: Path) -> bool:
    """True if `d` contains images at any depth, short-circuiting on the first hit.

    Directly-nested counting is not enough: class-labelled datasets keep their images
    in per-class leaf folders, so the root of such a tree has none of its own.
    """
    try:
        for p in d.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTS:
                return True
    except OSError:
        pass
    return False


def resolve_dataset_dir(
    kind: str,
    override: str | os.PathLike[str] | None = None,
    required_subdirs: Sequence[str] = (),
) -> Path:
    """Locate a dataset directory, on Kaggle or locally.

    `kind` is one of the keys in `_DATASET_CANDIDATES`. On Kaggle we try the known
    mount points first, then fall back to a recursive scan of /kaggle/input picking the
    directory with the most images — Kaggle occasionally nests dataset archives one
    level deeper than the slug suggests, and that fallback has already saved this
    project once.
    """
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"dataset override path does not exist: {p}")
        return p

    candidates = _DATASET_CANDIDATES.get(kind)
    if candidates is None:
        raise KeyError(f"unknown dataset kind {kind!r}; known: {sorted(_DATASET_CANDIDATES)}")

    roots = [Path("/kaggle/input")] if on_kaggle() else [REPO_ROOT / "data"]

    def ok(d: Path) -> bool:
        if not d.is_dir():
            return False
        if required_subdirs and not all((d / s).is_dir() for s in required_subdirs):
            return False
        return required_subdirs != () or _count_images(d) > 0

    for root in roots:
        for rel in candidates:
            c = root / rel
            if ok(c):
                print(f"[data] {kind}: using {c}")
                return c

    # Kaggle has changed how it nests mounts over time — it used to be
    # /kaggle/input/<slug>/, and can now be /kaggle/input/datasets/<owner>/<slug>/.
    # So rather than assuming a depth, look for a directory whose *name* matches the
    # leaf of any candidate, anywhere in the tree. This is checked before the
    # "most images wins" heuristic because that heuristic picks a leaf class folder
    # for class-nested datasets, which is exactly the wrong answer.
    wanted = {Path(rel).name for rel in candidates}

    def name_ok(d: Path) -> bool:
        if required_subdirs and not all((d / s).is_dir() for s in required_subdirs):
            return False
        # Images at ANY depth: a class-nested tree has none in its own root.
        return bool(required_subdirs) or _has_images(d)

    for root in roots:
        if not root.exists():
            continue
        by_name = [
            d for d in root.rglob("*") if d.is_dir() and d.name in wanted and name_ok(d)
        ]
        if by_name:
            best = sorted(by_name, key=lambda p: (len(p.parts), str(p)))[0]
            print(f"[data] {kind}: using {best} (matched by directory name)")
            return best

    # Last resort: the directory holding the most images.
    for root in roots:
        if not root.exists():
            continue
        print(f"[data] {kind}: candidates missed, scanning {root} recursively ...")
        matches: list[tuple[Path, int]] = []
        for sub in root.rglob("*"):
            if sub.is_dir() and ok(sub):
                matches.append((sub, _count_images(sub, recursive=True)))
        matches.sort(key=lambda t: -t[1])
        for p, n in matches[:10]:
            print(f"[data]   candidate: {p}  ({n} images)")
        if matches:
            print(f"[data] {kind}: using {matches[0][0]}")
            return matches[0][0]

    raise FileNotFoundError(
        f"Could not locate dataset {kind!r} under {[str(r) for r in roots]}. "
        f"Tried {candidates}. On Kaggle, attach it via kernel-metadata.json -> "
        "dataset_sources; locally, place it under data/."
    )


_DATASET_CANDIDATES: dict[str, tuple[str, ...]] = {
    # Our own pre-resized 256px corpus, built by scripts/build_hyperkvasir_dataset.py
    "hyperkvasir": (
        "hyperkvasir-unlabeled-256/hk256",
        "hyperkvasir-unlabeled-256",
        "hyper-kvasir-unlabeled-256",
        "hyperkvasir/unlabeled-images",
        "HyperKvasir/unlabeled-images",
    ),
    # HyperKvasir labelled split (23 classes) — used only for k-NN / linear probe.
    "hyperkvasir_labeled": (
        "hyperkvasir-labeled-256/hk_labeled256",
        "hyperkvasir-labeled-256",
        "hyperkvasir/labeled-images",
    ),
    # kaggle.com/datasets/debeshjha1/kvasirseg
    "kvasir_seg": (
        "kvasirseg/Kvasir-SEG",
        "kvasir-seg/Kvasir-SEG",
        "kvasirseg",
        "Kvasir-SEG",
    ),
    # External generalisation set, inference only.
    "cvc_clinicdb": (
        "cvcclinicdb/PNG",
        "cvc-clinicdb/PNG",
        "cvcclinicdb",
        "CVC-ClinicDB",
    ),
}


# ------------------------------------------------------------------- dataclasses

@dataclass
class ModelCfg:
    arch: str = "vit_small"
    patch_size: int = 16
    img_size: int = 224
    drop_path_rate: float = 0.0

    @property
    def embed_dim(self) -> int:
        return VIT_ARCHS[self.arch]["embed_dim"]

    @property
    def depth(self) -> int:
        return VIT_ARCHS[self.arch]["depth"]

    @property
    def num_heads(self) -> int:
        return VIT_ARCHS[self.arch]["num_heads"]

    @property
    def grid_size(self) -> int:
        if self.img_size % self.patch_size:
            raise ValueError(f"img_size {self.img_size} not divisible by patch {self.patch_size}")
        return self.img_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size**2


@dataclass
class OptimCfg:
    """Shared optimisation budget. Held identical across all four SSL methods.

    The controlled variable in the thesis is *samples seen* = epochs * corpus_size,
    not epochs, since every method sees the same corpus.
    """

    global_batch: int = 512
    accum_steps: int = 1
    epochs: int = 100
    warmup_epochs: int = 10
    start_lr: float = 2.0e-4
    ref_lr: float = 2.5e-4
    final_lr: float = 1.0e-6
    weight_decay: float = 0.04
    final_weight_decay: float = 0.4
    exclude_bias_and_norm_from_wd: bool = True
    ema: tuple[float, float] = (0.996, 1.0)
    ipe_scale: float = 1.0
    grad_clip: float = 3.0


@dataclass
class MaskCfg:
    """i-jepa multiblock masking. Defaults are the released in1k config values."""

    enc_mask_scale: tuple[float, float] = (0.85, 1.0)
    pred_mask_scale: tuple[float, float] = (0.15, 0.2)
    aspect_ratio: tuple[float, float] = (0.75, 1.5)
    num_enc_masks: int = 1
    num_pred_masks: int = 4
    min_keep: int = 10
    allow_overlap: bool = False


@dataclass
class JEPAHeadCfg:
    # 192 == 0.5 * 384, the same predictor/encoder width ratio the paper uses for ViT-B
    # (384/768). Copying the absolute 384 would be a ViT-B-sized head on a ViT-S body.
    pred_emb_dim: int = 192
    pred_depth: int = 6


@dataclass
class MAEHeadCfg:
    mask_ratio: float = 0.75
    decoder_embed_dim: int = 256  # 0.67 * 384, matching MAE's own 512/768 ratio
    decoder_depth: int = 8
    decoder_num_heads: int = 8
    norm_pix_loss: bool = True


@dataclass
class ContrastiveHeadCfg:
    proj_hidden_dim: int = 2048
    proj_out_dim: int = 256
    proj_num_layers: int = 3
    pred_hidden_dim: int = 2048  # MoCo v3 only
    temperature: float = 0.2  # MoCo v3; SimCLR overrides to 0.1
    moco_momentum: tuple[float, float] = (0.99, 1.0)
    freeze_patch_embed: bool = True  # MoCo v3's own fix for ViT training collapse
    symmetric_loss: bool = True


@dataclass
class AugCfg:
    """Pretraining view generation.

    I-JEPA/MAE defaults mirror the released i-jepa configs, where
    `use_horizontal_flip`, `use_color_distortion` and `use_gaussian_blur` are all
    False — RandomResizedCrop is genuinely the only augmentation.
    """

    crop_scale: tuple[float, float] = (0.3, 1.0)
    horizontal_flip: bool = False
    color_jitter_strength: float = 0.0
    color_distortion: bool = False
    gaussian_blur: bool = False
    solarize: bool = False
    two_views: bool = False
    # Corpus statistics; overwritten from configs/norm_stats.json once computed.
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class RuntimeCfg:
    seed: int = 0
    num_workers: int = 3  # 4 vCPUs on Kaggle; leave one for the main process
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    session_guard_hours: float = 7.5
    ckpt_push_minutes: float = 45.0
    log_every_steps: int = 50
    ckpt_dataset_slug: str = ""  # e.g. "morsalin101/jepa-thesis-ckpt"
    weights_dataset_slug: str = ""


@dataclass
class PretrainCfg:
    method: str = "ijepa"  # ijepa | mae | simclr | mocov3
    # Cap the pretraining corpus. 0 = use everything. Intended for fast end-to-end
    # validation on a few thousand images before committing GPU-hours to the real run.
    # It is a top-level field (not runtime) on purpose: it changes the experiment, so it
    # belongs in the config hash and in run_id — a subset run can then never resume from,
    # or be mistaken for, the full run.
    max_images: int = 0
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    mask: MaskCfg = field(default_factory=MaskCfg)
    jepa: JEPAHeadCfg = field(default_factory=JEPAHeadCfg)
    mae: MAEHeadCfg = field(default_factory=MAEHeadCfg)
    contrastive: ContrastiveHeadCfg = field(default_factory=ContrastiveHeadCfg)
    aug: AugCfg = field(default_factory=AugCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)

    @property
    def is_subset_run(self) -> bool:
        return self.max_images > 0

    @property
    def run_id(self) -> str:
        m, o = self.model, self.optim
        # The n<N> tag keeps a quick subset run in its own checkpoint namespace and makes
        # it obvious in every filename, log line and metrics record that it is not the
        # full-corpus result.
        subset = f"_n{self.max_images}" if self.is_subset_run else ""
        return (
            f"{self.method}_{m.arch}_p{m.patch_size}_{m.img_size}"
            f"_ep{o.epochs}_bs{o.global_batch}{subset}_s{self.runtime.seed}"
        )


@dataclass
class SegCfg:
    """Segmentation fine-tuning. Held identical across every encoder, including
    random-init — that identity is what makes the comparison a comparison."""

    encoder: str = "ijepa"  # ijepa | mae | simclr | mocov3 | random | imagenet
    decoder: str = "segformer"  # segformer | unet  (unet kept as a decoder ablation)
    model: ModelCfg = field(default_factory=lambda: ModelCfg(img_size=352))
    fpn_layers: tuple[int, int, int, int] = (2, 5, 8, 11)  # 0-indexed blocks {3,6,9,12}
    decoder_embed_dim: int = 256
    epochs: int = 100
    batch_size: int = 16
    enc_lr: float = 1.0e-4
    dec_lr: float = 1.0e-3
    layer_decay: float = 0.75
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    early_stop_patience: int = 20
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    label_fraction: float = 1.0  # low-label ablation: 0.1 / 0.25 / 0.5 / 1.0
    split: str = "800_100_100"  # or "880_120"
    pretrained_ckpt: str = ""
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)

    @property
    def run_id(self) -> str:
        frac = "" if self.label_fraction == 1.0 else f"_lf{self.label_fraction}"
        return f"seg_{self.encoder}_{self.decoder}_{self.model.img_size}{frac}_s{self.runtime.seed}"


# ------------------------------------------------------------------ YAML overlay

def _coerce(value: Any, ref: Any) -> Any:
    """Make YAML values match the dataclass field's type where it matters.

    YAML gives lists; several fields are tuples that end up in the config hash, and
    `[0.15, 0.2] != (0.15, 0.2)` would make an otherwise-identical run refuse to resume.
    """
    if isinstance(ref, tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(ref, float) and isinstance(value, int):
        return float(value)
    if isinstance(ref, Path):
        return Path(value)
    return value


def apply_overrides(cfg: Any, data: dict[str, Any], path: str = "") -> None:
    """Recursively overlay a dict onto a dataclass instance, in place.

    Unknown keys raise. A silently ignored typo in a config file is a whole wasted
    Kaggle session.
    """
    known = {f.name for f in fields(cfg)}
    for key, value in data.items():
        where = f"{path}{key}"
        if key not in known:
            raise KeyError(f"unknown config key {where!r} for {type(cfg).__name__}")
        current = getattr(cfg, key)
        if is_dataclass(current) and isinstance(value, dict):
            apply_overrides(current, value, path=f"{where}.")
        else:
            setattr(cfg, key, _coerce(value, current))


def load_yaml(name_or_path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(name_or_path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / "configs" / p
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    import yaml

    with open(p) as f:
        return yaml.safe_load(f) or {}


def build_pretrain_cfg(
    method: str,
    config_file: str | os.PathLike[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> PretrainCfg:
    cfg = PretrainCfg(method=method)
    apply_overrides(cfg, load_yaml(config_file or f"pretrain_{method}.yaml"))
    if overrides:
        apply_overrides(cfg, overrides)
    if cfg.method != method:
        raise ValueError(f"config declares method={cfg.method!r} but {method!r} was requested")
    return cfg


def build_seg_cfg(
    config_file: str | os.PathLike[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> SegCfg:
    cfg = SegCfg()
    apply_overrides(cfg, load_yaml(config_file or "segment_segformer.yaml"))
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg


# -------------------------------------------------------------------- hashing

def config_dict(cfg: Any) -> dict[str, Any]:
    """Dataclass -> plain JSON-able dict, tuples flattened to lists."""

    def enc(o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, tuple):
            return list(o)
        return o

    return json.loads(json.dumps(asdict(cfg), default=enc, sort_keys=True))


def config_hash(cfg: Any, ignore: Sequence[str] = ("runtime",)) -> str:
    """Stable hash of the experiment-defining config.

    `runtime` is excluded because num_workers / push cadence / guard hours legitimately
    differ between sessions and must not block a resume. Everything else must match or
    the checkpoint is from a different experiment.
    """
    d = config_dict(cfg)
    for k in ignore:
        d.pop(k, None)
    # seed does define the experiment, so keep it even though it lives under runtime
    seed = getattr(getattr(cfg, "runtime", None), "seed", None)
    if seed is not None:
        d["_seed"] = seed
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def describe_env() -> str:
    amp = amp_config()
    return (
        f"[env] kaggle={on_kaggle()}  device={amp.device}  gpu={amp.name} "
        f"(sm_{amp.sm})  autocast={amp.dtype}  scaler={amp.use_scaler}"
    )

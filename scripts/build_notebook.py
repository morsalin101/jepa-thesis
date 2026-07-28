"""Generate the Kaggle notebooks and their kernel-metadata.

One notebook per job, because Kaggle's ~9h session cap means a job is the unit of work:

    data_prep      CPU session — build the 256px corpus, dedup, generate splits (free)
    pretrain_<m>   GPU session — one per SSL method, resumable across sessions
    segment        GPU session — 5 encoders x 5 seeds + ablations
    analysis       CPU session — probe, stats, tables, figures (no GPU needed)

Each notebook clones the repo at a pinned branch, then writes every source file out as its
own `%%writefile` cell, then runs the job. That means when you open the notebook on Kaggle
you can **read and edit every line of the code in the UI** and step through it cell by
cell — while the clone still guarantees the starting point is exactly what is in git.

Editing a source cell and re-running it patches that file for the session. Re-running the
clone cell discards those edits and returns to the committed state.

    python scripts/build_notebook.py --user <kaggle-username>
    python scripts/build_notebook.py --user <name> --no-embed   # smaller, clone-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

# Where the notebooks clone the repo on Kaggle. Baked into the %%writefile paths so
# those cells do not depend on the current working directory.
WORKDIR = "/kaggle/working/jepa-thesis"

SRC_FILES = [
    "src/__init__.py",
    "src/config.py",
    "src/data/__init__.py",
    "src/data/transforms.py",
    "src/data/hyperkvasir.py",
    "src/data/kvasir_seg.py",
    "src/data/splits.py",
    "src/masks/__init__.py",
    "src/masks/utils.py",
    "src/masks/multiblock.py",
    "src/model/__init__.py",
    "src/model/vit.py",
    "src/model/predictor.py",
    "src/model/jepa.py",
    "src/model/mae.py",
    "src/model/heads.py",
    "src/model/simclr.py",
    "src/model/mocov3.py",
    "src/model/simple_fpn.py",
    "src/model/segformer_head.py",
    "src/model/unet.py",
    "src/utils/__init__.py",
    "src/utils/schedulers.py",
    "src/utils/checkpoint.py",
    "src/utils/ddp.py",
    "src/utils/kaggle_io.py",
    "src/engine/__init__.py",
    "src/engine/pretrain.py",
    "src/engine/segment.py",
    "src/eval/__init__.py",
    "src/eval/metrics.py",
    "src/eval/stats.py",
    "src/eval/tables.py",
    "src/eval/probe.py",
    "src/viz/__init__.py",
    "src/viz/style.py",
    "src/viz/figures.py",
    "src/viz/make_all.py",
    # The configs come last on purpose: they are the most likely thing to tweak in the
    # Kaggle UI (epochs, batch size, LR), so they sit right next to the job cells.
    "configs/pretrain_ijepa.yaml",
    "configs/pretrain_mae.yaml",
    "configs/pretrain_simclr.yaml",
    "configs/pretrain_mocov3.yaml",
    "configs/segment_segformer.yaml",
]


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def setup_cells(repo_url: str, branch: str, gpu: bool) -> list[dict]:
    cells = [
        code(
            f'REPO_URL = "{repo_url}"\n'
            f'BRANCH = "{branch}"\n'
            f'WORKDIR = "{WORKDIR}"\n'
        ),
        code(
            "import os, subprocess, sys\n"
            "\n"
            "def sh(cmd, check=True):\n"
            "    print('$', cmd)\n"
            "    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)\n"
            "    if r.stdout: print(r.stdout)\n"
            "    if r.stderr: print(r.stderr)\n"
            "    if check and r.returncode != 0:\n"
            "        raise RuntimeError(f'command failed (exit {r.returncode}): {cmd}')\n"
            "    return r\n"
            "\n"
            "if subprocess.run(f'git ls-remote {REPO_URL}', shell=True,\n"
            "                  capture_output=True).returncode != 0:\n"
            "    raise RuntimeError('Cannot reach GitHub — turn Internet ON in the session options.')\n"
            "\n"
            "if os.path.exists(WORKDIR):\n"
            "    sh(f'cd {WORKDIR} && git fetch origin && git reset --hard origin/{BRANCH}')\n"
            "else:\n"
            "    sh(f'git clone --branch {BRANCH} {REPO_URL} {WORKDIR}')\n"
            "sh(f'cd {WORKDIR} && git log -1 --oneline')\n"
            "os.chdir(WORKDIR)\n"
            "sys.path.insert(0, WORKDIR)\n"
        ),
        code("sh('pip install -q -r requirements.txt')\n"),
    ]
    if gpu:
        cells.append(
            code(
                "# Report the accelerator and the precision that follows from it.\n"
                "# T4 (sm_75) has fp16 tensor cores but NO bf16 hardware; P100 (sm_60) has\n"
                "# neither and runs ~2x slower. The code adapts either way — this cell is\n"
                "# here so you know what you were given before spending 8 hours on it.\n"
                "import torch\n"
                "from src.config import amp_config\n"
                "print('CUDA devices:', torch.cuda.device_count())\n"
                "amp = amp_config()\n"
                "print(amp)\n"
                "if torch.cuda.device_count() < 2:\n"
                "    print('\\n*** Only one GPU. I-JEPA and MAE will still run correctly, but\\n'\n"
                "          '    SimCLR/MoCo v3 need 2 GPUs to preserve global_batch=512.\\n'\n"
                "          '    Set Session options -> Accelerator -> GPU T4 x2 and re-run. ***')\n"
            )
        )
    return cells


def embed_cells() -> list[dict]:
    dirs = sorted({str(Path(f).parent) for f in SRC_FILES if str(Path(f).parent) != "."})
    # `%%writefile` raises "cell body is empty" on a blank file, which aborts Run All at
    # the very first cell. Empty package markers get created by the guard cell instead.
    empty = [f for f in SRC_FILES if (ROOT / f).is_file() and not (ROOT / f).read_text().strip()]
    out = [
        md(
            "### Source files (editable)\n\n"
            "Each cell below writes one file. Edit a cell and re-run it to patch the code "
            "in this session without a git round-trip. Re-running the clone cell above "
            "discards these edits.\n\n"
            "Run the guard cell first — `%%writefile` writes relative to the working "
            "directory and will not create missing folders, so it fails with "
            "`FileNotFoundError` if the clone has not run or the kernel was restarted."
        ),
        code(
            "# Guard for the %%writefile cells below. Safe to re-run at any time.\n"
            "#   1. %%writefile resolves paths relative to os.getcwd(), and a kernel\n"
            "#      restart resets that to /kaggle/working — so re-assert it.\n"
            "#   2. %%writefile will NOT create parent directories; it raises\n"
            "#      FileNotFoundError instead. So create them up front.\n"
            "#   3. Empty __init__.py files cannot be written by %%writefile at all\n"
            "#      ('cell body is empty'), so they are created here.\n"
            "import os, pathlib\n"
            "\n"
            "if not os.path.isdir(WORKDIR):\n"
            "    raise RuntimeError(\n"
            "        f'{WORKDIR} does not exist — run the git clone cell above first.')\n"
            "os.chdir(WORKDIR)\n"
            f"for d in {dirs!r}:\n"
            "    pathlib.Path(WORKDIR, d).mkdir(parents=True, exist_ok=True)\n"
            f"for f in {empty!r}:\n"
            "    pathlib.Path(WORKDIR, f).touch(exist_ok=True)\n"
            "print('cwd:', os.getcwd())\n"
            "print('ready for the %%writefile cells')\n"
        ),
    ]
    for rel in SRC_FILES:
        p = ROOT / rel
        if not p.is_file() or rel in empty:
            continue
        body = p.read_text()
        # Absolute path, not relative: %%writefile resolves against os.getcwd(), and a
        # kernel restart or running a cell out of order silently resets that to
        # /kaggle/working, which made every one of these cells fail with a
        # FileNotFoundError that pointed at the file rather than at the real cause.
        src = f"%%writefile {WORKDIR}/{rel}\n{body}"
        out.append(code(src if src.endswith("\n") else src + "\n"))
    return out


def assemble(intro: dict, repo_url: str, branch: str, gpu: bool, embed: bool, body: list[dict]) -> list[dict]:
    """intro -> clone/install/(gpu check) -> optional source cells -> the actual job.

    The source cells must land *after* the clone (which would otherwise overwrite them)
    and *before* the job cells (which import the code).
    """
    return [intro, *setup_cells(repo_url, branch, gpu), *(embed_cells() if embed else []), *body]


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def metadata(user: str, slug: str, gpu: bool, datasets: list[str], self_source: bool) -> dict:
    m = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": True,
        "dataset_sources": datasets,
        "competition_sources": [],
        # Mount this kernel's own previous output as a free fallback checkpoint mirror.
        "kernel_sources": [f"{user}/{slug}"] if self_source else [],
    }
    return m


# ------------------------------------------------------------------ notebooks


def build_data_prep(user: str, repo_url: str, branch: str, embed: bool) -> tuple[str, dict, dict]:
    intro = md(
            "# Data preparation (CPU session — free, no GPU quota)\n\n"
            "1. Download HyperKvasir unlabeled (~24 GB) into `/kaggle/tmp` scratch and "
            "stream-resize to 256px (~3 GB), then publish as a private Kaggle Dataset.\n"
            "2. Perceptual-hash the corpus against Kvasir-SEG and exclude near-duplicates "
            "— this prevents pretraining/test contamination and the count goes in the thesis.\n"
            "3. Generate the group-aware, stratified Kvasir-SEG splits.\n\n"
            "**Set the accelerator to None.** This is pure CPU work and a GPU session "
            "would burn quota for nothing."
    )
    body = [
        code(
            "# Credentials for publishing the resized corpora as Kaggle Datasets.\n"
            "# Labels must be exactly KAGGLE_USERNAME and KAGGLE_KEY (Add-ons -> Secrets).\n"
            "from kaggle_secrets import UserSecretsClient\n"
            "s = UserSecretsClient()\n"
            "os.environ['KAGGLE_USERNAME'] = s.get_secret('KAGGLE_USERNAME')\n"
            "os.environ['KAGGLE_KEY'] = s.get_secret('KAGGLE_KEY')\n"
            "print('credentials loaded for', os.environ['KAGGLE_USERNAME'])\n"
        ),
        code(
            "# Locate the mounted HyperKvasir source. Reading the attached dataset beats\n"
            "# downloading from datasets.simula.no, whose TLS chain is missing an\n"
            "# intermediate certificate (curl exit 60) — and a mount costs no disk quota.\n"
            "import glob\n"
            "\n"
            "def find_source(*patterns):\n"
            "    for pat in patterns:\n"
            "        hits = sorted(glob.glob(pat))\n"
            "        if hits:\n"
            "            return hits[0]\n"
            "    raise FileNotFoundError(\n"
            "        f'no HyperKvasir source found for {patterns}. Attach '\n"
            "        'faisalmahmud69/hyper-kvasir-unlabeled-images and '\n"
            "        'faisalmahmud69/hyper-kvasir-labeled-images via + Add Input.')\n"
            "\n"
            "UNLABELED_SRC = find_source('/kaggle/input/*/unlabeled-images/images',\n"
            "                           '/kaggle/input/*/unlabeled-images',\n"
            "                           '/kaggle/input/*/*/unlabeled-images/images')\n"
            "LABELED_SRC = find_source('/kaggle/input/*/labeled-images',\n"
            "                          '/kaggle/input/*/*/labeled-images')\n"
            "print('unlabeled:', UNLABELED_SRC, len(os.listdir(UNLABELED_SRC)), 'entries')\n"
            "print('labeled  :', LABELED_SRC)\n"
        ),
        code(
            "# ~99,417 images -> 256px short side, ~3 GB. 20-40 min on 4 vCPUs.\n"
            "# The resize is what makes the dataloader keep up with the GPU later:\n"
            "# decoding 1280x1024 JPEGs at 400 views/s is not possible on 4 cores.\n"
            "sh(f'python scripts/build_hyperkvasir_dataset.py --source-dir {UNLABELED_SRC} "
            f"--publish {user}/hyperkvasir-unlabeled-256')\n"
        ),
        code(
            "# The labelled split (10,662 images, 23 classes) — ~400 MB, a few minutes.\n"
            "# Never trained on; used only by the k-NN / linear probe in the analysis\n"
            "# notebook, which is the cheapest check that pretraining actually worked.\n"
            "sh(f'python scripts/build_hyperkvasir_dataset.py --split labeled "
            f"--source-dir {{LABELED_SRC}} --publish {user}/hyperkvasir-labeled-256')\n"
        ),
        code(
            "# Point at the corpus we just built in /kaggle/working — the published dataset\n"
            "# only appears under /kaggle/input on a *later* session, so auto-resolution\n"
            "# would not find it yet in this one.\n"
            "sh('python scripts/dedup_phash.py --threshold 6 "
            "--pretrain-root /kaggle/working/hk256')\n"
        ),
        code("sh('python -m src.data.splits')\n"),
        code(
            "print(open('splits/dedup_report.json').read())\n"
            "print(open('splits/800_100_100/stats.json').read())\n"
        ),
        md(
            "### Copy `splits/` back into git — required before segmentation\n\n"
            "The later notebooks clone the repo from GitHub, so the split files and the "
            "dedup exclusion list have to be **committed**, not just present here. The "
            "cell below copies them to `/kaggle/working/splits_to_commit/`; download that "
            "from the notebook's Output tab (or `kaggle kernels output`), drop it into "
            "`splits/` locally, and push.\n\n"
            "This is what guarantees every run — yours and anyone reproducing it — uses "
            "byte-identical splits."
        ),
        code(
            "import shutil\n"
            "shutil.copytree('splits', '/kaggle/working/splits_to_commit', dirs_exist_ok=True)\n"
            "for root, _, files in os.walk('/kaggle/working/splits_to_commit'):\n"
            "    for f in sorted(files):\n"
            "        print(os.path.join(root, f))\n"
            "print('\\nRetrieve with:\\n'\n"
            f"      '  kaggle kernels output {user}/data-prep -p /tmp/dp\\n'\n"
            "      '  cp -r /tmp/dp/splits_to_commit/* splits/\\n'\n"
            "      '  git add splits && git commit -m \\'data: splits + dedup list\\' && git push')\n"
        ),
    ]
    cells = assemble(intro, repo_url, branch, False, embed, body)
    datasets = [
        "debeshjha1/kvasirseg",
        # Mounted rather than downloaded: datasets.simula.no serves an incomplete TLS
        # chain, so curl fails from Kaggle. These mirrors carry the same images.
        "faisalmahmud69/hyper-kvasir-unlabeled-images",
        "faisalmahmud69/hyper-kvasir-labeled-images",
    ]
    return "data-prep", nb(cells), metadata(user, "data-prep", False, datasets, False)


def build_pretrain(
    user: str, repo_url: str, branch: str, method: str, embed: bool
) -> tuple[str, dict, dict]:
    slug = f"pretrain-{method}"
    cost = {"ijepa": "~8.8", "mae": "~4.5", "simclr": "~15.8", "mocov3": "~21.3"}[method]
    sessions = {"ijepa": 2, "mae": 1, "simclr": 3, "mocov3": 3}[method]
    intro = md(
            f"# Pretrain {method} on HyperKvasir unlabeled\n\n"
            f"ViT-S/16 @ 224, global batch 512, 100 epochs. Estimated **{cost} GPU-hours** "
            f"(~{sessions} session(s) at the 7.5h guard).\n\n"
            "**This notebook is resumable.** It stops cleanly before the session cap, saves "
            "full training state (model, EMA target, optimizer, scaler, schedule position) "
            "to a Kaggle Dataset, and picks up exactly where it left off next run. Just "
            "*Save & Run All* again until it prints `run complete`.\n\n"
            "Requires **GPU T4 x2** and Internet ON, plus `KAGGLE_USERNAME`/`KAGGLE_KEY` "
            "under Add-ons → Secrets for cross-session checkpointing."
    )
    body = [
        code(
            "from kaggle_secrets import UserSecretsClient\n"
            "s = UserSecretsClient()\n"
            "os.environ['KAGGLE_USERNAME'] = s.get_secret('KAGGLE_USERNAME')\n"
            "os.environ['KAGGLE_KEY'] = s.get_secret('KAGGLE_KEY')\n"
        ),
        code(
            f"sh('python -m src.engine.pretrain --method {method} "
            f"--ckpt-slug {user}/jepa-thesis-ckpt --guard-hours 7.5', check=False)\n"
        ),
        code(
            "# If the cell above printed 'N epochs remaining', the session guard stopped it\n"
            "# cleanly — just Save & Run All again to continue. If it printed 'run complete',\n"
            "# the exported encoder is in /kaggle/working/weights/ and the next cell ships it.\n"
            "!ls -la /kaggle/working/weights/ 2>/dev/null || echo 'not finished yet — re-run'\n"
        ),
        code(
            "# Publish the finished encoder (~88 MB) to the shared weights dataset that the\n"
            "# segmentation notebook mounts. Safe to re-run; a no-op until the run completes.\n"
            "import glob, json, pathlib, shutil, subprocess\n"
            "\n"
            f"SLUG = '{user}/jepa-thesis-weights'\n"
            "found = glob.glob('/kaggle/working/weights/*.pt')\n"
            "if not found:\n"
            "    print('nothing to publish yet — pretraining has not finished')\n"
            "else:\n"
            "    stage = pathlib.Path('/kaggle/working/weights_upload')\n"
            "    stage.mkdir(exist_ok=True)\n"
            "    # Carry over any encoders already in the dataset so a new version never\n"
            "    # drops the other methods' weights.\n"
            "    for p in glob.glob('/kaggle/input/jepa-thesis-weights/*.pt'):\n"
            "        shutil.copy(p, stage)\n"
            "    for p in found:\n"
            "        shutil.copy(p, stage)\n"
            "    (stage / 'dataset-metadata.json').write_text(json.dumps(\n"
            "        {'title': 'jepa-thesis-weights', 'id': SLUG,\n"
            "         'licenses': [{'name': 'CC0-1.0'}]}, indent=2))\n"
            "    # `datasets status` exits 0 even on a 403/404, so read stdout not the code.\n"
            "    probe = subprocess.run(['kaggle','datasets','status',SLUG],\n"
            "                           capture_output=True, text=True)\n"
            "    pout = (probe.stdout or '') + (probe.stderr or '')\n"
            "    exists = bool(pout.strip()) and 'error' not in pout.lower()\n"
            "    # Private is the default; there is no --private flag.\n"
            "    cmd = (['kaggle','datasets','version','-p',str(stage),'-m',\n"
            f"            'add {method}','--dir-mode','zip','--delete-old-versions']\n"
            "           if exists else\n"
            "           ['kaggle','datasets','create','-p',str(stage),'--dir-mode','zip'])\n"
            "    r = subprocess.run(cmd, capture_output=True, text=True)\n"
            "    print((r.stdout or '') + (r.stderr or ''))\n"
            "    print('contents:', sorted(p.name for p in stage.glob('*.pt')))\n"
        ),
    ]
    cells = assemble(intro, repo_url, branch, True, embed, body)
    datasets = [
        f"{user}/hyperkvasir-unlabeled-256",
        f"{user}/jepa-thesis-ckpt",
        # Mounted so the publish cell can carry over encoders from earlier methods;
        # a new dataset version replaces its whole contents otherwise.
        f"{user}/jepa-thesis-weights",
    ]
    return slug, nb(cells), metadata(user, slug, True, datasets, True)


def build_segment(user: str, repo_url: str, branch: str, embed: bool) -> tuple[str, dict, dict]:
    intro = md(
            "# Segmentation fine-tuning on Kvasir-SEG\n\n"
            "SegFormer all-MLP decoder on a ViT simple feature pyramid, 352px, 100 epochs.\n"
            "Five encoders x five seeds = 25 runs, ~4 GPU-hours total.\n\n"
            "Every arm uses the **identical** decoder, recipe, splits and seeds — only the "
            "encoder weights differ. Test is scored exactly once, on the best-val checkpoint."
    )
    body = [
        code(
            "# Pull the four exported encoders published by the pretraining notebooks.\n"
            "import shutil, glob, pathlib\n"
            "pathlib.Path('/kaggle/working/weights').mkdir(parents=True, exist_ok=True)\n"
            f"for p in glob.glob('/kaggle/input/jepa-thesis-weights/*.pt'):\n"
            "    shutil.copy(p, '/kaggle/working/weights/')\n"
            "print(os.listdir('/kaggle/working/weights'))\n"
        ),
        code(
            "ENCODERS = ['ijepa', 'mae', 'simclr', 'mocov3', 'random']\n"
            "SEEDS = [0, 1, 2, 3, 4]\n"
            "for enc in ENCODERS:\n"
            "    for seed in SEEDS:\n"
            "        sh(f'python -m src.engine.segment --encoder {enc} --seed {seed}', check=False)\n"
        ),
        md("## Ablations: low-label regime and decoder swap"),
        code(
            "for enc in ENCODERS:\n"
            "    for lf in [0.1, 0.25, 0.5]:\n"
            "        for seed in [0, 1, 2]:\n"
            "            sh(f'python -m src.engine.segment --encoder {enc} "
            "--label-fraction {lf} --seed {seed}', check=False)\n"
        ),
        code(
            "for enc in ENCODERS:\n"
            "    sh(f'python -m src.engine.segment --encoder {enc} --decoder unet --seed 0', check=False)\n"
        ),
        code(
            "# Comparability table against published Kvasir-SEG numbers.\n"
            "for enc in ENCODERS:\n"
            "    sh(f'python -m src.engine.segment --encoder {enc} --split 880_120 --seed 0', check=False)\n"
        ),
    ]
    cells = assemble(intro, repo_url, branch, True, embed, body)
    datasets = ["debeshjha1/kvasirseg", f"{user}/jepa-thesis-weights"]
    return "segment", nb(cells), metadata(user, "segment", True, datasets, True)


def build_analysis(user: str, repo_url: str, branch: str, embed: bool) -> tuple[str, dict, dict]:
    intro = md(
            "# Analysis: probe, statistics, tables and figures\n\n"
            "Everything here reads the JSON/JSONL artefacts written during training, so it "
            "needs **no GPU** (except the frozen-feature probe, which is cheap). You can run "
            "the same commands on your laptop to iterate on figures."
    )
    body = [
        code(
            "# k-NN + linear probe on frozen features. The cheapest signal about\n"
            "# representation quality — run it before trusting any segmentation number.\n"
            "sh('python -m src.eval.probe --save-embeddings', check=False)\n"
        ),
        code("sh('python -m src.eval.stats')\n"),
        code("sh('python -m src.eval.tables')\n"),
        code("sh('python -m src.viz.make_all --out /kaggle/working/figures')\n"),
        code(
            "from IPython.display import Image as IPyImage, display, Markdown\n"
            "import glob\n"
            "for f in sorted(glob.glob('/kaggle/working/figures/*.png')):\n"
            "    display(Markdown(f'### {os.path.basename(f)}'))\n"
            "    display(IPyImage(filename=f))\n"
        ),
        code(
            "display(Markdown(open('/kaggle/working/results/tables/all_tables.md').read()))\n"
        ),
    ]
    cells = assemble(intro, repo_url, branch, True, embed, body)
    datasets = [
        "debeshjha1/kvasirseg",
        f"{user}/jepa-thesis-weights",
        f"{user}/hyperkvasir-labeled-256",
    ]
    return "analysis", nb(cells), metadata(user, "analysis", True, datasets, False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Kaggle notebooks")
    ap.add_argument("--user", default="morsalin101")
    ap.add_argument("--repo", default="https://github.com/morsalin101/jepa-thesis.git")
    ap.add_argument("--branch", default="main")
    ap.add_argument(
        "--no-embed",
        action="store_true",
        help="omit the %%writefile source cells (smaller notebooks; code comes from the clone)",
    )
    args = ap.parse_args()
    args.embed = not args.no_embed

    builders = [
        build_data_prep(args.user, args.repo, args.branch, args.embed),
        *[
            build_pretrain(args.user, args.repo, args.branch, m, args.embed)
            for m in ("ijepa", "mae", "simclr", "mocov3")
        ],
        build_segment(args.user, args.repo, args.branch, args.embed),
        build_analysis(args.user, args.repo, args.branch, args.embed),
    ]

    for slug, notebook, meta in builders:
        d = NB_DIR / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{slug}.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"wrote {d}/  ({len(notebook['cells'])} cells)")

    print(
        "\nPush one with:  kaggle kernels push -p notebooks/<slug>\n"
        "Order: data-prep -> pretrain-* -> segment -> analysis"
    )


if __name__ == "__main__":
    main()

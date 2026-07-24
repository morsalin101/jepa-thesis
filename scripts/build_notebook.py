"""Build notebooks/kaggle_runner.ipynb by embedding every src/ file as a
%%writefile cell. The resulting notebook lets you run cells one by one in
the Kaggle UI and see/edit every line of code.

Re-run this after editing src/ files locally:
    python3 scripts/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "kaggle_runner.ipynb"


SRC_FILES = [
    "src/__init__.py",
    "src/config.py",
    "src/data.py",
    "src/utils/__init__.py",
    "src/utils/masking.py",
    "src/model/__init__.py",
    "src/model/vit.py",
    "src/model/predictor.py",
    "src/model/jepa.py",
    "src/engine/__init__.py",
    "src/engine/pretrain.py",
    "src/train.py",
    "configs/pretrain_jepa.yaml",
]


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def writefile_cell(rel_path: str, body: str) -> dict:
    src = f"%%writefile {rel_path}\n{body}"
    if not src.endswith("\n"):
        src += "\n"
    return code_cell(src)


def build() -> dict:
    cells: list[dict] = []

    cells.append(md_cell(
        "# JEPA thesis — Kaggle UI runner\n"
        "\n"
        "This notebook embeds every file in `src/` and `configs/` as its own "
        "cell so you can:\n"
        "\n"
        "1. **See** all the code in the Kaggle UI.\n"
        "2. **Edit** any cell in-place and re-run just that cell to update the file.\n"
        "3. **Run** the cells top-to-bottom, or one at a time to debug.\n"
        "\n"
        "**First time:** right sidebar → Accelerator → **GPU T4 x2**, Internet → **On**, "
        "then *Save & Run All*.\n"
        "\n"
        "**Re-running locally?** Regenerate this notebook with "
        "`python3 scripts/build_notebook.py` after editing `src/`.\n"
    ))

    cells.append(code_cell(
        "REPO_URL = \"https://github.com/morsalin101/jepa-thesis.git\"\n"
        "BRANCH = \"main\"\n"
        "WORKDIR = \"/kaggle/working/jepa-thesis\"\n"
        "REWRITE_FROM_NOTEBOOK = True  # if False, skip the %%writefile cells below\n"
    ))

    cells.append(code_cell(
        "import os, subprocess\n"
        "\n"
        "def sh(cmd, check=True):\n"
        "    print('$', cmd)\n"
        "    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)\n"
        "    if r.stdout: print(r.stdout)\n"
        "    if r.stderr: print(r.stderr)\n"
        "    if check and r.returncode != 0:\n"
        "        raise RuntimeError(f\"command failed (exit {r.returncode}): {cmd}\")\n"
        "    return r\n"
        "\n"
        "net = subprocess.run(f\"git ls-remote {REPO_URL}\", shell=True, capture_output=True, text=True)\n"
        "if net.returncode != 0:\n"
        "    raise RuntimeError(\"Cannot reach GitHub. Turn Internet ON.\")\n"
        "\n"
        "if os.path.exists(WORKDIR):\n"
        "    sh(f'cd {WORKDIR} && git fetch origin && git reset --hard origin/{BRANCH}')\n"
        "else:\n"
        "    sh(f'git clone --branch {BRANCH} {REPO_URL} {WORKDIR}')\n"
        "sh(f'cd {WORKDIR} && git log -1 --oneline')\n"
        "os.chdir(WORKDIR)\n"
    ))

    cells.append(code_cell(
        "sh(f'pip install -q -r {WORKDIR}/requirements.txt')\n"
    ))

    cells.append(code_cell(
        "import torch\n"
        "print('CUDA available:', torch.cuda.is_available())\n"
        "if torch.cuda.is_available():\n"
        "    name = torch.cuda.get_device_name(0)\n"
        "    major, _ = torch.cuda.get_device_capability(0)\n"
        "    print(f'Device: {name}  (sm_{major}0)')\n"
        "    if major < 7:\n"
        "        raise RuntimeError(\n"
        "            f'GPU {name} (sm_{major}0) is too old for the installed PyTorch. '\n"
        "            'Right sidebar -> Accelerator -> GPU T4 x2.'\n"
        "        )\n"
        "else:\n"
        "    print('Device: CPU (no GPU)')\n"
    ))

    cells.append(code_cell(
        "import os\n"
        "for root, dirs, files in os.walk('.'):\n"
        "    if '.git' in root or '__pycache__' in root:\n"
        "        continue\n"
        "    for f in sorted(files):\n"
        "        print(os.path.join(root, f))\n"
    ))

    if True:  # placeholder so we can later conditionally skip
        for rel in SRC_FILES:
            body = (ROOT / rel).read_text()
            cells.append(writefile_cell(rel, body))

    cells.append(code_cell(
        "import importlib\n"
        "import torch\n"
        "\n"
        "import src.config\n"
        "import src.data\n"
        "import src.model.vit\n"
        "import src.model.predictor\n"
        "import src.model.jepa\n"
        "import src.engine.pretrain\n"
        "importlib.reload(src.config)\n"
        "importlib.reload(src.data)\n"
        "importlib.reload(src.model.vit)\n"
        "importlib.reload(src.model.predictor)\n"
        "importlib.reload(src.model.jepa)\n"
        "importlib.reload(src.engine.pretrain)\n"
        "from src.config import CONFIG\n"
        "from src.model.vit import ViT\n"
        "from src.model.jepa import IJEPA\n"
        "from src.utils.masking import sample_target_block\n"
        "\n"
        "print('device:', CONFIG.device)\n"
        "print('img:', CONFIG.jepa.img_size, 'patch:', CONFIG.jepa.patch_size,\n"
        "      'tokens:', CONFIG.jepa.n_h * CONFIG.jepa.n_w)\n"
        "\n"
        "vit = ViT(img_size=96, patch_size=8, dim=192, depth=12, heads=3).to(CONFIG.device)\n"
        "x = torch.randn(2, 3, 96, 96, device=CONFIG.device)\n"
        "out = vit(x)\n"
        "print('ViT forward shape:', tuple(out.shape))\n"
        "\n"
        "m = sample_target_block(12, 12)\n"
        "print('mask sample: ctx=', m['n_ctx'], 'tgt=', m['n_tgt'])\n"
        "\n"
        "model = IJEPA(\n"
        "    img_size=CONFIG.jepa.img_size, patch_size=CONFIG.jepa.patch_size,\n"
        "    enc_dim=CONFIG.jepa.enc_dim, enc_depth=CONFIG.jepa.enc_depth, enc_heads=CONFIG.jepa.enc_heads,\n"
        "    pred_dim=CONFIG.jepa.pred_dim, pred_depth=CONFIG.jepa.pred_depth, pred_heads=CONFIG.jepa.pred_heads,\n"
        ").to(CONFIG.device)\n"
        "n = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
        "print(f'IJEPA trainable params: {n/1e6:.2f}M')\n"
    ))

    cells.append(code_cell(
        "!python -m src.train --mode pretrain --epochs 1\n"
    ))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    nb = build()
    NOTEBOOK.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {NOTEBOOK}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()

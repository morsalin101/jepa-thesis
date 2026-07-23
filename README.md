# JEPA thesis

Write code on your Mac, run training on **Kaggle's free GPU**. Source of truth
is GitHub; a Kaggle notebook pulls the latest code and runs on GPU.

```
src/            your code (single entrypoint: src/train.py)
configs/        experiment configs
notebooks/      kaggle_runner.ipynb  -> the notebook Kaggle runs
                kernel-metadata.json -> lets you push that notebook from local
scripts/        setup_kaggle_auth.sh, run_on_kaggle.sh
```

## One-time setup

1. **Kaggle CLI auth** (already installed). Get a token at
   kaggle.com -> Settings -> API -> *Create New Token* (downloads `kaggle.json`).
   ```bash
   ./scripts/setup_kaggle_auth.sh morsalin101 <api_key>
   # or drop kaggle.json into ~/.kaggle/ yourself, then rerun to verify
   ```
2. **Fill in your identifiers** (replace `morsalin101`):
   - `notebooks/kaggle_runner.ipynb`  -> `REPO_URL`
   - `notebooks/kernel-metadata.json` -> `id`  (e.g. `myuser/jepa-thesis-runner`)
3. **Create the GitHub repo** and push:
   ```bash
   git add -A && git commit -m "scaffold: local dev + kaggle runner"
   git branch -M main
   git remote add origin https://github.com/morsalin101/jepa-thesis.git
   git push -u origin main
   ```

## Daily loop

```bash
# 1. edit code locally, smoke-test on CPU/MPS
python -m src.train --epochs 1

# 2. ship it
git add -A && git commit -m "..." && git push

# 3. run on Kaggle GPU (headless) — notebook git-pulls then trains
./scripts/run_on_kaggle.sh
python3 -m kaggle kernels status  morsalin101/jepa-thesis-runner
python3 -m kaggle kernels output  morsalin101/jepa-thesis-runner -p ./outputs
```

Prefer the UI? Open the pushed notebook on kaggle.com, set Accelerator=GPU +
Internet=On, and hit *Run All*. Same notebook either way.

## Notes
- Kaggle images already ship a CUDA-matched `torch` — don't reinstall it.
- Only `/kaggle/working` persists; checkpoints go there (handled in `src/config.py`).
- Attach datasets under `dataset_sources` in `kernel-metadata.json`; they mount at `/kaggle/input`.
- Free GPU quota is ~30 hrs/week. Keep `src/train.py` as the one entrypoint.


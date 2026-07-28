# Kaggle runbook

Total budget ≈ **75 GPU-hours**. Kaggle gives ~30 GPU-h/week with a ~9h session cap, so
plan **4–5 calendar weeks**. CPU sessions are free and don't touch the GPU quota.

Replace `morsalin101` below if your Kaggle username differs.

---

## Step 0 — one-time setup (~10 min, no GPU)

**a. API token.** You need a valid one at `~/.kaggle/kaggle.json` (chmod 600) — you
already have this. It is gitignored and untracked, so it has never reached GitHub; keep it
that way. Rotating it (Settings → API → Expire Token → Create New Token) is optional
housekeeping, not a prerequisite.

**b. Add Kaggle Secrets.** In any Kaggle notebook: Add-ons → Secrets → add
`KAGGLE_USERNAME` and `KAGGLE_KEY`. Secrets are per-notebook — attach them to each one you
run.

This is not optional for pretraining. Without it the checkpoint pusher is disabled and
state only lives in `/kaggle/working`, which is wiped when the session ends — so a
100-epoch run could never get past the first 7.5 h.

**c. Create the two datasets** (they must exist before the notebooks can reference them):

```bash
python3 scripts/setup_kaggle_datasets.py --user morsalin101
```

**d. Push the code.** The notebooks `git clone` from GitHub, so nothing runs until this
is on `main`:

```bash
git add -A
git commit -m "feat: faithful I-JEPA + MAE/SimCLR/MoCo v3 baselines, SegFormer decoder, full eval"
git push origin main
```

**e. Generate and push the notebooks:**

```bash
python3 scripts/build_notebook.py --user morsalin101
for nb in data-prep pretrain-ijepa pretrain-mae pretrain-simclr pretrain-mocov3 segment analysis; do
  kaggle kernels push -p notebooks/$nb
done
```

Each notebook embeds all 38 source files as `%%writefile` cells, so once you open it on
Kaggle you can read and edit every line in the UI and step through cell by cell. Cell
order is: clone → install → **all source files** → the job. Edit a source cell and re-run
it to patch that file for the session; re-run the clone cell to go back to what is in git.

Re-run `build_notebook.py` and push again after any local code change, otherwise the
embedded cells go stale relative to the clone. (`--no-embed` gives clone-only notebooks
that are ~10× smaller if you ever prefer that.)

---

## Step 1 — data prep (CPU session, **free**, ~1–1.5 h)

Open `morsalin101/data-prep` on kaggle.com.

> **Session settings: Accelerator = None, Internet = On, Secrets attached.**
> A GPU here would burn quota for pure CPU work.

Run all. It downloads HyperKvasir (~24 GB) to scratch, stream-resizes to 256px (~3 GB),
publishes `hyperkvasir-unlabeled-256` and `hyperkvasir-labeled-256`, de-duplicates against
Kvasir-SEG, and generates the splits.

**Then bring the splits back into git** — the later notebooks clone the repo, so these
must be committed:

```bash
kaggle kernels output morsalin101/data-prep -p /tmp/dp
cp -r /tmp/dp/splits_to_commit/* splits/
git add splits && git commit -m "data: group-aware splits + dedup exclusion list" && git push
```

Check `splits/dedup_report.json` — the number of excluded near-duplicates goes in your
thesis. It pre-empts the worst possible reviewer objection.

---

## Step 2 — pretraining (GPU, ~50 GPU-h, 9 sessions)

Run these in whatever order you like; MAE first is a good smoke test since it's cheapest.

| Notebook | GPU-h | Sessions |
|---|---|---|
| `pretrain-mae` | ~4.5 | 1 |
| `pretrain-ijepa` | ~8.8 | 2 |
| `pretrain-simclr` | ~15.8 | 3 |
| `pretrain-mocov3` | ~21.3 | 3 |

> **Session settings: Accelerator = GPU T4 x2, Internet = On, Secrets attached.**
>
> Set this **in the UI** — `enable_gpu: true` in the metadata may resolve to a single
> P100, which is ~2× slower and, for SimCLR/MoCo v3, will refuse to run (their InfoNCE
> loss is batch-coupled, so one GPU can't preserve `global_batch=512`; I-JEPA and MAE
> absorb it via gradient accumulation).

**Save & Run All.** When it finishes, read the last output cell:

- `N epochs remaining` → the session guard stopped it cleanly at 7.5 h. **Just Save &
  Run All again.** It restores the model, EMA target, optimizer, GradScaler and schedule
  position and continues from where it stopped. Repeat until done.
- `run complete` → the encoder is exported and the final cell publishes it to
  `jepa-thesis-weights`.

Between methods, delete and recreate `jepa-thesis-ckpt` to reclaim version storage
(~400 MB × ~10 pushes per run adds up).

---

## Step 3 — segmentation (GPU, ~7 GPU-h)

Open `morsalin101/segment`. **Accelerator = GPU T4 x2.** Run all.

25 main runs (5 encoders × 5 seeds) plus the low-label, decoder and 880/120 ablations.
Each arm uses the identical decoder, recipe and splits — only the encoder weights differ.

If a session times out, comment out the arms already in `/kaggle/working/seg/` and re-run;
`summary.json` in each run directory tells you what completed.

---

## Step 4 — analysis (~2 GPU-h)

Open `morsalin101/analysis`. Run all. It produces:

- k-NN + linear probe on frozen features (run this early — if an encoder is at chance,
  its pretraining failed and no fine-tuning will hide it)
- Wilcoxon signed-rank + Holm-Bonferroni + BCa bootstrap CIs
- five LaTeX tables in `results/tables/*.tex`
- eleven figures in `figures/*.pdf`, rendered inline in the notebook

Pull them down:

```bash
kaggle kernels output morsalin101/analysis -p ./outputs
```

Figures and tables regenerate with **no GPU**, so iterate on them locally:

```bash
python3 -m src.eval.stats
python3 -m src.eval.tables
python3 -m src.viz.make_all
```

---

## Checkpoints to hit

- **After step 1** — check the dedup count and split stats look sane before spending GPU
  quota on a contaminated corpus.
- **After the first pretraining finishes** — run just the probe cell in `analysis`. A k-NN
  accuracy near chance means something is wrong; better to find that after 4.5 h (MAE)
  than after 50.
- **After the first 5 segmentation seeds** — that's your first real number.

## If something goes wrong

| Symptom | Cause |
|---|---|
| `Cannot reach GitHub` | Internet is Off in session settings |
| SimCLR/MoCo abort about world size | Single-GPU accelerator; switch to T4 x2 |
| `refusing to resume: config drift` | A config changed mid-run, which would shift every schedule. Revert the config, or change the seed to start a genuinely new run. |
| `ignoring foreign checkpoint` | Harmless — a different method's checkpoint was mounted |
| `no exported encoder for 'X'` | That method's pretraining hasn't finished, or its publish cell didn't run |
| Loss goes NaN | The watchdog aborts after 20 consecutive steps. Reload the last checkpoint and halve `ref_lr`. |

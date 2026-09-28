# Reproducing the Results

Step-by-step instructions to reproduce the Phase 4 four-arm AUC comparison
and the kernel concentration analysis reported in `results/RESULTS.md`.

---

## Prerequisites

**Python 3.11** (see `.python-version`). Install dependencies:

```bash
pip install -r requirements.txt
```

**Data:** Follow `data/README.md` to obtain N24News and build the four intermediate
Parquet / NPZ files (`index.parquet`, `labeled.parquet`, `finbert_labeled.parquet`,
`data/embeddings/n24news_finance.npz`). The French factors download automatically.

---

## Step 1 — Verify data

```bash
python scripts/build_index.py          # expects 61,218 rows, 10,206 finance
python scripts/build_labels.py         # expects 10,206 labeled rows
python scripts/build_finbert_labels.py # adds finbert_label / finbert_score
python scripts/embed_images.py         # produces (10206, 512) float32 NPZ
```

Each script prints a validation report and exits non-zero on failure.

---

## Step 2 — Classical dry run (sanity check, ~10 min)

Runs the five classical classifiers on CLIP image embeddings, market-direction labels,
expanding-window CV (101 folds). Expected AUC ≈ 0.50–0.51 (market direction from
images alone is near-chance — this is the correct baseline).

```bash
python scripts/run_classical_dryrun.py
# -> results/n24news_dryrun/metrics_image.csv
```

---

## Step 3 — Phase 4A: fold inventory

Selects the 3,612 non-neutral FinBERT rows (positive + negative captions), builds
the rolling 252-day / 21-day walk-forward fold structure, and reports fold counts
and balanced train sizes. Stops before any classifier fit.

```bash
python scripts/phase4_run.py --stage fold_inventory
# -> results/phase4/fold_structure_report.csv
```

Expected: 72 folds, min balanced train ≥ 60.

---

## Step 4 — Phase 4B: four-arm comparison (~45–60 min on CPU)

Runs all four arms across 72 folds. The VQC arm is the slow one (~2700 s total).

```bash
python scripts/phase4_run.py --stage four_arm
# -> results/phase4/metrics_four_arm.csv
# -> results/phase4/metrics_per_fold.csv
# -> results/phase4/checkpoint_per_fold.csv   (crash recovery)
```

Expected headline AUC (pooled, 72 folds):

| Arm | AUC |
|-----|-----|
| Classical RBF (PCA-50) | ~0.608 |
| Classical RBF (PCA-8) | ~0.586 |
| Quantum Kernel SVM | ~0.576 |
| VQC | ~0.574 |

---

## Step 5 — Kernel concentration sweep (~5 min)

Sweeps qubit count (4, 6, 8, 10) × reps (1–4) and records off-diagonal kernel
statistics. Generates the three heatmap figures.

```bash
python scripts/concentration_sweep.py
# -> results/concentration_sweep/concentration_sweep.csv
# -> results/concentration_sweep/heatmap_*.png
```

---

## Step 6 — Assemble main figure

```bash
python scripts/assemble_figure.py
# -> results/figure_main.png
```

---

## Step 7 — Verification run (optional, ~45–60 min)

Independent re-run with a different random seed to confirm numerical stability.

```bash
python scripts/quantum_verify2.py
python scripts/classical_verify2.py
# -> results/phase4/*_verify2.csv
```

---

## Quantum benchmark on synthetic data (no data required)

```bash
python demo_quantum.py
```

Runs the 4-way benchmark on synthetic 512-dim embeddings. No N24News or French
factors needed. Useful for verifying PennyLane installation.

---

## Tests

```bash
pytest tests/ -v
```

29 tests covering quantum math properties, leakage contracts, and end-to-end
classifier learning. All should pass without any data files.

---

## IBM hardware runs (optional)

The `hardware/` directory contains scripts used to compute kernel matrices on real
IBM quantum hardware. See `hardware/README.md` for details. Reproducing these
requires an IBM Quantum account with access to `ibm_miami` or `ibm_marrakesh`.

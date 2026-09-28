# Experimental Results

> All experiments use walk-forward expanding-window cross-validation (72 folds) on N24News
> financial articles. No future data is seen during training at any stage.

---

## 1. Headline: Four-Arm AUC Comparison (Phase 4)

Four classifiers compared on the same 72-fold walk-forward splits.
Quantum arms use PCA-8 input features; the classical-50d arm uses PCA-50 as a stronger baseline.

| Arm | Pooled AUC | Mean AUC | Std AUC | Runtime (s) | Folds |
|-----|-----------|----------|---------|-------------|-------|
| Classical RBF (PCA-50) | **0.608** | 0.604 | 0.119 | 0.85 | 72 |
| Classical RBF (PCA-8) | 0.586 | 0.587 | 0.113 | 0.64 | 72 |
| Quantum Kernel SVM (r=1) | 0.576 | 0.583 | 0.107 | 108 | 72 |
| VQC (r=2) | 0.574 | 0.583 | 0.118 | 2701 | 72 |

Source: `phase4/metrics_four_arm.csv` (independently verified in `metrics_four_arm_verify2.csv`)

**Key finding:** At 8 PCA features, quantum kernel SVM and VQC match classical RBF (within noise).
The classical-50d arm outperforms all 8-feature arms, consistent with dimensionality being the
binding constraint rather than kernel expressiveness.

---

## 2. Quantum Kernel Reps Sweep

Effect of encoding repetitions on AUC and kernel concentration (8 qubits).

| Reps | Pooled AUC | Mean off-diag | Off-diag var | Runtime (s) |
|------|-----------|--------------|--------------|-------------|
| 1 | **0.576** | 0.144 | 0.0301 | 108 |
| 2 | 0.573 | 0.088 | 0.0145 | 194 |
| 3 | 0.548 | 0.034 | 0.0063 | 280 |
| 4 | 0.532 | 0.019 | 0.0031 | 371 |

Source: `phase4/metrics_reps_sweep.csv`

More repetitions cause kernel concentration (off-diagonal entries collapse toward zero),
which degrades classification. Reps=1 is optimal.

---

## 3. Kernel Concentration Analysis

Concentration ratio (off-diagonal mean / diagonal mean) as a function of qubit count and reps.
A ratio near 0 means the kernel matrix is near-identity — no useful structure.

| Qubits | Reps=1 | Reps=2 | Reps=3 | Reps=4 |
|--------|--------|--------|--------|--------|
| 4 | 0.386 | 0.308 | 0.190 | 0.152 |
| 6 | 0.263 | 0.192 | 0.091 | 0.057 |
| 8 | 0.199 | 0.134 | 0.052 | 0.028 |
| 10 | 0.156 | 0.098 | 0.032 | 0.015 |

Source: `concentration_sweep/concentration_sweep.csv`
Figures: `concentration_sweep/heatmap_off_diag_mean.png`, `heatmap_off_diag_var.png`, `heatmap_conc_ratio.png`

Concentration worsens exponentially with both qubit count and circuit depth —
the known "barren plateau" / exponential concentration phenomenon.

---

## 4. Classical Text Classifier Metrics

Walk-forward CV on FinBERT / MiniLM text embeddings, five classical models.

| Model | Accuracy | F1 | AUC-ROC | Precision | Recall |
|-------|----------|----|---------|-----------| -------|
| XGBoost | **0.577** | **0.601** | 0.558 | **0.536** | **0.684** |
| Random Forest | 0.546 | 0.560 | 0.553 | 0.511 | 0.618 |
| SVM-RBF | 0.534 | 0.449 | **0.569** | 0.500 | 0.408 |
| SVM-Poly | 0.515 | 0.248 | 0.553 | 0.448 | 0.171 |
| MLP | 0.503 | 0.557 | 0.548 | 0.477 | 0.671 |

Source: `classifiers/metrics_text.csv`

---

## 5. Portfolio and Factor Analysis

Long-short quintile portfolio and Fama-French alpha analysis require a full CRSP
daily returns panel matched to the article sample. Results pending data access.

See `portfolio/` and relevant CSVs for schema; values will populate once CRSP merge is complete.

---

## Figures

| File | Description |
|------|-------------|
| `figure_main.png` | Main four-arm AUC comparison figure |
| `concentration_sweep/heatmap_off_diag_mean.png` | Kernel off-diagonal mean by qubits × reps |
| `concentration_sweep/heatmap_off_diag_var.png` | Kernel off-diagonal variance by qubits × reps |
| `concentration_sweep/heatmap_conc_ratio.png` | Concentration ratio by qubits × reps |

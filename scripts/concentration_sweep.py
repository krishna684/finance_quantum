#!/usr/bin/env python
"""Concentration sweep on real N24News CLIP embeddings.

PART 1 (runs always, no labels needed):
  Grid qubits{4,6,8,10} x reps{1,2,3,4}.  For each cell, build the quantum
  kernel via the statevector fast path (shots=None) and record off-diagonal
  mean, off-diagonal variance, and concentration ratio (off-diag mean / diag
  mean).  Expect ratio -> 1 and variance -> 0 as qubits * reps grows.

  Outputs:
    results/concentration_sweep/subset_indices.npy     -- 200 row-indices (fixed)
    results/concentration_sweep/concentration_sweep.csv
    results/concentration_sweep/heatmap_off_diag_mean.png
    results/concentration_sweep/heatmap_off_diag_var.png
    results/concentration_sweep/heatmap_conc_ratio.png

PART 2 (requires data/labeled_sentiment.parquet):
  For each grid point: quantum-kernel SVM (precomputed) + classical RBF-SVM
  under expanding walk-forward CV on label_sent; report AUC mean / std.
  FAIL LOUDLY on NaN AUC or unrecoverable single-class folds.

  Outputs:
    results/concentration_sweep/auc_vs_expressibility.csv
    results/concentration_sweep/auc_overlay.png

Fixed subset: 200 points stratified by label_up, seed=42.
Feature map: angle encoding (pipeline/quantum/feature_maps.py::angle_map),
             SAME map as the quantum simulator pipeline.
Reducer: pipeline/quantum/reducer.py::EmbeddingReducer (StandardScaler ->
         PCA(n_qubits) -> MinMax to [0, pi]).

Usage:
    python scripts/concentration_sweep.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Make pipeline importable from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.quantum.reducer import EmbeddingReducer
from pipeline.quantum.kernel import QuantumKernel

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_OUT  = _ROOT / "results" / "concentration_sweep"

QUBITS_GRID : list[int] = [4, 6, 8, 10]
REPS_GRID   : list[int] = [1, 2, 3, 4]
N_SUBSET    : int = 200
SEED        : int = 42

# Part 2 walk-forward parameters (count-based on 200-point subset)
_WF_MIN_TRAIN : int = 80   # minimum training points per fold
_WF_TEST_SIZE : int = 20   # test window size (points, not days)
_WF_STEP      : int = 20   # how far to advance the window each fold


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load labeled.parquet + CLIP embeddings, verify row alignment."""
    labeled = pd.read_parquet(_ROOT / "data" / "labeled.parquet")
    npz = np.load(_ROOT / "data" / "embeddings" / "n24news_finance.npz")
    embeddings = npz["embeddings"].astype(np.float64)   # (10206, 512)
    if not (labeled["image_id"].values == npz["image_ids"]).all():
        raise RuntimeError("Alignment mismatch between labeled.parquet and npz image_ids")
    return labeled, embeddings


def _draw_subset(labeled: pd.DataFrame, n: int = N_SUBSET, seed: int = SEED) -> np.ndarray:
    """Stratified sample by label_up, fixed seed. Returns sorted row indices."""
    y   = labeled["label_up"].values
    rng = np.random.default_rng(seed)
    half = n // 2
    sel = np.concatenate([
        rng.choice(np.where(y == 1)[0], half,      replace=False),
        rng.choice(np.where(y == 0)[0], n - half,  replace=False),
    ])
    return np.sort(sel)


# ---------------------------------------------------------------------------
# Kernel statistics
# ---------------------------------------------------------------------------

def _kernel_stats(K: np.ndarray) -> dict:
    """Off-diagonal mean, variance, and concentration ratio."""
    n        = K.shape[0]
    diag_m   = float(np.diag(K).mean())
    off_mask = ~np.eye(n, dtype=bool)
    off      = K[off_mask]
    off_m    = float(off.mean())
    off_v    = float(off.var())
    ratio    = off_m / diag_m if diag_m > 0 else float("nan")
    return {
        "off_diag_mean" : off_m,
        "off_diag_var"  : off_v,
        "conc_ratio"    : ratio,
        "diag_mean"     : diag_m,
    }


# ---------------------------------------------------------------------------
# PART 1
# ---------------------------------------------------------------------------

def run_part1(labeled: pd.DataFrame, embeddings: np.ndarray,
              subset_idx: np.ndarray) -> pd.DataFrame:
    """Concentration sweep — no labels, statevector fast path."""
    print("\n=== PART 1: CONCENTRATION SWEEP ===")
    n_cells = len(QUBITS_GRID) * len(REPS_GRID)
    print(f"Grid     : qubits={QUBITS_GRID} x reps={REPS_GRID} = {n_cells} cells")
    print(f"Subset   : {len(subset_idx)} points")
    print(f"Fast path: shots=None (exact statevector via default.qubit)")
    print()

    rows: list[dict] = []

    for n_q in QUBITS_GRID:
        # Fit reducer on ALL 10 k embeddings for stable PCA directions.
        # Part 1 is characterisation, not prediction — no leakage concern.
        reducer = EmbeddingReducer(n_components=n_q, seed=SEED)
        reducer.fit(embeddings)
        X_sub = reducer.transform(embeddings[subset_idx])   # (200, n_q)
        ev = reducer.explained_variance_ratio.sum()
        print(f"n_qubits={n_q:2d}: PCA explained_variance={ev:.4f}  "
              f"X_sub shape={X_sub.shape}  range=[{X_sub.min():.3f},{X_sub.max():.3f}]")

        for reps in REPS_GRID:
            qk = QuantumKernel(
                n_qubits=n_q, feature_map="angle",
                reps=reps, shots=None, seed=SEED,
            )
            K     = qk.train_matrix(X_sub)
            stats = _kernel_stats(K)
            row   = {"n_qubits": n_q, "reps": reps, **stats}
            rows.append(row)
            print(
                f"  reps={reps}: off_mean={stats['off_diag_mean']:.4f}  "
                f"off_var={stats['off_diag_var']:.6f}  "
                f"conc_ratio={stats['conc_ratio']:.4f}  "
                f"diag_mean={stats['diag_mean']:.4f}"
            )

        print()

    df = pd.DataFrame(rows)
    csv_path = _OUT / "concentration_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path}")
    return df


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def plot_heatmaps(df: pd.DataFrame) -> None:
    """Three heatmaps: off_diag_mean, off_diag_var, conc_ratio."""
    specs = [
        ("off_diag_mean", "Off-diagonal Mean",               "YlOrRd",   ".4f"),
        ("off_diag_var",  "Off-diagonal Variance",           "YlOrBr",   ".5f"),
        ("conc_ratio",    "Concentration Ratio\n(off-diag mean / diag mean)", "RdYlGn_r", ".4f"),
    ]

    for col, title, cmap, fmt in specs:
        pivot = df.pivot(index="n_qubits", columns="reps", values=col)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            pivot, annot=True, fmt=fmt, cmap=cmap,
            ax=ax, linewidths=0.5, cbar_kws={"label": col},
        )
        ax.set_title(
            f"{title}\nN24News CLIP, angle map, 200 samples (seed={SEED})"
        )
        ax.set_xlabel("reps")
        ax.set_ylabel("n_qubits")
        path = _OUT / f"heatmap_{col}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved -> {path}")


# ---------------------------------------------------------------------------
# PART 2 (gated on labeled_sentiment.parquet)
# ---------------------------------------------------------------------------

def run_part2(
    labeled_sent: pd.DataFrame,
    embeddings: np.ndarray,
    subset_idx: np.ndarray,
    sweep_df: pd.DataFrame,
) -> pd.DataFrame:
    """Walk-forward SVM AUC for each grid point; classical RBF control."""
    from sklearn.svm import SVC
    from sklearn.metrics import roc_auc_score

    print("\n=== PART 2: AUC vs CONCENTRATION ===")

    # Align subset with labeled_sentiment
    if "label_sent" not in labeled_sent.columns:
        raise RuntimeError(
            "FAIL: labeled_sentiment.parquet missing column 'label_sent'"
        )
    sub      = labeled_sent.iloc[subset_idx].copy().reset_index(drop=True)
    y_sub    = sub["label_sent"].values
    # Sort subset by date for expanding window
    order    = np.argsort(sub["date"].values)
    sub_ord  = sub.iloc[order].reset_index(drop=True)
    y_ord    = y_sub[order]
    emb_ord  = embeddings[subset_idx[order]]   # (200, 512)
    n_sub    = len(y_ord)

    print(f"Subset  : {n_sub} points  "
          f"({(y_ord==1).sum()} class-1 / {(y_ord==0).sum()} class-0)")
    print(f"CV      : expanding window, min_train={_WF_MIN_TRAIN}, "
          f"test={_WF_TEST_SIZE}, step={_WF_STEP}")
    print()

    if n_sub < _WF_MIN_TRAIN + _WF_TEST_SIZE:
        raise RuntimeError(
            f"FAIL: only {n_sub} subset points, need "
            f">= {_WF_MIN_TRAIN + _WF_TEST_SIZE}"
        )

    auc_rows: list[dict] = []

    for n_q in QUBITS_GRID:
        for reps in REPS_GRID:
            conc = sweep_df.loc[
                (sweep_df.n_qubits == n_q) & (sweep_df.reps == reps),
                "conc_ratio"
            ].iloc[0]

            qk = QuantumKernel(
                n_qubits=n_q, feature_map="angle",
                reps=reps, shots=None, seed=SEED,
            )

            q_aucs: list[float] = []
            c_aucs: list[float] = []
            n_skipped = 0

            for t in range(_WF_MIN_TRAIN, n_sub - _WF_TEST_SIZE + 1, _WF_STEP):
                X_tr_raw = emb_ord[:t]
                X_te_raw = emb_ord[t : t + _WF_TEST_SIZE]
                y_tr     = y_ord[:t]
                y_te     = y_ord[t : t + _WF_TEST_SIZE]

                # FAIL LOUDLY: single-class folds
                if len(np.unique(y_tr)) < 2:
                    print(f"  [WARN] SINGLE CLASS train fold t={t} "
                          f"(n_q={n_q}, reps={reps}) — SKIPPING")
                    n_skipped += 1
                    continue
                if len(np.unique(y_te)) < 2:
                    print(f"  [WARN] SINGLE CLASS test fold t={t} "
                          f"(n_q={n_q}, reps={reps}) — SKIPPING")
                    n_skipped += 1
                    continue

                # Fit reducer on train ONLY (no leakage)
                reducer = EmbeddingReducer(n_components=n_q, seed=SEED)
                reducer.fit(X_tr_raw)
                X_tr_q = reducer.transform(X_tr_raw)
                X_te_q = reducer.transform(X_te_raw)

                # Quantum kernel SVM
                K_tr   = qk.train_matrix(X_tr_q)
                K_te   = qk.cross_matrix(X_te_q, X_tr_q)
                svm_q  = SVC(kernel="precomputed", C=1.0, probability=True)
                svm_q.fit(K_tr, y_tr)
                auc_q  = roc_auc_score(y_te, svm_q.predict_proba(K_te)[:, 1])
                if np.isnan(auc_q):
                    raise RuntimeError(
                        f"FAIL: NaN quantum AUC at t={t}, n_q={n_q}, reps={reps}"
                    )

                # Classical RBF-SVM on same n_q-dim reduced features (fair comparison)
                svm_c  = SVC(kernel="rbf", C=1.0, probability=True)
                svm_c.fit(X_tr_q, y_tr)
                auc_c  = roc_auc_score(y_te, svm_c.predict_proba(X_te_q)[:, 1])
                if np.isnan(auc_c):
                    raise RuntimeError(
                        f"FAIL: NaN classical AUC at t={t}, n_q={n_q}, reps={reps}"
                    )

                q_aucs.append(auc_q)
                c_aucs.append(auc_c)

            if not q_aucs:
                raise RuntimeError(
                    f"FAIL: 0 valid folds for n_q={n_q}, reps={reps} "
                    f"(all {n_skipped} folds skipped — check label distribution)"
                )

            row = {
                "n_qubits"          : n_q,
                "reps"              : reps,
                "conc_ratio"        : conc,
                "quantum_auc_mean"  : float(np.mean(q_aucs)),
                "quantum_auc_std"   : float(np.std(q_aucs)),
                "classical_auc_mean": float(np.mean(c_aucs)),
                "classical_auc_std" : float(np.std(c_aucs)),
                "n_folds"           : len(q_aucs),
                "n_skipped"         : n_skipped,
            }
            auc_rows.append(row)
            print(
                f"n_q={n_q:2d} reps={reps}: "
                f"Q-AUC={row['quantum_auc_mean']:.3f}+/-{row['quantum_auc_std']:.3f}  "
                f"C-AUC={row['classical_auc_mean']:.3f}+/-{row['classical_auc_std']:.3f}  "
                f"conc={conc:.4f}  folds={len(q_aucs)}  skipped={n_skipped}"
            )

    df_auc  = pd.DataFrame(auc_rows)
    auc_csv = _OUT / "auc_vs_expressibility.csv"
    df_auc.to_csv(auc_csv, index=False)
    print(f"\nSaved -> {auc_csv}")

    _plot_auc_overlay(df_auc)
    return df_auc


def _plot_auc_overlay(df: pd.DataFrame) -> None:
    """Concentration ratio (x) vs AUC (y): quantum + classical lines."""
    df_s = df.sort_values("conc_ratio").reset_index(drop=True)
    x    = df_s["conc_ratio"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        x, df_s["quantum_auc_mean"], yerr=df_s["quantum_auc_std"],
        fmt="o-", color="purple", label="Quantum kernel SVM", capsize=3, linewidth=1.5,
    )
    ax.errorbar(
        x, df_s["classical_auc_mean"], yerr=df_s["classical_auc_std"],
        fmt="s--", color="steelblue", label="Classical RBF-SVM", capsize=3, linewidth=1.5,
    )
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance (AUC=0.50)")
    ax.set_xlabel("Concentration ratio  (off-diag mean / diag mean)", fontsize=11)
    ax.set_ylabel("AUC (walk-forward CV)", fontsize=11)
    ax.set_title(
        "AUC vs Quantum Kernel Concentration\n"
        "N24News CLIP, angle map, 200 samples, expanding walk-forward CV"
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0.3, 0.85)

    for _, row in df_s.iterrows():
        ax.annotate(
            f"q{int(row['n_qubits'])}r{int(row['reps'])}",
            (row["conc_ratio"], row["quantum_auc_mean"]),
            textcoords="offset points", xytext=(5, 4), fontsize=7, color="purple",
        )

    path = _OUT / "auc_overlay.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("Loading N24News CLIP embeddings...")
    labeled, embeddings = _load_data()
    print(f"  embeddings : {embeddings.shape}  dtype={embeddings.dtype}")
    print(f"  labeled    : {labeled.shape}")

    # ---- Fixed subset ----
    idx_path = _OUT / "subset_indices.npy"
    if idx_path.exists():
        subset_idx = np.load(idx_path)
        print(f"Loaded existing subset: {len(subset_idx)} indices from {idx_path}")
    else:
        subset_idx = _draw_subset(labeled, n=N_SUBSET, seed=SEED)
        np.save(idx_path, subset_idx)
        print(f"Drew new subset: {len(subset_idx)} indices -> {idx_path}")

    y_sub = labeled["label_up"].values[subset_idx]
    print(
        f"Subset labels: {(y_sub==1).sum()} up / {(y_sub==0).sum()} down  "
        f"(target {N_SUBSET//2}/{N_SUBSET//2})"
    )

    # ---- PART 1 ----
    sweep_df = run_part1(labeled, embeddings, subset_idx)
    print("\n--- Summary table ---")
    print(sweep_df.to_string(index=False, float_format="%.5f"))

    print()
    plot_heatmaps(sweep_df)
    print("\n=== PART 1 COMPLETE ===")

    # ---- PART 2 (conditional) ----
    sent_path = _ROOT / "data" / "labeled_sentiment.parquet"
    if not sent_path.exists():
        print(f"\nPART 2 SKIPPED: {sent_path} does not exist.")
        print("Re-run this script after data/labeled_sentiment.parquet is available.")
        return

    labeled_sent = pd.read_parquet(sent_path)
    print(f"\nloaded labeled_sentiment: {labeled_sent.shape}")
    run_part2(labeled_sent, embeddings, subset_idx, sweep_df)
    print("\n=== PART 2 COMPLETE ===")


if __name__ == "__main__":
    main()

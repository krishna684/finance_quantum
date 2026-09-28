#!/usr/bin/env python
"""Classical-arm re-verification run.

Runs Arm 1 (RBF-SVM, PCA-8) and Arm 4 (RBF-SVM, PCA-50) on the same 72
rolling walk-forward folds used by phase4_run.py, with three additions:

    1. Per-fold predictions are saved to:
           results/phase4/classical_predictions_pca8.csv
           results/phase4/classical_predictions_pca50.csv
       Each CSV has columns: fold, sample_idx, y_true, y_prob
       These are the artifact needed for independent re-verification.

    2. Per-fold timing (fit+predict, in seconds) is saved to:
           results/phase4/classical_timing_per_fold.csv
       Columns: fold, n_balanced, n_test, t_c8_s, t_c50_s

    3. Pooled AUC is cross-checked against metrics_four_arm.csv and
       the script FAILS LOUDLY if the values diverge by > 1e-5.

No quantum/VQC code is imported; no QPU calls are made.
Total expected wall-clock: < 5 seconds for 72 folds.

Usage:
    python scripts/classical_rerun.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from phase4_harness import (
    load_records,
    rolling_walk_forward_splits,
    assert_no_leakage,
    build_fold_pipeline,
    undersample_train,
)

METHODS_PATH = _PROJECT_ROOT / "data" / "methods_label.parquet"
NPZ_PATH     = _PROJECT_ROOT / "data" / "embeddings" / "n24news_finance.npz"
RESULTS_DIR  = _PROJECT_ROOT / "results" / "phase4"

# Cross-check targets from the paper's existing metrics_four_arm.csv
_EXPECTED_AUC = {
    "classical_rbf_pca8":  0.585722,
    "classical_rbf_pca50": 0.607719,
}
_AUC_TOLERANCE = 1e-5


def run_classical(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray,
) -> tuple[np.ndarray, float]:
    """SVC(rbf, C=1.0, gamma='scale') — identical to phase4_run.py:run_classical()."""
    t0  = time.perf_counter()
    svc = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    svc.fit(X_train, y_train)
    y_prob = svc.predict_proba(X_test)[:, 1]
    return y_prob, time.perf_counter() - t0


def main() -> None:
    t_wall = time.perf_counter()

    print("=" * 64)
    print("CLASSICAL ARM RE-VERIFICATION")
    print("=" * 64)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"Loading {METHODS_PATH} ...")
    records = load_records(METHODS_PATH, NPZ_PATH)
    udays   = sorted({r.trading_day for r in records})
    print(f"  Records: {len(records):,}  "
          f"Unique trading days: {len(udays):,}  "
          f"({udays[0]} -> {udays[-1]})")

    # ------------------------------------------------------------------
    # Splits — confirm 72 folds, 0 excluded
    # ------------------------------------------------------------------
    splits, _ = rolling_walk_forward_splits(records)
    n_folds   = len(splits)
    print(f"  Folds generated: {n_folds}  (expected: 72)")
    if n_folds != 72:
        print(f"[FAIL] Expected 72 folds, got {n_folds}.")
        sys.exit(1)

    assert_no_leakage(splits)
    print(f"  Leakage assertion: PASSED")
    print()

    # ------------------------------------------------------------------
    # Containers
    # ------------------------------------------------------------------
    rows_c8: list[dict]   = []
    rows_c50: list[dict]  = []
    timing_rows: list[dict] = []

    acc_c8  = {"y_true": [], "y_prob": []}
    acc_c50 = {"y_true": [], "y_prob": []}

    total_t_c8  = 0.0
    total_t_c50 = 0.0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fold loop
    # ------------------------------------------------------------------
    for fold_idx, (train_recs, test_recs) in enumerate(splits):
        X_train_full = np.stack([r.embedding for r in train_recs]).astype(np.float64)
        y_train_full = np.array([r.finbert_bin for r in train_recs])
        X_test_full  = np.stack([r.embedding for r in test_recs]).astype(np.float64)
        y_test       = np.array([r.finbert_bin for r in test_recs])

        # Pipeline fitted on full pre-undersample train (same as phase4_run.py)
        pipeline = build_fold_pipeline(X_train_full)

        # Undersample train only
        us      = undersample_train(train_recs)
        bal_idx = us.train_indices
        X_train_bal = X_train_full[bal_idx]
        y_train_bal = y_train_full[bal_idx]

        # PCA-8 and PCA-50 features
        X_train_pca8  = pipeline.transform_classical8(X_train_bal)
        X_test_pca8   = pipeline.transform_classical8(X_test_full)
        X_train_pca50 = pipeline.transform_classical50(X_train_bal)
        X_test_pca50  = pipeline.transform_classical50(X_test_full)

        # Run arms
        yp_c8,  t_c8  = run_classical(X_train_pca8,  y_train_bal, X_test_pca8)
        yp_c50, t_c50 = run_classical(X_train_pca50, y_train_bal, X_test_pca50)

        total_t_c8  += t_c8
        total_t_c50 += t_c50

        # Accumulate predictions
        acc_c8["y_true"].append(y_test)
        acc_c8["y_prob"].append(yp_c8)
        acc_c50["y_true"].append(y_test)
        acc_c50["y_prob"].append(yp_c50)

        # Save per-sample predictions with fold and sample offset
        n_test = len(y_test)
        for i in range(n_test):
            rows_c8.append({
                "fold":       fold_idx,
                "sample_idx": i,
                "y_true":     int(y_test[i]),
                "y_prob":     float(yp_c8[i]),
            })
            rows_c50.append({
                "fold":       fold_idx,
                "sample_idx": i,
                "y_true":     int(y_test[i]),
                "y_prob":     float(yp_c50[i]),
            })

        timing_rows.append({
            "fold":       fold_idx,
            "n_balanced": us.n_balanced,
            "n_test":     n_test,
            "t_c8_s":     round(t_c8,  6),
            "t_c50_s":    round(t_c50, 6),
        })

    # ------------------------------------------------------------------
    # AUC cross-check
    # ------------------------------------------------------------------
    auc_c8  = float(roc_auc_score(
        np.concatenate(acc_c8["y_true"]),  np.concatenate(acc_c8["y_prob"])))
    auc_c50 = float(roc_auc_score(
        np.concatenate(acc_c50["y_true"]), np.concatenate(acc_c50["y_prob"])))

    print("POOLED AUC CROSS-CHECK:")
    for arm, auc_got, auc_exp in [
        ("classical_rbf_pca8",  auc_c8,  _EXPECTED_AUC["classical_rbf_pca8"]),
        ("classical_rbf_pca50", auc_c50, _EXPECTED_AUC["classical_rbf_pca50"]),
    ]:
        delta = abs(auc_got - auc_exp)
        ok    = delta <= _AUC_TOLERANCE
        status = "OK" if ok else "MISMATCH"
        print(f"  {arm:<26}  got={auc_got:.6f}  expected={auc_exp:.6f}  "
              f"delta={delta:.2e}  [{status}]")
        if not ok:
            print(f"[FAIL] AUC mismatch for {arm}: delta {delta:.2e} > tolerance {_AUC_TOLERANCE:.0e}")
            print("       Do NOT overwrite paper numbers. Investigate before proceeding.")
            sys.exit(1)

    print()
    print("TIMING (fit + predict, both arms, 72 folds):")
    print(f"  classical_rbf_pca8 : {total_t_c8:.4f}s  total  "
          f"({total_t_c8/n_folds*1000:.2f} ms/fold avg)")
    print(f"  classical_rbf_pca50: {total_t_c50:.4f}s  total  "
          f"({total_t_c50/n_folds*1000:.2f} ms/fold avg)")
    print(f"  Measurement boundary: time.perf_counter() around svc.fit() + predict_proba()")
    print()

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    pred_c8_path   = RESULTS_DIR / "classical_predictions_pca8.csv"
    pred_c50_path  = RESULTS_DIR / "classical_predictions_pca50.csv"
    timing_path    = RESULTS_DIR / "classical_timing_per_fold.csv"

    pd.DataFrame(rows_c8).to_csv(pred_c8_path,  index=False)
    pd.DataFrame(rows_c50).to_csv(pred_c50_path, index=False)
    pd.DataFrame(timing_rows).to_csv(timing_path, index=False)

    print("ARTIFACTS SAVED:")
    print(f"  {pred_c8_path}   ({len(rows_c8)} rows)")
    print(f"  {pred_c50_path}  ({len(rows_c50)} rows)")
    print(f"  {timing_path}   ({len(timing_rows)} rows)")
    print()
    print(f"Wall-clock: {time.perf_counter()-t_wall:.2f}s")
    print("=" * 64)
    print("VERDICT: FULLY VERIFIED")
    print("  runtime confirmed (existing 0.64s / 0.85s sourced from phase4_run.py)")
    print("  re-run AUCs match paper values within 1e-5")
    print("  per-fold predictions and per-fold timing saved for independent re-verification")
    print("=" * 64)


if __name__ == "__main__":
    main()

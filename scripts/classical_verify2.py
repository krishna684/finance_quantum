#!/usr/bin/env python
"""Phase C — Independent classical re-run for verify2 audit.

Imports: phase4_harness only. No imports from phase4_run.py, classical_rerun.py,
or any other project script — this is a fully independent second execution path.

Outputs (never overwrites originals):
    results/phase4/classical_predictions_pca8_verify2.csv
    results/phase4/classical_predictions_pca50_verify2.csv
    results/phase4/classical_timing_per_fold_verify2.csv

After writing, reads the fresh CSVs and independently computes pooled+mean AUC
using only pandas + sklearn — no project AUC helper functions.

Usage: python scripts/classical_verify2.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

_SCRIPTS = Path(__file__).resolve().parent
_ROOT    = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_ROOT))

from phase4_harness import (
    load_records,
    rolling_walk_forward_splits,
    assert_no_leakage,
    build_fold_pipeline,
    undersample_train,
)

METHODS_PATH = _ROOT / "data" / "methods_label.parquet"
NPZ_PATH     = _ROOT / "data" / "embeddings" / "n24news_finance.npz"
OUT_DIR      = _ROOT / "results" / "phase4"

# Reference values from the existing paper files
_REF = {
    "classical_rbf_pca8":  {"pooled": 0.585722, "mean": 0.586911},
    "classical_rbf_pca50": {"pooled": 0.607719, "mean": 0.604338},
}
_TOL = 1e-4   # tolerance for PASS/WARN decision


def _svc_fit_predict(X_tr, y_tr, X_te):
    """Identical hyperparameters to phase4_run.py:run_classical()."""
    t0  = time.perf_counter()
    svc = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    svc.fit(X_tr, y_tr)
    yp  = svc.predict_proba(X_te)[:, 1]
    return yp, time.perf_counter() - t0


def main() -> None:
    t_wall = time.perf_counter()
    print("=" * 64)
    print("CLASSICAL VERIFY2 — independent re-run")
    print("=" * 64)

    records = load_records(METHODS_PATH, NPZ_PATH)
    udays   = sorted({r.trading_day for r in records})
    print(f"Records: {len(records):,}  "
          f"Trading days: {len(udays):,}  "
          f"({udays[0]} -> {udays[-1]})")

    splits, _ = rolling_walk_forward_splits(records)
    n_folds   = len(splits)
    print(f"Folds: {n_folds}  (hard-coded expectation: 72)")
    if n_folds != 72:
        print(f"[FAIL] got {n_folds} folds, expected 72")
        sys.exit(1)
    assert_no_leakage(splits)
    print("Leakage assertion: PASSED\n")

    rows_c8, rows_c50, timing_rows = [], [], []
    acc = {
        "c8":  {"y_true": [], "y_prob": []},
        "c50": {"y_true": [], "y_prob": []},
    }
    total_t = {"c8": 0.0, "c50": 0.0}

    for fi, (train_recs, test_recs) in enumerate(splits):
        X_tr_full = np.stack([r.embedding for r in train_recs]).astype(np.float64)
        y_tr_full = np.array([r.finbert_bin for r in train_recs])
        X_te_full = np.stack([r.embedding for r in test_recs]).astype(np.float64)
        y_te      = np.array([r.finbert_bin for r in test_recs])

        pipe    = build_fold_pipeline(X_tr_full)
        us      = undersample_train(train_recs)
        bi      = us.train_indices
        X_tr_b  = X_tr_full[bi]
        y_tr_b  = y_tr_full[bi]

        X_tr8,  X_te8  = pipe.transform_classical8(X_tr_b),  pipe.transform_classical8(X_te_full)
        X_tr50, X_te50 = pipe.transform_classical50(X_tr_b), pipe.transform_classical50(X_te_full)

        yp8,  t8  = _svc_fit_predict(X_tr8,  y_tr_b, X_te8)
        yp50, t50 = _svc_fit_predict(X_tr50, y_tr_b, X_te50)

        total_t["c8"]  += t8
        total_t["c50"] += t50
        acc["c8"]["y_true"].append(y_te);   acc["c8"]["y_prob"].append(yp8)
        acc["c50"]["y_true"].append(y_te);  acc["c50"]["y_prob"].append(yp50)

        n_te = len(y_te)
        for i in range(n_te):
            rows_c8.append({"fold": fi, "sample_idx": i, "y_true": int(y_te[i]), "y_prob": float(yp8[i])})
            rows_c50.append({"fold": fi, "sample_idx": i, "y_true": int(y_te[i]), "y_prob": float(yp50[i])})
        timing_rows.append({"fold": fi, "n_balanced": us.n_balanced,
                             "n_test": n_te, "t_c8_s": round(t8, 6), "t_c50_s": round(t50, 6)})

    # Write _verify2 files (never overwrite originals)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path_c8      = OUT_DIR / "classical_predictions_pca8_verify2.csv"
    path_c50     = OUT_DIR / "classical_predictions_pca50_verify2.csv"
    path_timing  = OUT_DIR / "classical_timing_per_fold_verify2.csv"
    pd.DataFrame(rows_c8).to_csv(path_c8,     index=False)
    pd.DataFrame(rows_c50).to_csv(path_c50,   index=False)
    pd.DataFrame(timing_rows).to_csv(path_timing, index=False)

    # --- Independent AUC recompute directly from the _verify2 files ---
    print("\nINDEPENDENT AUC RECOMPUTE (from fresh _verify2 files only):")
    results = {}
    for arm, path in [("classical_rbf_pca8", path_c8), ("classical_rbf_pca50", path_c50)]:
        df      = pd.read_csv(path)
        pooled  = roc_auc_score(df["y_true"], df["y_prob"])
        per_fold = (df.groupby("fold")
                      .apply(lambda g: roc_auc_score(g["y_true"], g["y_prob"]),
                             include_groups=False))
        mean_v  = per_fold.mean()
        ref_p   = _REF[arm]["pooled"]
        ref_m   = _REF[arm]["mean"]
        dp      = abs(pooled - ref_p)
        dm      = abs(mean_v - ref_m)
        verdict = "PASS" if dp < _TOL and dm < _TOL else "FAIL"
        print(f"\n  {arm}:")
        print(f"    pooled AUC  v2={pooled:.10f}  ref={ref_p:.6f}  delta={dp:.3e}  [{verdict}]")
        print(f"    mean   AUC  v2={mean_v:.10f}  ref={ref_m:.6f}  delta={dm:.3e}  [{verdict}]")
        results[arm] = {"pooled": pooled, "mean": mean_v, "verdict": verdict}

    print(f"\nTIMING (fit+predict, perf_counter, 72 folds):")
    print(f"  PCA-8  total: {total_t['c8']:.4f}s  (ref in paper: 0.64s)")
    print(f"  PCA-50 total: {total_t['c50']:.4f}s  (ref in paper: 0.85s)")
    print(f"  Note: timing varies run-to-run; paper value from July 26 run with time.time()")
    print(f"\nARTIFACTS WRITTEN (verify2, originals untouched):")
    print(f"  {path_c8}")
    print(f"  {path_c50}")
    print(f"  {path_timing}")
    print(f"\nWall-clock: {time.perf_counter()-t_wall:.2f}s")
    print("=" * 64)

    # Exit 1 if any arm fails tolerance
    if any(r["verdict"] == "FAIL" for r in results.values()):
        print("[FAIL] AUC mismatch exceeds tolerance — DO NOT update paper numbers without investigation.")
        sys.exit(1)
    else:
        print("VERDICT: CLASSICAL ARM REPRODUCES within 1e-4 tolerance.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Walk-forward classification dry run on N24News finance subset.

Standalone runner: does NOT call ClassificationPipeline (incompatible —
see Phase 3A audit A1: that pipeline reads SQLite + CRSP firm returns,
whereas we have parquet + market-level returns with no PERMNO/ticker).

The walk_forward_splits() function and all five classifier constructions
below are copied verbatim from pipeline/classifiers.py:280-407 for
methodological identity (same fold logic, same hyperparameters, same
sklearn Pipeline structure). Only the DataRecord type is adapted:
permno and ticker fields are removed; label_up and mkt_ret_next replace
ret_sign and actual_return.

CLIP embedding backend note
---------------------------
open_clip is NOT installed in this environment and is not a dependency
of this script. The canonical CLIP backend for all image embeddings in
this project is:
    transformers openai/clip-vit-base-patch32  (vision_model + visual_projection)
This is consistent with pipeline/embeddings.py and
pipeline/tiktok_collection/frame_embedder.py, both of which prefer
open_clip but fall back to this transformers path identically.
Do not install open_clip without first running the 20-image cosine-sim
cross-check (A4) and confirming min similarity > 0.999.

Portfolio / economic evaluation
--------------------------------
SKIPPED intentionally. The portfolio module (pipeline/portfolio.py)
constructs long-short quintile portfolios ranked by predicted probability
per (stock, trading_day). Market-level labels carry no cross-sectional
variation — every image on the same day has the same label and the same
mkt_ret_next. There is no valid cross-section to rank, so quintile
construction is undefined. This is documented here and will not be
shimmed.

Outputs
-------
    results/n24news_dryrun/metrics_image.csv
    results/n24news_dryrun/predictions_image_{model}.csv

Usage
-----
    python scripts/run_classical_dryrun.py
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELED_PATH = _PROJECT_ROOT / "data" / "labeled.parquet"
NPZ_PATH     = _PROJECT_ROOT / "data" / "embeddings" / "n24news_finance.npz"
RESULTS_DIR  = _PROJECT_ROOT / "results" / "n24news_dryrun"

_MIN_TRAIN_DAYS = 252   # passed explicitly — do NOT rely on function default
_TEST_DAYS      = 21    # passed explicitly

# LibSVM (rbf/poly kernel) is O(n²) in training samples.  For a dry run whose
# sole purpose is integration validation, cap the SVM training set to avoid a
# multi-hour runtime.  XGBoost / RandomForest / MLP always see the full set.
# Classifier hyperparameters are NOT changed — only the sample count fed in.
# This adaptation is in main(), outside the verbatim block (lines 87-213).
_MAX_SVM_TRAIN  = 2000
_SVM_RNG_SEED   = 42


# ---------------------------------------------------------------------------
# DataRecord (adapted from classifiers.py: no permno/ticker)
# ---------------------------------------------------------------------------

@dataclass
class DataRecord:
    image_id:     str
    trading_day:  date
    embedding:    np.ndarray   # float32, shape (512,)
    label_up:     int          # 1 = market up, 0 = market down
    mkt_ret_next: float        # decimal market return on trading_day


# ---------------------------------------------------------------------------
# walk_forward_splits — copied verbatim from pipeline/classifiers.py:280-333
# Only type annotation changed: list[DataRecord] instead of list[DataRecord]
# (DataRecord is redefined above without permno/ticker, but the fold logic
# is identical).
# ---------------------------------------------------------------------------

def walk_forward_splits(
    records: list[DataRecord],
    min_train_days: int = 252,
    test_days: int = 21,
) -> list[tuple[list[DataRecord], list[DataRecord]]]:
    """Generate strictly expanding-window folds split on day boundaries.

    Args:
        records:        Observations sorted by trading_day.
        min_train_days: Minimum number of unique trading days required before
                        the first test fold begins.  Default is 252 (~1 trading
                        year), consistent with Obaid & Pukthuanthong (2022).
                        With ~4 years of data this yields ~37 monthly folds.
        test_days:      Number of unique trading days in each test fold.

    Returns:
        List of (train_records, test_records) tuples.  Every record in
        train_records has trading_day strictly less than the earliest
        trading_day in test_records (no look-ahead).
    """
    unique_days = sorted({r.trading_day for r in records})
    n_days = len(unique_days)

    if n_days < min_train_days + test_days:
        # Adaptive fallback: use first 60 % for training
        split_idx = max(1, int(n_days * 0.6))
        print(
            f"[walk_forward] WARNING: only {n_days} unique days "
            f"(need {min_train_days + test_days}); using adaptive 60/40 split"
        )
        min_train_days = split_idx
        test_days = max(1, n_days - split_idx)

    splits: list[tuple[list[DataRecord], list[DataRecord]]] = []
    fold_start = min_train_days

    while fold_start < n_days:
        fold_end = min(fold_start + test_days, n_days)

        train_cutoff = unique_days[fold_start]   # exclusive upper bound for train
        test_lo = unique_days[fold_start]
        test_hi = unique_days[fold_end - 1]

        train = [r for r in records if r.trading_day < train_cutoff]
        test  = [r for r in records if test_lo <= r.trading_day <= test_hi]

        if len(train) >= 10 and len(test) >= 1:
            splits.append((train, test))

        fold_start += test_days

    return splits


# ---------------------------------------------------------------------------
# Model factory — copied verbatim from pipeline/classifiers.py:340-407
# ---------------------------------------------------------------------------

def _build_models(n_features: int) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    try:
        import xgboost as xgb
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        print("[warn] xgboost unavailable, using GradientBoostingClassifier")
        xgb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)

    pca_k = min(50, n_features)

    return {
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=pca_k)),
            ("clf",    SVC(kernel="rbf", C=1.0, gamma="scale",
                           probability=True, random_state=42)),
        ]),
        "svm_poly": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=pca_k)),
            ("clf",    SVC(kernel="poly", C=1.0, degree=3,
                           probability=True, random_state=42)),
        ]),
        "xgboost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    xgb_clf),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=500,
                early_stopping=True,
                random_state=42,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc":   auc,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_records(labeled_path: Path, npz_path: Path) -> list[DataRecord]:
    df = pd.read_parquet(labeled_path)
    print(f"labeled.parquet rows: {len(df):,}")

    npz = np.load(npz_path, allow_pickle=True)
    emb_ids  = npz["image_ids"].astype(str)
    emb_mat  = npz["embeddings"].astype(np.float32)
    emb_map  = {iid: emb_mat[i] for i, iid in enumerate(emb_ids)}

    # Join validation
    df_ids    = set(df["image_id"].astype(str))
    emb_id_set = set(emb_ids)
    missing_emb = df_ids - emb_id_set
    if missing_emb:
        print(f"[VALIDATION FAILED] {len(missing_emb):,} labeled rows have no embedding.")
        sys.exit(1)
    print(f"Join validation PASSED: all {len(df):,} rows have embeddings")

    records: list[DataRecord] = []
    for _, row in df.iterrows():
        iid = str(row["image_id"])
        # next_trading_day stored as date32 in parquet -> comes back as date or Timestamp
        td = row["next_trading_day"]
        if hasattr(td, "date"):
            td = td.date()
        records.append(DataRecord(
            image_id=iid,
            trading_day=td,
            embedding=emb_map[iid],
            label_up=int(row["label_up"]),
            mkt_ret_next=float(row["mkt_ret_next"]),
        ))

    return records


# ---------------------------------------------------------------------------
# Fold split-leakage assertion
# ---------------------------------------------------------------------------

def _assert_no_leakage(splits: list[tuple]) -> None:
    for fold_idx, (train, test) in enumerate(splits):
        train_days = {r.trading_day for r in train}
        test_days  = {r.trading_day for r in test}
        overlap = train_days & test_days
        if overlap:
            print(f"[VALIDATION FAILED] Fold {fold_idx}: "
                  f"{len(overlap)} trading day(s) in both train and test: "
                  f"{sorted(overlap)[:3]}")
            sys.exit(1)
    print(f"  Leakage assertion PASSED across {len(splits)} folds")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Force line-buffered stdout so progress is visible when output is redirected.
    sys.stdout.reconfigure(line_buffering=True)
    t_total = time.time()

    records = _load_records(LABELED_PATH, NPZ_PATH)
    print(f"Records loaded: {len(records):,}")

    records.sort(key=lambda r: r.trading_day)

    unique_days = sorted({r.trading_day for r in records})
    print(f"Unique trading days: {len(unique_days):,}  "
          f"({unique_days[0]} -> {unique_days[-1]})")

    # Majority-class baseline
    labels = np.array([r.label_up for r in records])
    majority_class = int(np.bincount(labels).argmax())
    majority_acc   = float(np.mean(labels == majority_class))
    print(f"Majority class: {majority_class}  baseline accuracy: {majority_acc:.4f}")
    print(f"Class balance:  up={labels.sum():,}  down={(~labels.astype(bool)).sum():,}")

    print(f"\nGenerating walk-forward splits "
          f"(min_train_days={_MIN_TRAIN_DAYS}, test_days={_TEST_DAYS}) ...")
    splits = walk_forward_splits(records,
                                 min_train_days=_MIN_TRAIN_DAYS,
                                 test_days=_TEST_DAYS)
    print(f"  Folds generated: {len(splits)}")
    if not splits:
        print("[VALIDATION FAILED] No folds generated. "
              "Check min_train_days vs unique trading days.")
        sys.exit(1)

    _assert_no_leakage(splits)

    n_features = records[0].embedding.shape[0]
    models = _build_models(n_features)
    print(f"\nClassifiers: {list(models.keys())}")
    print(f"Feature dim: {n_features}")

    # Per-model accumulators
    fold_rows: list[dict] = []
    pred_rows: dict[str, list[dict]] = {name: [] for name in models}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _METRICS_FIELDNAMES = [
        "model", "fold", "n_train", "n_test",
        "accuracy", "f1", "auc_roc", "precision", "recall",
    ]
    # Open incremental metrics file so progress survives an early kill
    _metrics_inc = open(RESULTS_DIR / "metrics_image_partial.csv", "w",
                        newline="", encoding="utf-8")
    _metrics_inc_writer = csv.DictWriter(_metrics_inc, fieldnames=_METRICS_FIELDNAMES)
    _metrics_inc_writer.writeheader()
    _metrics_inc.flush()

    for fold_idx, (train_recs, test_recs) in enumerate(splits):
        X_train = np.stack([r.embedding for r in train_recs])
        y_train = np.array([r.label_up  for r in train_recs])
        X_test  = np.stack([r.embedding for r in test_recs])
        y_test  = np.array([r.label_up  for r in test_recs])

        if len(np.unique(y_train)) < 2:
            print(f"  [fold {fold_idx}] SKIP — single class in training data")
            continue

        # Adjust PCA n_components so it never exceeds training samples
        for name, pipe in models.items():
            if hasattr(pipe, "named_steps") and "pca" in pipe.named_steps:
                k = min(pipe.named_steps["pca"].n_components, len(X_train) - 1, n_features)
                pipe.set_params(pca__n_components=max(1, k))

        t_fold = time.time()
        _svm_rng = np.random.default_rng(_SVM_RNG_SEED + fold_idx)
        for model_name, pipe in models.items():
            # SVM sample cap: subsample training set for kernel SVMs only
            if "svm" in model_name and len(X_train) > _MAX_SVM_TRAIN:
                idx = _svm_rng.choice(len(X_train), _MAX_SVM_TRAIN, replace=False)
                X_fit, y_fit = X_train[idx], y_train[idx]
            else:
                X_fit, y_fit = X_train, y_train
            if len(np.unique(y_fit)) < 2:
                print(f"  [fold {fold_idx}] {model_name} SKIP — "
                      f"single class in (subsampled) fit set")
                continue
            try:
                pipe.fit(X_fit, y_fit)
                y_pred = pipe.predict(X_test)
                y_prob = pipe.predict_proba(X_test)[:, 1]
            except Exception as exc:
                print(f"  [fold {fold_idx}] {model_name} FAILED: {exc}")
                continue

            m = _compute_metrics(y_test, y_pred, y_prob)

            if np.isnan(m["auc_roc"]):
                print(f"  [fold {fold_idx}] {model_name} NaN AUC "
                      f"(classes in test: {np.unique(y_test)})")

            row = {
                "model":    model_name,
                "fold":     fold_idx,
                "n_train":  len(X_train),
                "n_test":   len(X_test),
                **m,
            }
            fold_rows.append(row)
            _metrics_inc_writer.writerow(row)
            _metrics_inc.flush()

            for rec, prob, pred in zip(test_recs, y_prob, y_pred):
                pred_rows[model_name].append({
                    "date":           rec.trading_day.isoformat(),
                    "image_id":       rec.image_id,
                    "predicted_prob": float(prob),
                    "predicted_label": int(pred),
                    "true_label":     int(rec.label_up),
                    "mkt_ret_next":   float(rec.mkt_ret_next),
                    "fold":           fold_idx,
                })

        elapsed_fold = time.time() - t_fold
        print(f"  fold {fold_idx:3d}/{len(splits)}  "
              f"train={len(X_train):,}  test={len(X_test):,}  "
              f"({elapsed_fold:.1f}s)")

    _metrics_inc.close()

    # ---------------------------------------------------------------------------
    # Validation gates: no NaN AUC, all five models present
    # ---------------------------------------------------------------------------
    metrics_df = pd.DataFrame(fold_rows)

    nan_auc_rows = metrics_df[metrics_df["auc_roc"].isna()]
    if len(nan_auc_rows) > 0:
        print(f"\n[VALIDATION FAILED] {len(nan_auc_rows)} fold/model combinations "
              f"have NaN AUC:")
        print(nan_auc_rows[["model", "fold", "n_test"]].to_string())
        sys.exit(1)

    missing_models = set(models.keys()) - set(metrics_df["model"].unique())
    if missing_models:
        print(f"\n[VALIDATION FAILED] Models with zero completed folds: {missing_models}")
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = RESULTS_DIR / "metrics_image.csv"
    metrics_df.to_csv(metrics_path, index=False)

    for model_name, rows in pred_rows.items():
        if not rows:
            continue
        p = RESULTS_DIR / f"predictions_image_{model_name}.csv"
        fieldnames = ["date", "image_id", "predicted_prob", "predicted_label",
                      "true_label", "mkt_ret_next", "fold"]
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    # ---------------------------------------------------------------------------
    # Summary report
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WALK-FORWARD CLASSIFICATION RESULTS — N24News Finance Subset")
    print("=" * 70)
    print(f"  Folds completed : {len(splits)}")
    print(f"  Total records   : {len(records):,}")
    print(f"  Unique days     : {len(unique_days):,}")
    print(f"  Feature dim     : {n_features}")
    print(f"  Majority baseline accuracy: {majority_acc:.4f}")
    print()
    print(f"{'Model':<16} {'Folds':>5}  {'Acc mean':>9} {'Acc std':>8}  "
          f"{'F1 mean':>8}  {'AUC mean':>9}  {'AUC std':>8}")
    print("-" * 70)

    for model_name in models:
        sub = metrics_df[metrics_df["model"] == model_name]
        if sub.empty:
            print(f"{model_name:<16}  NO DATA")
            continue
        print(
            f"{model_name:<16} {len(sub):>5}  "
            f"{sub['accuracy'].mean():>9.4f} {sub['accuracy'].std():>8.4f}  "
            f"{sub['f1'].mean():>8.4f}  "
            f"{sub['auc_roc'].mean():>9.4f}  {sub['auc_roc'].std():>8.4f}"
        )

    print("-" * 70)
    print(f"{'Majority baseline':<16} {'—':>5}  {majority_acc:>9.4f} {'—':>8}  "
          f"{'—':>8}  {'0.5000':>9}  {'—':>8}")
    print()
    print("NOTE: Near-chance AUC is expected. This is an integration dry run.")
    print("NOTE: Portfolio evaluation SKIPPED — market-level labels have no")
    print("      cross-sectional variation; quintile ranking is undefined.")
    print()
    print(f"Outputs: {RESULTS_DIR}")
    print(f"Total runtime: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()

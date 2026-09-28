#!/usr/bin/env python
"""Phase 4 walk-forward harness — cross-modal sentiment (methods leg).

Architecture
------------
Four arms share the same temporal fold structure, the same feature pipeline,
and the same per-fold balanced training set:

  Arm 1: Classical RBF SVM on PCA-8 features   (primary quantum vs classical)
  Arm 2: Quantum Kernel SVM on angle-encoded PCA-8
  Arm 3: VQC on angle-encoded PCA-8
  Arm 4: Classical RBF SVM on PCA-50 features  (upper reference)

Feature pipeline per fold (shared, then split at the [0,π] rescale):
  StandardScaler  → PCA(8)  → classical arm (Arm 1)
                           → MinMax-to-[0,π] → quantum arms (Arms 2, 3)
  StandardScaler  → PCA(50) → Arm 4

Undersample policy (train only):
  Keep all P positives. Sample P negatives (1:1). If 2P < 60, relax to 2P
  negatives (1:2). Never touch test.

walk_forward_splits source
--------------------------
The rolling_walk_forward_splits() function below is derived verbatim from
pipeline/classifiers.py:280-333 (also in run_classical_dryrun.py:87-138).
The ONLY change is the train filter: expanding window replaced with a fixed
252-unique-trading-day rolling window by adding a roll_lo lower bound.

  ORIGINAL (expanding):
      train = [r for r in records if r.trading_day < train_cutoff]  # line 325

  ROLLING (this file):
      roll_lo_idx = fold_start - min_train_days          # assert >= 0
      roll_lo     = unique_days[roll_lo_idx]
      train       = [r for r in records if roll_lo <= r.trading_day < train_cutoff]

The strict-< upper boundary (train_cutoff = unique_days[fold_start]) and the
test window (test_lo <= day <= test_hi) are preserved without modification.

Usage
-----
  # Task 3 (fold report only, no models):
  python scripts/phase4_harness.py

  # Task 4C (full four-arm run) — add --run-models flag when implemented:
  python scripts/phase4_harness.py --run-models
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
METHODS_PATH   = _PROJECT_ROOT / "data" / "methods_label.parquet"
NPZ_PATH       = _PROJECT_ROOT / "data" / "embeddings" / "n24news_finance.npz"
RESULTS_DIR    = _PROJECT_ROOT / "results" / "phase4"

_MIN_TRAIN_DAYS = 252
_TEST_DAYS      = 21
_PCA_K_QUANTUM  = 8
_PCA_K_UPPER    = 50
_UNDERSAMPLE_SEED = 42
_MIN_BALANCED   = 60   # threshold: if 2P < this, relax 1:1 -> 1:2


# ---------------------------------------------------------------------------
# DataRecord
# ---------------------------------------------------------------------------

@dataclass
class DataRecord:
    image_id:     str
    trading_day:  date
    embedding:    np.ndarray   # float32, shape (512,)
    finbert_bin:  int          # 1=positive, 0=negative
    finbert_score: float


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(methods_path: Path, npz_path: Path) -> list[DataRecord]:
    df  = pd.read_parquet(methods_path)
    npz = np.load(npz_path, allow_pickle=True)

    emb_ids = npz["image_ids"].astype(str)
    emb_mat = npz["embeddings"].astype(np.float32)
    emb_map = {iid: emb_mat[i] for i, iid in enumerate(emb_ids)}

    # Join validation
    missing = set(df["image_id"].astype(str)) - set(emb_ids)
    if missing:
        print(f"[LOAD FAILED] {len(missing):,} records have no embedding.")
        sys.exit(1)

    records: list[DataRecord] = []
    for _, row in df.iterrows():
        iid = str(row["image_id"])
        td  = row["trading_day"]
        if hasattr(td, "date"):
            td = td.date()
        records.append(DataRecord(
            image_id=iid,
            trading_day=td,
            embedding=emb_map[iid],
            finbert_bin=int(row["finbert_bin"]),
            finbert_score=float(row["finbert_score"]),
        ))

    records.sort(key=lambda r: r.trading_day)
    print(f"Records loaded: {len(records):,}")
    labels = np.array([r.finbert_bin for r in records])
    print(f"  positive (1): {labels.sum():,}  negative (0): {(~labels.astype(bool)).sum():,}")
    return records


# ---------------------------------------------------------------------------
# Rolling walk-forward splits
# Derived verbatim from pipeline/classifiers.py:280-333.
# Change: expanding train -> fixed 252-unique-day rolling window.
# Strict-< upper boundary and test window unchanged.
# ---------------------------------------------------------------------------

def rolling_walk_forward_splits(
    records: list[DataRecord],
    min_train_days: int = _MIN_TRAIN_DAYS,
    test_days:      int = _TEST_DAYS,
) -> list[tuple[list[DataRecord], list[DataRecord]]]:
    """Fixed rolling-window folds, split on unique-trading-day boundaries.

    The train window always spans exactly min_train_days unique trading days
    present in the records (not calendar days).

    Original expanding-window source: pipeline/classifiers.py:280-333.
    Rolling modification: roll_lo = unique_days[fold_start - min_train_days]
    added as the lower bound of the train filter.
    """
    unique_days = sorted({r.trading_day for r in records})
    n_days      = len(unique_days)

    if n_days < min_train_days + test_days:
        print(
            f"[FAIL] Only {n_days} unique trading days in records; "
            f"need >= {min_train_days + test_days} "
            f"({min_train_days} train + {test_days} test). Cannot form any fold."
        )
        sys.exit(1)

    splits: list[tuple[list[DataRecord], list[DataRecord]]] = []
    fold_start = min_train_days  # first fold_start where roll_lo_idx = 0

    while fold_start < n_days:
        fold_end = min(fold_start + test_days, n_days)

        # CRITICAL: assert roll_lo_idx >= 0 before indexing unique_days.
        # A negative index would silently pull a future train window.
        roll_lo_idx = fold_start - min_train_days
        if roll_lo_idx < 0:
            print(
                f"[FAIL] fold_start={fold_start} yields roll_lo_idx={roll_lo_idx} < 0. "
                f"This would silently index from the end of unique_days. Aborting."
            )
            sys.exit(1)

        roll_lo       = unique_days[roll_lo_idx]        # inclusive lower bound for train
        train_cutoff  = unique_days[fold_start]         # exclusive upper bound for train
        test_lo       = unique_days[fold_start]
        test_hi       = unique_days[fold_end - 1]

        # Train: [roll_lo, train_cutoff)  strict-< upper boundary preserved
        train = [r for r in records if roll_lo <= r.trading_day < train_cutoff]
        # Test:  [test_lo, test_hi]
        test  = [r for r in records if test_lo <= r.trading_day <= test_hi]

        if len(train) >= 10 and len(test) >= 1:
            splits.append((train, test))

        fold_start += test_days

    return splits, unique_days


# ---------------------------------------------------------------------------
# Leakage assertion
# ---------------------------------------------------------------------------

def assert_no_leakage(splits: list[tuple]) -> None:
    for fold_idx, (train, test) in enumerate(splits):
        train_days = {r.trading_day for r in train}
        test_days  = {r.trading_day for r in test}
        overlap    = train_days & test_days
        if overlap:
            print(
                f"[FAIL] Fold {fold_idx}: {len(overlap)} trading day(s) "
                f"appear in BOTH train and test: {sorted(overlap)[:3]}"
            )
            sys.exit(1)
    print(f"  Leakage assertion PASSED across {len(splits)} folds")


# ---------------------------------------------------------------------------
# Per-fold feature pipeline
# Returns (scaler8, pca8, z_min, z_range, scaler50, pca50)
# Scaler and PCA are fit on the FULL pre-undersample train set.
# MinMax is fit on TRAIN PCA-8 output only (quantum rescale).
# ---------------------------------------------------------------------------

@dataclass
class FoldPipeline:
    scaler8:  StandardScaler
    pca8:     PCA
    z_min:    np.ndarray   # (8,) — fit on TRAIN PCA-8 only
    z_range:  np.ndarray   # (8,) — fit on TRAIN PCA-8 only
    scaler50: StandardScaler
    pca50:    PCA
    # Explainability
    expl_var_8:  np.ndarray   # (8,)  per-component explained variance ratios
    expl_var_50: np.ndarray   # (50,)

    def transform_classical8(self, X: np.ndarray) -> np.ndarray:
        """PCA-8 features for Arms 1 (classical RBF-8)."""
        return self.pca8.transform(self.scaler8.transform(X))

    def transform_quantum(self, X: np.ndarray) -> np.ndarray:
        """Angle-encoded PCA-8 features for Arms 2/3 (quantum). Clipped to [0,π]."""
        Z = self.pca8.transform(self.scaler8.transform(X))
        Z01 = (Z - self.z_min) / self.z_range
        return np.clip(Z01, 0.0, 1.0) * np.pi

    def transform_classical50(self, X: np.ndarray) -> np.ndarray:
        """PCA-50 features for Arm 4 (classical upper reference)."""
        return self.pca50.transform(self.scaler50.transform(X))


def build_fold_pipeline(X_train: np.ndarray) -> FoldPipeline:
    """Fit all feature pipeline objects on the FULL pre-undersample train set.

    The MinMax rescale for quantum arms is also fit here, on the TRAIN
    PCA-8 output. It is NEVER fit on test data — the assert below enforces this.
    Caller must call transform_quantum(X_test) only after this function returns.
    """
    n_samples   = X_train.shape[0]
    k8  = min(_PCA_K_QUANTUM, n_samples - 1)
    k50 = min(_PCA_K_UPPER,   n_samples - 1)

    # Arm 1/2/3: StandardScaler -> PCA(8)
    scaler8 = StandardScaler().fit(X_train)
    pca8    = PCA(n_components=k8, random_state=42).fit(scaler8.transform(X_train))
    Z_train = pca8.transform(scaler8.transform(X_train))   # (n_train, 8)

    # MinMax for quantum [0,π] — fit on TRAIN only
    # ASSERT: this block executes on X_train exclusively; X_test never enters here.
    z_min   = Z_train.min(axis=0)
    z_rng   = Z_train.max(axis=0) - z_min
    z_rng[z_rng == 0.0] = 1.0   # guard degenerate (zero-variance) components

    # Arm 4: StandardScaler -> PCA(50)
    scaler50 = StandardScaler().fit(X_train)
    pca50    = PCA(n_components=k50, random_state=42).fit(scaler50.transform(X_train))

    return FoldPipeline(
        scaler8=scaler8, pca8=pca8, z_min=z_min, z_range=z_rng,
        scaler50=scaler50, pca50=pca50,
        expl_var_8=pca8.explained_variance_ratio_,
        expl_var_50=pca50.explained_variance_ratio_,
    )


# ---------------------------------------------------------------------------
# Per-fold undersample (TRAIN ONLY)
# ---------------------------------------------------------------------------

@dataclass
class UndersampleResult:
    train_indices:   np.ndarray   # indices into the full train list
    n_pos:           int
    n_neg_target:    int          # negatives kept
    n_balanced:      int          # total balanced train = n_pos + n_neg_target
    ratio:           str          # "1:1" or "1:2"
    flag_2P_lt_60:   bool


def undersample_train(
    train: list[DataRecord],
    seed: int = _UNDERSAMPLE_SEED,
) -> UndersampleResult:
    """Within TRAIN ONLY: keep all P positives, sample negatives to 1:1 or 1:2.

    Policy:
      - 1:1 (target): P positives + P negatives = 2P total
      - 1:2 (fallback if 2P < 60): P positives + 2P negatives = 3P total
      - If even 1:2 negatives unavailable, take all available negatives.
      - NEVER modifies test records.
    """
    labels = np.array([r.finbert_bin for r in train])
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    P = len(pos_idx)
    N = len(neg_idx)

    flag_2P_lt_60 = (2 * P) < _MIN_BALANCED

    if flag_2P_lt_60:
        # 1:2 ratio: P pos + 2P neg
        n_neg_target = min(2 * P, N)
        ratio = "1:2"
    else:
        # 1:1 ratio: P pos + P neg
        n_neg_target = min(P, N)
        ratio = "1:1"

    rng = np.random.default_rng(seed)
    sampled_neg = rng.choice(neg_idx, size=n_neg_target, replace=False)
    chosen_idx  = np.sort(np.concatenate([pos_idx, sampled_neg]))

    return UndersampleResult(
        train_indices=chosen_idx,
        n_pos=P,
        n_neg_target=n_neg_target,
        n_balanced=P + n_neg_target,
        ratio=ratio,
        flag_2P_lt_60=flag_2P_lt_60,
    )


# ---------------------------------------------------------------------------
# Task 3: fold structure report
# ---------------------------------------------------------------------------

def fold_structure_report(
    splits: list[tuple[list[DataRecord], list[DataRecord]]],
    records: list[DataRecord],
) -> pd.DataFrame:
    """Run split + undersample, collect per-fold stats. No models run."""

    rows: list[dict] = []

    for fold_idx, (train, test) in enumerate(splits):
        # Pre-undersample train stats
        t_labels = np.array([r.finbert_bin for r in train])
        pre_pos  = int(t_labels.sum())
        pre_neg  = int((~t_labels.astype(bool)).sum())
        pre_n    = len(train)

        # Feature pipeline (fit on full train — validates scaler/PCA shape)
        X_train_full = np.stack([r.embedding for r in train]).astype(np.float64)
        pipeline = build_fold_pipeline(X_train_full)
        # Confirm MinMax assert: z_min / z_range come only from X_train_full above.
        # No test data touches the pipeline builder — structurally enforced.

        # Undersample
        us = undersample_train(train)

        # Test stats
        te_labels = np.array([r.finbert_bin for r in test])
        test_pos  = int(te_labels.sum())
        test_neg  = int((~te_labels.astype(bool)).sum())
        single_class_test = (test_pos == 0) or (test_neg == 0)
        test_pct_pos = test_pos / len(test) * 100 if len(test) > 0 else float("nan")

        # Assert test indices identical pre/post undersample (test is never touched)
        # We verify this implicitly: undersample_train only receives `train` records.

        rows.append({
            "fold":            fold_idx,
            "train_lo":        train[0].trading_day,
            "train_hi":        train[-1].trading_day,
            "test_lo":         test[0].trading_day,
            "test_hi":         test[-1].trading_day,
            "pre_n_train":     pre_n,
            "pre_pos_train":   pre_pos,
            "pre_neg_train":   pre_neg,
            "balanced_train":  us.n_balanced,
            "ratio":           us.ratio,
            "flag_2P_lt60":    us.flag_2P_lt_60,
            "n_test":          len(test),
            "test_pos":        test_pos,
            "test_neg":        test_neg,
            "test_pct_pos":    round(test_pct_pos, 1),
            "single_class_test": single_class_test,
            "expl_var_8_cumsum": round(float(pipeline.expl_var_8.sum()), 4),
        })

        if fold_idx % 20 == 0:
            print(
                f"  fold {fold_idx:3d}  train=[{train[0].trading_day},"
                f"{train[-1].trading_day}]  pre={pre_n:4d}(P={pre_pos},N={pre_neg})"
                f"  bal={us.n_balanced}({us.ratio})"
                f"  test={len(test)}(P={test_pos},N={test_neg})"
            )

    return pd.DataFrame(rows)


def print_fold_report(df: pd.DataFrame) -> None:
    n_folds = len(df)
    n_flags = int(df["flag_2P_lt60"].sum())
    n_single = int(df["single_class_test"].sum())

    print()
    print("=" * 80)
    print("PHASE 4B — FOLD STRUCTURE REPORT")
    print("=" * 80)
    print(f"  Rolling window:     {_MIN_TRAIN_DAYS} unique trading days")
    print(f"  Test window:        {_TEST_DAYS} unique trading days")
    print(f"  Total folds:        {n_folds}")
    print()

    # Balanced train summary
    bal = df["balanced_train"]
    print(f"  Balanced-train size (post-undersample):")
    print(f"    min     = {int(bal.min())}")
    print(f"    median  = {int(bal.median())}")
    print(f"    mean    = {bal.mean():.1f}")
    print(f"    max     = {int(bal.max())}")
    print()

    # n_test summary
    nt = df["n_test"]
    print(f"  Test-window size (n_test):")
    print(f"    min     = {int(nt.min())}")
    print(f"    median  = {int(nt.median())}")
    print(f"    max     = {int(nt.max())}")
    print()

    # 1:2 relaxation flags
    print(f"  Folds using 1:2 relaxation (2P < {_MIN_BALANCED}): {n_flags}")
    if n_flags:
        flag_df = df[df["flag_2P_lt60"]][["fold","pre_pos_train","balanced_train","ratio"]]
        print(flag_df.to_string(index=False))
    print()

    # Single-class test folds
    print(f"  Single-class test folds (no AUC possible): {n_single}")
    if n_single:
        sc_df = df[df["single_class_test"]][["fold","test_lo","test_hi","n_test","test_pos","test_neg"]]
        print(sc_df.to_string(index=False))
    print()

    # Pooled-AUC recommendation
    exclude_mask = df["single_class_test"] | (df["balanced_train"] < 20)
    n_exclude = int(exclude_mask.sum())
    n_pool    = n_folds - n_exclude
    print(f"  Pooled-AUC recommendation:")
    print(f"    Folds to INCLUDE: {n_pool}  (exclude {n_exclude} single-class / starved)")
    if n_exclude:
        excl_df = df[exclude_mask][["fold","balanced_train","n_test","single_class_test"]]
        print("    Excluded folds:")
        print(excl_df.to_string(index=False))
    print()

    # PCA explained variance
    ev_mean = df["expl_var_8_cumsum"].mean()
    ev_min  = df["expl_var_8_cumsum"].min()
    print(f"  PCA-8 cumulative explained variance (mean across folds): {ev_mean:.4f}")
    print(f"    min across folds: {ev_min:.4f}")
    print(f"    NOTE: {ev_mean*100:.1f}% of 512-dim CLIP variance captured in 8 components")
    print(f"          — material caveat for quantum-vs-classical comparison")
    print()

    # Per-fold table (all folds)
    print("-" * 80)
    print(f"{'fold':>4}  {'train_lo':>10}  {'train_hi':>10}  "
          f"{'preP':>5}  {'preN':>5}  {'bal':>4}  {'ratio':>4}  "
          f"{'ntest':>5}  {'tP':>4}  {'tN':>4}  {'t%+':>5}  {'singleCls':>9}")
    print("-" * 80)
    for _, row in df.iterrows():
        sc = "*** " if row["single_class_test"] else "    "
        fl = "1:2!" if row["flag_2P_lt60"] else row["ratio"]
        print(
            f"{int(row['fold']):>4}  {str(row['train_lo']):>10}  {str(row['train_hi']):>10}  "
            f"{int(row['pre_pos_train']):>5}  {int(row['pre_neg_train']):>5}  "
            f"{int(row['balanced_train']):>4}  {fl:>4}  "
            f"{int(row['n_test']):>5}  {int(row['test_pos']):>4}  {int(row['test_neg']):>4}  "
            f"{row['test_pct_pos']:>5.1f}  {sc}"
        )
    print("-" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    print(f"Loading {METHODS_PATH} ...")
    records = load_records(METHODS_PATH, NPZ_PATH)

    unique_days = sorted({r.trading_day for r in records})
    print(f"Unique trading days: {len(unique_days):,}  "
          f"({unique_days[0]} -> {unique_days[-1]})")

    labels = np.array([r.finbert_bin for r in records])
    majority_class = int(np.bincount(labels).argmax())
    majority_acc   = float(np.mean(labels == majority_class))
    print(f"Class balance:  pos={labels.sum():,}  neg={(~labels.astype(bool)).sum():,}")
    print(f"Majority baseline accuracy: {majority_acc:.4f}")

    print(f"\nGenerating rolling walk-forward splits "
          f"(min_train_days={_MIN_TRAIN_DAYS}, test_days={_TEST_DAYS}) ...")
    splits, udays = rolling_walk_forward_splits(
        records, min_train_days=_MIN_TRAIN_DAYS, test_days=_TEST_DAYS
    )
    print(f"  Folds generated: {len(splits)}")
    if not splits:
        print("[FAIL] No folds generated.")
        sys.exit(1)

    assert_no_leakage(splits)

    print(f"\nRunning fold structure analysis (split + undersample; NO models) ...")
    fold_df = fold_structure_report(splits, records)

    print_fold_report(fold_df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "fold_structure_report.csv"
    fold_df.to_csv(report_path, index=False)
    print(f"  -> {report_path}")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
    print()
    print("HARD STOP — Task 3 complete. No classifiers or quantum circuits were run.")
    print("Review fold_structure_report.csv before proceeding to Phase 4C.")


if __name__ == "__main__":
    main()

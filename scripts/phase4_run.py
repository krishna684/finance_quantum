#!/usr/bin/env python
"""Phase 4C — Four-arm comparison and concentration sweep (methods leg).

Arms
----
  classical_rbf_pca8   SVC(rbf, C=1.0) on raw PCA-8             FAIR arm
  qkernel_pca8         QuantumKernel + SVC(precomputed) on [0,π] PCA-8
  vqc_pca8             VQC on [0,π] PCA-8                        feature_map_reps=2
  classical_rbf_pca50  SVC(rbf) on PCA-50                        upper reference

Arms 1/2/3 share the SAME underlying PCA-8 (scaler+PCA fitted per fold on full
pre-undersample train). Arms 2/3 additionally apply MinMax→[0,π] to the PCA-8
output. Identity is asserted per fold before any arm runs.

Concentration sweep
-------------------
feature_map_reps ∈ {1,2,3,4} at fixed n_qubits=8.
reps=1 qkernel result IS the four-arm table's quantum-kernel entry.

QuantumSettings (explicit, per spec)
-------------------------------------
n_qubits=8, max_train_samples=400, vqc_layers=3, seed=42, shots=None.
shots=None is asserted before every train_matrix call (statevector path).

VQC note
--------
VQC's internal EmbeddingReducer is replaced with PassThroughReducer because
the shared pipeline already provides [0,π]-reduced PCA-8 features. The VQC
circuit receives these directly. feature_map_reps=2 (stated in report).

Undersample
-----------
Balanced TRAIN only (all P positives + P negatives, random_state=42).
Test is evaluated on the NATURAL (non-undersampled) distribution.

Usage
-----
    python scripts/phase4_run.py

Output
------
    results/phase4/metrics_four_arm.csv
    results/phase4/metrics_reps_sweep.csv
    results/phase4/predictions_{arm}.csv
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Ensure project root and scripts/ are on sys.path
# ---------------------------------------------------------------------------
_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT_SYSPATH = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))          # for phase4_harness
sys.path.insert(0, str(_PROJECT_ROOT_SYSPATH)) # for pipeline.*

from phase4_harness import (
    DataRecord,
    load_records,
    rolling_walk_forward_splits,
    assert_no_leakage,
    build_fold_pipeline,
    undersample_train,
    _MIN_TRAIN_DAYS,
    _TEST_DAYS,
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
METHODS_PATH  = _PROJECT_ROOT / "data" / "methods_label.parquet"
NPZ_PATH      = _PROJECT_ROOT / "data" / "embeddings" / "n24news_finance.npz"
RESULTS_DIR   = _PROJECT_ROOT / "results" / "phase4"

# ---------------------------------------------------------------------------
# Sweep parameters and fixed settings
# ---------------------------------------------------------------------------
_REPS_SWEEP     = [1, 2, 3, 4]
_VQC_REPS       = 2          # fixed for VQC arm; stated in report
_LEAN_POS_FOLDS = {40, 48, 52}   # flagged in 4B (test_pos ≤ 6)

_QS_BASE = dict(
    n_qubits=8,
    max_train_samples=400,
    vqc_layers=3,
    seed=42,
    shots=None,
)


# ---------------------------------------------------------------------------
# PassThroughReducer: drops EmbeddingReducer for VQC when features are
# already [0,π]-reduced by the shared pipeline.
# ---------------------------------------------------------------------------

class PassThroughReducer:
    """Identity transform replacing EmbeddingReducer inside VQC.

    The VQC's internal reducer calls fit_transform(X) on raw embeddings.
    Here X is already the [0,π]-rescaled PCA-8 output from the shared
    fold pipeline, so we return it unchanged.
    """
    n_components: int = 8
    fitted: bool = True

    def fit(self, X: np.ndarray) -> "PassThroughReducer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)


# ---------------------------------------------------------------------------
# Concentration metrics
# ---------------------------------------------------------------------------

def concentration_metrics(K: np.ndarray) -> dict:
    """var(offdiag) and mean(offdiag) from the upper triangle of K."""
    n = K.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    offdiag = K[rows, cols]
    return {
        "var_offdiag":  float(np.var(offdiag)),
        "mean_offdiag": float(np.mean(offdiag)),
    }


# ---------------------------------------------------------------------------
# Arm runners
# ---------------------------------------------------------------------------

def run_classical(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray,
) -> tuple[np.ndarray, float]:
    """SVC(rbf, C=1.0, gamma='scale') on pre-transformed features."""
    t0  = time.time()
    svc = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    svc.fit(X_train, y_train)
    y_prob = svc.predict_proba(X_test)[:, 1]
    return y_prob, time.time() - t0


def run_qkernel(
    X_train_angle: np.ndarray, y_train: np.ndarray,
    X_test_angle:  np.ndarray, reps: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """QuantumKernel (angle map, statevector) + SVC(precomputed).

    Returns (y_prob, K_train, elapsed_seconds).
    Asserts shots=None before train_matrix.
    Asserts training size <= max_train_samples.
    """
    from pipeline.quantum.config import QuantumSettings
    from pipeline.quantum.kernel import QuantumKernel

    qs = QuantumSettings(**{**_QS_BASE, "feature_map_reps": reps})
    qk = QuantumKernel(
        n_qubits=qs.n_qubits,
        feature_map=qs.feature_map,
        reps=reps,
        shots=qs.shots,
        seed=qs.seed,
    )

    # Fast-path assertion (statevector path engaged when shots=None)
    assert qk.shots is None, (
        f"[FAIL] qkernel reps={reps}: expected shots=None (statevector), "
        f"got shots={qk.shots}"
    )

    n_train = X_train_angle.shape[0]
    assert n_train <= qs.max_train_samples, (
        f"[FAIL] qkernel reps={reps}: training size {n_train} > "
        f"max_train_samples {qs.max_train_samples}"
    )

    t0      = time.time()
    K_train = qk.train_matrix(X_train_angle)
    svc     = SVC(kernel="precomputed", C=qs.svm_C, probability=True, random_state=qs.seed)
    svc.fit(K_train, y_train)
    K_test  = qk.cross_matrix(X_test_angle, X_train_angle)
    y_prob  = svc.predict_proba(K_test)[:, 1]

    return y_prob, K_train, time.time() - t0


def run_vqc(
    X_train_angle: np.ndarray, y_train: np.ndarray,
    X_test_angle:  np.ndarray,
    settings,
) -> tuple[np.ndarray, float]:
    """VQC with PassThroughReducer (features already [0,π]-reduced)."""
    from pipeline.quantum.vqc import VariationalQuantumClassifier

    vqc         = VariationalQuantumClassifier(settings)
    vqc.reducer = PassThroughReducer()   # bypass internal EmbeddingReducer

    t0     = time.time()
    vqc.fit(X_train_angle, y_train)
    y_prob = vqc.predict_proba(X_test_angle)[:, 1]
    return y_prob, time.time() - t0


# ---------------------------------------------------------------------------
# AUC helpers
# ---------------------------------------------------------------------------

def safe_auc(y_true: np.ndarray, y_prob: np.ndarray, tag: str, fold: int) -> float:
    """Compute AUC. FAIL LOUDLY on NaN or exception."""
    if len(np.unique(y_true)) < 2:
        print(f"[FAIL] Fold {fold} {tag}: single class in test "
              f"(classes: {np.unique(y_true)})")
        sys.exit(1)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception as exc:
        print(f"[FAIL] Fold {fold} {tag}: AUC error: {exc}")
        sys.exit(1)
    if np.isnan(auc):
        print(f"[FAIL] Fold {fold} {tag}: NaN AUC")
        sys.exit(1)
    return auc


def pooled_auc(accum: dict) -> float:
    y_true = np.concatenate(accum["y_true"])
    y_prob = np.concatenate(accum["y_prob"])
    return float(roc_auc_score(y_true, y_prob))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_wall = time.time()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"Loading {METHODS_PATH} ...")
    records = load_records(METHODS_PATH, NPZ_PATH)
    udays   = sorted({r.trading_day for r in records})
    print(f"Unique trading days: {len(udays):,}  ({udays[0]} -> {udays[-1]})")

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------
    print(f"\nGenerating rolling splits ...")
    splits, _ = rolling_walk_forward_splits(records)
    print(f"  Folds: {len(splits)}")
    assert_no_leakage(splits)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    from pipeline.quantum.config import QuantumSettings
    vqc_settings = QuantumSettings(**{**_QS_BASE, "feature_map_reps": _VQC_REPS})

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    def make_accum():
        return {"y_true": [], "y_prob": [], "aucs": [], "times": []}

    four_arm = {
        "classical_rbf_pca8":  make_accum(),
        "classical_rbf_pca50": make_accum(),
        "vqc_pca8":            make_accum(),
    }
    sweep = {
        reps: {**make_accum(), "var_offdiag": [], "mean_offdiag": []}
        for reps in _REPS_SWEEP
    }
    pca8_expl: list[float] = []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting four-arm + sweep across {len(splits)} folds ...")
    print(f"  Quantum kernel sweep: feature_map_reps={_REPS_SWEEP}")
    print(f"  VQC feature_map_reps (fixed): {_VQC_REPS}")
    print(f"  max_train_samples (asserted): {_QS_BASE['max_train_samples']}")
    print(f"  shots=None asserted per fold (statevector fast path)")
    print()

    # ------------------------------------------------------------------
    # Checkpoint file (written incrementally — survives crashes)
    # ------------------------------------------------------------------
    ckpt_path = RESULTS_DIR / "checkpoint_per_fold.csv"
    _ckpt_headers = (
        ["fold", "n_balanced", "auc_c8", "auc_c50", "auc_vqc"]
        + [f"auc_qk_r{r}" for r in _REPS_SWEEP]
        + [f"var_offdiag_r{r}" for r in _REPS_SWEEP]
        + [f"mean_offdiag_r{r}" for r in _REPS_SWEEP]
        + ["fold_time_s"]
    )
    with open(ckpt_path, "w", newline="", encoding="utf-8") as _f:
        csv.writer(_f).writerow(_ckpt_headers)

    # ------------------------------------------------------------------
    # Fold loop
    # ------------------------------------------------------------------
    for fold_idx, (train_recs, test_recs) in enumerate(splits):
        t_fold = time.time()

        # ---- Embeddings --------------------------------------------------
        X_train_full = np.stack([r.embedding for r in train_recs]).astype(np.float64)
        y_train_full = np.array([r.finbert_bin for r in train_recs])
        X_test_full  = np.stack([r.embedding for r in test_recs]).astype(np.float64)
        y_test       = np.array([r.finbert_bin for r in test_recs])

        # ---- Pipeline (fit on FULL pre-undersample train) ----------------
        pipeline = build_fold_pipeline(X_train_full)
        pca8_expl.append(float(pipeline.expl_var_8.sum()))

        # ---- Undersample (TRAIN ONLY) ------------------------------------
        us      = undersample_train(train_recs)
        bal_idx = us.train_indices
        X_train_bal = X_train_full[bal_idx]
        y_train_bal = y_train_full[bal_idx]

        # ---- Arm-specific features (from balanced train) -----------------
        X_train_pca8  = pipeline.transform_classical8(X_train_bal)
        X_test_pca8   = pipeline.transform_classical8(X_test_full)
        X_train_pca50 = pipeline.transform_classical50(X_train_bal)
        X_test_pca50  = pipeline.transform_classical50(X_test_full)
        X_train_angle = pipeline.transform_quantum(X_train_bal)
        X_test_angle  = pipeline.transform_quantum(X_test_full)

        # ---- Assert PCA-8 identity across arms ---------------------------
        # All of Arms 1/2/3 derive from the same scaler8+pca8.
        # Divergence happens ONLY at the [0,π] MinMax step for Arms 2/3.
        Z_check = pipeline.pca8.transform(pipeline.scaler8.transform(X_train_bal))
        if not np.allclose(X_train_pca8, Z_check, atol=1e-10):
            print(f"[FAIL] Fold {fold_idx}: PCA-8 identity check FAILED.")
            sys.exit(1)
        Z01         = (X_train_pca8 - pipeline.z_min) / pipeline.z_range
        angle_check = np.clip(Z01, 0.0, 1.0) * np.pi
        if not np.allclose(X_train_angle, angle_check, atol=1e-10):
            print(f"[FAIL] Fold {fold_idx}: angle-rescale identity check FAILED.")
            sys.exit(1)

        # ---- Arm 1: Classical RBF PCA-8 ----------------------------------
        yp_c8, t_c8   = run_classical(X_train_pca8, y_train_bal, X_test_pca8)
        auc_c8        = safe_auc(y_test, yp_c8,  "classical_rbf_pca8",  fold_idx)

        # ---- Arm 4: Classical RBF PCA-50 ---------------------------------
        yp_c50, t_c50 = run_classical(X_train_pca50, y_train_bal, X_test_pca50)
        auc_c50       = safe_auc(y_test, yp_c50, "classical_rbf_pca50", fold_idx)

        # ---- Arm 3: VQC --------------------------------------------------
        yp_vqc, t_vqc = run_vqc(X_train_angle, y_train_bal, X_test_angle, vqc_settings)
        auc_vqc       = safe_auc(y_test, yp_vqc, "vqc_pca8",            fold_idx)

        # ---- Sweep: QKernel at reps=1,2,3,4 (reps=1 IS the four-arm QK) -
        for reps in _REPS_SWEEP:
            yp_qk, K_train, t_qk = run_qkernel(
                X_train_angle, y_train_bal, X_test_angle, reps
            )
            auc_qk = safe_auc(y_test, yp_qk, f"qkernel_reps{reps}", fold_idx)
            cm     = concentration_metrics(K_train)
            sweep[reps]["y_true"].append(y_test)
            sweep[reps]["y_prob"].append(yp_qk)
            sweep[reps]["aucs"].append(auc_qk)
            sweep[reps]["var_offdiag"].append(cm["var_offdiag"])
            sweep[reps]["mean_offdiag"].append(cm["mean_offdiag"])
            sweep[reps]["times"].append(t_qk)

        # ---- Accumulate four-arm results ---------------------------------
        for arm, yp, auc, t_val in [
            ("classical_rbf_pca8",  yp_c8,  auc_c8,  t_c8),
            ("classical_rbf_pca50", yp_c50, auc_c50, t_c50),
            ("vqc_pca8",            yp_vqc, auc_vqc, t_vqc),
        ]:
            four_arm[arm]["y_true"].append(y_test)
            four_arm[arm]["y_prob"].append(yp)
            four_arm[arm]["aucs"].append(auc)
            four_arm[arm]["times"].append(t_val)

        # ---- Progress line -----------------------------------------------
        t_elapsed = time.time() - t_fold
        auc_qk1   = sweep[1]["aucs"][-1]
        if fold_idx % 10 == 0 or fold_idx == len(splits) - 1:
            print(
                f"  fold {fold_idx:3d}/{len(splits)}  "
                f"bal={us.n_balanced}  "
                f"c8={auc_c8:.3f}  qk(r=1)={auc_qk1:.3f}  "
                f"vqc={auc_vqc:.3f}  c50={auc_c50:.3f}  "
                f"({t_elapsed:.1f}s)"
            )

        # ---- Checkpoint (append one row; flush to disk) ------------------
        ckpt_row = (
            [fold_idx, us.n_balanced, auc_c8, auc_c50, auc_vqc]
            + [sweep[r]["aucs"][-1]        for r in _REPS_SWEEP]
            + [sweep[r]["var_offdiag"][-1] for r in _REPS_SWEEP]
            + [sweep[r]["mean_offdiag"][-1] for r in _REPS_SWEEP]
            + [t_elapsed]
        )
        with open(ckpt_path, "a", newline="", encoding="utf-8") as _f:
            csv.writer(_f).writerow(ckpt_row)

    # ------------------------------------------------------------------
    # Pool and report
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("PHASE 4C RESULTS")
    print("=" * 72)

    # Majority baseline
    all_y = np.concatenate(four_arm["classical_rbf_pca8"]["y_true"])
    maj_cls = int(np.bincount(all_y.astype(int)).argmax())
    maj_acc = float(np.mean(all_y == maj_cls))

    # ---- Four-arm table -------------------------------------------------
    print(f"\nFOUR-ARM TABLE  (qkernel at reps=1, VQC at reps={_VQC_REPS})")
    print(f"  Pooled AUC  = all 72 folds concatenated (rank-based, prevalence-robust)")
    print()
    # Build list of (label, y_true_list, y_prob_list, aucs_list, times_list)
    arm_rows = [
        ("classical_rbf_pca8",
         four_arm["classical_rbf_pca8"]["y_true"],
         four_arm["classical_rbf_pca8"]["y_prob"],
         four_arm["classical_rbf_pca8"]["aucs"],
         four_arm["classical_rbf_pca8"]["times"]),
        ("qkernel_pca8 (r=1)",
         sweep[1]["y_true"],
         sweep[1]["y_prob"],
         sweep[1]["aucs"],
         sweep[1]["times"]),
        (f"vqc_pca8 (r={_VQC_REPS})",
         four_arm["vqc_pca8"]["y_true"],
         four_arm["vqc_pca8"]["y_prob"],
         four_arm["vqc_pca8"]["aucs"],
         four_arm["vqc_pca8"]["times"]),
        ("classical_rbf_pca50",
         four_arm["classical_rbf_pca50"]["y_true"],
         four_arm["classical_rbf_pca50"]["y_prob"],
         four_arm["classical_rbf_pca50"]["aucs"],
         four_arm["classical_rbf_pca50"]["times"]),
    ]

    hdr = (f"  {'Arm':<26}  {'PooledAUC':>9}  {'MeanAUC':>8}  {'StdAUC':>7}"
           f"  {'MeanAUC-lean':>12}  {'StdAUC-lean':>11}  {'Runtime(s)':>10}")
    print(hdr)
    print("  " + "-" * 90)

    saved_arm_rows = []
    for label, yt_list, yp_list, aucs_list, times_list in arm_rows:
        p_auc     = float(roc_auc_score(np.concatenate(yt_list), np.concatenate(yp_list)))
        aucs_arr  = np.array(aucs_list)
        # Robustness: exclude lean-positive folds
        lean_mask = np.array([i in _LEAN_POS_FOLDS for i in range(len(aucs_arr))])
        aucs_nolean = aucs_arr[~lean_mask]
        rt        = sum(times_list)
        print(
            f"  {label:<26}  {p_auc:>9.4f}  {aucs_arr.mean():>8.4f}  "
            f"{aucs_arr.std():>7.4f}  {aucs_nolean.mean():>12.4f}  "
            f"{aucs_nolean.std():>11.4f}  {rt:>10.1f}"
        )
        saved_arm_rows.append({
            "arm": label,
            "pooled_auc":        round(p_auc, 6),
            "mean_auc_all":      round(float(aucs_arr.mean()), 6),
            "std_auc_all":       round(float(aucs_arr.std()),  6),
            "mean_auc_nolean":   round(float(aucs_nolean.mean()), 6),
            "std_auc_nolean":    round(float(aucs_nolean.std()),  6),
            "runtime_s":         round(rt, 2),
            "n_folds":           len(aucs_arr),
        })
    print(f"  {'Majority baseline':<26}  {'0.5000':>9}  {'—':>8}  {'—':>7}  {'—':>12}  {'—':>11}")
    print(f"  (maj_class={maj_cls}, baseline_acc={maj_acc:.4f})")

    # ---- Reps sweep table -----------------------------------------------
    print(f"\nCONCENTRATION SWEEP  (n_qubits=8, feature_map_reps in {_REPS_SWEEP})")
    print(f"  Expect: var(offdiag) -> 0 and pooled AUC -> 0.50 as reps rises")
    print()
    shdr = (f"  {'reps':>4}  {'PooledAUC':>9}  {'MeanAUC':>8}  {'StdAUC':>7}"
            f"  {'var(offdiag)':>12}  {'mean(offdiag)':>13}  {'Runtime(s)':>10}")
    print(shdr)
    print("  " + "-" * 70)

    saved_sweep_rows = []
    for reps in _REPS_SWEEP:
        sr       = sweep[reps]
        p_auc    = float(roc_auc_score(np.concatenate(sr["y_true"]), np.concatenate(sr["y_prob"])))
        aucs_arr = np.array(sr["aucs"])
        var_od   = float(np.mean(sr["var_offdiag"]))
        mean_od  = float(np.mean(sr["mean_offdiag"]))
        rt       = sum(sr["times"])
        print(
            f"  {reps:>4}  {p_auc:>9.4f}  {aucs_arr.mean():>8.4f}  {aucs_arr.std():>7.4f}"
            f"  {var_od:>12.6f}  {mean_od:>13.6f}  {rt:>10.1f}"
        )
        saved_sweep_rows.append({
            "reps":           reps,
            "pooled_auc":     round(p_auc, 6),
            "mean_auc":       round(float(aucs_arr.mean()), 6),
            "std_auc":        round(float(aucs_arr.std()),  6),
            "mean_var_offdiag":  round(var_od,  8),
            "mean_mean_offdiag": round(mean_od, 8),
            "runtime_s":      round(rt, 2),
        })

    # ---- PCA-8 explained variance ---------------------------------------
    ev_arr = np.array(pca8_expl)
    print(f"\nPCA-8 cumulative explained variance (across {len(ev_arr)} folds):")
    print(f"  mean={ev_arr.mean():.4f}  min={ev_arr.min():.4f}  max={ev_arr.max():.4f}")
    print(f"  => {ev_arr.mean()*100:.1f}% of 512-dim CLIP variance per fold")
    print( "  Framing: binding hardware constraint (8 qubits = 8 features);")
    print( "  both quantum and classical-RBF-pca8 arms see identical 21%.")
    print( "  PCA-50 arm quantifies compression cost separately.")

    # ---- Fast-path confirmation -----------------------------------------
    print(f"\nFast-path confirmation:")
    print(f"  shots=None asserted before every train_matrix call (all {len(splits)} folds,"
          f" {len(_REPS_SWEEP)} reps each = {len(splits)*len(_REPS_SWEEP)} assertions PASSED).")
    print(f"  Statevector fast path: O(n) state evals + |S@S*|^2 matmul (shots=None branch).")

    # ---- Integration fix log -------------------------------------------
    print(f"\nIntegration fixes made:")
    print(f"  1. QuantumKernelSVM bypassed: used QuantumKernel + SVC(precomputed) directly")
    print(f"     to enforce shared PCA-8 from the fold pipeline.")
    print(f"  2. VQC.reducer patched to PassThroughReducer: VQC's internal EmbeddingReducer")
    print(f"     would re-fit PCA on already-reduced features; PassThroughReducer returns")
    print(f"     them unchanged. VQC circuit and optimizer logic are untouched.")
    print(f"  3. PCA-8 identity asserted per fold (2 assertions: raw PCA-8 + angle rescale).")

    print(f"\nTotal runtime: {(time.time()-t_wall)/3600:.3f}h  "
          f"({time.time()-t_wall:.0f}s)")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    pd.DataFrame(saved_arm_rows).to_csv(RESULTS_DIR / "metrics_four_arm.csv", index=False)
    pd.DataFrame(saved_sweep_rows).to_csv(RESULTS_DIR / "metrics_reps_sweep.csv", index=False)

    # Per-fold metrics (all arms + all reps)
    fold_rows = []
    for fold_idx in range(len(splits)):
        row = {"fold": fold_idx}
        for arm_name in ("classical_rbf_pca8", "classical_rbf_pca50", "vqc_pca8"):
            row[f"auc_{arm_name}"] = four_arm[arm_name]["aucs"][fold_idx]
        for reps in _REPS_SWEEP:
            row[f"auc_qkernel_r{reps}"]  = sweep[reps]["aucs"][fold_idx]
            row[f"var_offdiag_r{reps}"]  = sweep[reps]["var_offdiag"][fold_idx]
            row[f"mean_offdiag_r{reps}"] = sweep[reps]["mean_offdiag"][fold_idx]
        fold_rows.append(row)
    pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "metrics_per_fold.csv", index=False)

    print(f"\nOutputs written to {RESULTS_DIR}/")
    print("STOP. Phase 4C complete. Awaiting review before any Phase 4D action.")


if __name__ == "__main__":
    main()

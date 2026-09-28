#!/usr/bin/env python
"""Phase D — Independent quantum re-run for verify2 audit.

Imports: phase4_harness (data/splits); pipeline.quantum.* (circuit execution).
NOT imported: phase4_run.py, classical_rerun.py, classical_verify2.py.

Off-diagonal kernel statistics (mean, variance) are reimplemented here with
pure numpy — no calls to any helper in pipeline/quantum/*.

Outputs (never overwrites originals):
    results/phase4/metrics_four_arm_verify2.csv
    results/phase4/metrics_reps_sweep_verify2.csv
    results/phase4/checkpoint_quantum_verify2.csv   (per-fold, written incrementally)

Usage: python scripts/quantum_verify2.py
Expected wall-clock: ~60 min (statevector simulator, 72 folds, 4 reps + VQC).
"""
from __future__ import annotations
import csv, sys, time
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

from pipeline.quantum.kernel import QuantumKernel
from pipeline.quantum.vqc    import VariationalQuantumClassifier
from pipeline.quantum.config import QuantumSettings

METHODS_PATH = _ROOT / "data" / "methods_label.parquet"
NPZ_PATH     = _ROOT / "data" / "embeddings" / "n24news_finance.npz"
OUT_DIR      = _ROOT / "results" / "phase4"

_REPS_SWEEP = [1, 2, 3, 4]
_VQC_REPS   = 2

# Reference values from existing paper files (never overwritten)
_REF_FOUR_ARM = {
    "classical_rbf_pca8":  0.585722,
    "qkernel_pca8 (r=1)":  0.575846,
    "vqc_pca8 (r=2)":      0.573847,
    "classical_rbf_pca50": 0.607719,
}
_REF_SWEEP_VAR  = {1: 0.03006228, 2: 0.01449074, 3: 0.00626683, 4: 0.00313312}
_REF_SWEEP_MEAN = {1: 0.14444648, 2: 0.08798104, 3: 0.03382237, 4: 0.01903112}
_TOL_AUC = 1e-3   # 1e-3 tolerance for quantum (statevector is deterministic but fold
                   # ordering/undersample seed must match exactly, so tight tolerance OK)
_TOL_KS  = 1e-4   # kernel stats tolerance


# ── Independent off-diagonal stats (pure numpy, no quantum module function) ──
def offdiag_stats(K: np.ndarray) -> tuple[float, float]:
    """Return (mean, variance) of off-diagonal elements. Pure numpy."""
    n = K.shape[0]
    mask  = ~np.eye(n, dtype=bool)
    off   = K[mask]
    return float(off.mean()), float(off.var())


# ── PassThroughReducer (reimplemented independently) ─────────────────────────
class _PassThrough:
    n_components: int = 8
    fitted: bool = True
    def fit(self, X): return self
    def transform(self, X): return np.asarray(X, dtype=np.float64)
    def fit_transform(self, X): return self.transform(X)


# ── QuantumSettings factory ───────────────────────────────────────────────────
_QS_BASE = dict(n_qubits=8, max_train_samples=400, vqc_layers=3, seed=42, shots=None)


def _make_qk(reps: int) -> QuantumKernel:
    qs = QuantumSettings(**{**_QS_BASE, "feature_map_reps": reps})
    qk = QuantumKernel(
        n_qubits=qs.n_qubits, feature_map=qs.feature_map,
        reps=reps, shots=qs.shots, seed=qs.seed,
    )
    assert qk.shots is None, f"shots must be None (statevector); got {qk.shots}"
    return qk


def run_qkernel(X_tr, y_tr, X_te, reps):
    qk     = _make_qk(reps)
    t0     = time.perf_counter()
    K_tr   = qk.train_matrix(X_tr)
    svc    = SVC(kernel="precomputed", C=1.0, probability=True, random_state=42)
    svc.fit(K_tr, y_tr)
    K_te   = qk.cross_matrix(X_te, X_tr)
    yp     = svc.predict_proba(K_te)[:, 1]
    return yp, K_tr, time.perf_counter() - t0


def run_vqc(X_tr, y_tr, X_te):
    qs      = QuantumSettings(**{**_QS_BASE, "feature_map_reps": _VQC_REPS})
    vqc     = VariationalQuantumClassifier(qs)
    vqc.reducer = _PassThrough()
    t0      = time.perf_counter()
    vqc.fit(X_tr, y_tr)
    yp      = vqc.predict_proba(X_te)[:, 1]
    return yp, time.perf_counter() - t0


def run_classical(X_tr, y_tr, X_te):
    t0  = time.perf_counter()
    svc = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    svc.fit(X_tr, y_tr)
    yp  = svc.predict_proba(X_te)[:, 1]
    return yp, time.perf_counter() - t0


def main() -> None:
    t_wall = time.perf_counter()
    print("=" * 68)
    print("QUANTUM VERIFY2 — independent re-run (statevector, zero QPU)")
    print("=" * 68)

    records = load_records(METHODS_PATH, NPZ_PATH)
    udays   = sorted({r.trading_day for r in records})
    print(f"Records: {len(records):,}  Trading days: {len(udays):,}  "
          f"({udays[0]} -> {udays[-1]})")

    splits, _ = rolling_walk_forward_splits(records)
    n_folds   = len(splits)
    print(f"Folds: {n_folds}  (expected 72)")
    if n_folds != 72:
        print(f"[FAIL] got {n_folds}")
        sys.exit(1)
    assert_no_leakage(splits)
    print("Leakage assertion: PASSED\n")

    # Accumulators
    def _acc(): return {"y_true": [], "y_prob": [], "times": []}
    sweep = {r: {**_acc(), "var": [], "mean": []} for r in _REPS_SWEEP}
    vqc_acc  = _acc()
    c8_acc   = _acc()
    c50_acc  = _acc()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / "checkpoint_quantum_verify2.csv"
    ckpt_hdrs = (
        ["fold", "n_balanced", "auc_c8", "auc_c50", "auc_vqc"]
        + [f"auc_qk_r{r}" for r in _REPS_SWEEP]
        + [f"var_r{r}" for r in _REPS_SWEEP]
        + [f"mean_r{r}" for r in _REPS_SWEEP]
        + ["fold_wall_s"]
    )
    with open(ckpt_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(ckpt_hdrs)

    for fi, (train_recs, test_recs) in enumerate(splits):
        t_fold = time.perf_counter()

        X_tr_full = np.stack([r.embedding for r in train_recs]).astype(np.float64)
        y_tr_full = np.array([r.finbert_bin for r in train_recs])
        X_te_full = np.stack([r.embedding for r in test_recs]).astype(np.float64)
        y_te      = np.array([r.finbert_bin for r in test_recs])

        pipe   = build_fold_pipeline(X_tr_full)
        us     = undersample_train(train_recs)
        bi     = us.train_indices
        X_tr_b = X_tr_full[bi];  y_tr_b = y_tr_full[bi]

        X_tr8,  X_te8  = pipe.transform_classical8(X_tr_b),  pipe.transform_classical8(X_te_full)
        X_tr50, X_te50 = pipe.transform_classical50(X_tr_b), pipe.transform_classical50(X_te_full)
        X_tr_a, X_te_a = pipe.transform_quantum(X_tr_b),     pipe.transform_quantum(X_te_full)

        # Classical arms (for four-arm table completeness)
        yp_c8,  t_c8  = run_classical(X_tr8,  y_tr_b, X_te8)
        yp_c50, t_c50 = run_classical(X_tr50, y_tr_b, X_te50)
        auc_c8  = float(roc_auc_score(y_te, yp_c8))
        auc_c50 = float(roc_auc_score(y_te, yp_c50))

        # VQC
        yp_vqc, t_vqc = run_vqc(X_tr_a, y_tr_b, X_te_a)
        auc_vqc = float(roc_auc_score(y_te, yp_vqc))

        # Quantum kernel sweep (reps=1,2,3,4) — reps=1 IS the four-arm QK entry
        sweep_aucs, sweep_vars, sweep_means = {}, {}, {}
        for reps in _REPS_SWEEP:
            yp_qk, K_tr, t_qk = run_qkernel(X_tr_a, y_tr_b, X_te_a, reps)
            auc_qk = float(roc_auc_score(y_te, yp_qk))
            m_od, v_od = offdiag_stats(K_tr)   # <-- pure numpy, no quantum module helper
            sweep[reps]["y_true"].append(y_te)
            sweep[reps]["y_prob"].append(yp_qk)
            sweep[reps]["times"].append(t_qk)
            sweep[reps]["var"].append(v_od)
            sweep[reps]["mean"].append(m_od)
            sweep_aucs[reps]  = auc_qk
            sweep_vars[reps]  = v_od
            sweep_means[reps] = m_od

        c8_acc["y_true"].append(y_te);   c8_acc["y_prob"].append(yp_c8)
        c50_acc["y_true"].append(y_te);  c50_acc["y_prob"].append(yp_c50)
        vqc_acc["y_true"].append(y_te);  vqc_acc["y_prob"].append(yp_vqc)

        fold_wall = time.perf_counter() - t_fold
        if fi % 10 == 0 or fi == n_folds - 1:
            elapsed = (time.perf_counter() - t_wall) / 60
            print(f"  fold {fi:3d}/{n_folds}  bal={us.n_balanced}  "
                  f"c8={auc_c8:.3f}  qk(r=1)={sweep_aucs[1]:.3f}  "
                  f"vqc={auc_vqc:.3f}  ({fold_wall:.1f}s fold)  "
                  f"[{elapsed:.1f} min elapsed]")

        # Incremental checkpoint
        row = (
            [fi, us.n_balanced, auc_c8, auc_c50, auc_vqc]
            + [sweep_aucs[r]  for r in _REPS_SWEEP]
            + [sweep_vars[r]  for r in _REPS_SWEEP]
            + [sweep_means[r] for r in _REPS_SWEEP]
            + [fold_wall]
        )
        with open(ckpt_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(row)

    print()

    # ── Pool results ─────────────────────────────────────────────────────────
    def pool_auc(acc): return float(roc_auc_score(
        np.concatenate(acc["y_true"]), np.concatenate(acc["y_prob"])))

    auc_c8_pool  = pool_auc(c8_acc)
    auc_c50_pool = pool_auc(c50_acc)
    auc_vqc_pool = pool_auc(vqc_acc)
    auc_qk1_pool = pool_auc({k: sweep[1][k] for k in ("y_true","y_prob")})
    t_vqc_total  = sum(sum(sweep[r]["times"]) for r in []) or \
                   sum(vqc_acc["times"]) if "times" in vqc_acc else float("nan")
    # VQC time tracked separately
    vqc_acc["times"] = vqc_acc.get("times", [])

    # ── Build four-arm output ─────────────────────────────────────────────────
    four_arm_rows = [
        {"arm": "classical_rbf_pca8",  "pooled_auc": round(auc_c8_pool, 6),  "n_folds": n_folds},
        {"arm": "qkernel_pca8 (r=1)",  "pooled_auc": round(auc_qk1_pool, 6), "n_folds": n_folds},
        {"arm": "vqc_pca8 (r=2)",      "pooled_auc": round(auc_vqc_pool, 6), "n_folds": n_folds},
        {"arm": "classical_rbf_pca50", "pooled_auc": round(auc_c50_pool, 6), "n_folds": n_folds},
    ]
    sweep_rows = []
    for reps in _REPS_SWEEP:
        sr = sweep[reps]
        sweep_rows.append({
            "reps":              reps,
            "pooled_auc":        round(pool_auc({k: sr[k] for k in ("y_true","y_prob")}), 6),
            "mean_var_offdiag":  round(float(np.mean(sr["var"])), 8),
            "mean_mean_offdiag": round(float(np.mean(sr["mean"])), 8),
            "runtime_s":         round(sum(sr["times"]), 2),
        })

    path_fa = OUT_DIR / "metrics_four_arm_verify2.csv"
    path_sw = OUT_DIR / "metrics_reps_sweep_verify2.csv"
    pd.DataFrame(four_arm_rows).to_csv(path_fa, index=False)
    pd.DataFrame(sweep_rows).to_csv(path_sw,   index=False)

    # ── Print and compare ─────────────────────────────────────────────────────
    print("=" * 68)
    print("FOUR-ARM COMPARISON vs REFERENCE:")
    all_pass = True
    for row in four_arm_rows:
        arm = row["arm"]
        v2  = row["pooled_auc"]
        ref = _REF_FOUR_ARM.get(arm, float("nan"))
        d   = abs(v2 - ref)
        ok  = d < _TOL_AUC
        all_pass = all_pass and ok
        print(f"  {arm:<26}  v2={v2:.6f}  ref={ref:.6f}  delta={d:.3e}  "
              f"[{'PASS' if ok else 'FAIL'}]")

    print()
    print("REPS SWEEP vs REFERENCE:")
    for row in sweep_rows:
        r   = row["reps"]
        v2v = row["mean_var_offdiag"]
        v2m = row["mean_mean_offdiag"]
        rv  = _REF_SWEEP_VAR[r]
        rm  = _REF_SWEEP_MEAN[r]
        dv  = abs(v2v - rv)
        dm  = abs(v2m - rm)
        okv = dv < _TOL_KS
        okm = dm < _TOL_KS
        all_pass = all_pass and okv and okm
        print(f"  reps={r}  var  v2={v2v:.8f}  ref={rv:.8f}  delta={dv:.3e}  "
              f"[{'PASS' if okv else 'FAIL'}]")
        print(f"        mean v2={v2m:.8f}  ref={rm:.8f}  delta={dm:.3e}  "
              f"[{'PASS' if okm else 'FAIL'}]")

    total_min = (time.perf_counter() - t_wall) / 60
    print(f"\nTotal wall-clock: {total_min:.1f} min")
    print(f"Outputs written: {path_fa}")
    print(f"              : {path_sw}")
    print(f"              : {ckpt_path}")

    if all_pass:
        print("\nVERDICT: QUANTUM ARM REPRODUCES within tolerance.")
    else:
        print("\n[FAIL] One or more values diverge — see FAIL flags above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

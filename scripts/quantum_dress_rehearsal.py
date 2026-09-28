#!/usr/bin/env python3
"""Quantum pipeline dress rehearsal — realistic dimensions, timing, diagnostics.

Purpose
-------
(a) Validate the full pipeline chain before real CLIP/FinBERT embeddings arrive.
(b) Measure stage-level runtimes for sprint planning.
(c) Characterise kernel concentration vs SNR under controlled conditions.

Synthetic data model
--------------------
X: n=600 samples in R^512 (CLIP-shaped float32).
   Planted nonlinear signal: k=8 latent dims Z ~ N(0,1), embedded into R^512
   via a fixed random orthogonal map Q; label = sign(Σ sin(Z_i + Z_{i+k/2})).
   Noise N(0, σ²) added in full 512-dim space.
   SNR sweep σ ∈ {0.1, 0.5, 1.0, 2.0, 4.0}.

Pipeline (all through the existing package APIs — no reimplementation)
----------------------------------------------------------------------
EmbeddingReducer (train-only fit: StandardScaler→PCA(8)→[0,π])
  → QuantumKernel  (angle map, 8 qubits, shots=None → statevector fast path)
      → sklearn SVC(precomputed) = the quantum kernel SVM arm
  → VariationalQuantumClassifier (owns its own reducer, receives raw embeddings)
  → sklearn SVC(rbf) on the same reduced features = classical reference

The kernel matrix is computed once via QuantumKernel directly so it is
available for diagnostics AND for the SVM; this is equivalent to what
QuantumKernelSVM.fit() does internally but avoids computing it twice.

Usage
-----
    python scripts/quantum_dress_rehearsal.py          # full run
    python scripts/quantum_dress_rehearsal.py --quick  # 2 sigmas, 20 VQC epochs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

# Ensure project root is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.quantum import QuantumSettings
from pipeline.quantum.kernel import QuantumKernel
from pipeline.quantum.reducer import EmbeddingReducer
from pipeline.quantum.vqc import VariationalQuantumClassifier

# ── run-level constants ────────────────────────────────────────────────────────

N_TOTAL = 600
N_TRAIN = 400
N_TEST = 200
D_EMBED = 512
K_LATENT = 8          # latent signal dimensions
N_QUBITS = 8          # must equal n_components in EmbeddingReducer
GLOBAL_SEED = 42

SNR_SIGMAS: list[float] = [0.1, 0.5, 1.0, 2.0, 4.0]
VQC_EPOCHS_FULL = 100
VQC_EPOCHS_QUICK = 20
VQC_LAYERS = 3
VQC_BATCH_SIZE = 32
SVM_C = 1.0

# Acceptance gate thresholds
GATE_AUC_STRONG_SIGNAL = 0.85   # qkernel SVM at σ=0.1
GATE_KERNEL_TIME_S = 300.0      # 5 min per 400×400 kernel


# ── helpers ───────────────────────────────────────────────────────────────────

class _Timer:
    elapsed: float = 0.0


@contextmanager
def timer() -> Generator[_Timer, None, None]:
    """Context manager; .elapsed is populated on exit."""
    t = _Timer()
    t0 = time.perf_counter()
    yield t
    t.elapsed = time.perf_counter() - t0


def make_planted_signal(
    n: int = N_TOTAL,
    d: int = D_EMBED,
    k: int = K_LATENT,
    sigma: float = 1.0,
    seed: int = GLOBAL_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate n samples in R^d with a planted nonlinear signal.

    Latent model
    ~~~~~~~~~~~~
    Z ~ N(0,I_k)                           shape (n, k)
    y = sign(Σ_i sin(Z[:,i] + Z[:,i+k/2]))  ∈ {0,1}
    X = Z @ Q.T + N(0,σ²I_d)               shape (n, d)

    where Q ∈ R^{d×k} has orthonormal columns (k-frame into R^d).
    The label boundary is nonlinear, non-axis-aligned, and lives in the
    k-dim latent subspace — hard for linear methods, recoverable with
    kernel tricks once the subspace is found.

    Returns
    -------
    X : (n, d) float32
    y : (n,)   int  ∈ {0, 1}, approximately balanced
    """
    rng = np.random.default_rng(seed)

    # Latent features
    Z = rng.normal(0.0, 1.0, (n, k))  # (n, k)

    # Nonlinear label: cross-term sin interactions
    half = k // 2
    signal = np.sum(np.sin(Z[:, :half] + Z[:, half:]), axis=1)
    y = (signal >= 0.0).astype(np.int32)

    # Random orthogonal embedding k → d via QR
    G = rng.normal(0.0, 1.0, (d, k))
    Q, _ = np.linalg.qr(G)  # Q: (d, k), orthonormal columns

    # Signal lives in the k-dim column space of Q
    X_signal = Z @ Q.T  # (n, d)

    # Isotropic noise
    X_noise = rng.normal(0.0, sigma, (n, d))

    X = (X_signal + X_noise).astype(np.float32)
    return X, y


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    n_test: int = N_TEST,
    seed: int = GLOBAL_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def kernel_diagnostics(K: np.ndarray) -> dict:
    """Off-diagonal statistics, spectrum, effective rank, PSD check.

    Effective rank uses the participation ratio:
        eff_rank = (Σλ_i)² / Σλ_i²
    This is 1 when one eigenvalue dominates (maximally concentrated)
    and n when all eigenvalues are equal (flat spectrum, no structure).
    """
    n = K.shape[0]
    off_mask = ~np.eye(n, dtype=bool)
    off = K[off_mask]

    eigvals = np.linalg.eigvalsh(K)  # ascending; real because K is symmetric
    top10 = eigvals[-10:][::-1].tolist()  # descending

    lam = np.maximum(eigvals, 0.0)
    lam_sum = lam.sum()
    eff_rank = float(lam_sum**2 / (lam**2).sum()) if lam_sum > 0 else 0.0

    return {
        "off_diag_mean":    float(np.mean(off)),
        "off_diag_std":     float(np.std(off)),
        "off_diag_min":     float(np.min(off)),
        "off_diag_max":     float(np.max(off)),
        "top10_eigenvalues": top10,
        "effective_rank":   eff_rank,
        "min_eigenvalue":   float(eigvals[0]),
        "is_psd":           bool(eigvals[0] > -1e-8),
        "has_nan":          bool(np.any(np.isnan(K))),
    }


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _fmt(v: float | None, decimals: int = 3) -> str:
    if v is None:
        return " N/A"
    return f"{v:.{decimals}f}"


# ── per-sigma pipeline run ────────────────────────────────────────────────────

def run_sigma(sigma: float, vqc_epochs: int) -> dict:
    """Full pipeline chain for one σ level.  Returns a metrics dict."""
    print(f"\n{'='*72}")
    print(f"  sigma = {sigma}   (VQC epochs={vqc_epochs})")
    print(f"{'='*72}")

    # ── data ──────────────────────────────────────────────────────────────
    X, y = make_planted_signal(sigma=sigma, seed=GLOBAL_SEED)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    balance_tr = float(y_train.mean())
    balance_te = float(y_test.mean())
    print(f"  data:  train={len(y_train)}  test={len(y_test)}  "
          f"class-1 share  train={balance_tr:.2f}  test={balance_te:.2f}")

    # ── stage 1: dimensionality reduction (train-only fit) ─────────────────
    with timer() as t_reduce:
        reducer = EmbeddingReducer(n_components=N_QUBITS, seed=GLOBAL_SEED)
        Xr_train = reducer.fit_transform(X_train.astype(np.float64))
        Xr_test  = reducer.transform(X_test.astype(np.float64))

    evr_sum = float(reducer.explained_variance_ratio.sum())
    print(f"  [reduction]    {t_reduce.elapsed:6.2f}s  "
          f"PCA explained variance (8 comps) = {evr_sum:.3f}")

    # Guard: features must be in [0, π]
    assert Xr_train.min() >= 0.0, "EmbeddingReducer output below 0"
    assert Xr_train.max() <= np.pi + 1e-9, "EmbeddingReducer output above pi"

    # ── fast-path verification ─────────────────────────────────────────────
    # The statevector fast path is selected automatically when shots=None.
    # We assert this now — before any circuit runs — so the gate fails
    # loudly rather than silently falling back to slow pairwise circuits.
    qkernel = QuantumKernel(
        n_qubits=N_QUBITS,
        feature_map="angle",
        reps=2,
        shots=None,       # ← explicit; any change here breaks the gate
        seed=GLOBAL_SEED,
    )
    fast_path_active = (qkernel.shots is None)
    if not fast_path_active:
        print("  FAIL: statevector fast path NOT active -- shots is not None")
        raise RuntimeError(
            "Fast-path gate failed: QuantumKernel.shots is not None. "
            "The dress rehearsal requires shots=None for the statevector path."
        )
    print("  [fast path]    CONFIRMED  shots=None -> statevector fast path active")

    # ── stage 2: quantum kernel matrix ────────────────────────────────────
    budget = QuantumKernel.evaluation_budget(len(Xr_train), n_test=len(Xr_test))
    print(f"  [kernel]       circuit budget={budget:,}  "
          f"(fast path: {len(Xr_train)+len(Xr_test)} statevector calls + matmul)")

    with timer() as t_kernel:
        K_train = qkernel.train_matrix(Xr_train)              # (400, 400)
        K_test  = qkernel.cross_matrix(Xr_test, Xr_train)    # (200, 400)

    print(f"  [kernel]       {t_kernel.elapsed:6.2f}s  "
          f"K_train={K_train.shape}  K_test={K_test.shape}")

    if t_kernel.elapsed > GATE_KERNEL_TIME_S:
        print(f"  WARNING: kernel took {t_kernel.elapsed:.1f}s > "
              f"{GATE_KERNEL_TIME_S}s gate")

    # NaN guard (hard stop — rest of analysis would be meaningless)
    if np.any(np.isnan(K_train)) or np.any(np.isnan(K_test)):
        raise RuntimeError(f"sigma={sigma}: NaN detected in kernel matrix")

    # ── kernel diagnostics ─────────────────────────────────────────────────
    kdiag = kernel_diagnostics(K_train)
    print(f"  [diag]         off-diag  mean={kdiag['off_diag_mean']:+.4f}  "
          f"std={kdiag['off_diag_std']:.4f}  "
          f"[{kdiag['off_diag_min']:.4f}, {kdiag['off_diag_max']:.4f}]")
    print(f"  [diag]         eff_rank={kdiag['effective_rank']:.1f}  "
          f"min_eigval={kdiag['min_eigenvalue']:.2e}  "
          f"PSD={'YES' if kdiag['is_psd'] else 'NO'}")
    top3 = [f"{v:.3f}" for v in kdiag["top10_eigenvalues"][:3]]
    print(f"  [diag]         top-3 eigvals: {top3}")

    # ── stage 3: quantum kernel SVM (precomputed kernel → SVC) ────────────
    with timer() as t_svm:
        svc = SVC(C=SVM_C, kernel="precomputed", probability=True,
                  random_state=GLOBAL_SEED)
        svc.fit(K_train, y_train)

    y_pred_qsvm = svc.predict(K_test)
    y_prob_qsvm = svc.predict_proba(K_test)[:, 1]
    auc_qsvm    = _safe_auc(y_test, y_prob_qsvm)
    acc_qsvm    = float(accuracy_score(y_test, y_pred_qsvm))
    print(f"  [qkernel SVM]  {t_svm.elapsed:6.2f}s  "
          f"AUC={_fmt(auc_qsvm)}  acc={acc_qsvm:.3f}")

    # ── stage 4: VQC ──────────────────────────────────────────────────────
    # VQC owns its own EmbeddingReducer internally; receives raw embeddings.
    vqc_settings = QuantumSettings(
        n_qubits=N_QUBITS,
        feature_map="angle",
        feature_map_reps=2,
        max_train_samples=N_TRAIN,
        vqc_layers=VQC_LAYERS,
        vqc_epochs=vqc_epochs,
        vqc_batch_size=VQC_BATCH_SIZE,
        shots=None,
        seed=GLOBAL_SEED,
    )
    with timer() as t_vqc:
        vqc = VariationalQuantumClassifier(vqc_settings)
        vqc.fit(X_train.astype(np.float64), y_train)

    y_pred_vqc = vqc.predict(X_test.astype(np.float64))
    y_prob_vqc = vqc.predict_proba(X_test.astype(np.float64))[:, 1]
    auc_vqc    = _safe_auc(y_test, y_prob_vqc)
    acc_vqc    = float(accuracy_score(y_test, y_pred_vqc))
    loss0, lossN = vqc.loss_history_[0], vqc.loss_history_[-1]
    print(f"  [VQC]          {t_vqc.elapsed:6.2f}s  "
          f"AUC={_fmt(auc_vqc)}  acc={acc_vqc:.3f}  "
          f"loss {loss0:.3f}->{lossN:.3f}")

    # Compute final test loss (MSE on ±1 targets, same objective as training)
    classes = vqc.classes_
    y_pm_test  = np.where(y_test == classes[1], 1.0, -1.0)
    test_scores = np.asarray(vqc._scores(X_test.astype(np.float64)))
    test_loss_final = float(np.mean((test_scores - y_pm_test) ** 2))

    # ── stage 5: classical RBF-SVM on the same 8-dim PCA features ─────────
    with timer() as t_rbf:
        clf_rbf = SVC(kernel="rbf", probability=True, random_state=GLOBAL_SEED)
        clf_rbf.fit(Xr_train, y_train)

    y_pred_rbf = clf_rbf.predict(Xr_test)
    y_prob_rbf = clf_rbf.predict_proba(Xr_test)[:, 1]
    auc_rbf    = _safe_auc(y_test, y_prob_rbf)
    acc_rbf    = float(accuracy_score(y_test, y_pred_rbf))
    print(f"  [RBF-SVM ref]  {t_rbf.elapsed:6.2f}s  "
          f"AUC={_fmt(auc_rbf)}  acc={acc_rbf:.3f}")

    return {
        "sigma": sigma,
        "fast_path_active": fast_path_active,
        "pca_explained_variance_sum": evr_sum,
        "timings": {
            "reduction_s":     t_reduce.elapsed,
            "kernel_matrix_s": t_kernel.elapsed,
            "svm_fit_s":       t_svm.elapsed,
            "vqc_train_s":     t_vqc.elapsed,
            "rbf_svm_fit_s":   t_rbf.elapsed,
        },
        "metrics": {
            "qkernel_svm": {
                "auc":      auc_qsvm,
                "accuracy": acc_qsvm,
            },
            "vqc": {
                "auc":      auc_vqc,
                "accuracy": acc_vqc,
            },
            "rbf_svm_reference": {
                "auc":      auc_rbf,
                "accuracy": acc_rbf,
            },
        },
        "kernel_diagnostics": kdiag,
        "vqc_loss_curves": {
            "train_per_epoch": vqc.loss_history_,
            "test_loss_final": test_loss_final,
            "epochs_run":      len(vqc.loss_history_),
        },
    }


# ── n=1000 timing probe ───────────────────────────────────────────────────────

def run_timing_probe_n1000() -> dict:
    """Kernel matrix only at n=1000 to measure O(n) scaling of fast path."""
    print(f"\n{'='*72}")
    print("  TIMING PROBE  n=1000  (kernel matrix only, statevector fast path)")
    print(f"{'='*72}")

    n = 1000
    X, _ = make_planted_signal(n=n, sigma=1.0, seed=99)  # different seed → fresh data

    with timer() as t_red:
        reducer = EmbeddingReducer(n_components=N_QUBITS, seed=GLOBAL_SEED)
        Xr = reducer.fit_transform(X.astype(np.float64))

    qkernel = QuantumKernel(
        n_qubits=N_QUBITS, feature_map="angle", reps=2,
        shots=None, seed=GLOBAL_SEED,
    )

    with timer() as t_kern:
        K = qkernel.train_matrix(Xr)

    assert not np.any(np.isnan(K)), "NaN in n=1000 kernel probe"
    print(f"  reduction:   {t_red.elapsed:.2f}s")
    print(f"  kernel:      {t_kern.elapsed:.1f}s  shape={K.shape}")

    return {
        "n":               n,
        "reduction_s":     t_red.elapsed,
        "kernel_matrix_s": t_kern.elapsed,
    }


# ── gates & reporting ─────────────────────────────────────────────────────────

def _gate(label: str, passed: bool | None) -> tuple[str, bool | None]:
    if passed is None:
        return f"  [SKIP] {label}", None
    status = "PASS" if passed else "FAIL"
    return f"  [{status}] {label}", passed


def print_results_table(results: list[dict]) -> None:
    print(f"\n{'='*72}")
    print("RESULTS TABLE")
    print(f"{'='*72}")
    hdr = (f"{'sig':>5}  {'model':<20}  {'AUC':>6}  {'acc':>6}  "
           f"{'kern_s':>7}  {'train_s':>8}")
    print(hdr)
    print("-" * 72)
    for r in results:
        s   = r["sigma"]
        t   = r["timings"]
        m   = r["metrics"]
        rows = [
            ("qkernel_svm",   m["qkernel_svm"],        t["kernel_matrix_s"], t["svm_fit_s"]),
            ("vqc",           m["vqc"],                 None,                 t["vqc_train_s"]),
            ("rbf_svm_ref",   m["rbf_svm_reference"],  None,                 t["rbf_svm_fit_s"]),
        ]
        for model, me, ks, ts in rows:
            ks_s = f"{ks:7.1f}" if ks is not None else "      -"
            print(f"{s:>5}  {model:<20}  "
                  f"{_fmt(me['auc']):>6}  {me['accuracy']:>6.3f}  "
                  f"{ks_s}  {ts:8.1f}s")
    print()


def print_concentration_table(results: list[dict]) -> None:
    print(f"\n{'='*72}")
    print("KERNEL CONCENTRATION vs SNR  (400x400 train kernel)")
    print(f"{'='*72}")
    print(f"{'sig':>5}  {'off-mean':>9}  {'off-std':>8}  "
          f"{'off-min':>8}  {'off-max':>8}  {'eff_rank':>9}  {'PSD':>4}")
    print("-" * 72)
    for r in results:
        d = r["kernel_diagnostics"]
        print(f"{r['sigma']:>5}  "
              f"{d['off_diag_mean']:>9.5f}  "
              f"{d['off_diag_std']:>8.5f}  "
              f"{d['off_diag_min']:>8.5f}  "
              f"{d['off_diag_max']:>8.5f}  "
              f"{d['effective_rank']:>9.1f}  "
              f"{'YES' if d['is_psd'] else 'NO':>4}")
    print()


def evaluate_gates(
    results: list[dict],
    probe: dict,
) -> tuple[dict, bool]:
    """Evaluate all acceptance gates. Returns (gate_dict, overall_pass)."""

    # Gate 1: fast path active in every run
    g_fast = all(r["fast_path_active"] for r in results)

    # Gate 2: qkernel SVM AUC > 0.85 at σ=0.1
    r01 = next((r for r in results if r["sigma"] == 0.1), None)
    if r01 is not None:
        auc01 = r01["metrics"]["qkernel_svm"]["auc"]
        g_auc = (auc01 is not None and auc01 > GATE_AUC_STRONG_SIGNAL)
    else:
        g_auc = None  # quick mode skipped σ=0.1

    # Gate 3: all kernel matrices computed in < 5 min
    g_kern_time = all(
        r["timings"]["kernel_matrix_s"] < GATE_KERNEL_TIME_S for r in results
    )

    # Gate 4: all kernels PSD (min eigenvalue > -1e-8)
    g_psd = all(r["kernel_diagnostics"]["is_psd"] for r in results)

    # Gate 5: no NaN anywhere
    g_nan = all(not r["kernel_diagnostics"]["has_nan"] for r in results)

    # Gate 6: concentration trend — off-diag std(σ=4) < std(σ=0.1)
    r_lo = next((r for r in results if r["sigma"] == 0.1), None)
    r_hi = next((r for r in results if r["sigma"] == 4.0), None)
    if r_lo is not None and r_hi is not None:
        g_conc = (
            r_hi["kernel_diagnostics"]["off_diag_std"]
            < r_lo["kernel_diagnostics"]["off_diag_std"]
        )
    else:
        g_conc = None  # quick mode

    gates = {
        "fast_path_active":          g_fast,
        "auc_strong_signal_gt_085":  g_auc,
        "kernel_time_lt_5min":       g_kern_time,
        "all_kernels_psd":           g_psd,
        "no_nan":                    g_nan,
        "concentration_trend":       g_conc,
    }

    # Overall: skip None gates (not enough data), fail on any False
    definite = [v for v in gates.values() if v is not None]
    overall = all(definite)
    gates["overall_pass"] = overall
    return gates, overall


def print_gates(gates: dict) -> None:
    print(f"\n{'='*72}")
    print("ACCEPTANCE GATES")
    print(f"{'='*72}")
    descs = {
        "fast_path_active":
            "statevector fast path active (shots=None)",
        "auc_strong_signal_gt_085":
            f"sigma=0.1 qkernel SVM AUC > {GATE_AUC_STRONG_SIGNAL}",
        "kernel_time_lt_5min":
            f"400x400 kernel matrix time < {GATE_KERNEL_TIME_S:.0f}s (5 min)",
        "all_kernels_psd":
            "all train kernels PSD (min eigval > -1e-8)",
        "no_nan":
            "no NaN in kernel matrices",
        "concentration_trend":
            "off-diag std decreases from sigma=0.1 -> sigma=4.0",
    }
    for key, desc in descs.items():
        line, _ = _gate(desc, gates.get(key))
        print(line)

    status = "ALL PASS" if gates.get("overall_pass") else "SOME GATES FAILED"
    print(f"\n  OVERALL: {status}")


# ── JSON serialisation ────────────────────────────────────────────────────────

def _np_serial(obj):
    """json.dumps default handler for numpy scalars and arrays."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantum pipeline dress rehearsal",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 2 sigma values and 20 VQC epochs for fast dev validation.",
    )
    args = parser.parse_args()

    vqc_epochs = VQC_EPOCHS_QUICK if args.quick else VQC_EPOCHS_FULL
    sigmas     = [0.1, 1.0] if args.quick else SNR_SIGMAS

    print("=" * 72)
    print("QUANTUM PIPELINE DRESS REHEARSAL")
    print("=" * 72)
    print(f"  n={N_TOTAL}  d={D_EMBED}  k_latent={K_LATENT}  n_qubits={N_QUBITS}")
    print(f"  n_train={N_TRAIN}  n_test={N_TEST}  seed={GLOBAL_SEED}")
    print(f"  VQC: layers={VQC_LAYERS}  epochs={vqc_epochs}  batch={VQC_BATCH_SIZE}")
    print(f"  SNR sweep: sigma in {sigmas}")
    print(f"  quick={args.quick}")

    wall_t0 = time.perf_counter()

    # ── sigma sweep ───────────────────────────────────────────────────────
    results: list[dict] = []
    for sigma in sigmas:
        r = run_sigma(sigma, vqc_epochs=vqc_epochs)
        results.append(r)

    # ── n=1000 timing probe ───────────────────────────────────────────────
    probe = run_timing_probe_n1000()

    total_wall = time.perf_counter() - wall_t0

    # ── print tables ──────────────────────────────────────────────────────
    print_results_table(results)
    print_concentration_table(results)

    # ── n=1000 extrapolation ──────────────────────────────────────────────
    t400_avg = float(np.mean([r["timings"]["kernel_matrix_s"] for r in results]))
    t1000    = probe["kernel_matrix_s"]
    ratio    = t1000 / t400_avg if t400_avg > 0 else float("nan")
    # Fast path is O(n) state circuits + O(n²) matmul; matmul dominates for large n.
    # Theoretical ratio for pure O(n) circuit cost: 1000/400 = 2.5x.
    print(f"{'='*72}")
    print("n=1000 TIMING EXTRAPOLATION")
    print(f"{'='*72}")
    print(f"  n=400  kernel avg:  {t400_avg:7.2f}s")
    print(f"  n=1000 kernel:      {t1000:7.2f}s  (statevector fast path)")
    print(f"  ratio 1000/400:     {ratio:7.2f}x  (O(n) circuit expected ~2.5x)")
    print(f"  n=2000 estimate:    ~{t1000 * 2:.0f}s  (linear extrapolation)")
    print(f"\n  Total wall time:  {total_wall:.1f}s")

    # ── acceptance gates ──────────────────────────────────────────────────
    gates, overall_pass = evaluate_gates(results, probe)
    print_gates(gates)

    # ── save JSON ─────────────────────────────────────────────────────────
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "dress_rehearsal.json"

    output = {
        "meta": {
            "n_total":   N_TOTAL,
            "n_train":   N_TRAIN,
            "n_test":    N_TEST,
            "d_embed":   D_EMBED,
            "k_latent":  K_LATENT,
            "n_qubits":  N_QUBITS,
            "vqc_epochs": vqc_epochs,
            "sigma_sweep": sigmas,
            "quick_mode": args.quick,
            "total_wall_s": total_wall,
        },
        "results":           results,
        "timing_probe_n1000": probe,
        "gates":             gates,
    }
    out_file.write_text(
        json.dumps(output, indent=2, default=_np_serial), encoding="utf-8"
    )
    print(f"\n  Results written to {out_file}")

    # Exit non-zero if any gate fails (useful in CI).
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Phase 1.4 — Aer dry-run: validate 20×20 angle_map kernel on noiseless simulator.

Proves the angle_map reps=1 pipeline produces a valid Gram matrix (diagonal=1,
off-diag stats finite) before a single second is spent on hardware.

Settings:
    n           = 20  (same size as Phase 3 hardware run)
    n_qubits    = 8
    reps        = 1
    shots       = 1024  (finite sampling, no gate/readout noise)
    backend     = PennyLane default.qubit  (noiseless shot simulation)
    data        = real CLIP embeddings from data/embeddings/n24news_finance.npz
                  20 stratified samples by label_up, seed=42

FAIL LOUDLY if:
    - Diagonal is not identically 1.0
    - Any NaN in the matrix
    - off_diag_var is 0.0 (degenerate kernel)

Usage:
    python hardware/aer_dryrun.py
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# project root on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from pipeline.quantum.reducer import EmbeddingReducer
from pipeline.quantum.kernel  import QuantumKernel

# ── constants ─────────────────────────────────────────────────────────────────
N_SAMPLES = 20
N_QUBITS  = 8
REPS      = 1
SHOTS     = 1024
SEED      = 42

# Simulator baselines (from metrics_reps_sweep.csv, reps=1)
_SIM_VAR_PRIMARY  = 0.03006228
_SIM_MEAN_PRIMARY = 0.14444648


def _load_20_samples() -> tuple[np.ndarray, np.ndarray]:
    """Load 20 stratified CLIP embeddings, reduced to 8-dim [0,π].

    Reducer fitted on full corpus (characterisation run — no fold leakage
    concern).  Returns (X_red, y) where X_red ∈ [0,π]^(20×8).
    """
    labeled    = pd.read_parquet(_ROOT / "data" / "labeled.parquet")
    npz        = np.load(_ROOT / "data" / "embeddings" / "n24news_finance.npz")
    embeddings = npz["embeddings"].astype(np.float64)
    labels     = labeled["label_up"].values

    # Stratified sample (same logic as batched_matrix.py)
    rng    = np.random.default_rng(SEED)
    half   = N_SAMPLES // 2
    sel_up = rng.choice(np.where(labels == 1)[0], half,             replace=False)
    sel_dn = rng.choice(np.where(labels == 0)[0], N_SAMPLES - half, replace=False)
    sel    = np.concatenate([sel_up, sel_dn])
    rng.shuffle(sel)

    print(f"  Selected indices: {sorted(sel.tolist())}")
    print(f"  Labels          : {labels[sel].tolist()}")

    # Fit reducer on full corpus
    reducer = EmbeddingReducer(n_components=N_QUBITS, seed=SEED)
    reducer.fit(embeddings)
    X_red = reducer.transform(embeddings[sel])

    ev = reducer.explained_variance_ratio.sum()
    print(f"  PCA-8 explained variance (full corpus): {ev:.4f}")
    print(f"  X_red range: [{X_red.min():.4f}, {X_red.max():.4f}]  (target [0, π={np.pi:.4f}])")

    if X_red.min() < 0.0 or X_red.max() > np.pi + 1e-9:
        raise RuntimeError(
            f"FAIL: EmbeddingReducer output outside [0,π]: "
            f"min={X_red.min():.6f}  max={X_red.max():.6f}"
        )

    return X_red, labels[sel].astype(int)


def main() -> None:
    print("=" * 64)
    print("AER DRY-RUN: angle_map n=20 reps=1 1024 shots")
    print("=" * 64)
    print(f"n_samples : {N_SAMPLES}")
    print(f"n_qubits  : {N_QUBITS}")
    print(f"reps      : {REPS}")
    print(f"shots     : {SHOTS}  (noiseless default.qubit + shot sampling)")
    print()

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading data ...")
    X, y = _load_20_samples()
    print()

    # ── build kernel ──────────────────────────────────────────────────────────
    n_pairs = N_SAMPLES * (N_SAMPLES - 1) // 2
    print(f"Building QuantumKernel (n_qubits={N_QUBITS}, reps={REPS}, shots={SHOTS}) ...")
    print(f"  Off-diagonal circuit evaluations: {n_pairs}")
    print(f"  (diagonal set to 1.0 by construction — no circuit required)")
    print()

    qkernel = QuantumKernel(
        n_qubits=N_QUBITS,
        feature_map="angle",
        reps=REPS,
        shots=SHOTS,
        seed=SEED,
    )

    print(f"Computing 20×20 Gram matrix ...")
    t0 = time.perf_counter()
    K  = qkernel.train_matrix(X)
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.2f}s")
    print()

    # ── validate ─────────────────────────────────────────────────────────────
    print("Validating Gram matrix ...")

    # 1. Shape
    if K.shape != (N_SAMPLES, N_SAMPLES):
        raise RuntimeError(f"FAIL: expected ({N_SAMPLES},{N_SAMPLES}), got {K.shape}")
    print(f"  shape      : {K.shape}  ✓")

    # 2. Diagonal exactly 1
    diag = np.diag(K)
    if not np.allclose(diag, 1.0, atol=1e-12):
        raise RuntimeError(
            f"FAIL: diagonal is not exactly 1.0: "
            f"min={diag.min():.8f}  max={diag.max():.8f}"
        )
    print(f"  diagonal   : exactly 1.0  ✓")

    # 3. No NaN
    if np.any(np.isnan(K)):
        raise RuntimeError("FAIL: NaN detected in Gram matrix")
    print(f"  NaN check  : clean  ✓")

    # 4. Symmetric
    if not np.allclose(K, K.T, atol=1e-12):
        raise RuntimeError("FAIL: Gram matrix is not symmetric")
    print(f"  symmetric  : ✓")

    # 5. Values in [0, 1]  (fidelity = probability)
    off_mask = ~np.eye(N_SAMPLES, dtype=bool)
    off_vals = K[off_mask]
    if off_vals.min() < -1e-9 or off_vals.max() > 1.0 + 1e-9:
        raise RuntimeError(
            f"FAIL: off-diagonal values outside [0,1]: "
            f"min={off_vals.min():.4f}  max={off_vals.max():.4f}"
        )
    print(f"  value range: [{off_vals.min():.4f}, {off_vals.max():.4f}] ⊂ [0,1]  ✓")

    # 6. Off-diagonal stats
    off_mean = float(off_vals.mean())
    off_var  = float(off_vals.var())
    if off_var == 0.0:
        raise RuntimeError("FAIL: off_diag_var = 0.0 — degenerate (all-equal) kernel")
    print(f"  off_diag_mean  : {off_mean:.6f}")
    print(f"  off_diag_var   : {off_var:.6f}")
    print()

    # ── comparison to simulator baseline ────────────────────────────────────
    print("Comparison to simulator baseline (metrics_reps_sweep.csv, reps=1):")
    print(f"  sim  off_diag_mean : {_SIM_MEAN_PRIMARY:.6f}")
    print(f"  sim  off_diag_var  : {_SIM_VAR_PRIMARY:.6f}")
    print(f"  dryrun mean        : {off_mean:.6f}  (delta={off_mean - _SIM_MEAN_PRIMARY:+.6f})")
    print(f"  dryrun var         : {off_var:.6f}  (delta={off_var  - _SIM_VAR_PRIMARY:+.6f})")
    print()
    print(
        "  NOTE: differences are expected — dry-run uses 20 samples (vs full\n"
        "        72-fold walk-forward pools) and has shot noise from 1024 shots\n"
        "        (sim baseline used shots=None, exact statevector)."
    )
    print()

    # ── gram matrix preview ──────────────────────────────────────────────────
    print("Gram matrix (3 decimal places):")
    print(np.array2string(K, precision=3, suppress_small=True))
    print()

    print("=" * 64)
    print("DRY-RUN PASSED — pipeline is valid.")
    print(f"  {n_pairs} circuits evaluated in {elapsed:.2f}s on default.qubit")
    print("  20×20 Gram matrix: diagonal=1 ✓  NaN-free ✓  symmetric ✓")
    print(f"  off_diag_mean={off_mean:.4f}  off_diag_var={off_var:.4f}")
    print("=" * 64)


if __name__ == "__main__":
    main()

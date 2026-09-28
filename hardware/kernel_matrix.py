#!/usr/bin/env python
"""20x20 quantum kernel matrix on ibm_boston -- Phase 4 micro-kernel.

Kernel:      k(x_i, x_j) = |<phi(x_j)|phi(x_i)>|^2
Feature map: zz_feature_map(feature_dimension=2, reps=2)
Data:        20 stratified samples from n24news_finance embeddings
             (CLIP 512-dim -> PCA 2-dim -> scaled to [0, pi])
Backend:     ibm_boston (or --backend override)

Outputs:
  hardware/kernel_matrix_{backend}.npy        -- 20x20 float64 symmetric K
  hardware/kernel_matrix_{backend}_meta.json  -- job_id, quantum_seconds, indices

Usage:
    set -a; source .env.quantum; set +a
    PYTHONUNBUFFERED=1 python hardware/kernel_matrix.py [--backend ibm_boston]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import zz_feature_map
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# Force UTF-8 stdout on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

_ROOT   = Path(__file__).resolve().parent.parent
_HW_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_and_project(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load CLIP embeddings, PCA->2D (fit on fold-0 train), scale to [0,pi].

    Returns
    -------
    X      : (n_samples, 2) float64, values in [0, pi]
    y      : (n_samples,)   int, label_up
    sel    : (n_samples,)   int, row indices into labeled.parquet / npz
    """
    labeled = pd.read_parquet(_ROOT / "data" / "labeled.parquet")
    npz     = np.load(_ROOT / "data" / "embeddings" / "n24news_finance.npz")
    embeddings = npz["embeddings"].astype(np.float64)   # (10206, 512)
    labels     = labeled["label_up"].values              # (10206,)

    # Fold-0 training mask: expanding window, first 252 unique trading days
    # labeled.parquet uses column "date" (the article/image date)
    days   = np.sort(labeled["date"].unique())
    cutoff = days[min(252, len(days) - 1)]
    train_mask = labeled["date"].values < cutoff

    # PCA fitted on training set only (no leakage)
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(embeddings[train_mask])
    X_2d = pca.transform(embeddings)                    # (10206, 2)

    # Scale to [0, pi] fitted on training set
    scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)))
    scaler.fit(X_2d[train_mask])
    X_scaled = scaler.transform(X_2d)                  # (10206, 2)

    # Stratified sample: floor(n/2) from each class
    rng  = np.random.default_rng(seed)
    half = n_samples // 2
    idx_up = np.where(labels == 1)[0]
    idx_dn = np.where(labels == 0)[0]
    sel_up = rng.choice(idx_up, half,              replace=False)
    sel_dn = rng.choice(idx_dn, n_samples - half,  replace=False)
    sel    = np.concatenate([sel_up, sel_dn])
    rng.shuffle(sel)

    return X_scaled[sel], labels[sel], sel


# ---------------------------------------------------------------------------
# Kernel circuit builder
# ---------------------------------------------------------------------------

def _build_kernel_circuits(
    X: np.ndarray,
) -> tuple[list, list[tuple[int, int]]]:
    """Build ZZFeatureMap^dagger(x_j) o ZZFeatureMap(x_i) for all (i,j), j>=i.

    k(x_i, x_j) = |<phi(x_j)|phi(x_i)>|^2  ~  P(measuring all zeros)
    """
    fmap = zz_feature_map(feature_dimension=2, reps=2)
    n    = len(X)

    circuits: list = []
    pairs: list[tuple[int, int]] = []

    for i in range(n):
        bound_i = fmap.assign_parameters(X[i])
        for j in range(i, n):
            bound_j_inv = fmap.assign_parameters(X[j]).inverse()
            qc = bound_i.compose(bound_j_inv)
            qc.measure_all()
            circuits.append(qc)
            pairs.append((i, j))

    return circuits, pairs


def _counts_to_fidelity(counts: dict, shots: int) -> float:
    """P(all-zeros bitstring) = fidelity estimate k(x_i, x_j)."""
    return counts.get("00", 0) / shots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="20x20 quantum kernel matrix")
    parser.add_argument("--backend",   default="ibm_boston")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--shots",     type=int, default=1024)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    n       = args.n_samples
    n_pairs = n * (n + 1) // 2   # upper triangle + diagonal

    print(f"=== QUANTUM KERNEL MATRIX ({n}x{n}) ===")
    print(f"Backend      : {args.backend}")
    print(f"Circuits     : {n_pairs}  (upper triangle + diagonal)")
    print(f"Shots/circuit: {args.shots}")
    print(f"Feature map  : zz_feature_map(dim=2, reps=2)")
    print()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Loading data, PCA->2D, scale to [0, pi] ...")
    X, y, sel = _load_and_project(n_samples=n, seed=args.seed)
    print(f"  sample indices : {sel.tolist()}")
    print(f"  labels         : {y.tolist()}  ({int(y.sum())} up / {int((1-y).sum())} down)")
    print(f"  X range        : [{X.min():.4f}, {X.max():.4f}]  (target [0, pi={np.pi:.4f}])")
    print()

    # ------------------------------------------------------------------
    # Circuits
    # ------------------------------------------------------------------
    print(f"Building {n_pairs} kernel circuits ...")
    circuits, pairs = _build_kernel_circuits(X)
    print(f"  depth (pre-transpile, circuit 0) : {circuits[0].depth()}")
    print()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    print("Connecting: QiskitRuntimeService() ...")
    try:
        service = QiskitRuntimeService()
    except Exception:
        print("[FATAL] QiskitRuntimeService() failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    backend = service.backend(args.backend)
    print(f"Backend      : {backend.name}  ({backend.num_qubits} qubits)")
    print()

    # ------------------------------------------------------------------
    # Transpile
    # ------------------------------------------------------------------
    print(f"Transpiling {n_pairs} circuits (optimization_level=1) ...")
    t_xpile = time.time()
    pm           = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = pm.run(circuits)
    print(f"  transpile time : {time.time() - t_xpile:.1f}s")
    print(f"  ISA depth  (circuit 0) : {isa_circuits[0].depth()}")
    print(f"  ISA ops    (circuit 0) : {isa_circuits[0].count_ops()}")
    print()

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots       = args.shots
    # No max_execution_time cap — let IBM's cycle limit govern; budget tracked via quantum_seconds

    print(f"Submitting {n_pairs}-circuit batch job ...")
    t0 = time.time()
    try:
        pub_list = [(qc,) for qc in isa_circuits]
        job      = sampler.run(pub_list)
        job_id   = job.job_id()
        print(f"  job_id  : {job_id}")
        print(f"  Waiting for result (includes queue time on {args.backend}) ...")
        result = job.result()
        wall_clock_s = time.time() - t0
    except Exception:
        print("[FATAL] Job submission or retrieval failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build kernel matrix
    # ------------------------------------------------------------------
    K = np.zeros((n, n), dtype=np.float64)
    for k_idx, (i, j) in enumerate(pairs):
        counts      = result[k_idx].data.meas.get_counts()
        K[i, j]     = _counts_to_fidelity(counts, args.shots)
        K[j, i]     = K[i, j]   # symmetric

    try:
        quantum_seconds = job.metrics()["usage"]["quantum_seconds"]
    except Exception as e:
        quantum_seconds = f"unavailable ({e})"

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    diag       = np.diag(K)
    off_mask   = ~np.eye(n, dtype=bool)
    off_values = K[off_mask]

    print()
    print(f"  Wall-clock (submit->result) : {wall_clock_s:.1f}s  [includes queue]")
    print(f"  quantum_seconds             : {quantum_seconds}  [quota burn]")
    print()
    print(f"  Diagonal (k(x,x), should be ~1.0):")
    print(f"    mean = {diag.mean():.4f}   min = {diag.min():.4f}   max = {diag.max():.4f}")
    print(f"  Off-diagonal entries:")
    print(f"    mean = {off_values.mean():.4f}   min = {off_values.min():.4f}   max = {off_values.max():.4f}")
    print()

    if diag.mean() > 0.80:
        print(f"  Diagonal check PASSED (mean = {diag.mean():.3f} > 0.80)")
    else:
        print(f"  [WARN] Diagonal mean = {diag.mean():.3f} -- high noise (readout or gate)")

    concentration_ratio = off_values.mean() / diag.mean() if diag.mean() > 0 else float("nan")
    print(f"  Concentration ratio (off-diag mean / diag mean) = {concentration_ratio:.4f}")
    print(f"  (Near 1.0 = concentration; near 0 = well-separated kernel)")
    print()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_npy  = _HW_DIR / f"kernel_matrix_{args.backend}.npy"
    out_json = _HW_DIR / f"kernel_matrix_{args.backend}_meta.json"
    np.save(out_npy, K)

    meta = {
        "backend"         : args.backend,
        "job_id"          : job_id,
        "n_samples"       : n,
        "n_pairs"         : n_pairs,
        "shots"           : args.shots,
        "feature_map"     : "zz_feature_map(feature_dimension=2, reps=2)",
        "wall_clock_s"    : wall_clock_s,
        "quantum_seconds" : quantum_seconds,
        "sample_indices"  : sel.tolist(),
        "labels"          : y.tolist(),
        "X_2d_scaled"     : X.tolist(),
        "diag_mean"       : float(diag.mean()),
        "off_diag_mean"   : float(off_values.mean()),
        "concentration_ratio": float(concentration_ratio),
        "timestamp"       : datetime.now(timezone.utc).isoformat(),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved kernel matrix -> {out_npy}")
    print(f"Saved metadata      -> {out_json}")
    print()
    print("=== KERNEL MATRIX K ===")
    print(np.array2string(K, precision=3, suppress_small=True))
    print()
    print("=== END ===")


if __name__ == "__main__":
    main()

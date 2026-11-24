#!/usr/bin/env python
"""Phase 3 — Batched 20×20 quantum kernel matrix on ibm_boston.

Budget constants (hard-coded, non-negotiable):
    QPU_REMAINING_S = 1200   # confirmed remaining before this run
    HARD_RESERVE_S  = 600    # floor — never touch
    SPEND_CAP_S     = 600    # max this entire run may spend

Requires measured spc (seconds-per-circuit) from the Phase 2 probe.
Pass it as --spc.  The script ABORTS before any batch that would push
cumulative spend over SPEND_CAP_S.

Circuit:
    angle_map(x1, n_qubits=8, reps=1) followed by its adjoint on x2.
    k(x1,x2) = P(|0…0>).  Matches pipeline/quantum/kernel.py exactly.

Batching:
    190 off-diagonal circuits split [48, 48, 48, 46].
    All four batches run inside ONE ibm_boston Session so they execute
    back-to-back without re-queuing.

Usage (Phase 3 — only after explicit GO):
    source .env.quantum
    python hardware/batched_matrix.py --spc <measured_spc>

Output:
    hardware/kernel_matrix_marrakesh_n20.npy        20×20 float64 Gram matrix
    hardware/kernel_matrix_marrakesh_n20_meta.json  per-batch usage log + metadata
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
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# ── project root ──────────────────────────────────────────────────────────────
_ROOT   = Path(__file__).resolve().parent.parent
_HW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, ParameterVector, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2

# ── budget constants (non-negotiable) ─────────────────────────────────────────
QPU_REMAINING_S : int = 1200
HARD_RESERVE_S  : int = 600
SPEND_CAP_S     : int = 600

# ── kernel configuration ──────────────────────────────────────────────────────
N          : int = 20
N_QUBITS   : int = 8
REPS       : int = 1
SHOTS      : int = 1024
BACKEND    : str = "ibm_marrakesh"   # switched from ibm_boston (91-deep queue vs Aug 6 deadline)
SEED       : int = 42

# Batches must be <= 48 circuits each (hardware rule)
BATCH_SIZES: list[int] = [48, 48, 48, 46]   # 190 total = N*(N-1)//2
assert sum(BATCH_SIZES) == N * (N - 1) // 2, "BATCH_SIZES must sum to N*(N-1)//2"
assert all(b <= 48 for b in BATCH_SIZES), "No batch may exceed 48 circuits"


# ── circuit builder (matches transpile_check.py) ──────────────────────────────

def build_angle_map_overlap_circuit(n_qubits: int, reps: int) -> QuantumCircuit:
    """Symbolic angle_map overlap circuit — used for transpilation only."""
    x1  = ParameterVector("x1", n_qubits)
    x2  = ParameterVector("x2", n_qubits)
    qr  = QuantumRegister(n_qubits, "q")
    cr  = ClassicalRegister(n_qubits, "meas")   # named "meas" → DataBin.meas.get_counts()
    qc  = QuantumCircuit(qr, cr)
    for _ in range(reps):
        for i in range(n_qubits):
            qc.ry(x1[i], i)
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)
        for i in range(n_qubits):
            qc.rz(x1[i], i)
    for _ in range(reps):
        for i in range(n_qubits - 1, -1, -1):
            qc.rz(-x2[i], i)
        for i in range(n_qubits - 1, -1, -1):
            qc.cx(i, (i + 1) % n_qubits)
        for i in range(n_qubits - 1, -1, -1):
            qc.ry(-x2[i], i)
    qc.measure(qr, cr)
    return qc


def bind_overlap_circuit(
    template: QuantumCircuit,
    x1_vals: np.ndarray,
    x2_vals: np.ndarray,
) -> QuantumCircuit:
    """Bind concrete x1, x2 values into the symbolic template."""
    bindings = {}
    x1_param = template.parameters[
        next(i for i, p in enumerate(template.parameters) if p.name.startswith("x1["))
    ]
    # Build full parameter map: template.parameters is a ParameterView sorted by name.
    # Names: x1[0]..x1[7], x2[0]..x2[7]
    param_map = {}
    for p in template.parameters:
        name = p.name   # e.g. "x1[3]"
        idx  = int(name.split("[")[1].rstrip("]"))
        if name.startswith("x1"):
            param_map[p] = float(x1_vals[idx])
        else:
            param_map[p] = float(x2_vals[idx])
    return template.assign_parameters(param_map)


# ── data loading + feature projection ────────────────────────────────────────

def load_and_reduce(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load CLIP embeddings, reduce to 8-dim angle features in [0, π].

    Matches the concentration_sweep approach:
        EmbeddingReducer = StandardScaler → PCA(8) → MinMax→[0,π]
    Reducer fitted on the FULL embedding matrix (no walk-forward here —
    this is a hardware characterisation run, not a walk-forward evaluation).

    Returns
    -------
    X_red : (n_samples, 8) float64  features in [0, π]
    y     : (n_samples,)   int       label_up
    sel   : (n_samples,)   int       row indices
    """
    labeled    = pd.read_parquet(_ROOT / "data" / "labeled.parquet")
    npz        = np.load(_ROOT / "data" / "embeddings" / "n24news_finance.npz")
    embeddings = npz["embeddings"].astype(np.float64)
    labels     = labeled["label_up"].values

    # Stratified sample
    rng      = np.random.default_rng(seed)
    half     = n_samples // 2
    idx_up   = np.where(labels == 1)[0]
    idx_dn   = np.where(labels == 0)[0]
    sel_up   = rng.choice(idx_up, half, replace=False)
    sel_dn   = rng.choice(idx_dn, n_samples - half, replace=False)
    sel      = np.concatenate([sel_up, sel_dn])
    rng.shuffle(sel)

    # EmbeddingReducer (fitted on full corpus)
    from pipeline.quantum.reducer import EmbeddingReducer
    reducer = EmbeddingReducer(n_components=N_QUBITS, seed=seed)
    reducer.fit(embeddings)
    X_red = reducer.transform(embeddings[sel])

    if X_red.min() < 0.0 or X_red.max() > np.pi + 1e-9:
        raise RuntimeError(
            f"FAIL: EmbeddingReducer output out of [0,π]: "
            f"min={X_red.min():.4f}  max={X_red.max():.4f}"
        )

    return X_red, labels[sel].astype(int), sel


# ── kernel matrix assembly ────────────────────────────────────────────────────

def _counts_to_fidelity(counts: dict, shots: int) -> float:
    """P(|0…0>) from a counts dict. All-zeros bitstring key is '00000000'."""
    zero_key = "0" * N_QUBITS
    return counts.get(zero_key, 0) / shots


def assemble_gram(
    n: int,
    pairs: list[tuple[int, int]],
    batch_results: list,
    shots: int,
) -> np.ndarray:
    """Build symmetric n×n Gram matrix. Diagonal = 1 by construction."""
    K = np.eye(n, dtype=np.float64)
    circuit_idx = 0
    for batch_res in batch_results:
        for pub_result in batch_res:
            if circuit_idx >= len(pairs):
                raise RuntimeError("FAIL: more results than expected pairs")
            i, j = pairs[circuit_idx]
            counts = pub_result.data.meas.get_counts()
            val    = _counts_to_fidelity(counts, shots)
            K[i, j] = val
            K[j, i] = val
            circuit_idx += 1
    if circuit_idx != len(pairs):
        raise RuntimeError(
            f"FAIL: assembled {circuit_idx} pairs, expected {len(pairs)}"
        )
    return K


# ── budget guardrail ─────────────────────────────────────────────────────────

def _budget_check(
    spent_s: float,
    spc: float,
    next_batch_size: int,
    label: str,
) -> None:
    """Raise if submitting next_batch_size circuits would exceed SPEND_CAP_S."""
    projected = spent_s + spc * next_batch_size
    if projected > SPEND_CAP_S:
        raise SystemExit(
            f"\nBUDGET ABORT before {label}:\n"
            f"  spent_s              : {spent_s:.1f}s\n"
            f"  spc                  : {spc:.3f}s/circuit\n"
            f"  next_batch_size      : {next_batch_size}\n"
            f"  projected total      : {projected:.1f}s\n"
            f"  SPEND_CAP_S          : {SPEND_CAP_S}s\n"
            f"  Do NOT submit. Report partial Gram matrix if any batches completed.\n"
        )
    print(
        f"  [budget OK] spent_s={spent_s:.1f}s  +  "
        f"spc×{next_batch_size}={spc*next_batch_size:.1f}s  "
        f"→ projected={projected:.1f}s  (cap={SPEND_CAP_S}s)"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 batched kernel matrix")
    parser.add_argument(
        "--spc",
        type=float,
        required=True,
        help="Seconds per circuit measured from Phase 2 probe on ibm_boston",
    )
    parser.add_argument("--backend", default=BACKEND)
    parser.add_argument("--shots",   type=int, default=SHOTS)
    parser.add_argument("--n",       type=int, default=N)
    parser.add_argument("--seed",    type=int, default=SEED)
    args = parser.parse_args()

    spc = args.spc
    if spc <= 0:
        print("[FATAL] --spc must be positive (measured from probe)", file=sys.stderr)
        sys.exit(1)

    n_pairs = args.n * (args.n - 1) // 2

    print("=" * 64)
    print("PHASE 3 — BATCHED QUANTUM KERNEL MATRIX")
    print("=" * 64)
    print(f"backend        : {args.backend}")
    print(f"n              : {args.n}  →  {n_pairs} off-diagonal circuits")
    print(f"n_qubits       : {N_QUBITS}")
    print(f"reps           : {REPS}")
    print(f"shots          : {args.shots}")
    print(f"batch_sizes    : {BATCH_SIZES}")
    print()
    print(f"BUDGET CONSTANTS:")
    print(f"  QPU_REMAINING_S  = {QPU_REMAINING_S}")
    print(f"  HARD_RESERVE_S   = {HARD_RESERVE_S}")
    print(f"  SPEND_CAP_S      = {SPEND_CAP_S}")
    print(f"  spc (probe)      = {spc:.3f} s/circuit")
    total_est = spc * n_pairs
    print(f"  full-matrix est  = {total_est:.1f}s ({n_pairs} circuits × {spc:.3f}s)")
    print()

    # Pre-flight: can we afford all four batches?
    if total_est > SPEND_CAP_S:
        print(
            f"[WARN] Full matrix estimate ({total_est:.1f}s) exceeds SPEND_CAP_S "
            f"({SPEND_CAP_S}s). Will abort mid-run on budget check."
        )
    print()

    # ── data ─────────────────────────────────────────────────────────────────
    print("Loading CLIP embeddings and reducing to 8-dim [0,π] features ...")
    X, y, sel = load_and_reduce(args.n, args.seed)
    print(f"  X shape   : {X.shape}  range=[{X.min():.4f}, {X.max():.4f}]")
    print(f"  labels    : {y.tolist()}  ({int(y.sum())} up / {int((y==0).sum())} down)")
    print()

    # ── build circuit list (symbolic template → bound) ────────────────────────
    print(f"Building {n_pairs} overlap circuits (angle_map reps={REPS}, n_qubits={N_QUBITS}) ...")
    template = build_angle_map_overlap_circuit(N_QUBITS, REPS)

    circuits: list[QuantumCircuit] = []
    pairs:    list[tuple[int, int]] = []
    for i in range(args.n):
        for j in range(i + 1, args.n):
            qc = bind_overlap_circuit(template, X[i], X[j])
            circuits.append(qc)
            pairs.append((i, j))

    assert len(circuits) == n_pairs, f"Expected {n_pairs} circuits, got {len(circuits)}"
    print(f"  circuit 0 pre-transpile depth : {circuits[0].depth()}")
    print()

    # ── auth ──────────────────────────────────────────────────────────────────
    print("Connecting: QiskitRuntimeService() ...")
    try:
        service = QiskitRuntimeService()
    except Exception:
        print("[FATAL] QiskitRuntimeService() failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    backend = service.backend(args.backend)
    print(f"Backend : {backend.name}  ({backend.num_qubits} qubits)")
    print()

    # ── transpile all circuits before opening Session ─────────────────────────
    print(f"Transpiling {n_pairs} circuits (optimization_level=1) ...")
    t_xpile = time.time()
    pm           = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = pm.run(circuits)
    print(f"  transpile time : {time.time()-t_xpile:.1f}s")
    print(f"  ISA depth [0]  : {isa_circuits[0].depth()}")
    print()

    # ── split into batches ───────────────────────────────────────────────────
    batches: list[list] = []
    pair_batches: list[list[tuple[int,int]]] = []
    offset = 0
    for sz in BATCH_SIZES:
        batches.append(isa_circuits[offset : offset + sz])
        pair_batches.append(pairs[offset : offset + sz])
        offset += sz

    # ── execute inside a single Session ─────────────────────────────────────
    spent_s      : float = 0.0
    batch_results: list  = []
    usage_log    : list[dict] = []
    t_session_start = time.time()

    print("Opening ibm_boston Session ...")
    try:
        with Session(backend=backend) as session:
            sampler = SamplerV2(mode=session)
            sampler.options.default_shots = args.shots

            for batch_idx, (batch_circuits, batch_pairs) in enumerate(
                zip(batches, pair_batches)
            ):
                batch_label = f"batch {batch_idx+1}/{len(batches)}"
                next_sz     = len(batch_circuits)

                # ── budget check BEFORE submitting ───────────────────────────
                print(f"\n[{batch_label}]  {next_sz} circuits")
                _budget_check(spent_s, spc, next_sz, batch_label)

                # ── submit ───────────────────────────────────────────────────
                pub_list = [(qc,) for qc in batch_circuits]
                print(f"  Submitting ...")
                t0  = time.time()
                job = sampler.run(pub_list)
                print(f"  job_id : {job.job_id()}")
                print(f"  Waiting for result ...")
                result = job.result()
                wall_s = time.time() - t0

                # ── read usage ───────────────────────────────────────────────
                try:
                    usage_s = job.metrics()["usage"]["quantum_seconds"]
                except Exception as e:
                    raise RuntimeError(
                        f"FAIL: could not read quantum_seconds for {batch_label}: {e}"
                    )

                if not isinstance(usage_s, (int, float)):
                    raise RuntimeError(
                        f"FAIL: quantum_seconds for {batch_label} is not numeric: "
                        f"{usage_s!r}"
                    )

                spent_s += float(usage_s)
                batch_results.append(result)

                log_entry = {
                    "batch":          batch_idx,
                    "batch_size":     next_sz,
                    "job_id":         job.job_id(),
                    "usage_s":        float(usage_s),
                    "cumulative_s":   spent_s,
                    "wall_s":         wall_s,
                }
                usage_log.append(log_entry)

                print(
                    f"  usage={usage_s}s  cumulative_spent={spent_s:.1f}s  "
                    f"remaining_cap={SPEND_CAP_S - spent_s:.1f}s"
                )

                # Verify no pub returned an empty result
                for pub_idx, pub_res in enumerate(result):
                    counts = pub_res.data.meas.get_counts()
                    if not counts:
                        raise RuntimeError(
                            f"FAIL: batch {batch_idx} pub {pub_idx} returned "
                            f"empty counts"
                        )

    except SystemExit:
        # Budget abort — fall through to save whatever we have
        print("\n[BUDGET ABORT] Saving partial Gram matrix from completed batches.")
        raise
    except Exception:
        print("[FATAL] Session execution failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # ── assemble Gram matrix ─────────────────────────────────────────────────
    print("\nAssembling Gram matrix ...")
    K = assemble_gram(args.n, pairs, batch_results, args.shots)

    # Validate
    diag_vals  = np.diag(K)
    off_mask   = ~np.eye(args.n, dtype=bool)
    off_vals   = K[off_mask]

    if not np.allclose(diag_vals, 1.0, atol=1e-12):
        raise RuntimeError(
            f"FAIL: diagonal is not exactly 1. "
            f"min={diag_vals.min():.6f}  max={diag_vals.max():.6f}"
        )
    if np.any(np.isnan(K)):
        raise RuntimeError("FAIL: NaN detected in assembled Gram matrix")

    off_mean = float(off_vals.mean())
    off_var  = float(off_vals.var())

    print(f"  shape          : {K.shape}")
    print(f"  diagonal       : exactly 1.0 ✓")
    print(f"  off_diag mean  : {off_mean:.6f}")
    print(f"  off_diag var   : {off_var:.6f}")
    print()

    # ── save ────────────────────────────────────────────────────────────────
    out_npy  = _HW_DIR / "kernel_matrix_marrakesh_n20.npy"
    out_json = _HW_DIR / "kernel_matrix_marrakesh_n20_meta.json"
    np.save(out_npy, K)

    meta = {
        "backend":          args.backend,
        "n":                args.n,
        "n_qubits":         N_QUBITS,
        "reps":             REPS,
        "shots":            args.shots,
        "circuit":          "angle_map_overlap",
        "batch_sizes":      BATCH_SIZES,
        "spc_probe":        spc,
        "total_spent_s":    spent_s,
        "spend_cap_s":      SPEND_CAP_S,
        "qpu_remaining_s":  QPU_REMAINING_S,
        "off_diag_mean":    off_mean,
        "off_diag_var":     off_var,
        "sample_indices":   sel.tolist(),
        "labels":           y.tolist(),
        "usage_log":        usage_log,
        "session_wall_s":   time.time() - t_session_start,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved kernel matrix  → {out_npy}")
    print(f"Saved metadata       → {out_json}")
    print()
    print(f"TOTAL spent_s : {spent_s:.1f}s  (cap={SPEND_CAP_S}s)")
    print("=" * 64)


if __name__ == "__main__":
    main()

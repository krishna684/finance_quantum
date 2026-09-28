#!/usr/bin/env python
"""Phase 2 — QPU probe on ibm_marrakesh.

Submits 10 angle_map overlap circuits (reps=1, n_qubits=8, 1024 shots)
as a standalone Sampler job (NOT Session) and reads the measured
quantum_seconds to establish the per-circuit rate (spc).

Budget tracking (cumulative across ALL jobs this run):
    QPU_REMAINING_S = 1200
    HARD_RESERVE_S  = 600    (floor — never touch)
    SPEND_CAP_S     = 600    (cap for this entire run)

This probe job counts toward SPEND_CAP_S.

Target switch note: originally planned for ibm_boston (queue depth 91 at
2026-08-07 00:08 UTC vs fixed Aug 6 submission deadline). Switched to
ibm_marrakesh (queue depth 0) for deadline compliance. 2Q error rate:
boston 1.48e-3 / marrakesh 2.86e-3. Physics claim (hardware noise
produces concentration collapse at reps=1) is if anything strengthened
by the higher noise on marrakesh — see Phase 5 draft note.

Usage:
    source .env.quantum && python hardware/probe_marrakesh.py
"""
from __future__ import annotations

import io
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

_ROOT   = Path(__file__).resolve().parent.parent
_HW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# ── budget constants ───────────────────────────────────────────────────────────
QPU_REMAINING_S : int = 1200
HARD_RESERVE_S  : int = 600
SPEND_CAP_S     : int = 600

# ── probe config ───────────────────────────────────────────────────────────────
N_PROBE_CIRCUITS : int = 10
N_QUBITS         : int = 8
REPS             : int = 1
SHOTS            : int = 1024
BACKEND          : str = "ibm_marrakesh"
SEED             : int = 42
N_FULL           : int = 20   # full matrix size (for projection)
N_FULL_PAIRS     : int = N_FULL * (N_FULL - 1) // 2   # = 190


def _build_overlap(n_qubits: int, reps: int,
                   x1: np.ndarray, x2: np.ndarray) -> QuantumCircuit:
    """Angle-map overlap circuit with concrete parameter values bound."""
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "meas")   # named "meas" → DataBin.meas.get_counts()
    qc = QuantumCircuit(qr, cr)
    for _ in range(reps):
        for i in range(n_qubits):
            qc.ry(float(x1[i]), i)
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)
        for i in range(n_qubits):
            qc.rz(float(x1[i]), i)
    for _ in range(reps):
        for i in range(n_qubits - 1, -1, -1):
            qc.rz(-float(x2[i]), i)
        for i in range(n_qubits - 1, -1, -1):
            qc.cx(i, (i + 1) % n_qubits)
        for i in range(n_qubits - 1, -1, -1):
            qc.ry(-float(x2[i]), i)
    qc.measure(qr, cr)
    return qc


def _load_reduced_embeddings(n_samples: int, seed: int) -> np.ndarray:
    """Load n_samples CLIP embeddings reduced to 8-dim [0,π]."""
    labeled    = pd.read_parquet(_ROOT / "data" / "labeled.parquet")
    npz        = np.load(_ROOT / "data" / "embeddings" / "n24news_finance.npz")
    embeddings = npz["embeddings"].astype(np.float64)
    labels     = labeled["label_up"].values

    rng    = np.random.default_rng(seed)
    half   = n_samples // 2
    sel_up = rng.choice(np.where(labels == 1)[0], half,             replace=False)
    sel_dn = rng.choice(np.where(labels == 0)[0], n_samples - half, replace=False)
    sel    = np.concatenate([sel_up, sel_dn])
    rng.shuffle(sel)

    from pipeline.quantum.reducer import EmbeddingReducer
    reducer = EmbeddingReducer(n_components=N_QUBITS, seed=seed)
    reducer.fit(embeddings)
    return reducer.transform(embeddings[sel])


def main() -> None:
    print("=" * 64)
    print("PHASE 2 — QPU PROBE")
    print("=" * 64)
    print(f"Backend          : {BACKEND}")
    print(f"Probe circuits   : {N_PROBE_CIRCUITS}")
    print(f"n_qubits         : {N_QUBITS},  reps={REPS},  shots={SHOTS}")
    print(f"SPEND_CAP_S      : {SPEND_CAP_S}")
    print()

    # ── data + circuits ───────────────────────────────────────────────────────
    print("Loading embeddings and building probe circuits ...")
    X = _load_reduced_embeddings(N_FULL, SEED)   # (20, 8)

    # First N_PROBE_CIRCUITS off-diagonal pairs
    pairs: list[tuple[int,int]] = []
    circuits: list[QuantumCircuit] = []
    for i in range(N_FULL):
        for j in range(i + 1, N_FULL):
            if len(circuits) >= N_PROBE_CIRCUITS:
                break
            circuits.append(_build_overlap(N_QUBITS, REPS, X[i], X[j]))
            pairs.append((i, j))
        if len(circuits) >= N_PROBE_CIRCUITS:
            break

    assert len(circuits) == N_PROBE_CIRCUITS
    print(f"  Pairs           : {pairs}")
    print(f"  Pre-transpile d : {circuits[0].depth()}")
    print()

    # ── auth ─────────────────────────────────────────────────────────────────
    print("Connecting: QiskitRuntimeService() ...")
    try:
        service = QiskitRuntimeService()
    except Exception:
        print("[FATAL] QiskitRuntimeService() failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    backend = service.backend(BACKEND)
    print(f"Backend          : {backend.name}  ({backend.num_qubits} qubits)")
    try:
        st = backend.status()
        print(f"  operational    : {st.operational}")
        print(f"  pending_jobs   : {st.pending_jobs}")
    except Exception as e:
        print(f"  status         : unavailable ({e})")
    print()

    # ── transpile ─────────────────────────────────────────────────────────────
    print(f"Transpiling {N_PROBE_CIRCUITS} circuits (optimization_level=1) ...")
    t_xp = time.time()
    pm           = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = pm.run(circuits)
    print(f"  transpile time : {time.time()-t_xp:.1f}s")
    print(f"  ISA depth [0]  : {isa_circuits[0].depth()}")
    print(f"  ISA 2Q count   : {sum(1 for g in isa_circuits[0].data if g.operation.num_qubits==2)}")
    print()

    # ── submit (standalone Sampler — NOT Session) ─────────────────────────────
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = SHOTS

    pub_list = [(qc,) for qc in isa_circuits]
    print(f"Submitting {N_PROBE_CIRCUITS}-circuit probe job (standalone, not Session) ...")
    t0  = time.time()
    job = sampler.run(pub_list)
    job_id = job.job_id()
    print(f"  job_id         : {job_id}")
    print(f"  Waiting for result (includes any remaining queue time) ...")

    try:
        result   = job.result()
        wall_s   = time.time() - t0
    except Exception:
        print("[FATAL] Job failed or was cancelled:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # ── read usage ─────────────────────────────────────────────────────────────
    try:
        usage_s = job.metrics()["usage"]["quantum_seconds"]
    except Exception as e:
        print(f"[FATAL] Could not read quantum_seconds: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(usage_s, (int, float)):
        print(f"[FATAL] quantum_seconds is not numeric: {usage_s!r}", file=sys.stderr)
        sys.exit(1)

    usage_s = float(usage_s)

    # ── probe metrics ─────────────────────────────────────────────────────────
    spc            = usage_s / N_PROBE_CIRCUITS
    matrix_est_s   = N_FULL_PAIRS * spc   # 190 circuits
    repeat_est_s   = N_FULL_PAIRS * spc   # second run (error bars)
    total_est_s    = usage_s + matrix_est_s

    print()
    print("=" * 64)
    print("PROBE RESULT")
    print("=" * 64)
    print(f"  job_id           : {job_id}")
    print(f"  probe circuits   : {N_PROBE_CIRCUITS}")
    print(f"  shots/circuit    : {SHOTS}")
    print(f"  usage (s)        : {usage_s}")
    print(f"  wall_clock (s)   : {wall_s:.1f}  [includes any queue time]")
    print()
    print(f"  spc              : {spc:.4f} s/circuit  (usage / {N_PROBE_CIRCUITS})")
    print()
    print(f"PROJECTIONS:")
    print(f"  Full matrix (190 circuits)  : {matrix_est_s:.1f}s")
    print(f"  Repeat run  (190 circuits)  : {repeat_est_s:.1f}s")
    print()
    print(f"BUDGET CHECK:")
    print(f"  SPEND_CAP_S                 : {SPEND_CAP_S}s")
    print(f"  probe spent_s               : {usage_s}s")
    print(f"  spent_s + matrix_est_s      : {usage_s + matrix_est_s:.1f}s")

    can_do_matrix = (usage_s + matrix_est_s) <= SPEND_CAP_S
    can_do_repeat = (usage_s + matrix_est_s + repeat_est_s) <= SPEND_CAP_S

    print()
    if can_do_matrix:
        print(f"  GO: matrix fits under cap.  spent_s={usage_s}s + {matrix_est_s:.1f}s = {usage_s+matrix_est_s:.1f}s <= {SPEND_CAP_S}s")
        if can_do_repeat:
            print(f"  GO: repeat also fits.        +{repeat_est_s:.1f}s → total={usage_s+matrix_est_s+repeat_est_s:.1f}s <= {SPEND_CAP_S}s")
        else:
            print(f"  NO-GO for repeat: would push to {usage_s+matrix_est_s+repeat_est_s:.1f}s > {SPEND_CAP_S}s")
    else:
        print(
            f"  NO-GO: matrix estimate ({usage_s+matrix_est_s:.1f}s) > SPEND_CAP_S ({SPEND_CAP_S}s).\n"
            f"  Possible reductions:\n"
            f"    n=15 → 105 circuits → {usage_s + 105*spc:.1f}s total\n"
            f"    512 shots → halves spc to ~{spc/2:.4f} → {usage_s + 190*(spc/2):.1f}s total"
        )

    print()

    # ── spot-check one result ─────────────────────────────────────────────────
    counts_0 = result[0].data.meas.get_counts()   # ClassicalRegister named "meas"
    zero_key  = "0" * N_QUBITS
    p0        = counts_0.get(zero_key, 0) / SHOTS
    print(f"Spot-check circuit 0: P(|0...0>) = {p0:.4f}  (counts keys: {len(counts_0)})")
    print()

    # ── save ─────────────────────────────────────────────────────────────────
    meta = {
        "backend":          BACKEND,
        "job_id":           job_id,
        "n_probe_circuits": N_PROBE_CIRCUITS,
        "shots":            SHOTS,
        "n_qubits":         N_QUBITS,
        "reps":             REPS,
        "usage_s":          usage_s,
        "wall_s":           wall_s,
        "spc":              spc,
        "matrix_est_s":     matrix_est_s,
        "repeat_est_s":     repeat_est_s,
        "spend_cap_s":      SPEND_CAP_S,
        "can_do_matrix":    can_do_matrix,
        "can_do_repeat":    can_do_repeat,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }
    out = _HW_DIR / "probe_marrakesh_meta.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved probe metadata → {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()

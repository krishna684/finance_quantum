#!/usr/bin/env python
"""Phase F post-run diagnostics — zero QPU cost, read-only.

Pulls raw unrounded metrics for all 5 completed jobs (probe + 4 batches),
inspects the exact physical qubits used in ibm_fez transpilation, and
reports per-qubit-pair 2Q error rates for those specific pairs.

No job submission. No re-run. Data-read only.

Usage:
    python hardware/phase_f_diagnostics.py
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

_ROOT   = Path(__file__).resolve().parent.parent
_HW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# ── credential loader (same as phase_f_run2.py) ───────────────────────────────
def _load_env_quantum() -> None:
    import os
    env_path = _ROOT / ".env.quantum"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip(); v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v

_load_env_quantum()

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService

# ── job IDs from meta.json ────────────────────────────────────────────────────
META = json.loads((_HW_DIR / "kernel_matrix_run2_n20_meta.json").read_text())
PROBE_JOB_ID  = META["probe_job_id"]     # d9r89tpdsedc73agaohg
BATCH_JOB_IDS = [e["job_id"] for e in META["usage_log"]]
BATCH_SIZES   = [e["batch_size"] for e in META["usage_log"]]
N_QUBITS = META["n_qubits"]
REPS     = META["reps"]
SHOTS    = META["shots"]
BACKEND_NAME = META["backend"]

N_PROBE = 10


# ── circuit builder (same as run2) ────────────────────────────────────────────
def _build_overlap_concrete(n_qubits: int, reps: int,
                             x1: np.ndarray, x2: np.ndarray) -> QuantumCircuit:
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "meas")
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


def main() -> None:
    print("=" * 72)
    print("PHASE F DIAGNOSTICS — raw metrics pull, zero QPU cost")
    print("=" * 72)
    print()

    service = QiskitRuntimeService()
    print("Auth: OK")
    backend = service.backend(BACKEND_NAME)
    print(f"Backend: {BACKEND_NAME}  ({backend.num_qubits} qubits)")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 1 + 2: Raw unrounded usage for probe + all four batches
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("ITEM 1+2: RAW UNROUNDED quantum_seconds FROM JOB METRICS")
    print("=" * 72)
    print()

    all_job_ids = [PROBE_JOB_ID] + BATCH_JOB_IDS
    all_labels  = [f"probe (n={N_PROBE})"] + [
        f"batch {i+1} (n={BATCH_SIZES[i]})" for i in range(len(BATCH_SIZES))
    ]

    raw_usages: dict[str, object] = {}
    for job_id, label in zip(all_job_ids, all_labels):
        job = service.job(job_id)
        metrics = job.metrics()
        # Print full metrics dict so nothing is hidden
        usage_section = metrics.get("usage", {})
        qs_raw = usage_section.get("quantum_seconds", "KEY_MISSING")
        # Also check for seconds/bss_seconds/etc
        all_usage_keys = {k: v for k, v in usage_section.items()}
        print(f"  [{label}]  job_id={job_id}")
        print(f"    metrics['usage'] full dict  : {all_usage_keys}")
        print(f"    quantum_seconds (raw Python): {repr(qs_raw)}")
        print(f"    type                        : {type(qs_raw).__name__}")
        raw_usages[label] = qs_raw
        print()

    # Summary table
    print("  SUMMARY — unrounded quantum_seconds:")
    print(f"  {'Job':<25}  {'circuits':>8}  {'raw_qs':>14}  {'qs/circuit':>12}")
    print(f"  {'-'*65}")
    total_qs = 0.0
    for (label, qs), n_circs in zip(
        raw_usages.items(),
        [N_PROBE] + BATCH_SIZES
    ):
        spc = (float(qs) / n_circs) if isinstance(qs, (int, float)) else float("nan")
        total_qs += float(qs) if isinstance(qs, (int, float)) else 0
        print(f"  {label:<25}  {n_circs:>8}  {str(qs):>14}  {spc:>12.6f}")

    print()
    # Is 46-circuit batch actually different from 48?
    b_usages = [raw_usages[f"batch {i+1} (n={BATCH_SIZES[i]})"] for i in range(4)]
    vals = [float(v) for v in b_usages]
    print(f"  Batch usage values: {vals}")
    if len(set(vals)) == 1:
        print(f"  -> All four values ARE IDENTICAL (not just rounded). Value = {vals[0]}")
        print(f"     This indicates IBM Runtime bills in fixed quantized units on ibm_fez.")
        print(f"     The 46-circuit batch billed the same quantum unit as the 48-circuit batches.")
    else:
        print(f"  -> Values are DISTINCT (rounding artifact only): {set(vals)}")

    # Overhead analysis: probe spc vs batch spc
    probe_qs    = float(raw_usages[f"probe (n={N_PROBE})"])
    batch_total = sum(vals)
    batch_n     = sum(BATCH_SIZES)
    probe_spc   = probe_qs / N_PROBE
    batch_spc   = batch_total / batch_n
    print()
    print(f"  Probe:  {probe_qs} s / {N_PROBE} circuits = {probe_spc:.6f} s/circuit")
    print(f"  Matrix: {batch_total} s / {batch_n} circuits = {batch_spc:.6f} s/circuit")
    overhead_s  = probe_qs - (probe_spc * N_PROBE)  # trivially 0 by construction
    # Better: compare probe_spc to batch_spc
    ratio = probe_spc / batch_spc if batch_spc > 0 else float("nan")
    print(f"  Ratio (probe_spc / batch_spc): {ratio:.4f}")
    fixed_overhead_est = probe_qs - batch_spc * N_PROBE
    print(f"  If probe has fixed per-job overhead: overhead_est = probe_qs - batch_spc*10 = "
          f"{probe_qs} - {batch_spc*N_PROBE:.4f} = {fixed_overhead_est:.4f}s")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 3A: ISA depth + 2Q count on ibm_fez (rebuild transpile, same circuit)
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("ITEM 3A: ibm_fez ISA CIRCUIT STATS (re-transpile circuit 0)")
    print("=" * 72)
    print()

    # Load same data as run2 (seed=42)
    labeled    = pd.read_parquet(_ROOT / "data" / "labeled.parquet")
    npz        = np.load(_ROOT / "data" / "embeddings" / "n24news_finance.npz")
    embeddings = npz["embeddings"].astype(np.float64)
    labels     = labeled["label_up"].values

    rng    = np.random.default_rng(42)
    half   = META["n"] // 2
    sel_up = rng.choice(np.where(labels == 1)[0], half,                  replace=False)
    sel_dn = rng.choice(np.where(labels == 0)[0], META["n"] - half,      replace=False)
    sel    = np.concatenate([sel_up, sel_dn])
    rng.shuffle(sel)

    from pipeline.quantum.reducer import EmbeddingReducer
    reducer = EmbeddingReducer(n_components=N_QUBITS, seed=42)
    reducer.fit(embeddings)
    X = reducer.transform(embeddings[sel])

    # Build circuit 0 (pair (0,1))
    qc0 = _build_overlap_concrete(N_QUBITS, REPS, X[0], X[1])
    print(f"  Pre-transpile depth   : {qc0.depth()}")
    pre_2q = sum(1 for g in qc0.data if g.operation.num_qubits == 2)
    print(f"  Pre-transpile 2Q count: {pre_2q}")
    print()

    print(f"  Transpiling on {BACKEND_NAME} (optimization_level=1) ...")
    t0 = time.time()
    pm  = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa = pm.run(qc0)
    print(f"  Transpile time: {time.time()-t0:.2f}s")
    print()

    isa_depth = isa.depth()
    isa_ops   = isa.count_ops()
    isa_2q_all = sum(v for op, v in isa_ops.items()
                     if any(g in op for g in ["cx","ecr","cz","rzz","rxx","cp"]))
    # More precise: count by num_qubits in data
    isa_2q_data = sum(1 for g in isa.data if g.operation.num_qubits == 2)

    print(f"  ISA depth            : {isa_depth}")
    print(f"  ISA op counts        : {dict(isa_ops)}")
    print(f"  ISA 2Q gate count    : {isa_2q_data}")
    print()

    # Extract the physical qubits used (layout)
    layout = isa.layout
    if layout is not None:
        final = layout.final_layout
        initial = layout.initial_layout
        print(f"  layout.initial_layout: {initial}")
        print(f"  layout.final_layout  : {final}")
        # Get the physical qubits that virtual qubits map to
        try:
            phys_qubits = [layout.final_index_layout()[i] for i in range(N_QUBITS)]
            print(f"  Physical qubits used : {phys_qubits}")
        except Exception as e:
            print(f"  Physical qubit extraction (final_index_layout): {e}")
            try:
                # Alternative
                from qiskit.transpiler import Layout
                phys_qubits = list(layout.initial_layout.get_physical_bits().keys())
                phys_qubits = [p for p in phys_qubits if isinstance(p, int)]
                print(f"  Physical qubits (alt): {phys_qubits}")
            except Exception as e2:
                print(f"  Physical qubit extraction (alt): {e2}")
                phys_qubits = None
    else:
        print("  layout: None (layout unavailable)")
        phys_qubits = None
    print()

    # Find which 2Q gates are in the ISA circuit and on which physical qubit pairs
    two_qubit_pairs_used: set[tuple[int, int]] = set()
    for inst in isa.data:
        if inst.operation.num_qubits == 2:
            q0 = isa.find_bit(inst.qubits[0]).index
            q1 = isa.find_bit(inst.qubits[1]).index
            pair = (min(q0, q1), max(q0, q1))
            two_qubit_pairs_used.add(pair)

    print(f"  Physical 2Q qubit pairs in ISA circuit: {sorted(two_qubit_pairs_used)}")
    print(f"  Unique 2Q pairs: {len(two_qubit_pairs_used)}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 3B: Per-pair 2Q error rates for the ACTUAL pairs used
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("ITEM 3B: 2Q ERROR RATES FOR ACTUAL TRANSPILED QUBIT PAIRS")
    print("=" * 72)
    print()

    try:
        props = backend.properties()
        # Build a lookup: frozenset({q0,q1}) -> error
        pair_errors: dict[frozenset, list[float]] = {}
        for gate in props.gates:
            if len(gate.qubits) != 2:
                continue
            fs = frozenset(gate.qubits)
            for p in gate.parameters:
                if p.name == "gate_error":
                    pair_errors.setdefault(fs, []).append(p.value)
                    break

        print(f"  {'Pair':<12}  {'Gate error(s)':>30}  {'Mean error':>12}")
        print(f"  {'-'*58}")

        used_errors = []
        for pair in sorted(two_qubit_pairs_used):
            fs = frozenset(pair)
            errs = pair_errors.get(fs, [])
            if errs:
                mean_e = float(np.mean(errs))
                used_errors.append(mean_e)
                print(f"  {str(pair):<12}  {str([f'{e:.6e}' for e in errs]):>30}  {mean_e:>12.6e}")
            else:
                print(f"  {str(pair):<12}  {'N/A (not in properties)':>30}  {'N/A':>12}")

        print()
        if used_errors:
            print(f"  Mean 2Q error across ALL used pairs      : {np.mean(used_errors):.6e}")
            print(f"  Min 2Q error across used pairs           : {np.min(used_errors):.6e}")
            print(f"  Max 2Q error across used pairs           : {np.max(used_errors):.6e}")
            print()

        # Compare to chip-wide average (what phase_f_run2.py reported)
        all_pair_errs = []
        for gate in props.gates:
            if len(gate.qubits) != 2:
                continue
            for p in gate.parameters:
                if p.name == "gate_error":
                    all_pair_errs.append(p.value)
                    break
        chip_mean = float(np.mean(all_pair_errs))
        chip_max  = float(np.max(all_pair_errs))
        chip_min  = float(np.min(all_pair_errs))
        print(f"  Chip-wide 2Q error (ALL {len(all_pair_errs)} pairs):")
        print(f"    mean = {chip_mean:.6e}  (previously reported as 4.208e-02)")
        print(f"    min  = {chip_min:.6e}")
        print(f"    max  = {chip_max:.6e}")
        print()

        if used_errors:
            used_mean = float(np.mean(used_errors))
            print(f"  VERDICT on 3B:")
            if abs(used_mean - chip_mean) / chip_mean < 0.1:
                print(f"    Used-pair mean ({used_mean:.4e}) is within 10% of chip-wide mean "
                      f"({chip_mean:.4e}). Chip-wide average was representative.")
            else:
                print(f"    Used-pair mean ({used_mean:.4e}) DIFFERS from chip-wide mean "
                      f"({chip_mean:.4e}). Chip-wide average was NOT representative "
                      f"of actual execution.")

    except Exception as e:
        import traceback
        print(f"  [ERROR] Could not retrieve backend properties: {e}")
        traceback.print_exc()

    print()

    # ─────────────────────────────────────────────────────────────────────────
    # ITEM 3C: Marrakesh vs fez ISA comparison (marrakesh re-transpile)
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("ITEM 3C: MARRAKESH ISA STATS (re-transpile same circuit 0 for comparison)")
    print("=" * 72)
    print()
    try:
        backend_mrk = service.backend("ibm_marrakesh")
        pm_mrk  = generate_preset_pass_manager(backend=backend_mrk, optimization_level=1)
        isa_mrk = pm_mrk.run(_build_overlap_concrete(N_QUBITS, REPS, X[0], X[1]))
        mrk_depth = isa_mrk.depth()
        mrk_2q    = sum(1 for g in isa_mrk.data if g.operation.num_qubits == 2)
        mrk_ops   = dict(isa_mrk.count_ops())

        mrk_pairs: set[tuple[int, int]] = set()
        for inst in isa_mrk.data:
            if inst.operation.num_qubits == 2:
                q0 = isa_mrk.find_bit(inst.qubits[0]).index
                q1 = isa_mrk.find_bit(inst.qubits[1]).index
                mrk_pairs.add((min(q0, q1), max(q0, q1)))

        print(f"  ibm_marrakesh ISA depth            : {mrk_depth}")
        print(f"  ibm_marrakesh ISA 2Q count         : {mrk_2q}")
        print(f"  ibm_marrakesh op counts            : {mrk_ops}")
        print(f"  ibm_marrakesh 2Q pairs             : {sorted(mrk_pairs)}")
        print()
        print(f"  ibm_fez       ISA depth            : {isa_depth}")
        print(f"  ibm_fez       ISA 2Q count         : {isa_2q_data}")
        print(f"  ibm_fez       op counts            : {dict(isa_ops)}")
        print(f"  ibm_fez       2Q pairs             : {sorted(two_qubit_pairs_used)}")
        print()

        if isa_depth == mrk_depth and isa_2q_data == mrk_2q:
            print("  VERDICT on depth/CX confound: ISA depth and 2Q count are IDENTICAL on "
                  "both backends. Off-diagonal difference is NOT confounded by circuit depth "
                  "or 2Q count. It is attributable SOLELY to per-gate error rate difference.")
        else:
            print(f"  VERDICT on depth/CX confound: ISA stats DIFFER.")
            print(f"    depth: marrakesh={mrk_depth}, fez={isa_depth}")
            print(f"    2Q:   marrakesh={mrk_2q},    fez={isa_2q_data}")
            print(f"    Off-diagonal difference is CONFOUNDED with circuit depth/2Q count. "
                  f"Cannot attribute solely to gate error rate.")

    except Exception as e:
        import traceback
        print(f"  [ERROR] Could not re-transpile on marrakesh: {e}")
        traceback.print_exc()

    print()
    print("=" * 72)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()

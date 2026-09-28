#!/usr/bin/env python
"""Phase 1.1 — Transpile angle_map reps=1 overlap circuit against ibm_boston.

Zero QPU cost: transpilation is local once the backend coupling map is
fetched. No circuits are submitted.

Circuit: U(x1) U(x2)†  where U(x) = angle_map(x, n_qubits=8, reps=1)
    angle_map layer (×reps):  RY(x[i]) ∀i → CNOT ring → RZ(x[i]) ∀i

Usage:
    source .env.quantum && python hardware/transpile_check.py
"""
from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService

N_QUBITS = 8
REPS = 1
BACKEND = "ibm_boston"


def build_angle_map_overlap(n_qubits: int, reps: int) -> QuantumCircuit:
    """Qiskit equivalent of QuantumKernel._build_overlap_qnode().

    Applies U(x1) then U(x2)† where U encodes via angle_map
    (pipeline/quantum/feature_maps.py::angle_map).

    Forward U(x), one rep:
        RY(x[i]) on qubit i  for i in 0..n-1
        CNOT(i, (i+1)%n)     for i in 0..n-1  [ring]
        RZ(x[i]) on qubit i  for i in 0..n-1

    Adjoint U(x2)†  =  reverse gate sequence, negate angles:
        RZ(-x2[i])  for i in n-1..0
        CNOT(i, (i+1)%n) for i in n-1..0  [ring reversed; CNOT = CNOT†]
        RY(-x2[i])  for i in n-1..0

    k(x1,x2) = P(|0…0>) = probs[0] of the output distribution.
    """
    x1 = ParameterVector("x1", n_qubits)
    x2 = ParameterVector("x2", n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)

    # ── U(x1): angle_map forward ──────────────────────────────────────────────
    for _ in range(reps):
        # RY layer
        for i in range(n_qubits):
            qc.ry(x1[i], i)
        # CNOT ring
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)
        # RZ layer
        for i in range(n_qubits):
            qc.rz(x1[i], i)

    # ── U(x2)†: adjoint of angle_map ─────────────────────────────────────────
    # The full forward sequence is:
    #   [RY(x2) layer, CNOT ring, RZ(x2) layer] × reps
    # Adjoint reverses rep order AND reverses within-rep gate order,
    # taking inverse (negated angle) for each rotation.
    for _ in range(reps):   # reps=1 so this is one pass; kept general
        # adjoint of RZ layer: RZ(-x2[i]) reversed qubit order
        # (same-qubit rotation so adjacent-qubit order doesn't matter)
        for i in range(n_qubits - 1, -1, -1):
            qc.rz(-x2[i], i)
        # adjoint of CNOT ring: reverse gate order (CNOT is self-inverse)
        for i in range(n_qubits - 1, -1, -1):
            qc.cx(i, (i + 1) % n_qubits)
        # adjoint of RY layer: RY(-x2[i]) reversed qubit order
        for i in range(n_qubits - 1, -1, -1):
            qc.ry(-x2[i], i)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def count_two_qubit_gates(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)


def main() -> None:
    print("=" * 60)
    print("TRANSPILE CHECK: angle_map reps=1 overlap circuit")
    print("=" * 60)
    print(f"n_qubits : {N_QUBITS}")
    print(f"reps     : {REPS}")
    print(f"backend  : {BACKEND}")
    print()

    qc = build_angle_map_overlap(N_QUBITS, REPS)

    pre_depth = qc.depth()
    pre_2q    = count_two_qubit_gates(qc)
    pre_total = len(qc.data)
    pre_ops   = dict(qc.count_ops())

    print("Pre-transpile (symbolic parameters):")
    print(f"  depth          : {pre_depth}")
    print(f"  2Q gate count  : {pre_2q}")
    print(f"  total gates    : {pre_total}")
    print(f"  gate breakdown : {pre_ops}")
    print()

    print(f"Connecting to QiskitRuntimeService ...")
    try:
        service = QiskitRuntimeService()
    except Exception:
        print("[FATAL] QiskitRuntimeService() failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    backend = service.backend(BACKEND)
    print(f"Backend          : {backend.name}  ({backend.num_qubits} qubits)")
    print()

    print("Transpiling (optimization_level=1) ...")
    pm  = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa = pm.run(qc)

    isa_depth = isa.depth()
    isa_2q    = count_two_qubit_gates(isa)
    isa_total = len(isa.data)
    isa_ops   = dict(isa.count_ops())

    print("Post-transpile (ISA — ibm_boston basis gates):")
    print(f"  depth          : {isa_depth}")
    print(f"  2Q gate count  : {isa_2q}")
    print(f"  total gates    : {isa_total}")
    print(f"  gate breakdown : {isa_ops}")
    print()

    if isa_2q <= 48:
        print(f"  [OK] 2Q gate count ({isa_2q}) is within typical fidelity range.")
    else:
        print(f"  [WARN] 2Q gate count ({isa_2q}) is high; expect significant noise.")

    print()
    print("No circuits submitted. Transpile check DONE.")
    print("=" * 60)


if __name__ == "__main__":
    main()

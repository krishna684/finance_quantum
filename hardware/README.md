# Hardware Quantum Runs

This directory contains the kernel matrices, metadata, and scripts from running
the quantum kernel circuit on IBM quantum processors. The hardware results appear
in Table 4 of the paper.

---

## Kernel matrices (paper Table 4)

| File | Backend | QPU-s | Off-diag mean | Off-diag var |
|------|---------|-------|---------------|--------------|
| `kernel_matrix_marrakesh_n20.npy` | ibm\_marrakesh | 63 | 0.117 | 0.0148 |
| `kernel_matrix_marrakesh_n20_meta.json` | ibm\_marrakesh | — | metadata | — |
| `kernel_matrix_run2_n20.npy` | ibm\_fez | 65 | 0.109 | 0.0126 |
| `kernel_matrix_run2_n20_meta.json` | ibm\_fez | — | metadata | — |
| `kernel_matrix_sim_n20_reps1.npy` | Simulator, 1 rep | — | 0.162 | 0.033 |
| `kernel_matrix_sim_n20_reps2.npy` | Simulator, 2 reps | — | 0.111 | 0.017 |
| `kernel_matrix_sim_n20_reps3.npy` | Simulator, 3 reps | — | 0.005 | 2.8e-5 |

All matrices are 20×20, computed on the same 20 photos (seed=42, indices stored in
each `_meta.json`). Circuit: angle-map overlap, 8 qubits, 1 repetition, 1024 shots.
ISA depth 106, 40 two-qubit gates (after transpilation to device topology).

**Probe and exploratory files**

| File | Description |
|------|-------------|
| `probe_marrakesh_meta.json` | Pre-run probe on ibm\_marrakesh: ISA depth, 2Q gate count, budget estimate |
| `kernel_matrix_ibm_miami.npy` | Exploratory 2-qubit run on ibm\_miami (128 shots) — not in paper |
| `kernel_matrix_ibm_miami_meta.json` | Metadata for ibm\_miami run |

---

## Scripts

| File | Description |
|------|-------------|
| `batched_matrix.py` | Submits angle-map overlap circuits in batches; produces the paper kernel matrices |
| `kernel_matrix.py` | Single-batch kernel matrix computation (earlier version) |
| `panel5_analysis.py` | Compares hardware and simulator matrices; computes bootstrap CIs |
| `aer_dryrun.py` | Validates circuits on the Aer simulator before hardware submission |
| `transpile_check.py` | Transpiles the feature map circuit and reports ISA depth and 2Q gate count |
| `probe_marrakesh.py` | Probe run to estimate QPU cost before committing to the full matrix |
| `phase_f_run2.py` | Script used for the ibm\_fez (run 2) submission |
| `phase_f_diagnostics.py` | Post-run diagnostics |
| `smoke_bell.py` | Bell-state smoke test to verify hardware connectivity |
| `queue_check.py` | Monitors IBM Quantum job status |
| `inventory.py` | Lists available backends and queue depths |

---

## Hardware runs

| Backend | Run date | QPU-s | Purpose |
|---------|----------|-------|---------|
| ibm\_marrakesh | 2026-08-07 | 63 | Full kernel matrix, 20 photos — paper Table 4 |
| ibm\_fez | 2026-08-08 | 65 | Same circuit, same photos, one day later — paper Table 4 |

Both processors: Heron r2, 8 qubits used, 2Q gate error ~2.8×10⁻³.
Total QPU time: 128 s (63 + 65). IBM list price for Heron r2: $1.60/QPU-s.

---

## Reproducing hardware runs

Requires an IBM Quantum account with access to a Heron r2 processor.

```bash
pip install qiskit qiskit-ibm-runtime

# Validate circuit locally first
python hardware/aer_dryrun.py

# Estimate cost before submitting
python hardware/probe_marrakesh.py

# Submit to hardware (ibm_fez or current equivalent)
python hardware/phase_f_run2.py
```

IBM Quantum backend names change as processors are retired. Use
`hardware/inventory.py` to list currently available backends. The circuit
parameters (8 qubits, reps=1, angle encoding) are set in `batched_matrix.py`.

# Hardware Quantum Runs

This directory contains scripts and results from running quantum kernel circuits
on real IBM quantum hardware via IBM Quantum (formerly IBM Q Experience).

---

## What is here

| File | Description |
|------|-------------|
| `kernel_matrix_ibm_miami.npy` | 20×20 kernel matrix computed on `ibm_miami` |
| `kernel_matrix_ibm_miami_meta.json` | Job metadata: backend, shots, feature map, timing |
| `kernel_matrix_boston_n20.npy` | 20×20 kernel matrix from earlier Boston backend run |
| `kernel_matrix_boston_n20_meta.json` | Job metadata |
| `kernel_matrix_run2_n20.npy` | Second independent run (reproducibility check) |
| `kernel_matrix_run2_n20_meta.json` | Job metadata |
| `probe_marrakesh_meta.json` | Probe run on `ibm_marrakesh`: ISA depth, 2Q gate count, budget |
| `kernel_matrix.py` | Computes a fidelity kernel matrix on hardware via Qiskit |
| `batched_matrix.py` | Batched version for larger sample sets within shot budget |
| `inventory.py` | Lists available backends and their qubit counts / queue depths |
| `queue_check.py` | Monitors job status in the IBM Quantum queue |
| `smoke_bell.py` | Bell-state smoke test — verifies hardware connectivity |
| `transpile_check.py` | Transpiles the feature map circuit and reports ISA depth / 2Q count |
| `aer_dryrun.py` | Aer simulator dry run — validates circuits before submitting to hardware |
| `phase_f_run2.py` | Main Phase F hardware run script |
| `phase_f_diagnostics.py` | Diagnostics for hardware run results |
| `panel5_analysis.py` | Analysis comparing hardware vs simulator kernel matrices |
| `probe_marrakesh.py` | Probe script for `ibm_marrakesh` — estimates cost before full run |

---

## Hardware used

| Backend | Qubits | Run date | Purpose |
|---------|--------|----------|---------|
| `ibm_miami` | 2 (feature_dim=2) | 2026-07-27 | Kernel matrix (20 samples, 128 shots) |
| `ibm_marrakesh` | 8 | 2026-07 | Budget probe (8 qubits, reps=1, 1024 shots) |

---

## Key results

**ibm_miami kernel matrix (20 samples, ZZFeatureMap, reps=2, 128 shots):**
- Off-diagonal mean: 0.347
- Concentration ratio: 0.357
- Wall clock: 691 s; quantum time: 110 s

**ibm_marrakesh probe (8 qubits, reps=1):**
- ISA circuit depth: 106; 2Q gates: 40
- Estimated full matrix cost: ~95 s quantum time (within 600 s budget)

---

## Reproducing hardware runs

Requires an IBM Quantum account with access to the relevant backend.

```bash
# Install Qiskit
pip install qiskit qiskit-ibm-runtime

# Set credentials
export IBMQ_TOKEN=<your_token>

# Dry run on Aer simulator first
python hardware/aer_dryrun.py

# Submit to hardware
python hardware/phase_f_run2.py
```

> Note: IBM Quantum backend names and availability change over time. The `ibm_miami`
> and `ibm_marrakesh` backends were available at time of submission. Use
> `hardware/inventory.py` to list currently available backends.

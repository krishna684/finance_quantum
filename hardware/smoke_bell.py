#!/usr/bin/env python
"""Bell-state smoke test — Part B.

PRE-FLIGHT (local, no quota spend):
    Runs the Bell circuit on qiskit.primitives.StatevectorSampler to validate
    the result-parsing code before touching any QPU.

HARDWARE:
    Submits to service.least_busy() (or --backend <name>) via SamplerV2.
    Reports BOTH wall-clock latency (queue info) and quantum_seconds (quota burn).

Usage:
    source .env.quantum && python hardware/smoke_bell.py [--backend ibm_xxx]
"""
from __future__ import annotations

import argparse
import io
import time
import traceback
import sys
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout on Windows (avoids cp1252 UnicodeEncodeError on em-dashes etc.)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

_LOG_DIR = Path(__file__).resolve().parent / "logs"


# ---------------------------------------------------------------------------
# Bell circuit
# ---------------------------------------------------------------------------

def _build_bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


# ---------------------------------------------------------------------------
# Pre-flight: local StatevectorSampler (no auth, no quota)
# ---------------------------------------------------------------------------

def _preflight(qc: QuantumCircuit) -> None:
    print("=== PRE-FLIGHT (local StatevectorSampler, no QPU spend) ===")
    sampler = StatevectorSampler()
    job = sampler.run([qc], shots=1024)
    result = job.result()
    counts = result[0].data.meas.get_counts()
    print(f"  Local counts: {counts}")

    total = sum(counts.values())
    frac_00 = counts.get("00", 0) / total
    frac_11 = counts.get("11", 0) / total
    leakage = sum(v for k, v in counts.items() if k not in ("00", "11")) / total

    print(f"  |00>: {frac_00:.3f}  |11>: {frac_11:.3f}  leakage: {leakage:.4f}")
    assert frac_00 > 0.45 and frac_11 > 0.45, (
        f"[PREFLIGHT FAILED] Expected ~50/50 on 00/11, got {counts}"
    )
    print("  Pre-flight PASSED — result parsing validated.\n")


# ---------------------------------------------------------------------------
# Hardware submission
# ---------------------------------------------------------------------------

def _hardware_run(backend_name: str | None, log_path: Path) -> None:
    print("=== HARDWARE SUBMISSION ===")

    print("Connecting: QiskitRuntimeService() ...")
    try:
        service = QiskitRuntimeService()
    except Exception:
        print("[FATAL] QiskitRuntimeService() failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    if backend_name:
        backend = service.backend(backend_name)
        print(f"Using requested backend: {backend.name}")
    else:
        try:
            backend = service.least_busy(operational=True, simulator=False)
            print(f"least_busy backend: {backend.name}")
        except Exception:
            print("[FATAL] service.least_busy() failed:", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)

    # Build and transpile
    qc = _build_bell()
    print(f"Transpiling for {backend.name} (optimization_level=1) ...")
    try:
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        isa = pm.run(qc)
    except Exception:
        print("[FATAL] Transpilation failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    print(f"  ISA depth: {isa.depth()}  ops: {isa.count_ops()}")

    # Submit
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = 1024
    sampler.options.max_execution_time = 60   # hard QPU-time cap (seconds)

    print("Submitting job ...")
    try:
        t0 = time.time()
        job = sampler.run([isa])
        job_id = job.job_id()
        print(f"  job_id: {job_id}")
        print("  Waiting for result (this is the queue latency) ...")
        result = job.result()
        wall_clock_s = time.time() - t0
    except Exception:
        print("[FATAL] Job submission or retrieval failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # Parse result
    counts = result[0].data.meas.get_counts()
    total = sum(counts.values())

    # Quantum seconds (actual quota burn)
    try:
        quantum_seconds = job.metrics()["usage"]["quantum_seconds"]
    except Exception as e:
        quantum_seconds = f"unavailable ({e})"

    # Report
    print()
    print(f"  Wall-clock (submit->result) : {wall_clock_s:.1f}s  [queue latency — NOT quota]")
    print(f"  quantum_seconds used       : {quantum_seconds}  [THIS is the quota burn]")
    print(f"  Counts: {counts}")

    frac_00 = counts.get("00", 0) / total
    frac_11 = counts.get("11", 0) / total
    leakage = sum(v for k, v in counts.items() if k not in ("00", "11")) / total
    print(f"  |00>: {frac_00:.3f}  |11>: {frac_11:.3f}  leakage: {leakage:.4f}")
    if frac_00 > 0.40 and frac_11 > 0.40:
        print("  Sanity check PASSED (≥40% each on 00/11)")
    else:
        print(f"  [WARN] Unexpected distribution — check noise / backend health")

    # Append to log
    ts = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== BELL SMOKE JOB — {ts} ===\n")
        f.write(f"backend         : {backend.name}\n")
        f.write(f"job_id          : {job_id}\n")
        f.write(f"wall_clock_s    : {wall_clock_s:.1f}\n")
        f.write(f"quantum_seconds : {quantum_seconds}\n")
        f.write(f"counts          : {counts}\n")
        f.write(f"|00|: {frac_00:.3f}  |11|: {frac_11:.3f}  leakage: {leakage:.4f}\n")
    print(f"\nAppended to log -> {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Bell-state smoke test")
    parser.add_argument("--backend", default=None,
                        help="Backend name (default: service.least_busy)")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Run only the local pre-flight check, no QPU submit")
    args = parser.parse_args()

    qc = _build_bell()
    _preflight(qc)

    if args.preflight_only:
        print("--preflight-only set. Stopping before hardware submit.")
        return

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    log_path = _LOG_DIR / f"inventory_{ts_str}.txt"   # append to same day's log
    _hardware_run(args.backend, log_path)


if __name__ == "__main__":
    main()

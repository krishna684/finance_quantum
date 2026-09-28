#!/usr/bin/env python
"""IBM Quantum backend inventory — Part A.

Reads credentials exclusively from environment variables (never from disk or
code).  Set these before running (e.g. `source .env.quantum`):

    QISKIT_IBM_TOKEN    — 44-char API key   (NEVER printed or logged)
    QISKIT_IBM_INSTANCE — instance CRN
    QISKIT_IBM_CHANNEL  — ibm_quantum_platform

Usage:
    source .env.quantum && python hardware/inventory.py
"""
from __future__ import annotations

import io
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import qiskit
import qiskit_ibm_runtime
from qiskit_ibm_runtime import QiskitRuntimeService

_LOG_DIR = Path(__file__).resolve().parent / "logs"

# ---------------------------------------------------------------------------
# Backend info collector
# ---------------------------------------------------------------------------

def _collect_backend_info(backend) -> dict:
    info: dict = dict(
        name=backend.name,
        num_qubits=None,
        operational=None,
        pending_jobs=None,
        median_2q_gate_error=None,
        median_readout_error=None,
        median_t1_us=None,
        median_t2_us=None,
    )

    try:
        info["num_qubits"] = backend.num_qubits
    except Exception as e:
        info["num_qubits"] = f"ERR:{e}"

    try:
        st = backend.status()
        info["operational"] = st.operational
        info["pending_jobs"] = st.pending_jobs
    except Exception as e:
        info["operational"] = f"ERR:{e}"
        info["pending_jobs"] = "N/A"

    try:
        props = backend.properties()
        if props is None:
            return info

        n = backend.num_qubits if isinstance(backend.num_qubits, int) else 0
        ro, t1s, t2s = [], [], []

        for q in range(n):
            try:
                ro.append(props.readout_error(q))
            except Exception:
                pass
            try:
                v = props.t1(q)
                if v is not None:
                    t1s.append(v * 1e6)
            except Exception:
                pass
            try:
                v = props.t2(q)
                if v is not None:
                    t2s.append(v * 1e6)
            except Exception:
                pass

        if ro:
            info["median_readout_error"] = statistics.median(ro)
        if t1s:
            info["median_t1_us"] = statistics.median(t1s)
        if t2s:
            info["median_t2_us"] = statistics.median(t2s)

        gate_errs_2q = []
        for gp in props.gates:
            if gp.gate in ("cx", "ecr", "cz", "rzx") and len(gp.qubits) == 2:
                for p in gp.parameters:
                    if p.name == "gate_error" and p.value is not None:
                        gate_errs_2q.append(p.value)
        if gate_errs_2q:
            info["median_2q_gate_error"] = statistics.median(gate_errs_2q)

    except Exception as e:
        info["median_2q_gate_error"] = f"ERR:{e}"

    return info


def _fmt(val, spec=".4f") -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return format(val, spec)
    return str(val)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc)
    log_path = _LOG_DIR / f"inventory_{ts.strftime('%Y%m%d')}.txt"

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    def flush_log() -> None:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    emit("=== IBM QUANTUM BACKEND INVENTORY ===")
    emit(f"Timestamp          : {ts.isoformat()}")
    emit(f"qiskit             : {qiskit.__version__}")
    emit(f"qiskit-ibm-runtime : {qiskit_ibm_runtime.__version__}")
    emit()

    # ------------------------------------------------------------------
    # Auth — reads QISKIT_IBM_TOKEN / QISKIT_IBM_INSTANCE / QISKIT_IBM_CHANNEL
    # from env; NEVER touches the token value itself.
    # ------------------------------------------------------------------
    print("Connecting: QiskitRuntimeService() ...")
    try:
        service = QiskitRuntimeService()
    except Exception:
        print("[FATAL] QiskitRuntimeService() raised:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    emit("Auth               : OK")
    emit()

    # ------------------------------------------------------------------
    # Account info — redact any key that could be a credential
    # ------------------------------------------------------------------
    _REDACT = {"token", "key", "secret", "password", "pass"}
    try:
        acct = service.active_account() or {}
        emit("Active account (credentials redacted):")
        for k, v in acct.items():
            if any(r in k.lower() for r in _REDACT):
                continue          # never log the token
            emit(f"  {k}: {v}")
    except Exception as e:
        emit(f"active_account() unavailable: {e}")
    emit()

    # ------------------------------------------------------------------
    # Plan / instances
    # ------------------------------------------------------------------
    try:
        instances = service.instances()
        emit(f"Accessible instances: {instances}")
    except Exception as e:
        emit(f"service.instances() unavailable: {e}")
    emit("Plan type / remaining QPU minutes: check the IBM Quantum dashboard.")
    emit("  Open Plan = 10 min QPU per 28-day rolling window (~5 min assumed per cycle).")
    emit()

    # ------------------------------------------------------------------
    # Real backends
    # ------------------------------------------------------------------
    try:
        real_backends = service.backends(simulator=False)
    except Exception:
        print("[FATAL] service.backends(simulator=False) raised:", file=sys.stderr)
        traceback.print_exc()
        flush_log()
        sys.exit(1)

    emit(f"Real (QPU) backends: {len(real_backends)}")
    emit()

    # Least busy
    try:
        lb = service.least_busy(operational=True, simulator=False)
        emit(f"least_busy         : {lb.name}")
    except Exception as e:
        emit(f"least_busy         : ERR — {e}")
    emit()

    # ------------------------------------------------------------------
    # Per-backend table
    # ------------------------------------------------------------------
    hdr = (
        f"{'Name':<32} {'Qubits':>6} {'Oper':>5} {'Queue':>6} "
        f"{'2Q-err':>10} {'RO-err':>10} {'T1(µs)':>10} {'T2(µs)':>10}"
    )
    emit("=== REAL BACKEND DETAILS ===")
    emit(hdr)
    emit("-" * len(hdr))

    for b in sorted(real_backends, key=lambda x: x.name):
        info = _collect_backend_info(b)
        emit(
            f"{info['name']:<32} "
            f"{_fmt(info['num_qubits'], 'd'):>6} "
            f"{str(info['operational']):>5} "
            f"{_fmt(info['pending_jobs'], 'd'):>6} "
            f"{_fmt(info['median_2q_gate_error'], '.2e'):>10} "
            f"{_fmt(info['median_readout_error'], '.2e'):>10} "
            f"{_fmt(info['median_t1_us'], '.1f'):>10} "
            f"{_fmt(info['median_t2_us'], '.1f'):>10}"
        )

    emit()
    emit("=== END OF INVENTORY ===")

    flush_log()
    print(f"\nLog -> {log_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Phase 1.5 — Queue check: ibm_boston status + comparable backends.

Zero QPU spend — read-only status queries only, no job submission.

Reports:
    1. ibm_boston pending_jobs + operational status
    2. All real backends sorted by pending_jobs (least busy first)
    3. Proceed vs. submit-and-wait recommendation

Usage:
    source .env.quantum && python hardware/queue_check.py
"""
from __future__ import annotations

import io
import sys
import traceback
from datetime import datetime, timezone

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

from qiskit_ibm_runtime import QiskitRuntimeService

TARGET_BACKEND = "ibm_boston"

# Rough throughput estimate for recommendation threshold.
# From inventory log 2026-07-26: bell smoke job = 6 QPU-s → suggests fast execution
# but queue depth is the wall-clock threat. Use 50 pending jobs as "short" threshold.
_QUEUE_SHORT_THRESHOLD = 50   # pending jobs; below this → proceed normally


def main() -> None:
    ts = datetime.now(timezone.utc)
    print("=" * 64)
    print("PHASE 1.5 — QUEUE CHECK")
    print("=" * 64)
    print(f"Timestamp : {ts.isoformat()}")
    print()

    print("Connecting: QiskitRuntimeService() ...")
    try:
        service = QiskitRuntimeService()
    except Exception:
        print("[FATAL] QiskitRuntimeService() failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    print("Auth      : OK")
    print()

    # ── fetch all real backends ───────────────────────────────────────────────
    try:
        real_backends = service.backends(simulator=False)
    except Exception:
        print("[FATAL] service.backends(simulator=False) failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # Collect status for all backends
    backend_info: list[dict] = []
    for b in real_backends:
        info = {"name": b.name, "num_qubits": None, "operational": None, "pending_jobs": None}
        try:
            info["num_qubits"] = b.num_qubits
        except Exception:
            pass
        try:
            st = b.status()
            info["operational"] = st.operational
            info["pending_jobs"] = st.pending_jobs
        except Exception as e:
            info["operational"] = f"ERR"
            info["pending_jobs"] = None
        backend_info.append(info)

    # Sort by pending_jobs (None → infinity for sorting)
    backend_info.sort(key=lambda x: x["pending_jobs"] if x["pending_jobs"] is not None else 999999)

    # ── ibm_boston detail ─────────────────────────────────────────────────────
    boston = next((b for b in backend_info if b["name"] == TARGET_BACKEND), None)
    if boston is None:
        print(f"[FATAL] {TARGET_BACKEND} not found in accessible backends", file=sys.stderr)
        sys.exit(1)

    print(f"TARGET: {TARGET_BACKEND}")
    print(f"  operational  : {boston['operational']}")
    print(f"  num_qubits   : {boston['num_qubits']}")
    print(f"  pending_jobs : {boston['pending_jobs']}")
    print()

    # ── all backends table ────────────────────────────────────────────────────
    print("ALL REAL BACKENDS (sorted by queue depth):")
    print(f"  {'Name':<28} {'Qubits':>6} {'Oper':>5} {'Pending':>8}")
    print(f"  {'-'*52}")
    for b in backend_info:
        pj = str(b["pending_jobs"]) if b["pending_jobs"] is not None else "N/A"
        print(
            f"  {b['name']:<28} "
            f"{str(b['num_qubits'] or 'N/A'):>6} "
            f"{str(b['operational']):>5} "
            f"{pj:>8}"
        )
    print()

    # ── least busy ───────────────────────────────────────────────────────────
    try:
        lb = service.least_busy(operational=True, simulator=False)
        lb_pending = next(
            (b["pending_jobs"] for b in backend_info if b["name"] == lb.name), "?"
        )
        print(f"least_busy : {lb.name}  (pending_jobs={lb_pending})")
        if lb.name != TARGET_BACKEND:
            print(
                f"  NOTE: {lb.name} is least busy but Panel 5's hardware claim is\n"
                f"        specifically about ibm_boston's noise profile.\n"
                f"        Do NOT switch targets without explicit user approval.\n"
                f"        If ibm_boston is operationally blocked (days-long queue),\n"
                f"        raise it for user decision before doing anything else."
            )
    except Exception as e:
        print(f"least_busy : ERR — {e}")
    print()

    # ── time-to-front estimate ────────────────────────────────────────────────
    # IBM Quantum does not expose an API-level time-to-front estimate.
    # We infer qualitatively from pending_jobs.
    # A 10-circuit probe job executes in roughly 1-30 QPU-seconds depending on
    # circuit depth. With O(N) pending jobs, wall-clock wait ~ N × avg_job_duration.
    # We cannot know avg_job_duration, so we flag thresholds:
    boston_pj = boston["pending_jobs"]
    print("TIME-TO-FRONT ESTIMATE:")
    print(f"  IBM API does not expose a direct time-to-front value.")
    print(f"  ibm_boston pending_jobs = {boston_pj}")
    if boston_pj is not None:
        if boston_pj < 10:
            ttf_note = "Very short queue — probe should reach the front within minutes."
        elif boston_pj < _QUEUE_SHORT_THRESHOLD:
            ttf_note = f"Short queue (< {_QUEUE_SHORT_THRESHOLD}) — probe likely reaches front within 30-60 min."
        elif boston_pj < 200:
            ttf_note = "Moderate queue — expect several hours of wall-clock wait."
        elif boston_pj < 400:
            ttf_note = "Deep queue — wall-clock wait likely 6-12+ hours. High deadline risk."
        else:
            ttf_note = "Very deep queue (>= 400) — wall-clock wait is a serious deadline threat."
        print(f"  Assessment: {ttf_note}")
    print()

    # ── recommendation ────────────────────────────────────────────────────────
    print("RECOMMENDATION:")
    if boston_pj is None:
        print(
            "  CANNOT ASSESS — pending_jobs unavailable.\n"
            "  Check IBM Quantum dashboard manually before proceeding."
        )
    elif boston_pj < _QUEUE_SHORT_THRESHOLD:
        print(
            f"  (a) PROCEED NORMALLY.\n"
            f"      Queue is short ({boston_pj} pending). Submit the 10-circuit probe\n"
            f"      now and wait for it to complete before Phase 3."
        )
    else:
        print(
            f"  (b) SUBMIT PROBE NOW AND CONTINUE ZERO-QPU WORK.\n"
            f"      Queue is deep ({boston_pj} pending). The probe job bills 0 s\n"
            f"      while queued (only bills on execution). Submit it into the queue\n"
            f"      immediately so it accrues queue time for free, and continue\n"
            f"      any remaining zero-QPU Phase 1/1.5 work in parallel.\n"
            f"      The probe result (measured spc) will be waiting when the queue\n"
            f"      clears. Do NOT submit the main Session until the probe returns."
        )
    print()
    print("=" * 64)


if __name__ == "__main__":
    main()

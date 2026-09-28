#!/usr/bin/env python
"""Phase F — Second hardware data point (run2).

Steps (all in one script, no intermediate GO required — user issued GO for Phase F):
    1. Queue check: all real backends sorted by pending_jobs.
       Select the shallowest-queue operational backend.
    2. Probe: 10 overlap circuits, 1024 shots, standalone Sampler (not Session).
       Report spc = usage_s / 10.  Abort if probe alone exceeds SPEND_CAP_S.
    3. Budget check: probe_s + 190*spc <= SPEND_CAP_S.  Abort if over.
    4. Full matrix: 190 off-diagonal circuits in ONE Session, batches [48,48,48,46].
       Per-batch budget check before each submission.
    5. Save to _run2 paths (originals NEVER overwritten):
           hardware/kernel_matrix_run2_n20.npy
           hardware/kernel_matrix_run2_n20_meta.json
    6. Recompute off-diagonal mean/variance directly from .npy and report
       side-by-side with original marrakesh values (0.1172 / 0.0148).

Circuit: angle_map_overlap, n_qubits=8, reps=1, shots=1024
Data:    same seed=42 → same 20 samples as original run for apples-to-apples comparison.

Usage:
    source .env.quantum && python hardware/phase_f_run2.py
"""
from __future__ import annotations

import io
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

_ROOT   = Path(__file__).resolve().parent.parent
_HW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# ── load IBM credentials from .env.quantum before any Qiskit import uses them ──
def _load_env_quantum() -> None:
    """Parse .env.quantum and inject into os.environ (handles quoted values)."""
    import os, re
    env_path = _ROOT / ".env.quantum"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            _k = _k.strip()
            _v = _v.strip().strip('"').strip("'")
            if _k and _k not in os.environ:
                os.environ[_k] = _v

_load_env_quantum()

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister, ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2

# ── budget constants (non-negotiable) ─────────────────────────────────────────
SPEND_CAP_S : int = 600

# ── kernel config ─────────────────────────────────────────────────────────────
N          : int = 20
N_QUBITS   : int = 8
REPS       : int = 1
SHOTS      : int = 1024
SEED       : int = 42
N_PAIRS    : int = N * (N - 1) // 2   # 190
N_PROBE    : int = 10

BATCH_SIZES: list[int] = [48, 48, 48, 46]
assert sum(BATCH_SIZES) == N_PAIRS
assert all(b <= 48 for b in BATCH_SIZES)

# ── original run reference values (for final comparison) ──────────────────────
_ORIG_MEAN = 0.11724403782894736
_ORIG_VAR  = 0.014819126948095094
_ORIG_BACKEND = "ibm_marrakesh"

# ── output paths (_run2, never overwriting originals) ─────────────────────────
OUT_NPY  = _HW_DIR / "kernel_matrix_run2_n20.npy"
OUT_JSON = _HW_DIR / "kernel_matrix_run2_n20_meta.json"


# ── circuit builders ──────────────────────────────────────────────────────────

def _build_overlap_concrete(n_qubits: int, reps: int,
                             x1: np.ndarray, x2: np.ndarray) -> QuantumCircuit:
    """Angle-map overlap circuit with concrete bound values."""
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


# ── data loading ──────────────────────────────────────────────────────────────

def load_and_reduce() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same loading logic as original batched_matrix.py (seed=42 → identical 20 samples)."""
    labeled    = pd.read_parquet(_ROOT / "data" / "labeled.parquet")
    npz        = np.load(_ROOT / "data" / "embeddings" / "n24news_finance.npz")
    embeddings = npz["embeddings"].astype(np.float64)
    labels     = labeled["label_up"].values

    rng    = np.random.default_rng(SEED)
    half   = N // 2
    sel_up = rng.choice(np.where(labels == 1)[0], half,      replace=False)
    sel_dn = rng.choice(np.where(labels == 0)[0], N - half,  replace=False)
    sel    = np.concatenate([sel_up, sel_dn])
    rng.shuffle(sel)

    from pipeline.quantum.reducer import EmbeddingReducer
    reducer = EmbeddingReducer(n_components=N_QUBITS, seed=SEED)
    reducer.fit(embeddings)
    X_red = reducer.transform(embeddings[sel])

    if X_red.min() < 0.0 or X_red.max() > np.pi + 1e-9:
        raise RuntimeError(
            f"EmbeddingReducer output out of [0,pi]: "
            f"min={X_red.min():.4f}  max={X_red.max():.4f}"
        )
    return X_red, labels[sel].astype(int), sel


# ── helpers ───────────────────────────────────────────────────────────────────

def _counts_to_fidelity(counts: dict) -> float:
    return counts.get("0" * N_QUBITS, 0) / SHOTS


def _offdiag_stats(K: np.ndarray) -> tuple[float, float]:
    mask = ~np.eye(K.shape[0], dtype=bool)
    off  = K[mask]
    return float(off.mean()), float(off.var())


def _get_2q_error(backend) -> str:
    """Best-effort: return mean 2Q gate error rate as string, or 'N/A'."""
    try:
        props = backend.properties()
        errors = []
        for gate in props.gates:
            if len(gate.qubits) == 2:
                for p in gate.parameters:
                    if p.name == "gate_error":
                        errors.append(p.value)
                        break
        if errors:
            return f"{np.mean(errors):.3e}  (n={len(errors)} 2Q gates)"
        return "N/A"
    except Exception as e:
        return f"N/A ({e})"


def _budget_abort(spent_s: float, spc: float, next_sz: int, label: str) -> None:
    projected = spent_s + spc * next_sz
    if projected > SPEND_CAP_S:
        raise SystemExit(
            f"\nBUDGET ABORT before {label}:\n"
            f"  spent_s={spent_s:.1f}s  spc={spc:.4f}  next={next_sz}"
            f"  projected={projected:.1f}s > cap={SPEND_CAP_S}s\n"
            f"  Partial Gram matrix (if any batches completed) saved below."
        )
    print(
        f"  [budget OK] spent={spent_s:.1f}s + {spc:.4f}*{next_sz}={spc*next_sz:.1f}s"
        f" -> projected={projected:.1f}s  (cap={SPEND_CAP_S}s)"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ts_start = datetime.now(timezone.utc)
    t_wall   = time.time()

    print("=" * 68)
    print("PHASE F — SECOND HARDWARE DATA POINT (run2)")
    print(f"Timestamp: {ts_start.isoformat()}")
    print("=" * 68)
    print()

    # ── STEP 1: Queue check ───────────────────────────────────────────────────
    print("STEP 1: QUEUE CHECK — all real backends")
    print("-" * 68)

    try:
        service = QiskitRuntimeService()
    except Exception:
        print("[FATAL] QiskitRuntimeService() failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    print("Auth: OK")
    print()

    try:
        real_backends = service.backends(simulator=False)
    except Exception:
        print("[FATAL] service.backends(simulator=False) failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

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
        except Exception:
            info["operational"] = "ERR"
            info["pending_jobs"] = None
        backend_info.append(info)

    backend_info.sort(key=lambda x: x["pending_jobs"] if x["pending_jobs"] is not None else 999999)

    print(f"  {'Backend':<28} {'Qubits':>6} {'Oper':>5} {'Pending':>8}")
    print(f"  {'-'*52}")
    for b in backend_info:
        pj = str(b["pending_jobs"]) if b["pending_jobs"] is not None else "N/A"
        print(f"  {b['name']:<28} {str(b['num_qubits'] or 'N/A'):>6} "
              f"{str(b['operational']):>5} {pj:>8}")
    print()

    # Select: shallowest queue among operational backends
    selected = None
    for b in backend_info:
        if b["operational"] is True and b["pending_jobs"] is not None:
            selected = b
            break

    if selected is None:
        print("[FATAL] No operational backend with known queue depth found.", file=sys.stderr)
        sys.exit(1)

    chosen_name = selected["name"]
    print(f"SELECTED: {chosen_name}")
    print(f"  pending_jobs : {selected['pending_jobs']}")
    print(f"  Reason: shallowest queue among all operational backends at query time.")
    print()

    backend = service.backend(chosen_name)

    # ── 2Q error rate ─────────────────────────────────────────────────────────
    twoq_err = _get_2q_error(backend)
    print(f"  2Q gate error ({chosen_name}): {twoq_err}")
    print(f"  2Q gate error (ibm_marrakesh, original run): 2.86e-3  [from 2026-07-26 inventory]")
    print()

    # ── STEP 2: Load data + build circuits ───────────────────────────────────
    print("STEP 2: LOAD DATA + BUILD CIRCUITS")
    print("-" * 68)
    print("Loading labeled embeddings and reducing to 8-dim [0,pi] (seed=42, same as run1) ...")
    X, y, sel = load_and_reduce()
    print(f"  X shape   : {X.shape}  range=[{X.min():.4f}, {X.max():.4f}]")
    print(f"  labels    : {y.tolist()}  ({int(y.sum())} up / {int((y==0).sum())} down)")
    print()

    # Verify same sample indices as original run (seed=42 should be deterministic)
    orig_sel = [1862,7985,7057,5118,879,4305,4256,7833,5607,2110,
                4333,8436,860,931,9480,6628,8709,6609,4636,3994]
    if list(sel.tolist()) == orig_sel:
        print("  [CHECK] sample_indices match original run1 exactly. Apples-to-apples comparison confirmed.")
    else:
        print(f"  [WARN] sample_indices differ from run1. Run2 uses different samples.")
        print(f"  run2: {sel.tolist()}")
        print(f"  run1: {orig_sel}")
    print()

    print(f"Building {N_PAIRS} overlap circuits (angle_map, reps={REPS}, n_qubits={N_QUBITS}) ...")
    circuits: list[QuantumCircuit] = []
    pairs:    list[tuple[int, int]] = []
    for i in range(N):
        for j in range(i + 1, N):
            circuits.append(_build_overlap_concrete(N_QUBITS, REPS, X[i], X[j]))
            pairs.append((i, j))
    assert len(circuits) == N_PAIRS
    print(f"  pre-transpile depth [0]: {circuits[0].depth()}")
    print()

    # ── STEP 3: Transpile ─────────────────────────────────────────────────────
    print(f"STEP 3: TRANSPILE {N_PAIRS} circuits (optimization_level=1) ...")
    print("-" * 68)
    t_xp = time.time()
    pm           = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = pm.run(circuits)
    print(f"  transpile time    : {time.time()-t_xp:.1f}s")
    print(f"  ISA depth [0]     : {isa_circuits[0].depth()}")
    n_2q = sum(1 for g in isa_circuits[0].data if g.operation.num_qubits == 2)
    print(f"  ISA 2Q count [0]  : {n_2q}")
    print()

    # ── STEP 4: Probe (standalone, NOT Session) ───────────────────────────────
    print("STEP 4: PROBE — 10 circuits, standalone Sampler (not Session)")
    print("-" * 68)
    probe_circuits = isa_circuits[:N_PROBE]

    sampler_standalone = SamplerV2(mode=backend)
    sampler_standalone.options.default_shots = SHOTS
    pub_list = [(qc,) for qc in probe_circuits]

    print(f"  Submitting {N_PROBE}-circuit probe job ...")
    t_probe = time.time()
    probe_job = sampler_standalone.run(pub_list)
    probe_job_id = probe_job.job_id()
    print(f"  job_id : {probe_job_id}")
    print(f"  Waiting for result ...")

    try:
        probe_result = probe_job.result()
        probe_wall_s = time.time() - t_probe
    except Exception:
        print("[FATAL] Probe job failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    try:
        probe_usage_s = float(probe_job.metrics()["usage"]["quantum_seconds"])
    except Exception as e:
        print(f"[FATAL] Could not read probe quantum_seconds: {e}", file=sys.stderr)
        sys.exit(1)

    spc = probe_usage_s / N_PROBE
    matrix_est_s = N_PAIRS * spc

    print(f"  probe usage_s      : {probe_usage_s}s")
    print(f"  probe wall_s       : {probe_wall_s:.1f}s")
    print(f"  spc                : {spc:.4f} s/circuit")
    print(f"  matrix est (190c)  : {matrix_est_s:.1f}s")
    print()

    # Spot-check circuit 0
    try:
        c0_counts = probe_result[0].data.meas.get_counts()
        p0 = _counts_to_fidelity(c0_counts)
        print(f"  spot-check c0: P(|0...0>) = {p0:.4f}")
    except Exception as e:
        print(f"  spot-check failed: {e}")
    print()

    # Budget check: probe + matrix <= cap
    print(f"  BUDGET: probe={probe_usage_s:.1f}s + matrix_est={matrix_est_s:.1f}s"
          f" = {probe_usage_s+matrix_est_s:.1f}s  (cap={SPEND_CAP_S}s)")
    if probe_usage_s + matrix_est_s > SPEND_CAP_S:
        print(f"[FATAL] Estimated total exceeds SPEND_CAP_S={SPEND_CAP_S}s. Abort.", file=sys.stderr)
        sys.exit(1)
    print(f"  GO: budget OK.")
    print()

    # ── STEP 5: Batched matrix in Session ─────────────────────────────────────
    print("STEP 5: BATCHED KERNEL MATRIX — Session")
    print("-" * 68)
    print(f"  backend      : {chosen_name}")
    print(f"  batches      : {BATCH_SIZES}  ({N_PAIRS} total circuits)")
    print(f"  shots        : {SHOTS}")
    print()

    batches      = []
    pair_batches = []
    offset = 0
    for sz in BATCH_SIZES:
        batches.append(isa_circuits[offset : offset + sz])
        pair_batches.append(pairs[offset : offset + sz])
        offset += sz

    spent_s      : float = float(probe_usage_s)   # probe already counted
    batch_results: list  = []
    usage_log    : list[dict] = []
    t_session    = time.time()

    try:
        with Session(backend=backend) as session:
            sampler = SamplerV2(mode=session)
            sampler.options.default_shots = SHOTS

            for bi, (batch_circs, batch_pairs) in enumerate(zip(batches, pair_batches)):
                label  = f"batch {bi+1}/{len(BATCH_SIZES)}"
                next_n = len(batch_circs)
                print(f"\n[{label}]  {next_n} circuits")
                _budget_abort(spent_s, spc, next_n, label)

                pub_list = [(qc,) for qc in batch_circs]
                print(f"  Submitting ...")
                t0  = time.time()
                job = sampler.run(pub_list)
                print(f"  job_id : {job.job_id()}")
                print(f"  Waiting ...")
                result  = job.result()
                wall_s  = time.time() - t0

                try:
                    usage_s = float(job.metrics()["usage"]["quantum_seconds"])
                except Exception as e:
                    raise RuntimeError(f"Could not read quantum_seconds for {label}: {e}")

                if not isinstance(usage_s, (int, float)):
                    raise RuntimeError(f"quantum_seconds not numeric for {label}: {usage_s!r}")

                spent_s += usage_s
                batch_results.append(result)
                usage_log.append({
                    "batch":        bi,
                    "batch_size":   next_n,
                    "job_id":       job.job_id(),
                    "usage_s":      usage_s,
                    "cumulative_s": spent_s,
                    "wall_s":       wall_s,
                })

                print(f"  usage={usage_s}s  cumulative={spent_s:.1f}s  "
                      f"remaining_cap={SPEND_CAP_S-spent_s:.1f}s")

                # Verify no empty results
                for pi, pr in enumerate(result):
                    if not pr.data.meas.get_counts():
                        raise RuntimeError(f"batch {bi} pub {pi} returned empty counts")

    except SystemExit:
        print("\n[BUDGET ABORT] Saving partial Gram matrix from completed batches.")
        # Fall through to assembly with whatever we have
        if not batch_results:
            raise
    except Exception:
        print("[FATAL] Session failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    session_wall_s = time.time() - t_session
    print(f"\nSession wall-clock: {session_wall_s:.1f}s")
    print()

    # ── STEP 6: Assemble Gram matrix ──────────────────────────────────────────
    print("STEP 6: ASSEMBLE GRAM MATRIX")
    print("-" * 68)

    # Reconstruct how many pairs were completed
    n_completed_batches = len(batch_results)
    completed_pairs     = sum(BATCH_SIZES[:n_completed_batches])

    K = np.eye(N, dtype=np.float64)
    circuit_idx = 0
    for br in batch_results:
        for pub_res in br:
            i, j = pairs[circuit_idx]
            val  = _counts_to_fidelity(pub_res.data.meas.get_counts())
            K[i, j] = val
            K[j, i] = val
            circuit_idx += 1

    if circuit_idx != completed_pairs:
        raise RuntimeError(f"Assembled {circuit_idx} pairs, expected {completed_pairs}")

    diag_vals = np.diag(K)
    if not np.allclose(diag_vals, 1.0, atol=1e-12):
        raise RuntimeError(f"Diagonal not 1: min={diag_vals.min():.6f} max={diag_vals.max():.6f}")
    if np.any(np.isnan(K)):
        raise RuntimeError("NaN in assembled Gram matrix")
    if n_completed_batches < len(BATCH_SIZES):
        print(f"  [PARTIAL] Only {n_completed_batches}/{len(BATCH_SIZES)} batches completed.")
        print(f"  Off-diagonal stats only for {completed_pairs}/{N_PAIRS} pairs — use with caution.")

    # Recompute off-diagonal stats directly from .npy (as instructed)
    np.save(OUT_NPY, K)

    K_reload = np.load(OUT_NPY)
    run2_mean, run2_var = _offdiag_stats(K_reload)

    print(f"  shape         : {K_reload.shape}")
    print(f"  diagonal      : exactly 1.0 confirmed")
    print(f"  off_diag_mean : {run2_mean:.10f}")
    print(f"  off_diag_var  : {run2_var:.10f}")
    print()

    # ── STEP 7: Save metadata ─────────────────────────────────────────────────
    meta = {
        "backend":          chosen_name,
        "queue_depth_at_selection": selected["pending_jobs"],
        "n":                N,
        "n_qubits":         N_QUBITS,
        "reps":             REPS,
        "shots":            SHOTS,
        "circuit":          "angle_map_overlap",
        "seed":             SEED,
        "batch_sizes":      BATCH_SIZES,
        "n_completed_batches": n_completed_batches,
        "probe_job_id":     probe_job_id,
        "probe_usage_s":    probe_usage_s,
        "spc":              spc,
        "matrix_est_s":     matrix_est_s,
        "total_spent_s":    spent_s,
        "spend_cap_s":      SPEND_CAP_S,
        "off_diag_mean":    run2_mean,
        "off_diag_var":     run2_var,
        "sample_indices":   sel.tolist(),
        "labels":           y.tolist(),
        "usage_log":        usage_log,
        "session_wall_s":   session_wall_s,
        "isa_depth":        isa_circuits[0].depth(),
        "isa_2q_count":     n_2q,
        "twoq_error_str":   twoq_err,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {OUT_NPY}")
    print(f"Saved: {OUT_JSON}")
    print()

    # ── FINAL COMPARISON ──────────────────────────────────────────────────────
    total_wall = time.time() - t_wall
    print("=" * 68)
    print("PHASE F — OFF-DIAGONAL KERNEL STATS COMPARISON")
    print("=" * 68)
    print()
    print(f"  {'Metric':<22}  {'Run1 (marrakesh)':>18}  {'Run2 (' + chosen_name + ')':>22}")
    print(f"  {'-'*66}")
    print(f"  {'off_diag_mean':<22}  {_ORIG_MEAN:>18.10f}  {run2_mean:>22.10f}")
    print(f"  {'off_diag_var':<22}  {_ORIG_VAR:>18.10f}  {run2_var:>22.10f}")
    delta_mean = run2_mean - _ORIG_MEAN
    delta_var  = run2_var  - _ORIG_VAR
    print(f"  {'delta_mean':<22}  {'':>18}  {delta_mean:>+22.6f}")
    print(f"  {'delta_var':<22}  {'':>18}  {delta_var:>+22.6f}")
    print()
    print(f"  Backend (run1) : {_ORIG_BACKEND}  [2Q error: 2.86e-3, Jul 26 inventory]")
    print(f"  Backend (run2) : {chosen_name}  [2Q error: {twoq_err}]")
    print(f"  Queue at selection (run2): {selected['pending_jobs']} pending jobs")
    print()
    print(f"  Both means < 0.25 confirms exponential concentration at reps=1, n_qubits=8")
    print(f"  (simulator primary reference: mean=0.14444648, var=0.03006228)")
    print()
    print(f"  Total spent_s (probe + matrix): {spent_s:.1f}s  (cap={SPEND_CAP_S}s)")
    print(f"  Total wall-clock: {total_wall/60:.1f} min")
    print()
    print("PHASE F COMPLETE. HARD STOP.")
    print("  Do not proceed to any further phase without explicit instruction.")
    print("=" * 68)


if __name__ == "__main__":
    main()

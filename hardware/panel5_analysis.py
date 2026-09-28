#!/usr/bin/env python
"""Phase 4 — Panel 5: hardware concentration vs simulator baseline.

Takes a completed 20×20 Gram matrix (from Phase 3 batched_matrix.py) and
computes off-diagonal var/mean, then compares to two simulator baselines:

  PRIMARY   (metrics_reps_sweep.csv, reps=1):
      var  = 0.03006228   mean = 0.14444648
      Source: 72 rolling walk-forward folds on methods_label.parquet,
              n_qubits=8, angle_map, shots=None (statevector).

  SECONDARY (concentration_sweep.csv, n_qubits=8, reps=1):
      var  = 0.03667887   mean = 0.19890544
      Source: fixed 200-point stratified subset, same angle_map, shots=None.
      Differs from PRIMARY because it uses a fixed 200-pt subset rather than
      per-fold variable training pools; the smaller pool yields higher mean
      overlap (more concentrated).

Physical question answered:
    Does device noise on ibm_boston at reps=1 reproduce the same
    concentration collapse that rising reps produce on the simulator?
    If hardware off_diag_mean ≈ PRIMARY mean (≈ 0.144) the noise is
    acting like more reps (extra concentration).  If it is higher, the
    noise floor is not adding to the concentration effect.

Usage:
    python hardware/panel5_analysis.py --gram-path hardware/kernel_matrix_boston_n20.npy
    python hardware/panel5_analysis.py --gram-path hardware/kernel_matrix_boston_n20.npy \
        --meta-path hardware/kernel_matrix_boston_n20_meta.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent

# ── simulator baselines ───────────────────────────────────────────────────────
# Primary: metrics_reps_sweep.csv reps=1 (72-fold walk-forward average)
_PRIMARY_VAR  = 0.03006228
_PRIMARY_MEAN = 0.14444648
_PRIMARY_SRC  = "results/phase4/metrics_reps_sweep.csv (reps=1, 72-fold WF avg)"

# Secondary: concentration_sweep.csv n_qubits=8, reps=1 (fixed 200-pt subset)
_SECONDARY_VAR  = 0.03667887
_SECONDARY_MEAN = 0.19890544
_SECONDARY_SRC  = "results/concentration_sweep/concentration_sweep.csv (n_q=8, reps=1, 200-pt subset)"


def _load_baseline_csv(csv_path: Path, reps_val: int) -> dict | None:
    """Re-read the CSV to confirm stored values match what we hard-coded."""
    try:
        import csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["reps"]) == reps_val:
                    return {
                        "mean_var_offdiag":  float(row["mean_var_offdiag"]),
                        "mean_mean_offdiag": float(row["mean_mean_offdiag"]),
                    }
    except Exception:
        pass
    return None


def _off_diag_stats(K: np.ndarray) -> tuple[float, float]:
    n        = K.shape[0]
    off_mask = ~np.eye(n, dtype=bool)
    off      = K[off_mask]
    return float(off.mean()), float(off.var())


def main() -> None:
    parser = argparse.ArgumentParser(description="Panel 5: hardware concentration analysis")
    parser.add_argument(
        "--gram-path",
        required=True,
        help="Path to .npy Gram matrix produced by batched_matrix.py",
    )
    parser.add_argument(
        "--meta-path",
        default=None,
        help="Optional .json metadata file (for logging job_id, shots, etc.)",
    )
    args = parser.parse_args()

    gram_path = Path(args.gram_path)
    if not gram_path.exists():
        print(f"[FATAL] Gram matrix not found: {gram_path}", file=sys.stderr)
        sys.exit(1)

    # ── load matrix ──────────────────────────────────────────────────────────
    K = np.load(gram_path)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise RuntimeError(f"FAIL: expected square matrix, got shape {K.shape}")
    n = K.shape[0]

    if np.any(np.isnan(K)):
        raise RuntimeError("FAIL: NaN in Gram matrix — check batched_matrix.py output")

    diag_mean = float(np.diag(K).mean())
    if diag_mean < 0.70:
        raise RuntimeError(
            f"FAIL: diagonal mean = {diag_mean:.4f} < 0.70 — "
            f"hardware readout or gate noise is catastrophic"
        )

    hw_mean, hw_var = _off_diag_stats(K)

    # ── load metadata if present ─────────────────────────────────────────────
    meta: dict = {}
    if args.meta_path:
        meta_path = Path(args.meta_path)
        if meta_path.exists():
            import json
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)

    # ── confirm baselines against CSVs on disk ───────────────────────────────
    primary_csv   = _ROOT / "results" / "phase4" / "metrics_reps_sweep.csv"
    secondary_csv = _ROOT / "results" / "concentration_sweep" / "concentration_sweep.csv"

    primary_disk   = _load_baseline_csv(primary_csv,   reps_val=1)
    secondary_disk = _load_baseline_csv(secondary_csv, reps_val=1)

    if primary_disk is None:
        raise RuntimeError(
            f"FAIL: could not read primary baseline from {primary_csv}. "
            f"Run scripts/phase4_run.py first."
        )

    # Confirm hard-coded values match disk (tolerance 1e-6)
    for key, stored, disk_val in [
        ("primary var",  _PRIMARY_VAR,  primary_disk["mean_var_offdiag"]),
        ("primary mean", _PRIMARY_MEAN, primary_disk["mean_mean_offdiag"]),
    ]:
        if abs(stored - disk_val) > 1e-6:
            raise RuntimeError(
                f"FAIL: hard-coded {key} ({stored}) does not match "
                f"metrics_reps_sweep.csv ({disk_val}). "
                f"Baseline file may have been regenerated."
            )

    # ── report ────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("PANEL 5: HARDWARE CONCENTRATION vs SIMULATOR BASELINE")
    print("=" * 70)
    print(f"Gram matrix  : {gram_path}  ({n}×{n})")
    if meta:
        print(f"Backend      : {meta.get('backend','?')}")
        print(f"Shots        : {meta.get('shots','?')}")
        print(f"Circuit      : {meta.get('circuit','?')}, reps={meta.get('reps','?')}")
        print(f"n_qubits     : {meta.get('n_qubits','?')}")
        print(f"QPU spent    : {meta.get('total_spent_s','?')} s")
    print()

    print("HARDWARE (ibm_boston, angle_map, reps=1):")
    print(f"  diag_mean    : {diag_mean:.6f}  (ideal = 1.0; noise floor = diag attenuation)")
    print(f"  off_diag_mean: {hw_mean:.6f}")
    print(f"  off_diag_var : {hw_var:.6f}")
    print()

    print("SIMULATOR BASELINE — PRIMARY")
    print(f"  Source: {_PRIMARY_SRC}")
    print(f"  off_diag_mean: {_PRIMARY_MEAN:.6f}")
    print(f"  off_diag_var : {_PRIMARY_VAR:.6f}")
    print()

    print("SIMULATOR BASELINE — SECONDARY")
    print(f"  Source: {_SECONDARY_SRC}")
    print(f"  off_diag_mean: {_SECONDARY_MEAN:.6f}")
    print(f"  off_diag_var : {_SECONDARY_VAR:.6f}")
    print(
        "  NOTE: secondary differs from primary because concentration_sweep.csv\n"
        "        uses a fixed 200-pt stratified subset; metrics_reps_sweep.csv\n"
        "        averages per-fold kernel stats across 72 rolling walk-forward\n"
        "        folds (variable training-pool sizes), giving a lower mean overlap."
    )
    print()

    # ── interpret ────────────────────────────────────────────────────────────
    print("INTERPRETATION:")
    delta_mean_primary = hw_mean - _PRIMARY_MEAN
    delta_var_primary  = hw_var  - _PRIMARY_VAR

    print(
        f"  hardware mean - primary mean = {delta_mean_primary:+.6f}  "
        f"({'higher' if delta_mean_primary>0 else 'lower'} on hardware)"
    )
    print(
        f"  hardware var  - primary var  = {delta_var_primary:+.6f}  "
        f"({'higher' if delta_var_primary>0 else 'lower'} on hardware)"
    )
    print()

    if hw_mean < _PRIMARY_MEAN and hw_var < _PRIMARY_VAR:
        verdict = (
            "Hardware noise INCREASES concentration beyond the reps=1 simulator\n"
            "    baseline: off_diag_mean is lower and variance is lower, consistent\n"
            "    with device noise acting as additional depolarising that flattens\n"
            "    the kernel toward the Haar random floor."
        )
    elif hw_mean > _PRIMARY_MEAN and hw_var > _PRIMARY_VAR:
        verdict = (
            "Hardware noise DOES NOT add concentration beyond the reps=1 simulator\n"
            "    baseline: off_diag_mean is higher and variance is higher, suggesting\n"
            "    the noise has not yet pushed the kernel to its Haar floor."
        )
    else:
        verdict = (
            "Mixed: hardware mean and variance move in opposite directions relative\n"
            "    to the simulator baseline. Interpret with caution."
        )

    print(f"  VERDICT: {verdict}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

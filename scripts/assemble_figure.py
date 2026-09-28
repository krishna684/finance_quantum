#!/usr/bin/env python
"""Assemble the six-panel main figure for the JFDS paper.

Panels:
    1  Four-arm pooled AUC bar chart
    2  Reps sweep — pooled AUC vs reps
    3  Reps sweep — off-diagonal kernel variance vs reps
    4  Concentration heatmap — off_diag_mean over (n_qubits x reps) grid
    5  Hardware vs simulator — off-diagonal mean comparison
    6  Hardware Gram matrix heatmap (20x20, ibm_marrakesh)

Output: results/figure_main.png  (300 dpi)

Usage:
    python scripts/assemble_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

_ROOT    = Path(__file__).resolve().parent.parent
_RESULTS = _ROOT / "results"
_HW_DIR  = _ROOT / "hardware"
_OUT     = _RESULTS / "figure_main.png"

# ── simulator baselines (hard-coded, cross-checked by panel5_analysis.py) ────
_PRIMARY_MEAN = 0.14444648   # metrics_reps_sweep.csv reps=1, 72-fold WF avg
_PRIMARY_VAR  = 0.03006228
_SECONDARY_MEAN = 0.19890544  # concentration_sweep.csv n_q=8, reps=1, 200-pt subset
_SECONDARY_VAR  = 0.03667887

# JFDS-style palette: muted, print-safe
_COL = {
    "qkernel":  "#2166ac",
    "classical_pca8":  "#92c5de",
    "vqc":      "#4dac26",
    "classical_pca50": "#d1e5f0",
    "hw":       "#d6604d",
    "sim_primary":   "#2166ac",
    "sim_secondary": "#92c5de",
}


def _load_four_arm() -> pd.DataFrame:
    return pd.read_csv(_RESULTS / "phase4" / "metrics_four_arm.csv")


def _load_reps_sweep() -> pd.DataFrame:
    return pd.read_csv(_RESULTS / "phase4" / "metrics_reps_sweep.csv")


def _load_concentration_sweep() -> pd.DataFrame:
    return pd.read_csv(_RESULTS / "concentration_sweep" / "concentration_sweep.csv")


def _load_gram() -> tuple[np.ndarray, dict]:
    K    = np.load(_HW_DIR / "kernel_matrix_boston_n20.npy")
    with open(_HW_DIR / "kernel_matrix_boston_n20_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return K, meta


# ── individual panel builders ─────────────────────────────────────────────────

def _panel1_four_arm(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Bar chart: pooled AUC for the four arms."""
    labels_raw = df["arm"].tolist()
    auc        = df["pooled_auc"].tolist()
    std        = df["std_auc_all"].tolist()

    short = [
        "RBF\nPCA-8",
        "QKernel\nPCA-8 r=1",
        "VQC\nPCA-8 r=2",
        "RBF\nPCA-50",
    ]
    colors = [
        _COL["classical_pca8"],
        _COL["qkernel"],
        _COL["vqc"],
        _COL["classical_pca50"],
    ]
    x = np.arange(len(short))
    bars = ax.bar(x, auc, yerr=std, capsize=3, color=colors,
                  edgecolor="k", linewidth=0.6, width=0.55, error_kw={"linewidth": 0.8})
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=7)
    ax.set_ylabel("Pooled AUC", fontsize=8)
    ax.set_ylim(0.45, 0.70)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.set_title("(a) Four-arm comparison", fontsize=8, fontweight="bold")
    ax.legend(fontsize=6, framealpha=0.7)
    for bar, v in zip(bars, auc):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6)


def _panel2_reps_auc(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Line chart: pooled AUC vs reps."""
    reps = df["reps"].tolist()
    auc  = df["pooled_auc"].tolist()
    std  = df["std_auc"].tolist()
    ax.errorbar(reps, auc, yerr=std, marker="o", markersize=4,
                color=_COL["qkernel"], linewidth=1.2, capsize=3,
                elinewidth=0.8, label="QKernel PCA-8")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Reps", fontsize=8)
    ax.set_ylabel("Pooled AUC", fontsize=8)
    ax.set_xticks(reps)
    ax.set_ylim(0.48, 0.63)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
    ax.set_title("(b) AUC vs circuit reps", fontsize=8, fontweight="bold")
    ax.legend(fontsize=6, framealpha=0.7)


def _panel3_reps_var(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Line chart: off-diagonal kernel variance vs reps."""
    reps = df["reps"].tolist()
    var  = df["mean_var_offdiag"].tolist()
    ax.plot(reps, var, marker="s", markersize=4,
            color=_COL["qkernel"], linewidth=1.2, label="Off-diag var (sim)")
    ax.set_xlabel("Reps", fontsize=8)
    ax.set_ylabel("Off-diagonal variance", fontsize=8)
    ax.set_xticks(reps)
    ax.set_title("(c) Concentration vs reps", fontsize=8, fontweight="bold")
    ax.legend(fontsize=6, framealpha=0.7)
    # annotate monotonic collapse
    for r, v in zip(reps, var):
        ax.annotate(f"{v:.4f}", xy=(r, v), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=5.5)


def _panel4_concentration_heatmap(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Heatmap: off_diag_mean over (n_qubits x reps) grid."""
    pivot = df.pivot(index="n_qubits", columns="reps", values="off_diag_mean")
    nq   = pivot.index.tolist()
    reps = pivot.columns.tolist()
    Z    = pivot.values

    im = ax.imshow(Z, aspect="auto", cmap="Blues_r",
                   vmin=0.0, vmax=0.45)
    ax.set_xticks(range(len(reps)))
    ax.set_xticklabels([str(r) for r in reps], fontsize=7)
    ax.set_yticks(range(len(nq)))
    ax.set_yticklabels([str(q) for q in nq], fontsize=7)
    ax.set_xlabel("Reps", fontsize=8)
    ax.set_ylabel("n_qubits", fontsize=8)
    ax.set_title("(d) Off-diag mean (lower = more concentrated)", fontsize=8, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(nq)):
        for j in range(len(reps)):
            ax.text(j, i, f"{Z[i, j]:.3f}", ha="center", va="center",
                    fontsize=5.5, color="white" if Z[i, j] < 0.15 else "black")


def _panel5_hardware_vs_sim(ax: plt.Axes, K: np.ndarray, meta: dict) -> None:
    """Bar chart: off-diagonal mean for hardware vs two simulator baselines."""
    n        = K.shape[0]
    off_mask = ~np.eye(n, dtype=bool)
    hw_mean  = float(K[off_mask].mean())

    categories = ["HW\nmarrakesh\nr=1", "Sim PRIMARY\n72-fold WF\nr=1", "Sim SECONDARY\n200-pt subset\nr=1"]
    values     = [hw_mean, _PRIMARY_MEAN, _SECONDARY_MEAN]
    colors_bar = [_COL["hw"], _COL["sim_primary"], _COL["sim_secondary"]]

    x    = np.arange(len(categories))
    bars = ax.bar(x, values, color=colors_bar, edgecolor="k", linewidth=0.6, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=7)
    ax.set_ylabel("Off-diagonal mean kernel value", fontsize=8)
    ax.set_ylim(0.0, 0.28)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.set_title("(e) Hardware concentration vs simulator", fontsize=8, fontweight="bold")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=6.5)
    # annotate delta
    delta = hw_mean - _PRIMARY_MEAN
    ax.annotate(
        f"HW - Sim = {delta:+.4f}\n({'lower' if delta < 0 else 'higher'} on hardware)",
        xy=(0, hw_mean), xytext=(0.5, hw_mean - 0.04),
        fontsize=6, color=_COL["hw"],
        arrowprops=dict(arrowstyle="-", color=_COL["hw"], lw=0.6),
    )


def _panel6_gram_heatmap(ax: plt.Axes, K: np.ndarray, meta: dict) -> None:
    """Heatmap: 20x20 hardware Gram matrix."""
    labels = meta.get("labels", [])
    n      = K.shape[0]

    # sort by label so up/down blocks visible
    if labels and len(labels) == n:
        idx    = np.argsort(labels)
        K_sort = K[np.ix_(idx, idx)]
        labels_sorted = [labels[i] for i in idx]
    else:
        K_sort        = K
        labels_sorted = []

    im = ax.imshow(K_sort, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Sample index (sorted by label)", fontsize=7)
    ax.set_ylabel("Sample index (sorted by label)", fontsize=7)
    ax.set_title(
        f"(f) Hardware Gram matrix (ibm_marrakesh, n=20)\n"
        f"off-diag mean={meta.get('off_diag_mean', 0):.4f}  "
        f"var={meta.get('off_diag_var', 0):.4f}",
        fontsize=7, fontweight="bold",
    )
    ax.tick_params(labelsize=6)

    # draw a divider between label=0 and label=1 blocks if sorted
    if labels_sorted:
        n_down = labels_sorted.count(0)
        if 0 < n_down < n:
            ax.axhline(n_down - 0.5, color="red", linewidth=0.8)
            ax.axvline(n_down - 0.5, color="red", linewidth=0.8)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    df_four  = _load_four_arm()
    df_reps  = _load_reps_sweep()
    df_conc  = _load_concentration_sweep()
    K, meta  = _load_gram()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(
        "Quantum Kernel vs Classical Baseline — Finance Sentiment Classification",
        fontsize=10, fontweight="bold", y=0.995,
    )

    _panel1_four_arm(axes[0, 0], df_four)
    _panel2_reps_auc(axes[0, 1], df_reps)
    _panel3_reps_var(axes[0, 2], df_reps)
    _panel4_concentration_heatmap(axes[1, 0], df_conc)
    _panel5_hardware_vs_sim(axes[1, 1], K, meta)
    _panel6_gram_heatmap(axes[1, 2], K, meta)

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=300, bbox_inches="tight")
    print(f"Figure saved -> {_OUT}  ({_OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

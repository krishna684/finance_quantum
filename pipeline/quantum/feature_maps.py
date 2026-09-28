"""Stage 2: quantum feature maps.

Encodes reduced classical features x into a quantum state |phi(x)>.
Two maps, per proposal section 2.3:

  angle      — RY rotation per feature per qubit, followed by a ring of
               CNOTs and a data-reuploading RZ layer, repeated `reps`
               times. Comparable in spirit to Qiskit's ZZFeatureMap:
               the entanglement makes the induced kernel classically
               hard to simulate at scale (the whole point).
  amplitude  — features embedded directly as statevector amplitudes
               (normalized, zero-padded to 2^n_qubits). Shallow circuit,
               but state prep is expensive on real hardware.

These are pure circuit-builders: they apply gates inside an active
QNode context and hold no state, so kernel.py and vqc.py share them.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml


def angle_map(x: np.ndarray, n_qubits: int, reps: int = 2) -> None:
    """Angle encoding with entanglement and data re-uploading.

    Layer structure, repeated `reps` times:
        RY(x_i) on qubit i  ->  CNOT ring  ->  RZ(x_i) on qubit i

    Re-uploading the data after entanglement is what generates the
    nonlinear cross-feature terms in the induced kernel.

    Uses x[..., i] indexing so a batched input of shape (batch, n_qubits)
    broadcasts through the circuit (PennyLane parameter broadcasting) —
    the VQC trains on whole minibatches in one QNode call.
    """
    for _ in range(reps):
        for i in range(n_qubits):
            qml.RY(x[..., i], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        for i in range(n_qubits):
            qml.RZ(x[..., i], wires=i)


def amplitude_map(x: np.ndarray, n_qubits: int, reps: int = 1) -> None:
    """Amplitude encoding: x becomes the statevector (padded, normalized).

    `reps` is accepted for interface parity but ignored — repeating an
    amplitude embedding is not meaningful.
    """
    qml.AmplitudeEmbedding(
        features=x, wires=range(n_qubits), pad_with=0.0, normalize=True
    )


def get_feature_map(name: str):
    """Look up a feature map by config name."""
    maps = {"angle": angle_map, "amplitude": amplitude_map}
    if name not in maps:
        raise ValueError(f"Unknown feature map {name!r}; options: {sorted(maps)}")
    return maps[name]


def validate_features(x: np.ndarray, name: str, n_qubits: int) -> None:
    """Check feature dimensionality against the chosen map."""
    d = np.asarray(x).shape[-1]
    if name == "angle" and d != n_qubits:
        raise ValueError(
            f"angle map needs exactly n_qubits={n_qubits} features, got {d}"
        )
    if name == "amplitude" and d > 2**n_qubits:
        raise ValueError(
            f"amplitude map fits at most 2^{n_qubits}={2**n_qubits} features, got {d}"
        )

"""Stage 3a: quantum kernel estimation.

The fidelity kernel between two samples is the squared overlap of their
encoded states:

    k(x1, x2) = |<phi(x2)|phi(x1)>|^2

Computed with the adjoint trick: run U(x1) then U(x2)^dagger and measure
the probability of returning to |0...0>. One circuit evaluation per pair.

Complexity — the reason max_train_samples exists:
    train kernel:  n(n-1)/2 evaluations   (symmetric, unit diagonal)
    test kernel:   n_test * n_train evaluations
At n_train=400 that is ~80k circuit evals for the train block; a 252-day
walk-forward window with multiple images/day blows past that fast, hence
stratified subsampling. On hardware each evaluation is also multiplied
by `shots`, which is why the cap is a first-class config knob and not an
afterthought.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from pipeline.quantum.feature_maps import get_feature_map, validate_features


class QuantumKernel:
    """Computes fidelity kernel matrices for a chosen feature map.

    Args:
        n_qubits: Circuit width; must match reduced feature count for
            the angle map.
        feature_map: "angle" or "amplitude".
        reps: Feature-map repetitions.
        shots: None for exact analytic simulation; an int to emulate
            finite-sampling noise (hardware realism / noise analysis).
        seed: Seed for the shot-sampling RNG.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        feature_map: str = "angle",
        reps: int = 2,
        shots: int | None = None,
        seed: int = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.feature_map_name = feature_map
        self.reps = reps
        self.shots = shots
        self._map = get_feature_map(feature_map)
        dev_seed = np.random.default_rng(seed)
        # Pass shots to the device (PennyLane >=0.38 removed qml.set_shots)
        self._dev = qml.device("default.qubit", wires=n_qubits, shots=shots, seed=dev_seed)
        self._overlap = self._build_overlap_qnode()

    def _build_overlap_qnode(self):
        n, reps, fmap = self.n_qubits, self.reps, self._map

        @qml.qnode(self._dev)
        def overlap(x1, x2):
            fmap(x1, n, reps)
            qml.adjoint(fmap)(x2, n, reps)
            return qml.probs(wires=range(n))

        return overlap

    def _states(self, X: np.ndarray) -> np.ndarray:
        """Statevector per sample — the analytic fast path.

        On a simulator with shots=None, |<phi(x2)|phi(x1)>|^2 can be
        computed from the two statevectors directly, turning O(n^2)
        circuit evaluations into O(n) simulations plus a numpy matmul.
        Results are identical to the pairwise overlap circuit; only the
        shots!=None path needs real pair circuits (and on hardware,
        state readout doesn't exist, so the overlap circuit is also
        what would actually run there).
        """
        n, reps, fmap = self.n_qubits, self.reps, self._map

        @qml.qnode(self._dev)
        def state_circuit(x):
            fmap(x, n, reps)
            return qml.state()

        return np.stack([np.asarray(state_circuit(x)) for x in X])

    def pair(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Kernel value for a single pair: P(|0...0>) after U(x1) U(x2)†."""
        return float(self._overlap(x1, x2)[0])

    def train_matrix(self, X: np.ndarray) -> np.ndarray:
        """Symmetric train kernel, n(n-1)/2 circuit evaluations.

        Diagonal is exactly 1 (state overlaps with itself), so it is set
        analytically rather than evaluated — free accuracy under shot
        noise and n fewer circuit runs.
        """
        X = np.asarray(X, dtype=np.float64)
        validate_features(X, self.feature_map_name, self.n_qubits)
        n = X.shape[0]
        if self.shots is None:
            S = self._states(X)
            K = np.abs(S @ S.conj().T) ** 2
            np.fill_diagonal(K, 1.0)
            return K
        K = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                K[i, j] = K[j, i] = self.pair(X[i], X[j])
        return K

    def cross_matrix(self, X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """Rectangular test-vs-train kernel, n_test * n_train evaluations."""
        X_test = np.asarray(X_test, dtype=np.float64)
        X_train = np.asarray(X_train, dtype=np.float64)
        validate_features(X_test, self.feature_map_name, self.n_qubits)
        if self.shots is None:
            S_test = self._states(X_test)
            S_train = self._states(X_train)
            return np.abs(S_test @ S_train.conj().T) ** 2
        K = np.empty((X_test.shape[0], X_train.shape[0]))
        for i in range(X_test.shape[0]):
            for j in range(X_train.shape[0]):
                K[i, j] = self.pair(X_test[i], X_train[j])
        return K

    @staticmethod
    def evaluation_budget(n_train: int, n_test: int = 0) -> int:
        """Circuit evaluations needed — use to sanity-check a config
        before committing to a run (multiply by shots for hardware)."""
        return n_train * (n_train - 1) // 2 + n_test * n_train


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, max_samples: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cap the training set for kernel estimation, preserving class balance.

    Returns (X_sub, y_sub, indices) where indices map back into the
    original arrays — keep them so predictions can be joined back to
    article/frame IDs downstream.
    """
    X, y = np.asarray(X), np.asarray(y)
    n = X.shape[0]
    if n <= max_samples:
        idx = np.arange(n)
        return X, y, idx
    rng = np.random.default_rng(seed)
    picked: list[np.ndarray] = []
    classes, counts = np.unique(y, return_counts=True)
    # Proportional allocation, remainder to the largest classes.
    alloc = np.floor(counts / n * max_samples).astype(int)
    shortfall = max_samples - alloc.sum()
    order = np.argsort(-counts)
    for k in range(shortfall):
        alloc[order[k % len(order)]] += 1
    for cls, take in zip(classes, alloc):
        cls_idx = np.flatnonzero(y == cls)
        take = min(take, cls_idx.size)
        picked.append(rng.choice(cls_idx, size=take, replace=False))
    idx = np.sort(np.concatenate(picked))
    return X[idx], y[idx], idx

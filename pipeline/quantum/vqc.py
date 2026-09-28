"""Stage 3c: variational quantum classifier (VQC).

Architecture: feature map (data encoding) -> StronglyEntanglingLayers
(trainable ansatz) -> <Z> expectation on qubit 0, mapped to a binary
class probability. Parameters are optimized end-to-end with Adam via
PennyLane's autograd — the "hybrid classical-quantum optimization" of
proposal section 2.3.

Unlike kernel estimation (O(n^2) in samples), VQC training is O(n) per
epoch, so it scales to larger training windows — but it is non-convex
and seed-sensitive, which is itself a finding worth reporting. Runs are
seeded and loss curves retained for exactly that reason.

Binary classification only, matching the O&P negative/positive sentiment
framing. Extend to multiclass via one-vs-rest if ever needed.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from pipeline.quantum.config import QuantumSettings
from pipeline.quantum.feature_maps import get_feature_map
from pipeline.quantum.reducer import EmbeddingReducer


class VariationalQuantumClassifier:
    """Binary VQC with an sklearn-style fit/predict surface.

    Owns its reducer (train-only fit), like QuantumKernelSVM, so it
    accepts raw embeddings.
    """

    def __init__(self, settings: QuantumSettings | None = None) -> None:
        self.settings = settings or QuantumSettings()
        s = self.settings
        self.reducer = EmbeddingReducer(n_components=s.n_qubits, seed=s.seed)
        self._map = get_feature_map(s.feature_map)
        self._dev = qml.device("default.qubit", wires=s.n_qubits)
        self._qnode = self._build_qnode()
        self.weights_: pnp.ndarray | None = None
        self.bias_: pnp.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.loss_history_: list[float] = []
        self.fitted = False

    def _build_qnode(self):
        s = self.settings
        fmap = self._map

        @qml.qnode(self._dev, interface="autograd")
        def circuit(x, weights):
            fmap(x, s.n_qubits, s.feature_map_reps)
            qml.StronglyEntanglingLayers(weights, wires=range(s.n_qubits))
            return qml.expval(qml.PauliZ(0))

        return circuit

    # --- training ---------------------------------------------------

    def _raw_score(self, x, weights, bias):
        """Unbounded score in ~[-1-|b|, 1+|b|]; sign gives the class.

        Accepts a single sample (n_qubits,) or a batch
        (batch, n_qubits) — the circuit broadcasts over the batch.
        """
        return self._qnode(x, weights) + bias

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VariationalQuantumClassifier":
        s = self.settings
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError(
                f"VQC is binary; got {self.classes_.size} classes {self.classes_}"
            )
        # Map labels to {-1, +1} to match the <Z> expectation range.
        y_pm = np.where(y == self.classes_[1], 1.0, -1.0)

        Xr = self.reducer.fit_transform(X)

        shape = qml.StronglyEntanglingLayers.shape(
            n_layers=s.vqc_layers, n_wires=s.n_qubits
        )
        rng = np.random.default_rng(s.seed)
        weights = pnp.array(
            rng.uniform(0, 2 * np.pi, size=shape), requires_grad=True
        )
        bias = pnp.array(0.0, requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=s.vqc_learning_rate)

        def batch_loss(w, b, Xb, yb):
            # Square loss on the raw score against +/-1 targets — smooth,
            # robust, and standard for NISQ-era VQC training. One
            # broadcasted QNode call per batch, not one per sample.
            scores = self._raw_score(Xb, w, b)
            return pnp.mean((scores - yb) ** 2)

        n = Xr.shape[0]
        self.loss_history_ = []
        for _ in range(s.vqc_epochs):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, s.vqc_batch_size):
                sel = perm[start : start + s.vqc_batch_size]
                (weights, bias), l = opt.step_and_cost(
                    lambda w, b: batch_loss(w, b, Xr[sel], y_pm[sel]),
                    weights,
                    bias,
                )
                epoch_loss += float(l)
                n_batches += 1
            self.loss_history_.append(epoch_loss / max(n_batches, 1))

        self.weights_, self.bias_ = weights, bias
        self.fitted = True
        return self

    # --- inference --------------------------------------------------

    def _scores(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("VQC not fitted")
        Xr = self.reducer.transform(np.asarray(X, dtype=np.float64))
        return np.asarray(self._raw_score(Xr, self.weights_, self.bias_))

    def predict(self, X: np.ndarray) -> np.ndarray:
        s = self._scores(X)
        return np.where(s >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Squash raw scores through a sigmoid to (0, 1).

        Uncalibrated — fine for ranking/AUC and quintile sorts; apply
        Platt scaling downstream if calibrated probabilities matter.
        """
        p1 = 1.0 / (1.0 + np.exp(-3.0 * self._scores(X)))
        return np.column_stack([1.0 - p1, p1])

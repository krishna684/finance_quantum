"""Stage 3b: quantum kernel SVM.

Wraps sklearn's SVC(kernel='precomputed') around QuantumKernel so the
quantum arm exposes the same fit/predict/predict_proba surface as the
five classical baselines — it can be registered in the existing
classifier suite and flow through walk-forward CV, quintile portfolio
construction, and factor regressions without special-casing.

The full pre-classifier chain (reduce -> subsample -> kernel) lives in
fit(), so callers hand it raw embeddings exactly like they hand raw
embeddings to XGBoost.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC

from pipeline.quantum.config import QuantumSettings
from pipeline.quantum.kernel import QuantumKernel, stratified_subsample
from pipeline.quantum.reducer import EmbeddingReducer


class QuantumKernelSVM:
    """Quantum fidelity kernel + classical SVM, sklearn-style.

    Owns its own EmbeddingReducer (fit on the training window inside
    fit(), never on test data) and its own training-size cap.
    """

    def __init__(self, settings: QuantumSettings | None = None) -> None:
        self.settings = settings or QuantumSettings()
        s = self.settings
        self.reducer = EmbeddingReducer(n_components=s.n_qubits, seed=s.seed)
        self.kernel = QuantumKernel(
            n_qubits=s.n_qubits,
            feature_map=s.feature_map,
            reps=s.feature_map_reps,
            shots=s.shots,
            seed=s.seed,
        )
        self._svc = SVC(C=s.svm_C, kernel="precomputed", probability=True,
                        random_state=s.seed)
        self._X_train_reduced: np.ndarray | None = None
        self.train_indices_: np.ndarray | None = None
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelSVM":
        """Reduce, cap, build the train kernel, fit the SVM.

        Args:
            X: (n, embed_dim) raw embeddings (e.g. CLIP 512-dim).
            y: (n,) binary or multiclass labels.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        Xr = self.reducer.fit_transform(X)
        Xr, y_sub, idx = stratified_subsample(
            Xr, y, self.settings.max_train_samples, seed=self.settings.seed
        )
        self.train_indices_ = idx
        K = self.kernel.train_matrix(Xr)
        self._svc.fit(K, y_sub)
        self._X_train_reduced = Xr
        self.fitted = True
        return self

    def _cross_kernel(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("QuantumKernelSVM not fitted")
        Xr = self.reducer.transform(np.asarray(X, dtype=np.float64))
        return self.kernel.cross_matrix(Xr, self._X_train_reduced)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._svc.predict(self._cross_kernel(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._svc.predict_proba(self._cross_kernel(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._svc.decision_function(self._cross_kernel(X))

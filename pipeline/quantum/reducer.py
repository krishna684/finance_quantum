"""Stage 1: dimensionality reduction of image/text embeddings.

CLIP ViT-B/32 embeddings are 512-dim; FinBERT are 768-dim. Current
simulators/hardware want 8-20 features. This module compresses via PCA
and rescales each component to [0, pi] for angle encoding.

Leakage discipline: fit() must only ever see the training window of a
walk-forward split. transform() is then applied to both train and test.
This mirrors how the classical pipeline handles scaler fitting, so the
quantum arm inherits the same walk-forward hygiene.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class EmbeddingReducer:
    """StandardScaler -> PCA -> MinMax rescale to [0, pi].

    The [0, pi] range maps each feature to a rotation angle covering the
    Bloch sphere meridian exactly once, avoiding the wrap-around aliasing
    you get if features span more than 2*pi.
    """

    def __init__(self, n_components: int = 8, seed: int = 42) -> None:
        self.n_components = n_components
        self.seed = seed
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=n_components, random_state=seed)
        self._post_min: np.ndarray | None = None
        self._post_range: np.ndarray | None = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> "EmbeddingReducer":
        """Fit on the TRAINING window only.

        Args:
            X: (n_samples, embed_dim) float array of embeddings.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[0] <= self.n_components:
            raise ValueError(
                f"Need > {self.n_components} samples to fit PCA with "
                f"{self.n_components} components; got {X.shape[0]}"
            )
        Z = self._pca.fit_transform(self._scaler.fit_transform(X))
        self._post_min = Z.min(axis=0)
        rng = Z.max(axis=0) - self._post_min
        # Guard degenerate components (zero variance in train window).
        rng[rng == 0.0] = 1.0
        self._post_range = rng
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project to n_components features scaled to ~[0, pi].

        Test-window values can fall slightly outside the train min/max;
        they are clipped to [0, pi] rather than re-scaled, preserving
        the train-only fit contract.
        """
        if not self.fitted:
            raise RuntimeError("EmbeddingReducer.transform called before fit")
        X = np.asarray(X, dtype=np.float64)
        Z = self._pca.transform(self._scaler.transform(X))
        Z01 = (Z - self._post_min) / self._post_range
        return np.clip(Z01, 0.0, 1.0) * np.pi

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Per-component variance share — report this in the paper.

        If 8 components capture only ~30% of CLIP embedding variance,
        that is a material caveat for the quantum-vs-classical comparison
        (the classical models see the full 512 dims).
        """
        if not self.fitted:
            raise RuntimeError("Reducer not fitted")
        return self._pca.explained_variance_ratio_

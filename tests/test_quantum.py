"""Tests for pipeline.quantum.

Test sizes are deliberately small (4 qubits, tens of samples) so the
suite stays fast; correctness properties are size-independent.
"""

import numpy as np
import pytest

from pipeline.quantum.config import QuantumSettings
from pipeline.quantum.reducer import EmbeddingReducer
from pipeline.quantum.feature_maps import get_feature_map, validate_features
from pipeline.quantum.kernel import QuantumKernel, stratified_subsample
from pipeline.quantum.classifiers import QuantumKernelSVM
from pipeline.quantum.vqc import VariationalQuantumClassifier


# ---------------------------------------------------------------- helpers

def make_blobs(n_per_class: int, dim: int = 32, sep: float = 3.0, seed: int = 0):
    """Two Gaussian blobs in embedding space — linearly separable-ish,
    a sanity floor any working classifier must clear."""
    rng = np.random.default_rng(seed)
    X0 = rng.normal(0.0, 1.0, size=(n_per_class, dim))
    X1 = rng.normal(sep / np.sqrt(dim), 1.0, size=(n_per_class, dim))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def small_settings(**overrides) -> QuantumSettings:
    base = dict(
        n_qubits=4,
        feature_map_reps=1,
        max_train_samples=40,
        vqc_layers=2,
        vqc_epochs=15,
        vqc_batch_size=16,
        seed=7,
    )
    base.update(overrides)
    return QuantumSettings(**base)


# ---------------------------------------------------------------- config

class TestQuantumSettings:
    def test_defaults_valid(self):
        s = QuantumSettings()
        assert s.n_qubits == 8
        assert s.shots is None

    def test_rejects_bad_feature_map(self):
        with pytest.raises(ValueError, match="feature_map"):
            QuantumSettings(feature_map="teleportation")

    def test_rejects_tiny_qubit_count(self):
        with pytest.raises(ValueError, match="n_qubits"):
            QuantumSettings(n_qubits=1)

    def test_rejects_tiny_train_cap(self):
        with pytest.raises(ValueError, match="max_train_samples"):
            QuantumSettings(max_train_samples=5)


# ---------------------------------------------------------------- reducer

class TestEmbeddingReducer:
    def test_output_shape_and_range(self):
        X, _ = make_blobs(30, dim=64)
        r = EmbeddingReducer(n_components=4)
        Z = r.fit_transform(X)
        assert Z.shape == (60, 4)
        assert Z.min() >= 0.0 and Z.max() <= np.pi + 1e-12

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            EmbeddingReducer(n_components=4).transform(np.zeros((5, 64)))

    def test_test_window_clipped_not_refit(self):
        """Out-of-range test values must clip to [0, pi], proving the
        scaler was NOT refit on test data (leakage contract)."""
        X_train, _ = make_blobs(30, dim=64, seed=1)
        X_test = X_train * 10.0  # far outside the train range
        r = EmbeddingReducer(n_components=4).fit(X_train)
        Z = r.transform(X_test)
        assert Z.min() >= 0.0 and Z.max() <= np.pi + 1e-12

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="samples"):
            EmbeddingReducer(n_components=8).fit(np.zeros((5, 64)))

    def test_explained_variance_available(self):
        X, _ = make_blobs(30, dim=64)
        r = EmbeddingReducer(n_components=4).fit(X)
        evr = r.explained_variance_ratio
        assert evr.shape == (4,) and np.all(evr >= 0)


# ---------------------------------------------------------------- feature maps

class TestFeatureMaps:
    def test_lookup(self):
        assert get_feature_map("angle") is not None
        assert get_feature_map("amplitude") is not None
        with pytest.raises(ValueError):
            get_feature_map("nope")

    def test_angle_dimension_check(self):
        with pytest.raises(ValueError, match="n_qubits"):
            validate_features(np.zeros((3, 5)), "angle", n_qubits=4)

    def test_amplitude_dimension_check(self):
        # 2^3 = 8 max features on 3 qubits
        with pytest.raises(ValueError, match="at most"):
            validate_features(np.zeros((3, 9)), "amplitude", n_qubits=3)
        validate_features(np.zeros((3, 8)), "amplitude", n_qubits=3)  # ok


# ---------------------------------------------------------------- kernel

@pytest.fixture(scope="module")
def kernel():
    return QuantumKernel(n_qubits=4, feature_map="angle", reps=1)


@pytest.fixture(scope="module")
def Xr():
    rng = np.random.default_rng(3)
    return rng.uniform(0, np.pi, size=(10, 4))


class TestQuantumKernel:
    def test_self_overlap_is_one(self, kernel, Xr):
        assert kernel.pair(Xr[0], Xr[0]) == pytest.approx(1.0, abs=1e-9)

    def test_train_matrix_properties(self, kernel, Xr):
        K = kernel.train_matrix(Xr)
        assert K.shape == (10, 10)
        assert np.allclose(K, K.T), "kernel must be symmetric"
        assert np.allclose(np.diag(K), 1.0), "unit diagonal"
        assert np.all(K >= -1e-12) and np.all(K <= 1.0 + 1e-12), "probabilities"
        eigvals = np.linalg.eigvalsh(K)
        assert eigvals.min() > -1e-8, "fidelity kernel must be PSD"

    def test_cross_matrix_consistent_with_pair(self, kernel, Xr):
        C = kernel.cross_matrix(Xr[:3], Xr[3:8])
        assert C.shape == (3, 5)
        assert C[0, 0] == pytest.approx(kernel.pair(Xr[0], Xr[3]), abs=1e-9)

    def test_fast_path_matches_pairwise_circuit(self, kernel, Xr):
        """The statevector fast path must reproduce the overlap-circuit
        kernel exactly — it's an optimization, not an approximation."""
        K_fast = kernel.train_matrix(Xr[:5])
        K_pair = np.eye(5)
        for i in range(5):
            for j in range(i + 1, 5):
                K_pair[i, j] = K_pair[j, i] = kernel.pair(Xr[i], Xr[j])
        assert np.allclose(K_fast, K_pair, atol=1e-9)

    def test_evaluation_budget(self):
        assert QuantumKernel.evaluation_budget(100) == 4950
        assert QuantumKernel.evaluation_budget(100, n_test=10) == 4950 + 1000

    def test_shot_noise_mode_runs(self):
        k = QuantumKernel(n_qubits=4, reps=1, shots=256, seed=0)
        rng = np.random.default_rng(0)
        x1, x2 = rng.uniform(0, np.pi, size=(2, 4))
        v = k.pair(x1, x2)
        assert 0.0 <= v <= 1.0


class TestStratifiedSubsample:
    def test_no_op_when_under_cap(self):
        X, y = make_blobs(10, dim=4)
        Xs, ys, idx = stratified_subsample(X, y, max_samples=100)
        assert Xs.shape[0] == 20 and idx.shape[0] == 20

    def test_caps_and_preserves_balance(self):
        X, y = make_blobs(50, dim=4)
        Xs, ys, idx = stratified_subsample(X, y, max_samples=30, seed=1)
        assert Xs.shape[0] == 30
        counts = np.bincount(ys)
        assert abs(counts[0] - counts[1]) <= 1, "classes stay balanced"
        # Indices map back into the original arrays.
        assert np.array_equal(X[idx], Xs)

    def test_deterministic_under_seed(self):
        X, y = make_blobs(50, dim=4)
        _, _, i1 = stratified_subsample(X, y, 30, seed=9)
        _, _, i2 = stratified_subsample(X, y, 30, seed=9)
        assert np.array_equal(i1, i2)


# ---------------------------------------------------------------- classifiers

class TestQuantumKernelSVM:
    def test_learns_separable_blobs(self):
        X, y = make_blobs(30, dim=32, sep=4.0, seed=5)
        X_train, y_train = X[:40], y[:40]
        X_test, y_test = X[40:], y[40:]
        clf = QuantumKernelSVM(small_settings()).fit(X_train, y_train)
        acc = (clf.predict(X_test) == y_test).mean()
        assert acc >= 0.85, f"expected >=0.85 on separable blobs, got {acc:.2f}"

    def test_respects_train_cap(self):
        X, y = make_blobs(40, dim=32, seed=6)
        clf = QuantumKernelSVM(small_settings(max_train_samples=20)).fit(X, y)
        assert clf.train_indices_.shape[0] == 20

    def test_predict_proba_shape(self):
        X, y = make_blobs(25, dim=32, seed=7)
        clf = QuantumKernelSVM(small_settings()).fit(X, y)
        P = clf.predict_proba(X[:5])
        assert P.shape == (5, 2)
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            QuantumKernelSVM(small_settings()).predict(np.zeros((2, 32)))


class TestVQC:
    def test_learns_separable_blobs(self):
        X, y = make_blobs(30, dim=32, sep=4.0, seed=11)
        X_train, y_train = X[:40], y[:40]
        X_test, y_test = X[40:], y[40:]
        clf = VariationalQuantumClassifier(small_settings()).fit(X_train, y_train)
        acc = (clf.predict(X_test) == y_test).mean()
        assert acc >= 0.75, f"expected >=0.75 on separable blobs, got {acc:.2f}"

    def test_loss_decreases(self):
        X, y = make_blobs(20, dim=32, sep=4.0, seed=12)
        clf = VariationalQuantumClassifier(small_settings()).fit(X, y)
        assert clf.loss_history_[-1] < clf.loss_history_[0]

    def test_rejects_multiclass(self):
        X = np.random.default_rng(0).normal(size=(30, 32))
        y = np.array([0, 1, 2] * 10)
        with pytest.raises(ValueError, match="binary"):
            VariationalQuantumClassifier(small_settings()).fit(X, y)

    def test_predict_proba_shape(self):
        X, y = make_blobs(20, dim=32, seed=13)
        clf = VariationalQuantumClassifier(small_settings()).fit(X, y)
        P = clf.predict_proba(X[:4])
        assert P.shape == (4, 2)
        assert np.allclose(P.sum(axis=1), 1.0)

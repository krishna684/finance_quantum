"""End-to-end demo: quantum vs classical on synthetic embeddings.

Generates two overlapping Gaussian classes in 512-dim space (stand-in
for CLIP embeddings with a weak sentiment signal — deliberately NOT
trivially separable), then benchmarks on identical splits:

    1. Classical SVM-RBF on full 512-dim embeddings
    2. Classical SVM-RBF on the same 8-dim PCA features the quantum
       models see (isolates the cost of dimensionality reduction)
    3. QuantumKernelSVM  (fidelity kernel, angle map, 8 qubits)
    4. VariationalQuantumClassifier

Run:  python demo_quantum.py
"""

import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC

from pipeline.quantum import (
    EmbeddingReducer,
    QuantumKernel,
    QuantumKernelSVM,
    QuantumSettings,
    VariationalQuantumClassifier,
)


def make_data(n_per_class=120, dim=512, sep=3.0, seed=42):
    rng = np.random.default_rng(seed)
    # Signal lives in a random low-dim subspace, like real sentiment
    # structure inside CLIP space.
    direction = rng.normal(size=dim)
    direction /= np.linalg.norm(direction)
    X0 = rng.normal(0, 1, (n_per_class, dim))
    X1 = rng.normal(0, 1, (n_per_class, dim)) + sep * direction
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def evaluate(name, y_true, y_pred, y_score, elapsed):
    print(
        f"{name:<38} acc={accuracy_score(y_true, y_pred):.3f}  "
        f"f1={f1_score(y_true, y_pred):.3f}  "
        f"auc={roc_auc_score(y_true, y_score):.3f}  "
        f"({elapsed:.1f}s)"
    )


def main():
    settings = QuantumSettings(
        n_qubits=8, feature_map_reps=2, max_train_samples=150,
        vqc_layers=3, vqc_epochs=40, seed=42,
    )
    X, y = make_data()
    split = 160
    X_tr, y_tr, X_te, y_te = X[:split], y[:split], X[split:], y[split:]
    print(f"train={len(y_tr)}  test={len(y_te)}  dim={X.shape[1]}")
    budget = QuantumKernel.evaluation_budget(
        min(len(y_tr), settings.max_train_samples), n_test=len(y_te)
    )
    print(f"quantum kernel circuit-evaluation budget: {budget:,}\n")

    # 1. Classical, full embeddings
    t0 = time.time()
    clf = SVC(kernel="rbf", probability=True, random_state=42).fit(X_tr, y_tr)
    evaluate("SVM-RBF (512-dim, classical)", y_te, clf.predict(X_te),
             clf.predict_proba(X_te)[:, 1], time.time() - t0)

    # 2. Classical, same reduced features as quantum
    t0 = time.time()
    red = EmbeddingReducer(n_components=settings.n_qubits, seed=42)
    Ztr, Zte = red.fit_transform(X_tr), red.transform(X_te)
    clf = SVC(kernel="rbf", probability=True, random_state=42).fit(Ztr, y_tr)
    evaluate("SVM-RBF (8-dim PCA, classical)", y_te, clf.predict(Zte),
             clf.predict_proba(Zte)[:, 1], time.time() - t0)
    print(f"  PCA explained variance (8 comps): "
          f"{red.explained_variance_ratio.sum():.1%}")

    # 3. Quantum kernel SVM
    t0 = time.time()
    qsvm = QuantumKernelSVM(settings).fit(X_tr, y_tr)
    evaluate("Quantum kernel SVM (8 qubits)", y_te, qsvm.predict(X_te),
             qsvm.predict_proba(X_te)[:, 1], time.time() - t0)

    # 4. VQC
    t0 = time.time()
    vqc = VariationalQuantumClassifier(settings).fit(X_tr, y_tr)
    evaluate("VQC (8 qubits, 3 layers)", y_te, vqc.predict(X_te),
             vqc.predict_proba(X_te)[:, 1], time.time() - t0)
    print(f"  VQC loss: {vqc.loss_history_[0]:.3f} -> {vqc.loss_history_[-1]:.3f}")


if __name__ == "__main__":
    main()

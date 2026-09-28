"""Quantum classification pipeline.

Benchmarks quantum kernel methods and variational quantum classifiers
against the classical baselines, per the QIC proposal (section 2.3).

Stages:
    1. EmbeddingReducer   — PCA compression of CLIP/FinBERT embeddings
                            to n_qubits features, scaled for angle encoding.
    2. Feature maps       — parameterized circuits encoding classical
                            features into quantum states.
    3. QuantumKernel      — fidelity kernel matrices from circuit overlaps,
                            fed into a precomputed-kernel SVM.
    4. VQC                — variational quantum classifier trained via
                            hybrid classical-quantum optimization.

All computation runs on simulators (PennyLane default.qubit). Hardware
execution via IBM Quantum is a device-swap, not a rewrite.
"""

from pipeline.quantum.config import QuantumSettings
from pipeline.quantum.reducer import EmbeddingReducer
from pipeline.quantum.kernel import QuantumKernel
from pipeline.quantum.classifiers import QuantumKernelSVM
from pipeline.quantum.vqc import VariationalQuantumClassifier

__all__ = [
    "QuantumSettings",
    "EmbeddingReducer",
    "QuantumKernel",
    "QuantumKernelSVM",
    "VariationalQuantumClassifier",
]

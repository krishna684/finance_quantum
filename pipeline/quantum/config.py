"""Configuration for the quantum pipeline.

Standalone dataclass so it can be merged into pipeline/config.py Settings
(as nested fields or flattened with a quantum_ prefix) without touching
this module's internals.
"""

from dataclasses import dataclass


@dataclass
class QuantumSettings:
    """All knobs for the quantum pipeline, with simulator-safe defaults.

    Attributes:
        n_qubits: Number of qubits = number of PCA features. Proposal
            range is 8-20; default 8 keeps simulator statevectors small
            (2^8 = 256 amplitudes) and kernel estimation fast.
        feature_map: "angle" (one rotation per feature per qubit, with
            entangling layers) or "amplitude" (features as statevector
            amplitudes; requires len(x) <= 2^n_qubits).
        feature_map_reps: Number of encoding + entanglement repetitions.
            More reps = more expressive kernel, more circuit depth.
        max_train_samples: Hard cap on training-set size for kernel
            estimation. The symmetric train kernel needs n(n-1)/2 circuit
            overlap evaluations — O(n^2). At n=500 that is ~125k
            evaluations; at n=2000 it is ~2M. Cap early, subsample
            stratified. This is the binding computational constraint.
        vqc_layers: Depth of the StronglyEntanglingLayers ansatz.
        vqc_epochs: Max optimizer epochs for VQC training.
        vqc_learning_rate: Adam step size.
        vqc_batch_size: Minibatch size for VQC gradient steps.
        svm_C: Regularization for the precomputed-kernel SVM.
        seed: RNG seed threaded through subsampling, init, and shuffling.
        shots: None = exact analytic simulation (default; deterministic
            kernels). Set to an int (e.g. 1024) to emulate hardware
            sampling noise for the noise-sensitivity analysis.
    """

    n_qubits: int = 8
    feature_map: str = "angle"
    feature_map_reps: int = 2
    max_train_samples: int = 400
    vqc_layers: int = 3
    vqc_epochs: int = 60
    vqc_learning_rate: float = 0.05
    vqc_batch_size: int = 32
    svm_C: float = 1.0
    seed: int = 42
    shots: int | None = None

    def __post_init__(self) -> None:
        if self.n_qubits < 2:
            raise ValueError("n_qubits must be >= 2")
        if self.feature_map not in ("angle", "amplitude"):
            raise ValueError(
                f"feature_map must be 'angle' or 'amplitude', got {self.feature_map!r}"
            )
        if self.max_train_samples < 10:
            raise ValueError("max_train_samples must be >= 10")

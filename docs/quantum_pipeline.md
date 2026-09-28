# Quantum Pipeline (`pipeline/quantum/`)

Implements quantum kernel SVM and variational quantum classifier (VQC) as drop-in
alternatives to the classical classifiers. No external credentials required;
runs on PennyLane simulators.

## Module overview

| File | Purpose |
|------|---------|
| `pipeline/quantum/config.py` | `QuantumSettings` dataclass — env-var driven configuration |
| `pipeline/quantum/reducer.py` | `EmbeddingReducer` — StandardScaler → PCA → [0, π] rescale, train-only fit |
| `pipeline/quantum/feature_maps.py` | Angle encoding (entanglement + data re-uploading) and amplitude encoding |
| `pipeline/quantum/kernel.py` | `QuantumKernel` — fidelity kernel matrices; `stratified_subsample` for O(n²) cap |
| `pipeline/quantum/classifiers.py` | `QuantumKernelSVM` — sklearn-style `fit`/`predict`/`predict_proba` |
| `pipeline/quantum/vqc.py` | `VariationalQuantumClassifier` — batched hybrid training, loss history retained |
| `tests/test_quantum.py` | 29 tests: math properties, leakage contract, end-to-end learning |
| `demo_quantum.py` | 4-way benchmark on synthetic 512-dim embeddings |

## Integration

Both `QuantumKernelSVM` and `VariationalQuantumClassifier` expose `fit` / `predict` /
`predict_proba` (and `decision_function` on the SVM), so walk-forward CV, quintile
portfolio construction, and factor regressions need no changes.

1. `pennylane` must be in `requirements.txt` (pin it — quantum APIs change often).
2. Register quantum classifiers in the classifier suite alongside the five classical models.
3. Merge `QuantumSettings` fields into `Settings` (nested or `quantum_`-prefixed).
4. See `.env.example` for all tunable env vars.
5. Optional: `quantum-bench` subcommand in `run_pipeline.py` modelled on `demo_quantum.py`.

## Configuration (`.env` / env vars)

| Variable | Default | Description |
|----------|---------|-------------|
| `QUANTUM_N_QUBITS` | 8 | PCA output features = number of qubits |
| `QUANTUM_FEATURE_MAP` | `angle` | `angle` or `amplitude` |
| `QUANTUM_FEATURE_MAP_REPS` | 2 | Encoding + entanglement repetitions |
| `QUANTUM_MAX_TRAIN_SAMPLES` | 400 | O(n²) cap; stratified subsample |
| `QUANTUM_VQC_LAYERS` | 3 | StronglyEntanglingLayers depth |
| `QUANTUM_VQC_EPOCHS` | 60 | Training epochs |
| `QUANTUM_VQC_LEARNING_RATE` | 0.05 | Adam learning rate |
| `QUANTUM_VQC_BATCH_SIZE` | 32 | Minibatch size |
| `QUANTUM_SVM_C` | 1.0 | SVM regularisation |
| `QUANTUM_SEED` | 42 | RNG seed |
| `QUANTUM_SHOTS` | None | Unset = analytic; integer = hardware-noise emulation |

## Design notes

**Reducers live inside classifiers.** Each quantum model owns an `EmbeddingReducer` fit
inside `fit()` — train-window only, so walk-forward hygiene is preserved automatically.

**Analytic fast path.** With `shots=None`, kernels are computed from per-sample statevectors
(O(n) simulations + numpy matmul) instead of O(n²) overlap circuits — mathematically
identical (tested), ~100× faster. `QuantumKernel.evaluation_budget(n)` reports hardware cost
before you commit.

**`[0, π]` feature range** avoids rotation-angle wraparound aliasing. Test-window values
outside the train range clip rather than refit.

**VQC is binary-only**, matching the negative/positive sentiment framing. Square loss on ±1
targets; `predict_proba` is sigmoid-squashed and uncalibrated (fine for AUC and quintile
sorts; Platt-scale if calibrated probabilities are needed).

## Known issue: kernel concentration

At high qubit counts or circuit depths, quantum kernel off-diagonal entries concentrate near
a constant and AUC collapses toward 0.5. This is the barren-plateau / exponential
concentration phenomenon. The concentration sweep in `results/concentration_sweep/` maps
the safe operating region. Reps=1, qubits ≤ 8 is the current recommended setting.

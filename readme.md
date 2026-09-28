# Quantum-Enhanced Sentiment Analysis for Financial Market Prediction

A research pipeline that collects multimodal financial media (institutional news + TikTok consumer demand signals), extracts sentiment using classical and quantum ML classifiers, and evaluates predictive power against CRSP stock returns.

Based on the Obaid & Pukthuanthong (2022) framework, extended with quantum kernel SVM and variational quantum classifiers (VQC) via PennyLane.

---

## Key Results

Four-arm walk-forward comparison across 72 folds (N24News financial articles):

| Arm | Pooled AUC |
|-----|-----------|
| Classical RBF (PCA-50) | **0.608** |
| Classical RBF (PCA-8) | 0.586 |
| Quantum Kernel SVM (PCA-8, r=1) | 0.576 |
| VQC (PCA-8, r=2) | 0.574 |

Quantum classifiers match classical performance at the same dimensionality. The classical-50d
arm outperforms, pointing to dimensionality as the binding constraint rather than kernel type.

See [results/RESULTS.md](results/RESULTS.md) for full tables, figures, and the kernel
concentration analysis.

---

## Repository Structure

```
finance_quantum/
├── README.md                        # This file
├── .env.example                     # All configuration variables
├── requirements.txt                 # Python dependencies
├── run_pipeline.py                  # CLI entry point
├── demo_quantum.py                  # 4-way quantum vs classical benchmark
│
├── pipeline/                        # Core pipeline modules
│   ├── collector.py                 # News article collection
│   ├── config.py                    # Settings (env-var driven)
│   ├── embeddings.py                # CLIP + FinBERT / MiniLM embeddings
│   ├── classifiers.py               # Classical classifier suite
│   ├── sentiment.py                 # Sentiment scoring
│   ├── signals.py                   # Signal construction
│   ├── portfolio.py                 # Quintile portfolio + factor regressions
│   ├── returns.py                   # Return alignment
│   ├── report.py                    # Results reporting
│   ├── quantum/                     # Quantum classifiers (PennyLane)
│   │   ├── config.py
│   │   ├── reducer.py               # PCA + angle rescaling
│   │   ├── feature_maps.py          # Angle and amplitude encoding
│   │   ├── kernel.py                # Fidelity kernel matrices
│   │   ├── classifiers.py           # QuantumKernelSVM
│   │   └── vqc.py                   # VariationalQuantumClassifier
│   └── tiktok_collection/           # TikTok Research API + Bright Data pipeline
│
├── tests/                           # pytest suite (29 quantum + classical tests)
├── scripts/                         # Utility scripts
│
├── results/                         # Experiment outputs
│   ├── RESULTS.md                   # Results summary
│   ├── figure_main.png              # Main AUC comparison figure
│   ├── phase4/                      # Four-arm comparison CSVs
│   ├── concentration_sweep/         # Kernel concentration analysis + heatmaps
│   ├── classifiers/                 # Per-model predictions and metrics
│   └── portfolio/                   # Portfolio returns and factor alphas
│
└── docs/                            # Extended documentation
    ├── quantum_pipeline.md          # Quantum module design and configuration
    └── data_collection_protocol.md  # Full data collection protocol
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in credentials — see .env.example for all options
```

**Required credentials** (depending on which pipeline you run):

| Variable | Required for |
|----------|-------------|
| `TIKTOK_CLIENT_KEY` + `TIKTOK_CLIENT_SECRET` | TikTok Research API collection |
| `BRIGHTDATA_API_KEY` | Bright Data TikTok collection |
| `OPENAI_API_KEY` | Whisper transcription |
| `ANTHROPIC_API_KEY` | LLM entity extraction |

No credentials are needed to run the quantum benchmark (`demo_quantum.py`) or tests.

---

## Running

### Quantum benchmark (no credentials needed)

```bash
python demo_quantum.py
```

Runs the 4-way quantum vs classical benchmark on synthetic 512-dim embeddings.

### Full pipeline

```bash
# Collect news articles
python run_pipeline.py collect --env-file .env

# Run classification + portfolio analysis
python run_pipeline.py analyze --env-file .env

# Load brand-to-firm mapping (TikTok pipeline)
python run_pipeline.py load-mapping --csv mapping_template.csv --env-file .env
```

### Tests

```bash
pytest tests/
```

29 tests covering quantum math properties, leakage contracts, and end-to-end learning.

---

## Pipeline Architecture

```
Data Collection
  WSJ / NYT articles (MU Library TDM access)
  TikTok videos + comments (Research API / Bright Data)
        |
Embedding
  CLIP ViT-B/32 (images)
  FinBERT / all-MiniLM-L6-v2 (text)
        |
Classification (walk-forward expanding-window CV)
  Classical: SVM-RBF, SVM-Poly, XGBoost, Random Forest, MLP
  Quantum:   QuantumKernelSVM, VQC  (PennyLane)
        |
Signal Construction
  Sentiment scores -> long-short quintile portfolios
        |
Evaluation
  AUC-ROC, Fama-French 5-factor + UMD alphas
```

---

## Data Sources

- **Wall Street Journal / New York Times** — institutional media, via MU Library TDM license
- **TikTok Research API** — retail consumer demand signals (purchase intent + negative sentiment)
- **CRSP** — daily stock returns, PERMNO identifiers
- **Compustat** — firm fundamentals, segment data
- **Kenneth French Data Library** — Fama-French factors

See [docs/data_collection_protocol.md](docs/data_collection_protocol.md) for the full protocol.

---

## Quantum Module

See [docs/quantum_pipeline.md](docs/quantum_pipeline.md) for design decisions,
configuration reference, and the kernel concentration analysis.

---

## Citation

If you use this code or data pipeline, please cite:

```
Obaid, K., & Pukthuanthong, K. (2022). A picture is worth a thousand words:
Measuring investor sentiment by combining machine learning and photos from news.
Journal of Financial Economics.
```

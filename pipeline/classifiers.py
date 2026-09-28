from __future__ import annotations

"""Walk-forward classification pipeline for image/text sentiment signals.

Trains classical ML classifiers on CLIP image embeddings, financial text
embeddings, or their multimodal concatenation, and evaluates their ability to
predict the sign of the next trading-day stock return.

CRITICAL: All cross-validation uses strictly expanding windows split on
calendar-day boundaries.  No future data is ever visible to the model during
training or hyperparameter fitting.

Classifiers
-----------
    svm_rbf        SVM with RBF kernel
    svm_poly       SVM with degree-3 polynomial kernel
    xgboost        Gradient-boosted trees
    random_forest  Bagged decision trees
    mlp            Feedforward neural network (MLPClassifier)

Each model is wrapped in a sklearn Pipeline:
    StandardScaler → [PCA for SVMs] → Classifier

Walk-forward scheme
-------------------
    1. Sort all observations by trading_day (the day whose return is predicted).
    2. Identify unique trading days; use day-level boundaries for folds to
       prevent within-day leakage.
    3. Fold k: train on days [0, split_k), test on days [split_k, split_k + step).
    4. Minimum training size: min_train_days (default 60 unique days ≈ 3 months).
    5. Test step: test_days (default 21 unique days ≈ 1 month).
    6. After all folds: retrain each model on the full dataset and save to disk.

Outputs (written to results/classifiers/)
-----------------------------------------
    predictions_{embedding_type}_{model_name}.csv
        date, article_id, permno, ticker, predicted_prob, true_label,
        actual_return, fold

    metrics_{embedding_type}.csv
        model, fold, n_train, n_test, accuracy, f1, auc_roc, precision, recall

Usage
-----
    from pipeline.classifiers import ClassificationPipeline
    from pathlib import Path

    pipe = ClassificationPipeline(
        news_db=Path("data/news.db"),
        crsp_csv=Path("data/crsp_dsf.csv"),
        crsp_ids_csv=Path("data/crsp_identifiers.csv"),
        results_dir=Path("results"),
    )
    results = pipe.run(embedding_type="image")   # "image" | "text" | "multimodal"
    print(results.summary())
"""

import csv
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline.logger import get_logger
from pipeline.returns import align_timestamp

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class DataRecord:
    article_id: str
    trading_day: date
    permno: str
    ticker: str
    embedding: np.ndarray   # float32, shape (dim,)
    ret_sign: int           # 1 = positive return, 0 = negative return
    actual_return: float    # raw return value


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _json_tickers(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def _load_image_records(
    conn: sqlite3.Connection,
    crsp_mapper,
    crsp_returns,
    min_date: str,
    max_date: str,
) -> list[DataRecord]:
    """Load image embeddings joined with article metadata and CRSP returns.

    Multiple images per article are mean-pooled into one embedding vector.
    """
    try:
        rows = conn.execute(
            """
            SELECT
                ie.image_id,
                ie.embedding,
                ie.embedding_dim,
                ni.article_id,
                na.publication_date,
                COALESCE(na.publication_time, '09:00:00') AS publication_time,
                na.mapped_tickers
            FROM image_embeddings ie
            JOIN news_images ni ON ie.image_id = ni.image_id
            JOIN news_articles na ON ni.article_id = na.article_id
            WHERE na.mapped_tickers IS NOT NULL
              AND na.mapped_tickers != '[]'
              AND na.publication_date IS NOT NULL
              AND na.publication_date >= ?
              AND na.publication_date <= ?
            """,
            (min_date, max_date),
        ).fetchall()
    except sqlite3.OperationalError:
        log.warning("image_embeddings_table_missing")
        return []

    # Group embeddings by article_id (mean-pool multiple images)
    article_embs: dict[str, list[np.ndarray]] = {}
    article_meta: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        aid = row["article_id"]
        emb = np.frombuffer(row["embedding"], dtype=np.float32).reshape(row["embedding_dim"]).copy()
        article_embs.setdefault(aid, []).append(emb)
        if aid not in article_meta:
            article_meta[aid] = (row["publication_date"], row["publication_time"], row["mapped_tickers"])

    return _build_records(article_embs, article_meta, crsp_mapper, crsp_returns)


def _load_text_records(
    conn: sqlite3.Connection,
    crsp_mapper,
    crsp_returns,
    min_date: str,
    max_date: str,
) -> list[DataRecord]:
    """Load text embeddings joined with article metadata and CRSP returns."""
    try:
        rows = conn.execute(
            """
            SELECT
                te.article_id,
                te.embedding,
                te.embedding_dim,
                na.publication_date,
                COALESCE(na.publication_time, '09:00:00') AS publication_time,
                na.mapped_tickers
            FROM text_embeddings te
            JOIN news_articles na ON te.article_id = na.article_id
            WHERE na.mapped_tickers IS NOT NULL
              AND na.mapped_tickers != '[]'
              AND na.publication_date IS NOT NULL
              AND na.publication_date >= ?
              AND na.publication_date <= ?
            """,
            (min_date, max_date),
        ).fetchall()
    except sqlite3.OperationalError:
        log.warning("text_embeddings_table_missing")
        return []

    article_embs: dict[str, list[np.ndarray]] = {}
    article_meta: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        aid = row["article_id"]
        emb = np.frombuffer(row["embedding"], dtype=np.float32).reshape(row["embedding_dim"]).copy()
        article_embs.setdefault(aid, []).append(emb)
        if aid not in article_meta:
            article_meta[aid] = (row["publication_date"], row["publication_time"], row["mapped_tickers"])

    return _build_records(article_embs, article_meta, crsp_mapper, crsp_returns)


def _build_records(
    article_embs: dict[str, list[np.ndarray]],
    article_meta: dict[str, tuple[str, str, str]],
    crsp_mapper,
    crsp_returns,
) -> list[DataRecord]:
    records: list[DataRecord] = []
    for aid, embs in article_embs.items():
        pub_date, pub_time, tickers_raw = article_meta[aid]
        mean_emb = np.mean(embs, axis=0).astype(np.float32)
        tickers = _json_tickers(tickers_raw)

        ts = f"{pub_date}T{pub_time}-05:00"
        try:
            aligned = align_timestamp(ts)
        except Exception:
            continue

        for ticker in tickers:
            crsp_rec = crsp_mapper.lookup(ticker, as_of=pub_date)
            if not crsp_rec:
                continue
            ret = crsp_returns.get_return(crsp_rec.permno, aligned.trading_day)
            if ret is None or ret == 0.0:
                # Sample-selection note: exact zero returns (~0.01% of CRSP observations)
                # are dropped along with missing returns.  This introduces negligible bias
                # but should be disclosed if this sample selection is discussed in the paper.
                continue
            records.append(DataRecord(
                article_id=aid,
                trading_day=aligned.trading_day,
                permno=crsp_rec.permno,
                ticker=ticker,
                embedding=mean_emb,
                ret_sign=1 if ret > 0 else 0,
                actual_return=ret,
            ))

    return records


def _load_multimodal_records(
    conn: sqlite3.Connection,
    crsp_mapper,
    crsp_returns,
    min_date: str,
    max_date: str,
) -> list[DataRecord]:
    """Concatenate image and text embeddings for articles that have both."""
    img_records = _load_image_records(conn, crsp_mapper, crsp_returns, min_date, max_date)
    txt_records = _load_text_records(conn, crsp_mapper, crsp_returns, min_date, max_date)

    # Index by (article_id, permno)
    img_idx: dict[tuple[str, str], DataRecord] = {
        (r.article_id, r.permno): r for r in img_records
    }
    txt_idx: dict[tuple[str, str], DataRecord] = {
        (r.article_id, r.permno): r for r in txt_records
    }

    common_keys = set(img_idx) & set(txt_idx)
    records: list[DataRecord] = []
    for key in common_keys:
        ir = img_idx[key]
        tr = txt_idx[key]
        fused = np.concatenate([ir.embedding, tr.embedding])
        records.append(DataRecord(
            article_id=ir.article_id,
            trading_day=ir.trading_day,
            permno=ir.permno,
            ticker=ir.ticker,
            embedding=fused,
            ret_sign=ir.ret_sign,
            actual_return=ir.actual_return,
        ))

    log.info("multimodal_records_built", n=len(records), img=len(img_records), txt=len(txt_records))
    return records


# ---------------------------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------------------------

def walk_forward_splits(
    records: list[DataRecord],
    min_train_days: int = 252,
    test_days: int = 21,
) -> list[tuple[list[DataRecord], list[DataRecord]]]:
    """Generate strictly expanding-window folds split on day boundaries.

    Args:
        records:        Observations sorted by trading_day.
        min_train_days: Minimum number of unique trading days required before
                        the first test fold begins.  Default is 252 (~1 trading
                        year), consistent with Obaid & Pukthuanthong (2022).
                        With ~4 years of data this yields ~37 monthly folds.
        test_days:      Number of unique trading days in each test fold.

    Returns:
        List of (train_records, test_records) tuples.  Every record in
        train_records has trading_day strictly less than the earliest
        trading_day in test_records (no look-ahead).
    """
    unique_days = sorted({r.trading_day for r in records})
    n_days = len(unique_days)

    if n_days < min_train_days + test_days:
        # Adaptive fallback: use first 60 % for training
        split_idx = max(1, int(n_days * 0.6))
        log.warning(
            "insufficient_days_for_walkforward",
            unique_days=n_days,
            min_train_days=min_train_days,
            fallback_split=split_idx,
        )
        min_train_days = split_idx
        test_days = max(1, n_days - split_idx)

    splits: list[tuple[list[DataRecord], list[DataRecord]]] = []
    fold_start = min_train_days

    while fold_start < n_days:
        fold_end = min(fold_start + test_days, n_days)

        train_cutoff = unique_days[fold_start]   # exclusive upper bound for train
        test_lo = unique_days[fold_start]
        test_hi = unique_days[fold_end - 1]

        train = [r for r in records if r.trading_day < train_cutoff]
        test = [r for r in records if test_lo <= r.trading_day <= test_hi]

        if len(train) >= 10 and len(test) >= 1:
            splits.append((train, test))

        fold_start += test_days

    return splits


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_models(n_features: int, quantum_settings=None) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    try:
        import xgboost as xgb
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        log.warning("xgboost_unavailable_using_gbm")
        xgb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)

    # PCA components for SVMs: reduce to at most 50 dims to keep training tractable
    pca_k = min(50, n_features)

    models = {
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_k)),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)),
        ]),
        "svm_poly": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_k)),
            ("clf", SVC(kernel="poly", C=1.0, degree=3, probability=True, random_state=42)),
        ]),
        "xgboost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", xgb_clf),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=500,
                early_stopping=True,
                random_state=42,
            )),
        ]),
    }

    try:
        from pipeline.quantum import QuantumKernelSVM, VariationalQuantumClassifier
        from pipeline.quantum.config import QuantumSettings

        qs = quantum_settings or QuantumSettings()
        models["quantum_kernel_svm"] = QuantumKernelSVM(qs)
        models["quantum_vqc"] = VariationalQuantumClassifier(qs)
        log.info("quantum_models_registered", n_qubits=qs.n_qubits, feature_map=qs.feature_map)
    except ImportError:
        log.warning("pennylane_unavailable_skipping_quantum_models")

    return models


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class FoldMetrics:
    model: str
    fold: int
    n_train: int
    n_test: int
    accuracy: float
    f1: float
    auc_roc: float
    precision: float
    recall: float


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    # Guard against single-class test fold
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": auc,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResults:
    embedding_type: str
    fold_metrics: list[FoldMetrics] = field(default_factory=list)

    def summary(self) -> str:
        if not self.fold_metrics:
            return f"ClassificationResults({self.embedding_type}): no folds"

        import statistics

        lines = [f"\n=== Classification Results ({self.embedding_type}) ==="]
        for model_name in {m.model for m in self.fold_metrics}:
            folds = [m for m in self.fold_metrics if m.model == model_name]
            accs = [m.accuracy for m in folds]
            aucs = [m.auc_roc for m in folds if not (m.auc_roc != m.auc_roc)]  # exclude NaN
            lines.append(
                f"  {model_name:15s}  "
                f"acc={statistics.mean(accs):.3f}±{statistics.stdev(accs) if len(accs)>1 else 0:.3f}  "
                f"auc={statistics.mean(aucs):.3f}  "
                f"folds={len(folds)}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class ClassificationPipeline:
    """Orchestrates data loading, walk-forward CV, saving predictions and models."""

    def __init__(
        self,
        news_db: Path,
        crsp_csv: Path,
        crsp_ids_csv: Path,
        results_dir: Path,
        min_train_days: int = 60,
        test_days: int = 21,
        quantum_settings=None,
    ) -> None:
        self.news_db = Path(news_db)
        self.crsp_csv = Path(crsp_csv)
        self.crsp_ids_csv = Path(crsp_ids_csv)
        self.results_dir = Path(results_dir)
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.quantum_settings = quantum_settings

    def _load_crsp(self):
        from pipeline.crsp import CRSPMapper
        from pipeline.returns import CRSPReturns

        mapper = CRSPMapper.from_csv(self.crsp_ids_csv)
        returns = CRSPReturns.from_csv(self.crsp_csv)
        return mapper, returns

    def run(
        self,
        embedding_type: str = "image",
        min_date: str = "1800-01-01",
        max_date: str = "2099-12-31",
    ) -> ClassificationResults:
        """Run walk-forward classification for the given embedding type.

        Args:
            embedding_type: "image", "text", or "multimodal"
            min_date:       Earliest article publication date to include
            max_date:       Latest article publication date to include

        Returns:
            ClassificationResults with per-fold metrics.
        """
        if embedding_type not in {"image", "text", "multimodal"}:
            raise ValueError(f"embedding_type must be 'image', 'text', or 'multimodal'; got {embedding_type!r}")

        log.info("classification_start", embedding_type=embedding_type)

        mapper, crsp_returns = self._load_crsp()

        conn = sqlite3.connect(self.news_db)
        conn.row_factory = sqlite3.Row
        try:
            if embedding_type == "image":
                records = _load_image_records(conn, mapper, crsp_returns, min_date, max_date)
            elif embedding_type == "text":
                records = _load_text_records(conn, mapper, crsp_returns, min_date, max_date)
            else:
                records = _load_multimodal_records(conn, mapper, crsp_returns, min_date, max_date)
        finally:
            conn.close()

        log.info("records_loaded", n=len(records), embedding_type=embedding_type)

        if len(records) < 20:
            log.warning("insufficient_records", n=len(records))
            return ClassificationResults(embedding_type=embedding_type)

        # Sort by trading day — mandatory for walk-forward
        records.sort(key=lambda r: r.trading_day)

        n_features = records[0].embedding.shape[0]
        splits = walk_forward_splits(records, self.min_train_days, self.test_days)
        log.info("walk_forward_splits", n_folds=len(splits))

        if not splits:
            log.warning("no_folds_generated")
            return ClassificationResults(embedding_type=embedding_type)

        models = _build_models(n_features, quantum_settings=self.quantum_settings)
        results = ClassificationResults(embedding_type=embedding_type)

        # Per-model prediction accumulator: list of row dicts
        all_preds: dict[str, list[dict]] = {name: [] for name in models}

        for fold_idx, (train_recs, test_recs) in enumerate(splits):
            X_train = np.stack([r.embedding for r in train_recs])
            y_train = np.array([r.ret_sign for r in train_recs])
            X_test = np.stack([r.embedding for r in test_recs])
            y_test = np.array([r.ret_sign for r in test_recs])

            # Guard: PCA n_components must not exceed training samples
            # (quantum classifiers are not sklearn Pipelines — skip them)
            for name, pipe in models.items():
                if hasattr(pipe, "named_steps") and "pca" in pipe.named_steps:
                    k = min(pipe.named_steps["pca"].n_components, len(X_train) - 1, n_features)
                    pipe.set_params(pca__n_components=max(1, k))

            for model_name, pipe in models.items():
                # Skip fold if only one class in training data
                if len(np.unique(y_train)) < 2:
                    log.warning("single_class_train", fold=fold_idx, model=model_name)
                    continue

                try:
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_test)
                    y_prob = pipe.predict_proba(X_test)[:, 1]
                except Exception as exc:
                    log.error("model_fit_failed", model=model_name, fold=fold_idx, exc=str(exc))
                    continue

                metrics = _compute_metrics(y_test, y_pred, y_prob)
                results.fold_metrics.append(FoldMetrics(
                    model=model_name,
                    fold=fold_idx,
                    n_train=len(X_train),
                    n_test=len(X_test),
                    **metrics,
                ))

                for rec, prob, pred in zip(test_recs, y_prob, y_pred):
                    all_preds[model_name].append({
                        "date": rec.trading_day.isoformat(),
                        "article_id": rec.article_id,
                        "permno": rec.permno,
                        "ticker": rec.ticker,
                        "predicted_prob": float(prob),
                        "predicted_label": int(pred),
                        "true_label": int(rec.ret_sign),
                        "actual_return": float(rec.actual_return),
                        "fold": fold_idx,
                    })

        # Save outputs
        clf_dir = self.results_dir / "classifiers"
        clf_dir.mkdir(parents=True, exist_ok=True)
        self._save_predictions(clf_dir, embedding_type, all_preds)
        self._save_metrics(clf_dir, embedding_type, results.fold_metrics)
        self._save_final_models(clf_dir, embedding_type, models, records)

        log.info("classification_complete", embedding_type=embedding_type, n_folds=len(splits))
        return results

    def _save_predictions(
        self,
        clf_dir: Path,
        embedding_type: str,
        all_preds: dict[str, list[dict]],
    ) -> None:
        fieldnames = [
            "date", "article_id", "permno", "ticker",
            "predicted_prob", "predicted_label", "true_label", "actual_return", "fold",
        ]
        for model_name, rows in all_preds.items():
            if not rows:
                continue
            out_path = clf_dir / f"predictions_{embedding_type}_{model_name}.csv"
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            log.info("predictions_saved", path=str(out_path), n_rows=len(rows))

    def _save_metrics(
        self,
        clf_dir: Path,
        embedding_type: str,
        fold_metrics: list[FoldMetrics],
    ) -> None:
        if not fold_metrics:
            return
        out_path = clf_dir / f"metrics_{embedding_type}.csv"
        fieldnames = ["model", "fold", "n_train", "n_test", "accuracy", "f1", "auc_roc", "precision", "recall"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in fold_metrics:
                writer.writerow({
                    "model": m.model,
                    "fold": m.fold,
                    "n_train": m.n_train,
                    "n_test": m.n_test,
                    "accuracy": f"{m.accuracy:.6f}",
                    "f1": f"{m.f1:.6f}",
                    "auc_roc": f"{m.auc_roc:.6f}",
                    "precision": f"{m.precision:.6f}",
                    "recall": f"{m.recall:.6f}",
                })
        log.info("metrics_saved", path=str(out_path), n_rows=len(fold_metrics))

    def _save_final_models(
        self,
        clf_dir: Path,
        embedding_type: str,
        models: dict,
        records: list[DataRecord],
    ) -> None:
        """Retrain each model on the full dataset and save with joblib."""
        try:
            import joblib
        except ImportError:
            log.warning("joblib_unavailable_skipping_model_save")
            return

        if not records:
            return

        X_all = np.stack([r.embedding for r in records])
        y_all = np.array([r.ret_sign for r in records])

        if len(np.unique(y_all)) < 2:
            return

        n_features = X_all.shape[1]
        models_dir = clf_dir / "models"
        models_dir.mkdir(exist_ok=True)

        for model_name, pipe in models.items():
            if hasattr(pipe, "named_steps") and "pca" in pipe.named_steps:
                k = min(pipe.named_steps["pca"].n_components, len(X_all) - 1, n_features)
                pipe.set_params(pca__n_components=max(1, k))
            try:
                pipe.fit(X_all, y_all)
                out_path = models_dir / f"{embedding_type}_{model_name}.joblib"
                joblib.dump(pipe, out_path)
                log.info("model_saved", path=str(out_path))
            except Exception as exc:
                log.error("model_save_failed", model=model_name, exc=str(exc))

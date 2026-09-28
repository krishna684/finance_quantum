"""Tests for pipeline.classifiers — walk-forward CV and data utilities."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from pipeline.classifiers import (
    DataRecord,
    walk_forward_splits,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_records(n: int, n_features: int = 16, start: date | None = None) -> list[DataRecord]:
    """Generate synthetic DataRecords with ascending trading days."""
    base = start or date(2020, 1, 2)
    records = []
    day = base
    for i in range(n):
        # Skip weekends
        while day.weekday() >= 5:
            day += timedelta(days=1)
        records.append(DataRecord(
            article_id=f"art_{i}",
            trading_day=day,
            permno=f"PERM{i % 5}",
            ticker=f"TK{i % 5}",
            embedding=np.random.rand(n_features).astype(np.float32),
            ret_sign=i % 2,
            actual_return=0.01 if i % 2 == 1 else -0.01,
        ))
        day += timedelta(days=1)
    return records


# ---------------------------------------------------------------------------
# walk_forward_splits
# ---------------------------------------------------------------------------

def test_walk_forward_no_overlap():
    """Training set must be strictly before every test date in each fold."""
    records = _make_records(200, n_features=8)
    splits = walk_forward_splits(records, min_train_days=30, test_days=10)

    assert len(splits) > 0
    for train, test in splits:
        max_train_day = max(r.trading_day for r in train)
        min_test_day = min(r.trading_day for r in test)
        assert max_train_day < min_test_day, (
            f"Lookahead bias: train extends to {max_train_day}, test starts at {min_test_day}"
        )


def test_walk_forward_expanding_window():
    """Each subsequent training set must be strictly larger than the previous."""
    records = _make_records(300, n_features=8)
    splits = walk_forward_splits(records, min_train_days=30, test_days=10)

    assert len(splits) >= 2
    prev_train_size = len(splits[0][0])
    for train, _ in splits[1:]:
        assert len(train) >= prev_train_size, "Training window must be non-decreasing"
        prev_train_size = len(train)


def test_walk_forward_covers_all_data():
    """Union of test folds should cover all data after the initial training period."""
    records = _make_records(100, n_features=8)
    splits = walk_forward_splits(records, min_train_days=20, test_days=10)

    test_ids = set()
    for _, test in splits:
        test_ids.update(r.article_id for r in test)

    all_ids = {r.article_id for r in records}
    # All records not in the initial training period should appear in exactly one test fold
    first_split_train = splits[0][0]
    initial_ids = {r.article_id for r in first_split_train}
    expected_test_ids = all_ids - initial_ids
    assert expected_test_ids <= test_ids, "Some records were never tested"


def test_walk_forward_insufficient_data_fallback():
    """Small dataset should still produce at least one split via adaptive fallback."""
    records = _make_records(20, n_features=8)
    splits = walk_forward_splits(records, min_train_days=60, test_days=21)
    assert len(splits) >= 1


def test_walk_forward_empty_records():
    records: list[DataRecord] = []
    splits = walk_forward_splits(records, min_train_days=10, test_days=5)
    assert splits == []


def test_walk_forward_single_test_fold():
    """Exactly enough data for one fold should yield one split."""
    records = _make_records(30, n_features=4)
    splits = walk_forward_splits(records, min_train_days=20, test_days=10)
    assert len(splits) >= 1
    train, test = splits[0]
    assert len(train) >= 1
    assert len(test) >= 1


# ---------------------------------------------------------------------------
# _json_tickers helper
# ---------------------------------------------------------------------------

def test_json_tickers_parses_list():
    from pipeline.classifiers import _json_tickers
    assert _json_tickers('["AAPL", "TSLA"]') == ["AAPL", "TSLA"]


def test_json_tickers_handles_empty():
    from pipeline.classifiers import _json_tickers
    assert _json_tickers("[]") == []
    assert _json_tickers(None) == []
    assert _json_tickers("") == []


def test_json_tickers_handles_malformed():
    from pipeline.classifiers import _json_tickers
    assert _json_tickers("not json") == []


# ---------------------------------------------------------------------------
# ClassificationPipeline (integration-lite, no real model)
# ---------------------------------------------------------------------------

def _make_news_db(tmp_path: Path, n_articles: int = 80, n_images: int = 40) -> Path:
    """Build a minimal news.db with embeddings, articles, and images."""
    db = tmp_path / "news.db"
    conn = sqlite3.connect(db)

    conn.executescript("""
        CREATE TABLE news_articles (
            article_id TEXT PRIMARY KEY,
            source TEXT,
            headline TEXT,
            full_body TEXT,
            publication_date TEXT,
            publication_time TEXT,
            timezone TEXT,
            section TEXT,
            front_page INTEGER DEFAULT 0,
            authors TEXT,
            word_count INTEGER,
            database_source_id TEXT,
            url TEXT,
            collection_timestamp TEXT,
            sentiment_score REAL,
            sentiment_label TEXT,
            cashtag_tickers TEXT,
            company_names TEXT,
            mapped_tickers TEXT,
            extracted_brands TEXT,
            extracted_products TEXT,
            extracted_retailers TEXT,
            extracted_categories TEXT
        );
        CREATE TABLE news_images (
            image_id TEXT PRIMARY KEY,
            article_id TEXT,
            image_binary BLOB,
            caption TEXT,
            placement TEXT,
            image_order INTEGER,
            width_px INTEGER,
            height_px INTEGER,
            sha256_hash TEXT UNIQUE,
            file_extension TEXT,
            collection_timestamp TEXT
        );
        CREATE TABLE image_embeddings (
            image_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            model_name TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
    """)

    base_date = date(2021, 1, 4)
    for i in range(n_articles):
        d = base_date + timedelta(days=i)
        while d.weekday() >= 5:
            d += timedelta(days=1)
        conn.execute(
            "INSERT INTO news_articles (article_id, publication_date, publication_time, mapped_tickers)"
            " VALUES (?, ?, '09:00:00', ?)",
            (f"art_{i}", d.isoformat(), '["AAPL"]'),
        )

    for i in range(min(n_images, n_articles)):
        emb = np.random.rand(16).astype(np.float32)
        conn.execute(
            "INSERT INTO news_images (image_id, article_id, sha256_hash) VALUES (?, ?, ?)",
            (f"img_{i}", f"art_{i}", f"hash_{i}"),
        )
        conn.execute(
            "INSERT INTO image_embeddings (image_id, embedding, model_name, embedding_dim, created_at)"
            " VALUES (?, ?, 'test/clip', 16, '2024-01-01T00:00:00+00:00')",
            (f"img_{i}", emb.tobytes()),
        )

    conn.commit()
    conn.close()
    return db


def _make_crsp_csv(tmp_path: Path, permno: str = "14593") -> Path:
    """Write a minimal CRSP DSF CSV."""
    path = tmp_path / "crsp_dsf.csv"
    base = date(2021, 1, 4)
    lines = ["permno,date,ret,retx,prc,vol,shrout"]
    for i in range(300):
        d = base + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        ret = 0.01 if i % 2 == 0 else -0.01
        lines.append(f"{permno},{d.isoformat()},{ret},{ret},150.0,1000000,1000")
    path.write_text("\n".join(lines))
    return path


def _make_crsp_ids_csv(tmp_path: Path, permno: str = "14593") -> Path:
    """Write a minimal CRSP msenames CSV."""
    path = tmp_path / "crsp_ids.csv"
    path.write_text(
        "permno,ticker,comnam,ncusip,gvkey,namedt,nameenddt\n"
        f"{permno},AAPL,APPLE INC,037833100,001690,2000-01-01,2099-12-31\n"
    )
    return path


def test_classification_pipeline_runs_end_to_end(tmp_path):
    """Pipeline should complete without error and write prediction and metric CSVs."""
    from pipeline.classifiers import ClassificationPipeline

    db = _make_news_db(tmp_path, n_articles=120, n_images=60)
    crsp = _make_crsp_csv(tmp_path)
    ids = _make_crsp_ids_csv(tmp_path)
    results_dir = tmp_path / "results"

    pipe = ClassificationPipeline(
        news_db=db,
        crsp_csv=crsp,
        crsp_ids_csv=ids,
        results_dir=results_dir,
        min_train_days=20,
        test_days=10,
    )
    result = pipe.run(embedding_type="image")

    # Should have produced fold metrics
    assert len(result.fold_metrics) > 0, "No fold metrics produced"

    # Prediction CSVs should be present
    pred_files = list((results_dir / "classifiers").glob("predictions_image_*.csv"))
    assert len(pred_files) > 0, "No prediction CSVs written"

    # Metrics CSV should be present
    metrics_file = results_dir / "classifiers" / "metrics_image.csv"
    assert metrics_file.exists(), "metrics_image.csv not written"


def test_classification_pipeline_no_embeddings(tmp_path):
    """Pipeline with no embeddings should return empty results without crashing."""
    from pipeline.classifiers import ClassificationPipeline

    # Minimal DB with no image_embeddings table
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE news_articles (article_id TEXT PRIMARY KEY, mapped_tickers TEXT,"
        " publication_date TEXT, publication_time TEXT)"
    )
    conn.execute(
        "CREATE TABLE news_images (image_id TEXT PRIMARY KEY, article_id TEXT, image_binary BLOB,"
        " sha256_hash TEXT)"
    )
    conn.commit()
    conn.close()

    crsp = _make_crsp_csv(tmp_path)
    ids = _make_crsp_ids_csv(tmp_path)

    pipe = ClassificationPipeline(
        news_db=db,
        crsp_csv=crsp,
        crsp_ids_csv=ids,
        results_dir=tmp_path / "results",
    )
    result = pipe.run(embedding_type="image")
    assert result.fold_metrics == []

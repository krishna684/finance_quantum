import json
import tempfile
from pathlib import Path

import pytest
from pipeline.storage import Storage


@pytest.fixture
def storage(tmp_path):
    s = Storage(tmp_path / "test.db")
    s.init_schema()
    yield s
    s.close()


def _video_payload(video_id="v001"):
    return {
        "video_id": video_id,
        "creator_username": "testuser",
        "post_datetime_utc": "2024-01-15T12:00:00+00:00",
        "full_transcript": "where can I buy this amazing product",
        "all_hashtags": ["tiktokmademebuyit", "skincare"],
        "number_of_views": 10000,
        "number_of_likes": 500,
        "number_of_shares": 100,
        "total_comments": 200,
        "signal_type": "high_purchase_intent",
        "query_text": "where can I buy this",
        "engagement_score": 10700,
        "extracted_brands": ["product"],
        "extracted_products": ["amazing product"],
        "extracted_retailers": [],
        "extracted_categories": ["skincare"],
        "sentiment_score": 0.6,
        "sentiment_label": "positive",
        "cashtag_tickers": [],
        "company_names": [],
        "mapped_tickers": [],
        "views_per_hour": 100.0,
        "likes_per_hour": 5.0,
        "comments_per_hour": 2.0,
        "engagement_per_hour": 107.0,
        "hours_since_post": 100.0,
        "velocity_tier": "high",
    }


def _comment_payload(comment_id="c001", video_id="v001"):
    return {
        "comment_id": comment_id,
        "video_id": video_id,
        "comment_text": "is this on amazon?",
        "comment_timestamp": "2024-01-15T13:00:00+00:00",
        "comment_likes": 50,
        "reply_indicator": False,
        "parent_comment_id": None,
        "extracted_brands": [],
        "extracted_products": [],
        "extracted_retailers": ["amazon"],
        "extracted_categories": [],
        "sentiment_score": 0.0,
        "sentiment_label": "neutral",
        "cashtag_tickers": [],
        "company_names": ["amazon"],
        "mapped_tickers": ["AMZN"],
    }


def test_upsert_video(storage):
    storage.upsert_video(_video_payload())
    storage.commit()
    row = storage.conn.execute("SELECT * FROM tiktok_videos WHERE video_id='v001'").fetchone()
    assert row is not None
    assert row["creator_username"] == "testuser"
    assert row["sentiment_label"] == "positive"
    assert row["velocity_tier"] == "high"
    assert json.loads(row["extracted_categories"]) == ["skincare"]


def test_upsert_video_idempotent(storage):
    storage.upsert_video(_video_payload())
    storage.upsert_video(_video_payload())  # second upsert should not raise or duplicate
    storage.commit()
    count = storage.conn.execute("SELECT COUNT(*) FROM tiktok_videos").fetchone()[0]
    assert count == 1


def test_upsert_comment(storage):
    storage.upsert_video(_video_payload())
    storage.upsert_comment(_comment_payload())
    storage.commit()
    row = storage.conn.execute("SELECT * FROM tiktok_comments WHERE comment_id='c001'").fetchone()
    assert row is not None
    assert row["sentiment_label"] == "neutral"
    assert json.loads(row["mapped_tickers"]) == ["AMZN"]


def test_insert_mentions(storage):
    storage.upsert_video(_video_payload())
    storage.insert_mentions("video", "v001", "ticker", ["AAPL", "TSLA"])
    storage.commit()
    rows = storage.conn.execute(
        "SELECT entity_text FROM entity_mentions WHERE source_id='v001' AND entity_type='ticker'"
    ).fetchall()
    texts = {r[0] for r in rows}
    assert "AAPL" in texts
    assert "TSLA" in texts


def test_schema_migration(tmp_path):
    """Existing DB without new columns should be migrated safely."""
    db_path = tmp_path / "old.db"
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE tiktok_videos (
            video_id TEXT PRIMARY KEY,
            creator_username TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE tiktok_comments (
            comment_id TEXT PRIMARY KEY,
            video_id TEXT
        )"""
    )
    conn.commit()
    conn.close()

    s = Storage(db_path)
    s.init_schema()  # should not raise
    cols = {row[1] for row in s.conn.execute("PRAGMA table_info(tiktok_videos)")}
    assert "sentiment_score" in cols
    assert "mapped_tickers" in cols
    assert "velocity_tier" in cols
    s.close()

"""Tests for pipeline.tiktok_collection.

Covers:
  - Pass 1 filter: cashtag regex edge cases, false positives, rule logic
  - Pass 2 classifier: prompt output parsing (no LLM calls)
  - Frame naming: frame_id / thumbnail_id conventions
  - Entity extraction output schema (no LLM calls — NER-only path)
  - Checkpoint resume via MockBrightDataClient + FinCollectionStorage
  - Full chain: collector-mock → filter → store (no real API calls)

All tests are self-contained; no credentials required.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from pipeline.tiktok_collection.classifier import parse_info_trading, parse_yes_no
from pipeline.tiktok_collection.collector import MockBrightDataClient, _make_synthetic_posts
from pipeline.tiktok_collection.entity_extractor import extract_financial_entities
from pipeline.tiktok_collection.filter import Pass1Filter, build_filter
from pipeline.tiktok_collection.storage import FinCollectionStorage
from pipeline.tiktok_collection.transcriber import frame_id, thumbnail_id


# ---------------------------------------------------------------------------
# Pass 1 — cashtag regex
# ---------------------------------------------------------------------------

class TestCashtagRegex:
    """Verify the $TICKER regex catches real cases and rejects false positives."""

    def setup_method(self):
        self.f = build_filter()

    def _video(self, description: str, hashtags=None, transcript: str = "") -> dict:
        return {"post_id": "1", "description": description,
                "hashtags": hashtags or [], "transcript": transcript}

    def test_single_letter_ticker(self):
        # $A is Agilent Technologies — valid cashtag
        r = self.f.apply(self._video("$A just broke out!"))
        assert r.passed
        assert r.matched_rule == "cashtag"
        assert "A" in r.matched_tickers

    def test_five_letter_ticker(self):
        r = self.f.apply(self._video("$GOOGL earnings beat!"))
        assert r.passed
        assert "GOOGL" in r.matched_tickers

    def test_dollar_amount_not_matched(self):
        # $5 and $100 must NOT be interpreted as cashtags
        r = self.f.apply(self._video("Only $5 left in my account"))
        # should not pass on cashtag rule; may pass on other rules if AAPL etc. present
        if r.passed:
            assert r.matched_rule != "cashtag"

    def test_dollar_amount_100(self):
        r = self.f.apply(self._video("Made $100 today!"))
        if r.passed:
            assert r.matched_rule != "cashtag"

    def test_cashtag_in_transcript(self):
        r = self.f.apply(self._video("", transcript="Check out $NVDA right now"))
        assert r.passed
        assert r.matched_rule == "cashtag"

    def test_multiple_cashtags(self):
        r = self.f.apply(self._video("$AAPL and $TSLA both moving"))
        assert r.passed
        assert "AAPL" in r.matched_tickers
        assert "TSLA" in r.matched_tickers

    def test_no_cashtag_no_company_no_genre(self):
        r = self.f.apply(self._video("Today I cooked pasta for dinner"))
        assert not r.passed

    def test_six_letter_ticker_not_matched(self):
        # $ABCDEF has 6 letters → should not match cashtag regex
        r = self.f.apply(self._video("I love $ABCDEF today"))
        # cashtag rule should not fire; might still pass via ticker_name if ABCDEF happens to be a known ticker
        if r.passed and r.matched_rule == "cashtag":
            # This would be wrong — $ABCDEF should not match
            assert False, "$ABCDEF has 6 letters and should NOT match cashtag regex"


class TestTickerNameRule:
    def setup_method(self):
        self.f = build_filter()

    def _video(self, description: str, hashtags=None, transcript: str = "") -> dict:
        return {"post_id": "2", "description": description,
                "hashtags": hashtags or [], "transcript": transcript}

    def test_company_name_apple(self):
        r = self.f.apply(self._video("apple is about to release earnings next week"))
        assert r.passed
        assert r.matched_rule == "ticker_name"

    def test_bare_ticker_in_uppercase(self):
        r = self.f.apply(self._video("NVDA is printing money for holders"))
        assert r.passed

    def test_company_name_in_transcript(self):
        r = self.f.apply(self._video("", transcript="I've been watching tesla all week"))
        assert r.passed


class TestGenreMarketRule:
    def setup_method(self):
        self.f = build_filter()

    def _video(self, description: str, hashtags: list, transcript: str = "") -> dict:
        return {"post_id": "3", "description": description,
                "hashtags": hashtags, "transcript": transcript}

    def test_genre_plus_market_term(self):
        r = self.f.apply(self._video(
            "Market crash incoming — buy the dip",
            hashtags=["stocktok"],
        ))
        assert r.passed
        assert r.matched_rule == "genre_market"

    def test_genre_hashtag_only_not_enough(self):
        r = self.f.apply(self._video(
            "Just chilling and vibing today",
            hashtags=["fintok"],
        ))
        # No market term → should NOT pass on genre_market rule
        # (might pass on ticker/cashtag if text contains one)
        if r.passed:
            assert r.matched_rule != "genre_market"

    def test_market_term_only_not_enough(self):
        r = self.f.apply(self._video(
            "Fed rate decision tomorrow is going to be huge",
            hashtags=["cooking"],  # Not a finance genre tag
        ))
        if r.passed:
            assert r.matched_rule != "genre_market"

    def test_hashtag_with_hash_prefix(self):
        # Hashtags may come in with # prefix
        r = self.f.apply(self._video(
            "earnings report coming up buy the dip now",
            hashtags=["#stocktok"],
        ))
        assert r.passed


# ---------------------------------------------------------------------------
# Pass 2 — prompt output parsing
# ---------------------------------------------------------------------------

class TestParseYesNo:
    def test_yes_exact(self):
        assert parse_yes_no("YES") == "YES"

    def test_no_exact(self):
        assert parse_yes_no("NO") == "NO"

    def test_yes_with_noise(self):
        assert parse_yes_no("YES, this is about stocks") == "YES"

    def test_no_with_noise(self):
        assert parse_yes_no("NO, this is unrelated") == "NO"

    def test_unknown_gibberish(self):
        assert parse_yes_no("MAYBE") == "UNKNOWN"

    def test_lowercase(self):
        assert parse_yes_no("yes") == "YES"

    def test_empty(self):
        assert parse_yes_no("") == "UNKNOWN"


class TestParseInfoTrading:
    def test_information_exact(self):
        assert parse_info_trading("INFORMATION") == "INFORMATION"

    def test_trading_exact(self):
        assert parse_info_trading("TRADING") == "TRADING"

    def test_information_with_noise(self):
        assert parse_info_trading("INFORMATION: this is analysis") == "INFORMATION"

    def test_trading_with_noise(self):
        assert parse_info_trading("The answer is TRADING") == "TRADING"

    def test_unknown(self):
        assert parse_info_trading("NEUTRAL") == "UNKNOWN"

    def test_lowercase(self):
        assert parse_info_trading("information") == "INFORMATION"


# ---------------------------------------------------------------------------
# Frame naming
# ---------------------------------------------------------------------------

class TestFrameNaming:
    def test_frame_id_padding(self):
        assert frame_id("7123456789", 2) == "7123456789_fr00002"

    def test_frame_id_large_ts(self):
        assert frame_id("abc", 99) == "abc_fr00099"

    def test_thumbnail_id(self):
        assert thumbnail_id("7000000001") == "7000000001_thumb"

    def test_frame_id_unique_per_ts(self):
        assert frame_id("X", 2) != frame_id("X", 4)
        assert frame_id("X", 2) != frame_id("Y", 2)


# ---------------------------------------------------------------------------
# Entity extraction (NER-only, no LLM)
# ---------------------------------------------------------------------------

class TestEntityExtraction:
    def test_cashtag_detected(self):
        rows = extract_financial_entities(
            post_id="p1",
            post_date="2025-01-15",
            description="$AAPL is going to moon!",
            transcript="",
            anthropic_client=None,
            model="",
        )
        tickers = [r.ticker for r in rows]
        assert "AAPL" in tickers

    def test_company_name_detected(self):
        rows = extract_financial_entities(
            post_id="p2",
            post_date="2025-01-15",
            description="",
            transcript="Tesla delivery numbers just dropped and they are incredible",
            anthropic_client=None,
            model="",
        )
        tickers = [r.ticker for r in rows]
        assert "TSLA" in tickers

    def test_no_entity_returns_empty(self):
        rows = extract_financial_entities(
            post_id="p3",
            post_date="2025-01-15",
            description="I made pasta for dinner today",
            transcript="",
            anthropic_client=None,
            model="",
        )
        assert rows == []

    def test_schema_fields(self):
        rows = extract_financial_entities(
            post_id="p4",
            post_date="2025-01-15",
            description="$NVDA is a buy",
            transcript="",
            anthropic_client=None,
            model="",
        )
        assert len(rows) > 0
        r = rows[0]
        assert hasattr(r, "post_id")
        assert hasattr(r, "post_date")
        assert hasattr(r, "ticker")
        assert hasattr(r, "company_name")
        assert hasattr(r, "stance")
        assert hasattr(r, "has_position")
        assert r.stance in ("BULLISH", "BEARISH", "NEUTRAL")
        assert isinstance(r.has_position, bool)

    def test_fallback_stance_neutral(self):
        rows = extract_financial_entities(
            post_id="p5",
            post_date="2025-01-15",
            description="$GME",
            transcript="",
            anthropic_client=None,  # No LLM → NEUTRAL
            model="",
        )
        for r in rows:
            assert r.stance == "NEUTRAL"

    def test_post_id_propagated(self):
        rows = extract_financial_entities(
            post_id="my_post_999",
            post_date="2025-03-01",
            description="$AAPL $TSLA",
            transcript="",
            anthropic_client=None,
            model="",
        )
        for r in rows:
            assert r.post_id == "my_post_999"
            assert r.post_date == "2025-03-01"


# ---------------------------------------------------------------------------
# Storage: schema + basic CRUD
# ---------------------------------------------------------------------------

class TestFinCollectionStorage:
    def setup_method(self):
        self._tmp = tempfile.mkdtemp()
        self.db_path = Path(self._tmp) / "test_fin.db"
        self.storage = FinCollectionStorage(self.db_path)
        self.storage.init_schema()

    def teardown_method(self):
        self.storage.close()

    def test_schema_creates_tables(self):
        tables = {row[0] for row in self.storage.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "tiktok_fin_posts" in tables
        assert "tiktok_frames" in tables
        assert "tiktok_entity_rows" in tables
        assert "image_embeddings" in tables
        assert "text_embeddings" in tables

    def test_upsert_post_round_trip(self):
        self.storage.upsert_post({
            "post_id": "111",
            "url": "https://tiktok.com/v/111",
            "description": "$AAPL is up",
            "hashtags": ["stocktok"],
            "post_date": "2025-01-15",
            "creator_username": "trader1",
            "play_count": 5000,
            "digg_count": 200,
            "pass1_matched": True,
            "llm_topic": "YES",
        })
        self.storage.commit()
        row = self.storage.get_post("111")
        assert row is not None
        assert row["post_id"] == "111"
        assert row["llm_topic"] == "YES"
        assert row["play_count"] == 5000

    def test_upsert_post_idempotent(self):
        payload = {"post_id": "222", "description": "test", "play_count": 100}
        self.storage.upsert_post(payload)
        payload["play_count"] = 200
        self.storage.upsert_post(payload)
        self.storage.commit()
        row = self.storage.get_post("222")
        assert row["play_count"] == 200

    def test_upsert_frame(self):
        self.storage.upsert_post({"post_id": "333", "description": "x"})
        self.storage.upsert_frame("333_fr00002", "333", 2, b"\xff\xd8\xff" + b"\x00" * 100)
        self.storage.commit()
        assert self.storage.frame_exists("333_fr00002")
        assert not self.storage.frame_exists("333_fr00004")

    def test_insert_entity_rows(self):
        self.storage.upsert_post({"post_id": "444", "description": "x"})
        self.storage.insert_entity_rows([
            {"post_id": "444", "post_date": "2025-01-15",
             "ticker": "AAPL", "company_name": "Apple",
             "stance": "BULLISH", "has_position": False},
        ])
        self.storage.commit()
        assert self.storage.entity_rows_exist("444")
        assert not self.storage.entity_rows_exist("999")

    def test_set_transcript(self):
        self.storage.upsert_post({"post_id": "555", "description": "x"})
        self.storage.set_transcript("555", "AAPL just reported earnings")
        self.storage.commit()
        row = self.storage.get_post("555")
        assert row["transcript"] == "AAPL just reported earnings"

    def test_set_mapped_tickers(self):
        self.storage.upsert_post({"post_id": "666", "description": "x"})
        self.storage.set_mapped_tickers("666", ["AAPL", "TSLA"])
        self.storage.commit()
        row = self.storage.get_post("666")
        assert json.loads(row["mapped_tickers"]) == ["AAPL", "TSLA"]

    def test_unembedded_transcripts_query(self):
        self.storage.upsert_post({
            "post_id": "777", "description": "x", "transcript": "hello"
        })
        self.storage.commit()
        pending = self.storage.unembedded_transcripts()
        pids = [p for p, _ in pending]
        assert "777" in pids

    def test_unembedded_frames_query(self):
        self.storage.upsert_post({"post_id": "888", "description": "x"})
        self.storage.upsert_frame("888_fr00002", "888", 2, b"\xff" * 50)
        self.storage.commit()
        pending = self.storage.unembedded_frames()
        fids = [f for f, _ in pending]
        assert "888_fr00002" in fids


# ---------------------------------------------------------------------------
# Collector mock
# ---------------------------------------------------------------------------

class TestMockCollector:
    def test_returns_synthetic_records(self):
        client = MockBrightDataClient()
        posts = client.search_posts("AAPL stock", limit=5)
        assert len(posts) == 5

    def test_fixture_path_loads(self):
        fixture = [
            {"post_id": "f1", "description": "$AAPL bullish", "hashtags": ["stocktok"],
             "url": "https://tiktok.com/v/f1", "play_count": 100},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(fixture, tmp)
            tmp_path = Path(tmp.name)
        try:
            client = MockBrightDataClient(fixture_path=tmp_path)
            posts = client.search_posts("x")
            assert len(posts) == 1
            assert posts[0]["post_id"] == "f1"
        finally:
            tmp_path.unlink()

    def test_limit_respected(self):
        client = MockBrightDataClient()
        posts = client.search_posts("TSLA stock", limit=3)
        assert len(posts) <= 3

    def test_has_required_fields(self):
        posts = _make_synthetic_posts("$GME", 1)
        p = posts[0]
        for field in ("post_id", "url", "description", "hashtags", "play_count"):
            assert field in p, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# Full chain integration (no external APIs)
# ---------------------------------------------------------------------------

class TestFullChainNoAPI:
    """Run collector-mock → filter → store with fixture data.

    No external APIs are called; validates the pipeline plumbing.
    """

    def setup_method(self):
        self._tmp = tempfile.mkdtemp()
        self.db_path = Path(self._tmp) / "test_fin_chain.db"
        self.storage = FinCollectionStorage(self.db_path)
        self.storage.init_schema()
        self.filt = build_filter()

    def teardown_method(self):
        self.storage.close()

    def _fixture_posts(self) -> list[dict]:
        return [
            {
                "post_id": f"chain_{i}",
                "url": f"https://tiktok.com/v/{i}",
                "description": f"$AAPL earnings report coming up #{['stocktok', 'fintok'][i % 2]}",
                "hashtags": ["stocktok", "earnings"],
                "create_time": "2025-01-15T14:30:00Z",
                "play_count": 1000 + i * 100,
                "digg_count": 50 + i * 10,
                "share_count": 5,
                "comment_count": 20,
                "profile_username": f"trader_{i}",
                "region": "US",
            }
            for i in range(5)
        ]

    def test_filter_then_store(self):
        posts = self._fixture_posts()

        # Pass 1
        survivors = self.filt.apply_batch(posts)
        assert len(survivors) > 0, "All fixture posts should pass Pass 1 (contain $AAPL)"

        # Store
        for post in survivors:
            self.storage.upsert_post({
                "post_id": str(post["post_id"]),
                "url": post.get("url"),
                "description": post.get("description"),
                "hashtags": post.get("hashtags") or [],
                "post_date": "2025-01-15",
                "creator_username": post.get("profile_username"),
                "play_count": post.get("play_count", 0),
                "digg_count": post.get("digg_count", 0),
                "share_count": post.get("share_count", 0),
                "comment_count": post.get("comment_count", 0),
                "pass1_matched": True,
                "llm_topic": "YES",  # simulated Pass 2 result
            })
        self.storage.commit()

        # Verify all stored
        for i in range(5):
            row = self.storage.get_post(f"chain_{i}")
            assert row is not None

    def test_entity_extraction_stored(self):
        post = {
            "post_id": "ent_test",
            "post_date": "2025-01-15",
            "description": "$TSLA and $NVDA are my top picks",
            "transcript": "",
        }
        self.storage.upsert_post({**post, "play_count": 0, "digg_count": 0})

        rows = extract_financial_entities(
            post_id="ent_test",
            post_date="2025-01-15",
            description=post["description"],
            transcript=post["transcript"],
            anthropic_client=None,
            model="",
        )
        assert len(rows) >= 1

        self.storage.insert_entity_rows([
            {"post_id": r.post_id, "post_date": r.post_date,
             "ticker": r.ticker, "company_name": r.company_name,
             "stance": r.stance, "has_position": r.has_position}
            for r in rows
        ])
        self.storage.commit()

        assert self.storage.entity_rows_exist("ent_test")

    def test_checkpoint_resume(self):
        """CheckpointStore prevents re-processing already-done windows."""
        from pipeline.checkpoint import CheckpointStore

        ckpt = CheckpointStore(self.db_path)
        try:
            ckpt.mark_done("fin:$AAPL", "20250101", "20250131", videos_saved=3)
            assert ckpt.is_done("fin:$AAPL", "20250101", "20250131")
            assert not ckpt.is_done("fin:$AAPL", "20250201", "20250228")
            assert not ckpt.is_done("fin:$TSLA", "20250101", "20250131")
        finally:
            ckpt.close()

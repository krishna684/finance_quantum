from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS tiktok_videos (
            video_id TEXT PRIMARY KEY,
            creator_username TEXT,
            post_datetime_utc TEXT,
            full_transcript TEXT,
            all_hashtags TEXT,
            number_of_views INTEGER,
            number_of_likes INTEGER,
            number_of_shares INTEGER,
            total_comments INTEGER,
            collection_timestamp TEXT,
            signal_type TEXT,
            query_text TEXT,
            engagement_score INTEGER,
            extracted_brands TEXT,
            extracted_products TEXT,
            extracted_retailers TEXT,
            extracted_categories TEXT,
            sentiment_score REAL,
            sentiment_label TEXT,
            cashtag_tickers TEXT,
            company_names TEXT,
            mapped_tickers TEXT,
            views_per_hour REAL,
            likes_per_hour REAL,
            comments_per_hour REAL,
            engagement_per_hour REAL,
            hours_since_post REAL,
            velocity_tier TEXT
        );

        CREATE TABLE IF NOT EXISTS tiktok_comments (
            comment_id TEXT PRIMARY KEY,
            video_id TEXT,
            comment_text TEXT,
            comment_timestamp TEXT,
            comment_likes INTEGER,
            reply_indicator INTEGER,
            parent_comment_id TEXT,
            extracted_brands TEXT,
            extracted_products TEXT,
            extracted_retailers TEXT,
            extracted_categories TEXT,
            sentiment_score REAL,
            sentiment_label TEXT,
            cashtag_tickers TEXT,
            company_names TEXT,
            mapped_tickers TEXT,
            FOREIGN KEY(video_id) REFERENCES tiktok_videos(video_id)
        );

        CREATE TABLE IF NOT EXISTS entity_mentions (
            mention_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT,
            source_id TEXT,
            entity_type TEXT,
            entity_text TEXT,
            normalized_entity TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS brand_firm_mapping (
            mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_company_name TEXT,
            ticker TEXT,
            permno TEXT,
            brand_name TEXT,
            product_line TEXT,
            product_category TEXT,
            source_type TEXT,
            source_ref TEXT,
            confidence_score REAL,
            is_public_company INTEGER DEFAULT 1,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_comments_video_id ON tiktok_comments(video_id);
        CREATE INDEX IF NOT EXISTS idx_entity_source ON entity_mentions(source_type, source_id);
        CREATE INDEX IF NOT EXISTS idx_mapping_brand ON brand_firm_mapping(brand_name);
        """
        self.conn.executescript(ddl)
        self.conn.commit()
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add columns introduced after initial schema without dropping existing data."""
        new_video_cols = [
            ("sentiment_score", "REAL"),
            ("sentiment_label", "TEXT"),
            ("cashtag_tickers", "TEXT"),
            ("company_names", "TEXT"),
            ("mapped_tickers", "TEXT"),
            ("views_per_hour", "REAL"),
            ("likes_per_hour", "REAL"),
            ("comments_per_hour", "REAL"),
            ("engagement_per_hour", "REAL"),
            ("hours_since_post", "REAL"),
            ("velocity_tier", "TEXT"),
        ]
        new_comment_cols = [
            ("sentiment_score", "REAL"),
            ("sentiment_label", "TEXT"),
            ("cashtag_tickers", "TEXT"),
            ("company_names", "TEXT"),
            ("mapped_tickers", "TEXT"),
        ]
        existing_video = {row[1] for row in self.conn.execute("PRAGMA table_info(tiktok_videos)")}
        existing_comment = {row[1] for row in self.conn.execute("PRAGMA table_info(tiktok_comments)")}

        for col, col_type in new_video_cols:
            if col not in existing_video:
                self.conn.execute(f"ALTER TABLE tiktok_videos ADD COLUMN {col} {col_type}")
        for col, col_type in new_comment_cols:
            if col not in existing_comment:
                self.conn.execute(f"ALTER TABLE tiktok_comments ADD COLUMN {col} {col_type}")
        self.conn.commit()

    def upsert_video(self, payload: dict[str, Any]) -> None:
        sql = """
        INSERT INTO tiktok_videos (
            video_id, creator_username, post_datetime_utc, full_transcript, all_hashtags,
            number_of_views, number_of_likes, number_of_shares, total_comments,
            collection_timestamp, signal_type, query_text, engagement_score,
            extracted_brands, extracted_products, extracted_retailers, extracted_categories,
            sentiment_score, sentiment_label,
            cashtag_tickers, company_names, mapped_tickers,
            views_per_hour, likes_per_hour, comments_per_hour,
            engagement_per_hour, hours_since_post, velocity_tier
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id) DO UPDATE SET
            creator_username=excluded.creator_username,
            post_datetime_utc=excluded.post_datetime_utc,
            full_transcript=excluded.full_transcript,
            all_hashtags=excluded.all_hashtags,
            number_of_views=excluded.number_of_views,
            number_of_likes=excluded.number_of_likes,
            number_of_shares=excluded.number_of_shares,
            total_comments=excluded.total_comments,
            collection_timestamp=excluded.collection_timestamp,
            signal_type=excluded.signal_type,
            query_text=excluded.query_text,
            engagement_score=excluded.engagement_score,
            extracted_brands=excluded.extracted_brands,
            extracted_products=excluded.extracted_products,
            extracted_retailers=excluded.extracted_retailers,
            extracted_categories=excluded.extracted_categories,
            sentiment_score=excluded.sentiment_score,
            sentiment_label=excluded.sentiment_label,
            cashtag_tickers=excluded.cashtag_tickers,
            company_names=excluded.company_names,
            mapped_tickers=excluded.mapped_tickers,
            views_per_hour=excluded.views_per_hour,
            likes_per_hour=excluded.likes_per_hour,
            comments_per_hour=excluded.comments_per_hour,
            engagement_per_hour=excluded.engagement_per_hour,
            hours_since_post=excluded.hours_since_post,
            velocity_tier=excluded.velocity_tier
        """
        self.conn.execute(
            sql,
            (
                str(payload["video_id"]),
                payload.get("creator_username"),
                payload.get("post_datetime_utc"),
                payload.get("full_transcript"),
                json.dumps(payload.get("all_hashtags", []), ensure_ascii=False),
                payload.get("number_of_views", 0),
                payload.get("number_of_likes", 0),
                payload.get("number_of_shares", 0),
                payload.get("total_comments", 0),
                payload.get("collection_timestamp", _utc_now_iso()),
                payload.get("signal_type"),
                payload.get("query_text"),
                payload.get("engagement_score", 0),
                json.dumps(payload.get("extracted_brands", []), ensure_ascii=False),
                json.dumps(payload.get("extracted_products", []), ensure_ascii=False),
                json.dumps(payload.get("extracted_retailers", []), ensure_ascii=False),
                json.dumps(payload.get("extracted_categories", []), ensure_ascii=False),
                payload.get("sentiment_score"),
                payload.get("sentiment_label"),
                json.dumps(payload.get("cashtag_tickers", []), ensure_ascii=False),
                json.dumps(payload.get("company_names", []), ensure_ascii=False),
                json.dumps(payload.get("mapped_tickers", []), ensure_ascii=False),
                payload.get("views_per_hour"),
                payload.get("likes_per_hour"),
                payload.get("comments_per_hour"),
                payload.get("engagement_per_hour"),
                payload.get("hours_since_post"),
                payload.get("velocity_tier"),
            ),
        )

    def upsert_comment(self, payload: dict[str, Any]) -> None:
        sql = """
        INSERT INTO tiktok_comments (
            comment_id, video_id, comment_text, comment_timestamp, comment_likes,
            reply_indicator, parent_comment_id, extracted_brands, extracted_products,
            extracted_retailers, extracted_categories, sentiment_score, sentiment_label,
            cashtag_tickers, company_names, mapped_tickers
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(comment_id) DO UPDATE SET
            video_id=excluded.video_id,
            comment_text=excluded.comment_text,
            comment_timestamp=excluded.comment_timestamp,
            comment_likes=excluded.comment_likes,
            reply_indicator=excluded.reply_indicator,
            parent_comment_id=excluded.parent_comment_id,
            extracted_brands=excluded.extracted_brands,
            extracted_products=excluded.extracted_products,
            extracted_retailers=excluded.extracted_retailers,
            extracted_categories=excluded.extracted_categories,
            sentiment_score=excluded.sentiment_score,
            sentiment_label=excluded.sentiment_label,
            cashtag_tickers=excluded.cashtag_tickers,
            company_names=excluded.company_names,
            mapped_tickers=excluded.mapped_tickers
        """
        self.conn.execute(
            sql,
            (
                str(payload["comment_id"]),
                str(payload["video_id"]),
                payload.get("comment_text"),
                payload.get("comment_timestamp"),
                payload.get("comment_likes", 0),
                1 if payload.get("reply_indicator") else 0,
                payload.get("parent_comment_id"),
                json.dumps(payload.get("extracted_brands", []), ensure_ascii=False),
                json.dumps(payload.get("extracted_products", []), ensure_ascii=False),
                json.dumps(payload.get("extracted_retailers", []), ensure_ascii=False),
                json.dumps(payload.get("extracted_categories", []), ensure_ascii=False),
                payload.get("sentiment_score"),
                payload.get("sentiment_label"),
                json.dumps(payload.get("cashtag_tickers", []), ensure_ascii=False),
                json.dumps(payload.get("company_names", []), ensure_ascii=False),
                json.dumps(payload.get("mapped_tickers", []), ensure_ascii=False),
            ),
        )

    def insert_mentions(self, source_type: str, source_id: str, entity_type: str, values: list[str]) -> None:
        now = _utc_now_iso()
        for value in values:
            self.conn.execute(
                """
                INSERT INTO entity_mentions(source_type, source_id, entity_type, entity_text, normalized_entity, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source_type, str(source_id), entity_type, value, value.lower().strip(), now),
            )

    def commit(self) -> None:
        self.conn.commit()

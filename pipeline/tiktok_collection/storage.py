from __future__ import annotations

"""SQLite storage for the financial TikTok collection pipeline.

Creates / manages a separate database (default: data/tiktok_fin.db) with
tables for posts, video frames, per-ticker entity rows, image embeddings, and
text embeddings.  All ALTER TABLE additions follow the safe migrate-on-startup
pattern used in the classical pipeline.

Tables
------
tiktok_fin_posts      — one row per Bright Data post (raw + enriched fields)
tiktok_frames         — JPEG frames extracted from downloaded videos
tiktok_entity_rows    — one row per (post, ticker) pair from entity extraction
image_embeddings      — CLIP embeddings keyed by frame_id (same schema as news.db)
text_embeddings       — FinBERT embeddings keyed by "tiktok_{post_id}" (same schema)
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class FinCollectionStorage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS tiktok_fin_posts (
            post_id              TEXT PRIMARY KEY,
            url                  TEXT,
            description          TEXT,
            hashtags             TEXT,
            transcript           TEXT,
            post_date            TEXT,
            creator_username     TEXT,
            play_count           INTEGER DEFAULT 0,
            digg_count           INTEGER DEFAULT 0,
            share_count          INTEGER DEFAULT 0,
            comment_count        INTEGER DEFAULT 0,
            region               TEXT,
            search_keyword       TEXT,
            pass1_matched        INTEGER DEFAULT 0,
            llm_topic            TEXT,
            content_type         TEXT,
            mapped_tickers       TEXT,
            collection_timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tiktok_frames (
            frame_id       TEXT PRIMARY KEY,
            post_id        TEXT NOT NULL,
            timestamp_sec  INTEGER NOT NULL,
            is_thumbnail   INTEGER DEFAULT 0,
            image_binary   BLOB NOT NULL,
            created_at     TEXT NOT NULL,
            FOREIGN KEY(post_id) REFERENCES tiktok_fin_posts(post_id)
        );

        CREATE TABLE IF NOT EXISTS tiktok_entity_rows (
            row_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id       TEXT NOT NULL,
            post_date     TEXT,
            ticker        TEXT NOT NULL,
            company_name  TEXT,
            stance        TEXT NOT NULL,
            has_position  INTEGER DEFAULT 0,
            created_at    TEXT NOT NULL,
            FOREIGN KEY(post_id) REFERENCES tiktok_fin_posts(post_id)
        );

        -- Same schema as news.db image_embeddings; image_id uses frame_id directly
        CREATE TABLE IF NOT EXISTS image_embeddings (
            image_id       TEXT PRIMARY KEY,
            embedding      BLOB NOT NULL,
            model_name     TEXT NOT NULL,
            embedding_dim  INTEGER NOT NULL,
            created_at     TEXT NOT NULL
        );

        -- Same schema as news.db text_embeddings; article_id = "tiktok_{post_id}"
        CREATE TABLE IF NOT EXISTS text_embeddings (
            article_id     TEXT PRIMARY KEY,
            embedding      BLOB NOT NULL,
            model_name     TEXT NOT NULL,
            embedding_dim  INTEGER NOT NULL,
            created_at     TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_fin_posts_date    ON tiktok_fin_posts(post_date);
        CREATE INDEX IF NOT EXISTS idx_fin_posts_ticker  ON tiktok_fin_posts(mapped_tickers);
        CREATE INDEX IF NOT EXISTS idx_frames_post       ON tiktok_frames(post_id);
        CREATE INDEX IF NOT EXISTS idx_entity_post       ON tiktok_entity_rows(post_id);
        CREATE INDEX IF NOT EXISTS idx_entity_ticker     ON tiktok_entity_rows(ticker);
        CREATE INDEX IF NOT EXISTS idx_img_emb           ON image_embeddings(image_id);
        CREATE INDEX IF NOT EXISTS idx_txt_emb           ON text_embeddings(article_id);
        """)
        self.conn.commit()
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Safely add columns that were introduced after initial schema."""
        new_post_cols: list[tuple[str, str]] = []
        existing = {row[1] for row in self.conn.execute("PRAGMA table_info(tiktok_fin_posts)")}
        for col, col_type in new_post_cols:
            if col not in existing:
                self.conn.execute(f"ALTER TABLE tiktok_fin_posts ADD COLUMN {col} {col_type}")
        self.conn.commit()

    # ------------------------------------------------------------------
    # Posts
    # ------------------------------------------------------------------

    def upsert_post(self, payload: dict[str, Any]) -> None:
        sql = """
        INSERT INTO tiktok_fin_posts (
            post_id, url, description, hashtags, transcript, post_date,
            creator_username, play_count, digg_count, share_count, comment_count,
            region, search_keyword, pass1_matched, llm_topic, content_type,
            mapped_tickers, collection_timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(post_id) DO UPDATE SET
            url=excluded.url,
            description=excluded.description,
            hashtags=excluded.hashtags,
            transcript=excluded.transcript,
            post_date=excluded.post_date,
            creator_username=excluded.creator_username,
            play_count=excluded.play_count,
            digg_count=excluded.digg_count,
            share_count=excluded.share_count,
            comment_count=excluded.comment_count,
            region=excluded.region,
            search_keyword=excluded.search_keyword,
            pass1_matched=excluded.pass1_matched,
            llm_topic=excluded.llm_topic,
            content_type=excluded.content_type,
            mapped_tickers=excluded.mapped_tickers,
            collection_timestamp=excluded.collection_timestamp
        """
        self.conn.execute(sql, (
            str(payload["post_id"]),
            payload.get("url"),
            payload.get("description"),
            json.dumps(payload.get("hashtags") or [], ensure_ascii=False),
            payload.get("transcript"),
            payload.get("post_date"),
            payload.get("creator_username"),
            int(payload.get("play_count") or 0),
            int(payload.get("digg_count") or 0),
            int(payload.get("share_count") or 0),
            int(payload.get("comment_count") or 0),
            payload.get("region"),
            payload.get("search_keyword"),
            1 if payload.get("pass1_matched") else 0,
            payload.get("llm_topic"),
            payload.get("content_type"),
            json.dumps(payload.get("mapped_tickers") or [], ensure_ascii=False),
            payload.get("collection_timestamp") or _utc_now_iso(),
        ))

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    def upsert_frame(
        self,
        frame_id: str,
        post_id: str,
        timestamp_sec: int,
        image_binary: bytes,
        is_thumbnail: bool = False,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO tiktok_frames
                (frame_id, post_id, timestamp_sec, is_thumbnail, image_binary, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (frame_id, post_id, timestamp_sec, 1 if is_thumbnail else 0,
             image_binary, _utc_now_iso()),
        )

    def frame_exists(self, frame_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM tiktok_frames WHERE frame_id=?", (frame_id,)
        ).fetchone() is not None

    # ------------------------------------------------------------------
    # Entity rows
    # ------------------------------------------------------------------

    def insert_entity_rows(self, rows: list[dict[str, Any]]) -> None:
        now = _utc_now_iso()
        self.conn.executemany(
            """
            INSERT INTO tiktok_entity_rows
                (post_id, post_date, ticker, company_name, stance, has_position, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["post_id"], r.get("post_date"), r["ticker"],
                    r.get("company_name"), r["stance"],
                    1 if r.get("has_position") else 0, now,
                )
                for r in rows
            ],
        )

    def entity_rows_exist(self, post_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM tiktok_entity_rows WHERE post_id=?", (post_id,)
        ).fetchone() is not None

    # ------------------------------------------------------------------
    # Transcript helpers
    # ------------------------------------------------------------------

    def set_transcript(self, post_id: str, transcript: str) -> None:
        self.conn.execute(
            "UPDATE tiktok_fin_posts SET transcript=? WHERE post_id=?",
            (transcript, post_id),
        )

    def set_content_type(self, post_id: str, content_type: str) -> None:
        self.conn.execute(
            "UPDATE tiktok_fin_posts SET content_type=? WHERE post_id=?",
            (content_type, post_id),
        )

    def set_mapped_tickers(self, post_id: str, tickers: list[str]) -> None:
        self.conn.execute(
            "UPDATE tiktok_fin_posts SET mapped_tickers=? WHERE post_id=?",
            (json.dumps(tickers, ensure_ascii=False), post_id),
        )

    # ------------------------------------------------------------------
    # Queries used by embedders
    # ------------------------------------------------------------------

    def unembedded_frames(self) -> list[tuple[str, bytes]]:
        """Return (frame_id, image_binary) for frames not yet in image_embeddings."""
        rows = self.conn.execute(
            """
            SELECT f.frame_id, f.image_binary
            FROM tiktok_frames f
            LEFT JOIN image_embeddings e ON e.image_id = f.frame_id
            WHERE e.image_id IS NULL AND f.image_binary IS NOT NULL
            """
        ).fetchall()
        return [(r["frame_id"], bytes(r["image_binary"])) for r in rows]

    def unembedded_transcripts(self) -> list[tuple[str, str]]:
        """Return (post_id, transcript) for posts not yet in text_embeddings."""
        rows = self.conn.execute(
            """
            SELECT p.post_id, p.transcript
            FROM tiktok_fin_posts p
            LEFT JOIN text_embeddings e ON e.article_id = 'tiktok_' || p.post_id
            WHERE e.article_id IS NULL
              AND p.transcript IS NOT NULL
              AND p.transcript != ''
            """
        ).fetchall()
        return [(r["post_id"], r["transcript"]) for r in rows]

    # ------------------------------------------------------------------
    # Commit / misc
    # ------------------------------------------------------------------

    def commit(self) -> None:
        self.conn.commit()

    def get_post(self, post_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM tiktok_fin_posts WHERE post_id=?", (post_id,)
        ).fetchone()
        return dict(row) if row else None

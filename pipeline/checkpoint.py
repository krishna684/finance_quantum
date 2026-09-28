from __future__ import annotations

"""Lightweight checkpoint store backed by a SQLite table.

Each (query_text, start_date, end_date) window that completes successfully is
recorded.  On the next run the collector skips any window that already has a
checkpoint row, enabling resume-on-crash without reprocessing.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class CheckpointStore:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(db_path)
        self._init()

    def _init(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS collection_checkpoints (
                checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                videos_saved INTEGER DEFAULT 0,
                UNIQUE(query_text, start_date, end_date)
            )
            """
        )
        self._conn.commit()

    def is_done(self, query_text: str, start_date: str, end_date: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM collection_checkpoints WHERE query_text=? AND start_date=? AND end_date=?",
            (query_text, start_date, end_date),
        ).fetchone()
        return row is not None

    def mark_done(self, query_text: str, start_date: str, end_date: str, videos_saved: int = 0) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO collection_checkpoints
                (query_text, start_date, end_date, completed_at, videos_saved)
            VALUES (?, ?, ?, ?, ?)
            """,
            (query_text, start_date, end_date, _utc_now_iso(), videos_saved),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

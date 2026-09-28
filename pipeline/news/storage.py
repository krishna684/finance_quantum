from __future__ import annotations

"""SQLite storage layer for institutional media (WSJ/NYT) articles and images."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.news.models import NewsArticle, NewsImage


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class NewsStorage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS news_articles (
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

        CREATE TABLE IF NOT EXISTS news_images (
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
            collection_timestamp TEXT,
            FOREIGN KEY(article_id) REFERENCES news_articles(article_id)
        );

        CREATE INDEX IF NOT EXISTS idx_articles_date ON news_articles(publication_date);
        CREATE INDEX IF NOT EXISTS idx_articles_source ON news_articles(source);
        CREATE INDEX IF NOT EXISTS idx_articles_tickers ON news_articles(mapped_tickers);
        CREATE INDEX IF NOT EXISTS idx_images_article ON news_images(article_id);
        CREATE INDEX IF NOT EXISTS idx_images_hash ON news_images(sha256_hash);
        """
        self.conn.executescript(ddl)
        self.conn.commit()

    def upsert_article(self, article: NewsArticle, enrichment: dict[str, Any] | None = None) -> None:
        enrich = enrichment or {}
        sql = """
        INSERT INTO news_articles (
            article_id, source, headline, full_body, publication_date, publication_time,
            timezone, section, front_page, authors, word_count, database_source_id, url,
            collection_timestamp, sentiment_score, sentiment_label,
            cashtag_tickers, company_names, mapped_tickers,
            extracted_brands, extracted_products, extracted_retailers, extracted_categories
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(article_id) DO UPDATE SET
            headline=excluded.headline,
            full_body=excluded.full_body,
            publication_date=excluded.publication_date,
            publication_time=excluded.publication_time,
            section=excluded.section,
            front_page=excluded.front_page,
            authors=excluded.authors,
            word_count=excluded.word_count,
            sentiment_score=excluded.sentiment_score,
            sentiment_label=excluded.sentiment_label,
            cashtag_tickers=excluded.cashtag_tickers,
            company_names=excluded.company_names,
            mapped_tickers=excluded.mapped_tickers,
            extracted_brands=excluded.extracted_brands,
            extracted_products=excluded.extracted_products,
            extracted_retailers=excluded.extracted_retailers,
            extracted_categories=excluded.extracted_categories
        """
        self.conn.execute(
            sql,
            (
                article.article_id,
                article.source,
                article.headline,
                article.full_body,
                article.publication_date,
                article.publication_time,
                article.timezone,
                article.section,
                1 if article.front_page else 0,
                json.dumps(article.authors, ensure_ascii=False),
                article.word_count,
                article.database_source_id,
                article.url,
                _utc_now_iso(),
                enrich.get("sentiment_score"),
                enrich.get("sentiment_label"),
                json.dumps(enrich.get("cashtag_tickers", []), ensure_ascii=False),
                json.dumps(enrich.get("company_names", []), ensure_ascii=False),
                json.dumps(enrich.get("mapped_tickers", []), ensure_ascii=False),
                json.dumps(enrich.get("extracted_brands", []), ensure_ascii=False),
                json.dumps(enrich.get("extracted_products", []), ensure_ascii=False),
                json.dumps(enrich.get("extracted_retailers", []), ensure_ascii=False),
                json.dumps(enrich.get("extracted_categories", []), ensure_ascii=False),
            ),
        )

    def upsert_image(self, image: NewsImage) -> None:
        sql = """
        INSERT OR IGNORE INTO news_images (
            image_id, article_id, image_binary, caption, placement, image_order,
            width_px, height_px, sha256_hash, file_extension, collection_timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.conn.execute(
            sql,
            (
                image.image_id,
                image.article_id,
                image.image_binary,
                image.caption,
                image.placement,
                image.image_order,
                image.width_px,
                image.height_px,
                image.sha256_hash,
                image.file_extension,
                _utc_now_iso(),
            ),
        )

    def article_exists(self, database_source_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM news_articles WHERE database_source_id=?",
            (database_source_id,),
        ).fetchone()
        return row is not None

    def image_hash_exists(self, sha256_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM news_images WHERE sha256_hash=?",
            (sha256_hash,),
        ).fetchone()
        return row is not None

    def commit(self) -> None:
        self.conn.commit()

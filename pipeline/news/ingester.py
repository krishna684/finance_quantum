from __future__ import annotations

"""Ingest institutional media articles into the news database.

Applies entity extraction, sentiment scoring, and NER at ingest time
(same discipline as TikTok pipeline).
"""

from pathlib import Path
from typing import Iterator

from pipeline.entities import extract_entities
from pipeline.logger import get_logger
from pipeline.ner import all_tickers, extract_ner
from pipeline.news.models import NewsArticle
from pipeline.news.parsers import parse_directory, parse_file
from pipeline.news.storage import NewsStorage
from pipeline.sentiment import score_text

log = get_logger(__name__)


def _enrich(article: NewsArticle) -> dict:
    """Run sentiment + NER + entity extraction on the article text."""
    full_text = f"{article.headline} {article.full_body}"

    sentiment_score, sentiment_label = score_text(full_text)
    ner = extract_ner(full_text)
    extraction = extract_entities(full_text)

    return {
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "cashtag_tickers": ner.cashtag_tickers,
        "company_names": ner.company_names,
        "mapped_tickers": ner.mapped_tickers,
        "extracted_brands": extraction.brands,
        "extracted_products": extraction.products,
        "extracted_retailers": extraction.retailers,
        "extracted_categories": extraction.categories,
    }


def ingest_file(
    path: Path,
    storage: NewsStorage,
    source_hint: str = "UNKNOWN",
    fmt: str | None = None,
    skip_existing: bool = True,
) -> int:
    """Parse *path* and upsert articles into *storage*.

    Returns the count of new articles ingested.
    """
    articles = parse_file(path, source_hint=source_hint, fmt=fmt)
    count = 0
    for article in articles:
        if skip_existing and storage.article_exists(article.database_source_id):
            log.debug("article_skipped", db_id=article.database_source_id)
            continue
        enrichment = _enrich(article)
        storage.upsert_article(article, enrichment)
        count += 1
    storage.commit()
    log.info("file_ingested", path=str(path), articles=count, source=source_hint)
    return count


def ingest_directory(
    directory: Path,
    storage: NewsStorage,
    source_hint: str = "UNKNOWN",
    fmt: str | None = None,
    skip_existing: bool = True,
    glob: str = "**/*",
) -> int:
    """Ingest all files under *directory* matching *glob*.

    Returns total article count ingested.
    """
    log.info("directory_ingest_started", directory=str(directory), source=source_hint)
    total = 0
    for path in sorted(directory.glob(glob)):
        if path.is_file() and path.suffix.lower() in (".xml", ".csv", ".tsv", ".jsonl", ".jsons", ".ndjson"):
            total += ingest_file(path, storage, source_hint=source_hint, fmt=fmt, skip_existing=skip_existing)
    log.info("directory_ingest_done", directory=str(directory), total_articles=total)
    return total

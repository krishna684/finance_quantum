from __future__ import annotations

"""Parsers for academic database export formats.

Supported formats
-----------------
ProQuest XML   -- most common MU Library format
Factiva XML    -- Dow Jones / WSJ primary feed
CSV/TSV        -- generic fallback for manual exports

Usage:
    from pipeline.news.parsers import parse_file
    articles = parse_file(Path("export.xml"), source_hint="WSJ")
"""

import csv
import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from pipeline.news.models import NewsArticle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_id() -> str:
    return str(uuid.uuid4())


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


def _infer_source(publication: str) -> str:
    pub = publication.lower()
    if "wall street" in pub or "wsj" in pub:
        return "WSJ"
    if "new york times" in pub or "nyt" in pub:
        return "NYT"
    return publication.upper()[:10]


def _split_date_time(raw: str) -> tuple[str, str | None]:
    """Return (YYYY-MM-DD, HH:MM:SS|None) from various date string formats."""
    raw = raw.strip()
    # ISO 8601: 2024-01-15T09:30:00
    m = re.match(r"(\d{4}-\d{2}-\d{2})(?:T(\d{2}:\d{2}:\d{2}))?", raw)
    if m:
        return m.group(1), m.group(2)
    # ProQuest: "January 15, 2024"
    from datetime import datetime
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%Y%m%d"):
        try:
            dt = datetime.strptime(raw[:len(fmt) + 4], fmt)
            return dt.strftime("%Y-%m-%d"), None
        except ValueError:
            pass
    return raw[:10], None


# ---------------------------------------------------------------------------
# ProQuest XML parser
#
# ProQuest exports a <RECORDS> root with one <Record> per article.
# Fields vary by database but the common ones are mapped below.
# ---------------------------------------------------------------------------

def _parse_proquest_xml(path: Path, source_hint: str) -> Iterator[NewsArticle]:
    tree = ET.parse(path)
    root = tree.getroot()

    # Handle both <RECORDS> wrapper and bare <Record> lists
    records = root.findall(".//Record") or root.findall(".//document")

    for rec in records:
        def txt(tag: str) -> str:
            el = rec.find(tag)
            return _clean(el.text) if el is not None else ""

        headline = txt("Title") or txt("title") or txt("Headline")
        body = txt("FullText") or txt("fulltext") or txt("Abstract") or txt("abstract")
        pub_title = txt("PublicationTitle") or txt("publication_title") or txt("SourceTitle")
        raw_date = txt("PubDate") or txt("SortDate") or txt("publication_date") or ""
        pub_date, pub_time = _split_date_time(raw_date) if raw_date else ("", None)
        section = txt("PublicationType") or txt("section") or None
        authors_raw = txt("Authors") or txt("Author") or txt("authors") or ""
        authors = [a.strip() for a in re.split(r"[;,]", authors_raw) if a.strip()]
        source = _infer_source(pub_title) if pub_title else source_hint
        db_id = txt("DocumentID") or txt("accession_number") or _new_id()
        front_page = "A1" in (txt("StartPage") or "") or "front" in section.lower() if section else False

        if not headline and not body:
            continue

        yield NewsArticle(
            article_id=_new_id(),
            source=source,
            headline=headline,
            full_body=body,
            publication_date=pub_date,
            publication_time=pub_time,
            timezone="America/New_York",
            section=section,
            front_page=front_page,
            authors=authors,
            word_count=_word_count(body),
            database_source_id=db_id,
            url=txt("url") or None,
        )


# ---------------------------------------------------------------------------
# Factiva XML parser
#
# Factiva uses <doc> elements with <an> (accession number), <hd> (headline),
# <td> (text), <pd> (publication date), <sn> (source name), etc.
# ---------------------------------------------------------------------------

def _parse_factiva_xml(path: Path, source_hint: str) -> Iterator[NewsArticle]:
    tree = ET.parse(path)
    root = tree.getroot()

    for doc in root.iter("doc"):
        def txt(tag: str) -> str:
            el = doc.find(tag)
            if el is None:
                return ""
            return _clean("".join(el.itertext()))

        headline = txt("hd") or txt("hl2")
        body_parts = [_clean("".join(p.itertext())) for p in doc.findall(".//p")]
        body = " ".join(body_parts)
        if not body:
            body = txt("td")
        pub_title = txt("sn")
        raw_date = txt("pd")
        pub_date, pub_time = _split_date_time(raw_date) if raw_date else ("", None)
        section = txt("sc") or None
        authors_raw = txt("by") or ""
        authors = [a.strip() for a in re.split(r"[;,&]|and\s", authors_raw) if a.strip() and a.lower() != "and"]
        source = _infer_source(pub_title) if pub_title else source_hint
        db_id = txt("an") or _new_id()

        if not headline and not body:
            continue

        yield NewsArticle(
            article_id=_new_id(),
            source=source,
            headline=headline,
            full_body=body,
            publication_date=pub_date,
            publication_time=pub_time,
            timezone="America/New_York",
            section=section,
            front_page=False,
            authors=authors,
            word_count=_word_count(body),
            database_source_id=db_id,
            url=None,
        )


# ---------------------------------------------------------------------------
# CSV / TSV fallback parser
#
# Expected columns (case-insensitive):
#   article_id, source, headline, full_body, publication_date,
#   publication_time, section, front_page, authors, database_source_id
# All columns optional except headline or full_body.
# ---------------------------------------------------------------------------

_CSV_FIELD_ALIASES = {
    "title": "headline",
    "body": "full_body",
    "text": "full_body",
    "content": "full_body",
    "date": "publication_date",
    "pub_date": "publication_date",
    "time": "publication_time",
    "pub_time": "publication_time",
    "author": "authors",
    "byline": "authors",
    "section_category": "section",
    "front_page_flag": "front_page",
}


def _parse_csv(path: Path, source_hint: str) -> Iterator[NewsArticle]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            # normalise keys — guard against None keys from trailing commas
            norm = {
                _CSV_FIELD_ALIASES.get(k.lower().strip(), k.lower().strip()): v
                for k, v in row.items()
                if k is not None
            }

            headline = _clean(norm.get("headline", ""))
            body = _clean(norm.get("full_body", ""))
            if not headline and not body:
                continue

            raw_date = norm.get("publication_date", "")
            pub_date, pub_time = _split_date_time(raw_date) if raw_date else ("", None)
            pub_time = pub_time or norm.get("publication_time") or None
            authors_raw = norm.get("authors", "")
            authors = [a.strip() for a in re.split(r"[;,]", authors_raw) if a.strip()]
            front_raw = norm.get("front_page", "").strip().lower()
            front_page = front_raw in {"1", "true", "yes", "t"}
            source = _infer_source(norm.get("source", source_hint)) or source_hint

            yield NewsArticle(
                article_id=norm.get("article_id") or _new_id(),
                source=source,
                headline=headline,
                full_body=body,
                publication_date=pub_date,
                publication_time=pub_time,
                timezone="America/New_York",
                section=norm.get("section") or None,
                front_page=front_page,
                authors=authors,
                word_count=_word_count(body),
                database_source_id=norm.get("database_source_id", "") or _new_id(),
                url=norm.get("url") or None,
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _parse_jsonlines(path: Path, source_hint: str) -> Iterator[NewsArticle]:
    """Parse JSONL / .jsons files (one JSON object per line).

    Expected keys: title, text, gmtDate, source_id, article_id, source
    Malformed lines are silently skipped.
    """
    import json as _json
    from datetime import datetime

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except _json.JSONDecodeError:
                continue

            headline = _clean(obj.get("title", ""))
            body = _clean(obj.get("text", ""))
            if not headline and not body:
                continue

            # Parse date from gmtDate "Sat, 10 Oct 1857 00:00:00 +0000"
            pub_date = ""
            pub_time: str | None = None
            gmt_date = obj.get("gmtDate", "")
            if gmt_date:
                try:
                    parts = gmt_date.split()
                    # parts: ["Sat,", "10", "Oct", "1857", "00:00:00", "+0000"]
                    if len(parts) >= 4:
                        dt = datetime.strptime(
                            f"{parts[1]} {parts[2]} {parts[3]}", "%d %b %Y"
                        )
                        pub_date = dt.strftime("%Y-%m-%d")
                    if len(parts) >= 5:
                        pub_time = parts[4]
                except ValueError:
                    pass

            source_raw = obj.get("source", source_hint)
            # Normalise source label
            if source_raw in ("NYTHistorical", "nythistorical"):
                source = "NYT"
            elif source_raw in ("WSJHistorical", "wsjhistorical"):
                source = "WSJ"
            else:
                source = source_raw or source_hint

            db_id = str(obj.get("source_id") or obj.get("article_id") or obj.get("id") or _new_id())
            art_id = db_id  # use same value as PK

            yield NewsArticle(
                article_id=art_id,
                source=source,
                headline=headline,
                full_body=body,
                publication_date=pub_date,
                publication_time=pub_time,
                timezone="UTC",
                section=None,
                front_page=False,
                authors=[],
                word_count=_word_count(body),
                database_source_id=db_id,
                url=obj.get("filename") or None,
            )


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".jsons", ".ndjson"):
        return "jsonl"
    if suffix in (".csv", ".tsv"):
        return "csv"
    if suffix not in (".xml",):
        return "csv"
    # Peek at root element to distinguish ProQuest vs Factiva
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        tag = root.tag.lower()
        if "records" in tag or "record" in tag:
            return "proquest"
        if root.find(".//doc") is not None or root.find(".//an") is not None:
            return "factiva"
        # If has <Record> children → proquest
        if root.find(".//Record") is not None:
            return "proquest"
    except ET.ParseError:
        pass
    return "proquest"  # safe default for XML


def parse_file(path: Path, source_hint: str = "UNKNOWN", fmt: str | None = None) -> list[NewsArticle]:
    """Parse *path* and return a list of NewsArticle objects.

    Args:
        path: Path to export file (.xml, .csv, .tsv, .jsonl, .jsons)
        source_hint: 'WSJ' or 'NYT' — used when source cannot be inferred
        fmt: 'proquest' | 'factiva' | 'csv' | 'jsonl' — auto-detected if None
    """
    detected = fmt or _detect_format(path)
    if detected == "jsonl":
        return list(_parse_jsonlines(path, source_hint))
    if detected == "factiva":
        return list(_parse_factiva_xml(path, source_hint))
    if detected == "csv":
        return list(_parse_csv(path, source_hint))
    return list(_parse_proquest_xml(path, source_hint))


def parse_directory(
    directory: Path,
    source_hint: str = "UNKNOWN",
    fmt: str | None = None,
    glob: str = "**/*",
) -> Iterator[NewsArticle]:
    """Yield articles from all matching files in *directory*."""
    for path in sorted(directory.glob(glob)):
        if path.is_file() and path.suffix.lower() in (".xml", ".csv", ".tsv"):
            yield from parse_file(path, source_hint=source_hint, fmt=fmt)

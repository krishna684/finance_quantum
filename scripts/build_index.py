#!/usr/bin/env python
"""Build a merge-ready Parquet index from N24News metadata.

Schema (source-agnostic — GoodNews loader can be appended without schema change):
    image_path       str   — absolute path to JPEG on disk
    source_dataset   str   — "n24news" | "goodnews"
    image_id         str   — dataset-native unique identifier
    date             date  — article publication date (parsed from article_url /YYYY/MM/DD/)
    next_trading_day date  — first NYSE session on or after date
    date_source      str   — "article_url" | "image_url" (provenance of parsed date)
    section          str   — editorial section label
    headline         str
    caption          str
    abstract         str
    is_finance       bool  — section in FINANCE_SECTIONS

To add GoodNews later
---------------------
1. Implement _build_goodnews_rows(sorted_days) returning list[dict] with the
   same keys as _build_n24news_rows().
2. Append rows: rows += _build_goodnews_rows(sorted_days)
3. Update EXPECTED_ROWS and EXPECTED_FINANCE.
4. Re-run; no schema changes required.

Usage
-----
    python scripts/build_index.py
    N24NEWS_ROOT=/alt/path python scripts/build_index.py
"""
from __future__ import annotations

import bisect
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
N24NEWS_ROOT = Path(os.environ.get("N24NEWS_ROOT", r"C:\Users\kkffz\Downloads\N24News"))
METADATA_JSON = N24NEWS_ROOT / "news" / "nytimes_dataset.json"
IMGS_DIR = N24NEWS_ROOT / "imgs"
OUTPUT = _PROJECT_ROOT / "data" / "index.parquet"

FINANCE_SECTIONS = frozenset({"Economy", "Global Business", "Your Money", "Technology", "Media"})
_DATE_RE = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/")

# Expected values — update when GoodNews is added
EXPECTED_ROWS = 61_218
EXPECTED_FINANCE = 10_206


# ---------------------------------------------------------------------------
# NYSE calendar
# ---------------------------------------------------------------------------

def _build_trading_day_index(start: str = "2009-01-01", end: str = "2022-01-01") -> list[date]:
    """Return sorted list of all NYSE valid trading days in [start, end]."""
    nyse = mcal.get_calendar("NYSE")
    valid = nyse.valid_days(start_date=start, end_date=end)
    return sorted(d.date() for d in valid)


def _next_trading_day(d: date, sorted_days: list[date]) -> date:
    """Return d if it is a trading day, else the next NYSE session after d."""
    idx = bisect.bisect_left(sorted_days, d)
    if idx >= len(sorted_days):
        raise ValueError(f"Date {d} is beyond the precomputed calendar range")
    return sorted_days[idx]


# ---------------------------------------------------------------------------
# N24News loader
# ---------------------------------------------------------------------------

def _build_n24news_rows(sorted_days: list[date]) -> list[dict]:
    print(f"[n24news] Loading {METADATA_JSON} ...")
    with open(METADATA_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"[n24news] {len(records):,} records read")

    rows: list[dict] = []
    skipped = 0
    for r in records:
        # Primary: /YYYY/MM/DD/ in article_url; fallback: same pattern in image_url
        m_article = _DATE_RE.search(r.get("article_url", ""))
        m_image   = _DATE_RE.search(r.get("image_url", ""))
        m = m_article or m_image
        if not m:
            skipped += 1
            continue
        pub_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        rows.append({
            "image_path": str((IMGS_DIR / (r["image_id"] + ".jpg")).resolve()),
            "source_dataset": "n24news",
            "image_id": r["image_id"],
            "date": pub_date,
            "next_trading_day": _next_trading_day(pub_date, sorted_days),
            "date_source": "article_url" if m_article else "image_url",
            "section": r["section"],
            "headline": r["headline"],
            "caption": r["caption"],
            "abstract": r["abstract"],
            "is_finance": r["section"] in FINANCE_SECTIONS,
        })

    if skipped:
        print(f"[n24news] WARNING: {skipped} records had no parseable date — excluded")
    print(f"[n24news] {len(rows):,} rows built")
    return rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame) -> None:
    errors: list[str] = []

    if len(df) != EXPECTED_ROWS:
        errors.append(f"Row count: expected {EXPECTED_ROWS:,}, got {len(df):,}")

    null_dates = int(df["date"].isna().sum())
    if null_dates:
        errors.append(f"Null dates: {null_dates}")

    null_ntd = int(df["next_trading_day"].isna().sum())
    if null_ntd:
        errors.append(f"Null next_trading_day: {null_ntd}")

    null_paths = int(df["image_path"].isna().sum())
    if null_paths:
        errors.append(f"Null image_path: {null_paths}")

    print(f"Checking {len(df):,} image paths exist on disk ...")
    missing_files = [p for p in df["image_path"] if not Path(p).exists()]
    if missing_files:
        errors.append(
            f"Missing image files: {len(missing_files):,} "
            f"(first: {missing_files[0]})"
        )

    finance_count = int(df["is_finance"].sum())
    if finance_count != EXPECTED_FINANCE:
        errors.append(
            f"is_finance count: expected {EXPECTED_FINANCE:,}, got {finance_count:,}"
        )

    image_url_count = int((df["date_source"] == "image_url").sum())
    if image_url_count != 431:
        errors.append(
            f"date_source=='image_url' count: expected 431, got {image_url_count}"
        )

    if errors:
        print("\n[VALIDATION FAILED]")
        for e in errors:
            print(f"  x {e}")
        sys.exit(1)

    print("\n[VALIDATION PASSED]")
    print(f"  + Rows:             {len(df):,}")
    print(f"  + Null dates:       0")
    print(f"  + Null next_td:     0")
    print(f"  + Missing files:    0")
    print(f"  + is_finance:       {finance_count:,}")
    print(f"  + date_source counts: {dict(df['date_source'].value_counts())}")
    print(f"  + Date range:       {df['date'].min()} -> {df['date'].max()}")
    print(f"  + Sources:          {dict(df['source_dataset'].value_counts())}")
    per_section = df.groupby("section").size().sort_values(ascending=False)
    print(f"  + Top sections:     {dict(per_section.head(5))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building NYSE trading-day calendar ...")
    sorted_days = _build_trading_day_index()
    print(f"  {len(sorted_days):,} valid trading days (2009-2022)")

    rows = _build_n24news_rows(sorted_days)
    # Future: rows += _build_goodnews_rows(sorted_days)

    df = pd.DataFrame(rows)

    # Enforce column order and types
    df = df[[
        "image_path", "source_dataset", "image_id",
        "date", "next_trading_day", "date_source",
        "section", "headline", "caption", "abstract",
        "is_finance",
    ]]
    df["is_finance"] = df["is_finance"].astype(bool)
    # date/next_trading_day stay as Python date objects -> pyarrow writes them as date32

    _validate(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    print(f"\n  -> {OUTPUT}  ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

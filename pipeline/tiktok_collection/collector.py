from __future__ import annotations

"""Bright Data TikTok scraper client.

Architecture
------------
The BrightDataClient protocol defines the interface.  Two implementations
are provided:

  BrightDataHTTPClient   — real Bright Data Dataset API (snapshot-based)
  MockBrightDataClient   — returns fixture data; use for tests and dry-runs

The snapshot-based collection flow:
  1. POST /datasets/v3/trigger  → {"snapshot_id": "..."}
  2. Poll GET /datasets/v3/snapshot/{id}?format=json until status=="ready"
  3. Download JSON payload

Bright Data dataset ID for TikTok Posts: gd_lu702nij2f790tmv9h
Discovery parameter: discover_by=keyword

Response fields returned by Bright Data TikTok Posts:
  post_id, url, description, create_time,
  digg_count (likes), share_count, collect_count, comment_count,
  play_count, video_duration, hashtags,
  profile_username, profile_followers, is_verified, region
"""

import json
import time
from pathlib import Path
from typing import Any, Protocol

import requests

from pipeline.logger import get_logger

log = get_logger(__name__)

_BRIGHTDATA_BASE = "https://api.brightdata.com"
_DATASET_ID = "gd_lu702nij2f790tmv9h"
_POLL_INTERVAL = 5   # seconds between snapshot polls
_POLL_TIMEOUT = 600  # maximum seconds to wait for a snapshot


class BrightDataClient(Protocol):
    """Interface for keyword-based TikTok post search."""

    def search_posts(self, keyword: str, limit: int = 500) -> list[dict[str, Any]]:
        """Return up to *limit* raw post dicts matching *keyword*."""
        ...


class BrightDataHTTPClient:
    """Real Bright Data Dataset API client (snapshot pattern)."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("BRIGHTDATA_API_KEY is required for BrightDataHTTPClient")
        self._api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _trigger(self, keyword: str, limit: int) -> str:
        """Trigger a dataset snapshot and return snapshot_id."""
        url = f"{_BRIGHTDATA_BASE}/datasets/v3/trigger"
        params = {
            "dataset_id": _DATASET_ID,
            "discover_by": "keyword",
            "limit": limit,
            "format": "json",
        }
        resp = self._session.post(url, params=params, json=[{"keyword": keyword}], timeout=60)
        resp.raise_for_status()
        data = resp.json()
        snapshot_id = data.get("snapshot_id")
        if not snapshot_id:
            raise RuntimeError(f"Bright Data trigger returned no snapshot_id: {data}")
        log.info("brightdata_triggered", keyword=keyword, snapshot_id=snapshot_id, limit=limit)
        return snapshot_id

    def _poll(self, snapshot_id: str) -> list[dict[str, Any]]:
        """Poll until the snapshot is ready, then return the records."""
        url = f"{_BRIGHTDATA_BASE}/datasets/v3/snapshot/{snapshot_id}"
        deadline = time.time() + _POLL_TIMEOUT
        while time.time() < deadline:
            resp = self._session.get(url, params={"format": "json"}, timeout=60)
            if resp.status_code == 202:
                # Still processing
                time.sleep(_POLL_INTERVAL)
                continue
            resp.raise_for_status()
            data = resp.json()
            # Bright Data returns either a list directly or {"status": ..., "data": [...]}
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                status = data.get("status", "")
                if status in ("ready", ""):
                    return data.get("data", [])
                if status in ("running", "pending"):
                    time.sleep(_POLL_INTERVAL)
                    continue
                raise RuntimeError(f"Unexpected snapshot status: {status}")
        raise TimeoutError(f"Snapshot {snapshot_id} did not finish in {_POLL_TIMEOUT}s")

    def search_posts(self, keyword: str, limit: int = 500) -> list[dict[str, Any]]:
        snapshot_id = self._trigger(keyword, limit)
        records = self._poll(snapshot_id)
        log.info("brightdata_collected", keyword=keyword, n=len(records))
        return records


class MockBrightDataClient:
    """Returns fixture data for testing and dry-runs without credentials.

    If *fixture_path* is provided and exists, loads JSON from that file.
    Otherwise generates synthetic records.
    """

    def __init__(self, fixture_path: Path | None = None) -> None:
        self._fixture_path = fixture_path

    def search_posts(self, keyword: str, limit: int = 500) -> list[dict[str, Any]]:
        if self._fixture_path and self._fixture_path.exists():
            with open(self._fixture_path, encoding="utf-8") as f:
                data = json.load(f)
            records = data if isinstance(data, list) else data.get("data", [])
            return records[:limit]

        return _make_synthetic_posts(keyword, min(limit, 5))


def _make_synthetic_posts(keyword: str, n: int) -> list[dict[str, Any]]:
    """Generate n fake Bright Data post records for a given keyword."""
    base_id = abs(hash(keyword)) % 10_000_000
    templates = [
        {
            "description": f"$AAPL earnings looking strong this quarter! #stocktok #earnings",
            "hashtags": ["stocktok", "earnings", "AAPL", "investing"],
            "digg_count": 1500,
        },
        {
            "description": f"TSLA stock analysis — buy the dip or avoid? #daytrading #fintok",
            "hashtags": ["daytrading", "fintok", "TSLA", "stockanalysis"],
            "digg_count": 890,
        },
        {
            "description": f"My NVDA position is up 40% — here's why I'm holding #stockmarket",
            "hashtags": ["stockmarket", "NVDA", "investing"],
            "digg_count": 2300,
        },
        {
            "description": "Fed rate decision tomorrow — what it means for your stocks #wallstreet",
            "hashtags": ["wallstreet", "stockmarket", "financetiktok"],
            "digg_count": 560,
        },
        {
            "description": "Short squeeze incoming on $GME? Short interest data says YES #wallstreetbets",
            "hashtags": ["wallstreetbets", "GME", "shortselling", "stocktok"],
            "digg_count": 4100,
        },
    ]
    records = []
    for i in range(n):
        tmpl = templates[i % len(templates)]
        records.append({
            "post_id": str(base_id + i),
            "url": f"https://www.tiktok.com/@user{i}/video/{base_id + i}",
            "description": tmpl["description"],
            "hashtags": tmpl["hashtags"],
            "create_time": "2025-01-15T14:30:00Z",
            "digg_count": tmpl["digg_count"],
            "share_count": tmpl["digg_count"] // 10,
            "collect_count": tmpl["digg_count"] // 20,
            "comment_count": tmpl["digg_count"] // 15,
            "play_count": tmpl["digg_count"] * 12,
            "video_duration": 45,
            "profile_username": f"stocktrader_{i}",
            "profile_followers": 12000 + i * 1000,
            "is_verified": False,
            "region": "US",
            "_search_keyword": keyword,
        })
    return records


def make_client(api_key: str, fixture_path: Path | None = None) -> BrightDataClient:
    """Factory: real client if api_key is set, else mock."""
    if api_key:
        return BrightDataHTTPClient(api_key)
    log.warning("brightdata_no_key_using_mock", fixture_path=str(fixture_path))
    return MockBrightDataClient(fixture_path=fixture_path)

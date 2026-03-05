from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    client_key: str
    client_secret: str
    access_token: str
    db_path: Path
    query_videos_target: int
    random_low_engagement_target: int
    comments_per_video_target: int
    comment_scan_limit: int
    window_days: int
    lookback_months: int
    region_codes: list[str]


def load_settings(env_file: str | None = None) -> Settings:
    load_dotenv(env_file)

    db_path = Path(os.getenv("DB_PATH", "data/tiktok_signals.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    regions = [
        code.strip().upper()
        for code in os.getenv("REGION_CODES", "US").split(",")
        if code.strip()
    ]

    return Settings(
        client_key=os.getenv("TIKTOK_CLIENT_KEY", ""),
        client_secret=os.getenv("TIKTOK_CLIENT_SECRET", ""),
        access_token=os.getenv("TIKTOK_ACCESS_TOKEN", ""),
        db_path=db_path,
        query_videos_target=int(os.getenv("QUERY_VIDEOS_TARGET", "75")),
        random_low_engagement_target=int(os.getenv("RANDOM_LOW_ENGAGEMENT_TARGET", "25")),
        comments_per_video_target=int(os.getenv("COMMENTS_PER_VIDEO_TARGET", "100")),
        comment_scan_limit=int(os.getenv("COMMENT_SCAN_LIMIT", "300")),
        window_days=int(os.getenv("WINDOW_DAYS", "30")),
        lookback_months=int(os.getenv("LOOKBACK_MONTHS", "24")),
        region_codes=regions,
    )

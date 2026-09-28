from __future__ import annotations

"""Engagement velocity and momentum metrics for TikTok videos.

All metrics are computed from a *snapshot* at collection time plus the video's
post timestamp.  We cannot re-fetch historical time-series data from the
Research API, so these are approximations based on elapsed time since posting.

Metrics produced
----------------
views_per_hour       : average view rate since posting
likes_per_hour       : average like rate since posting
comments_per_hour    : average comment rate since posting
engagement_per_hour  : (views + likes + comments) / hours_since_post
hours_since_post     : age of video at collection time
velocity_tier        : 'viral' | 'high' | 'medium' | 'low'

Velocity tier thresholds (engagement_per_hour):
  viral  : >= 10_000
  high   : >= 1_000
  medium : >= 100
  low    : < 100
"""

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class VelocityMetrics:
    views_per_hour: float
    likes_per_hour: float
    comments_per_hour: float
    engagement_per_hour: float
    hours_since_post: float
    velocity_tier: str  # 'viral' | 'high' | 'medium' | 'low'


_TIER_THRESHOLDS = [
    (10_000, "viral"),
    (1_000, "high"),
    (100, "medium"),
]


def _tier(engagement_per_hour: float) -> str:
    for threshold, label in _TIER_THRESHOLDS:
        if engagement_per_hour >= threshold:
            return label
    return "low"


def compute_velocity(
    views: int,
    likes: int,
    comments: int,
    post_datetime_utc: str | None,
    collection_timestamp: str | None = None,
) -> VelocityMetrics | None:
    """Compute velocity metrics from snapshot counts and timestamps.

    Returns None if post_datetime_utc is missing or unparseable.
    """
    if not post_datetime_utc:
        return None

    try:
        post_dt = datetime.fromisoformat(post_datetime_utc)
    except ValueError:
        return None

    if collection_timestamp:
        try:
            collect_dt = datetime.fromisoformat(collection_timestamp)
        except ValueError:
            collect_dt = datetime.now(timezone.utc)
    else:
        collect_dt = datetime.now(timezone.utc)

    # Ensure both are timezone-aware
    if post_dt.tzinfo is None:
        post_dt = post_dt.replace(tzinfo=timezone.utc)
    if collect_dt.tzinfo is None:
        collect_dt = collect_dt.replace(tzinfo=timezone.utc)

    elapsed_seconds = (collect_dt - post_dt).total_seconds()
    if elapsed_seconds <= 0:
        return None

    hours = elapsed_seconds / 3600.0

    v = views / hours
    l = likes / hours
    c = comments / hours
    e = (views + likes + comments) / hours

    return VelocityMetrics(
        views_per_hour=round(v, 4),
        likes_per_hour=round(l, 4),
        comments_per_hour=round(c, 4),
        engagement_per_hour=round(e, 4),
        hours_since_post=round(hours, 2),
        velocity_tier=_tier(e),
    )

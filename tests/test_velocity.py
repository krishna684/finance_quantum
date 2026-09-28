import pytest
from pipeline.velocity import VelocityMetrics, compute_velocity


def _make_post(hours_ago: float) -> str:
    from datetime import datetime, timedelta, timezone
    dt = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return dt.isoformat()


def test_basic_velocity():
    post = _make_post(24)
    vel = compute_velocity(views=24_000, likes=2_400, comments=240, post_datetime_utc=post)
    assert vel is not None
    assert abs(vel.views_per_hour - 1000.0) < 10
    assert abs(vel.likes_per_hour - 100.0) < 5
    assert vel.hours_since_post > 23.9


def test_viral_tier():
    post = _make_post(1)
    vel = compute_velocity(views=50_000, likes=5_000, comments=500, post_datetime_utc=post)
    assert vel is not None
    assert vel.velocity_tier == "viral"


def test_low_tier():
    post = _make_post(100)
    vel = compute_velocity(views=100, likes=10, comments=5, post_datetime_utc=post)
    assert vel is not None
    assert vel.velocity_tier == "low"


def test_missing_post_time():
    vel = compute_velocity(views=1000, likes=100, comments=10, post_datetime_utc=None)
    assert vel is None


def test_bad_timestamp():
    vel = compute_velocity(views=1000, likes=100, comments=10, post_datetime_utc="not-a-date")
    assert vel is None


def test_zero_counts():
    post = _make_post(10)
    vel = compute_velocity(views=0, likes=0, comments=0, post_datetime_utc=post)
    assert vel is not None
    assert vel.views_per_hour == 0.0
    assert vel.velocity_tier == "low"

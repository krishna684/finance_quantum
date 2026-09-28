from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from pipeline.config import Settings
from pipeline.constants import (
    EXCLUDED_FINANCE_TERMS,
    HIGH_PURCHASE_INTENT_PHRASES,
    NEGATIVE_SENTIMENT_PHRASES,
    TIKTOKMADEMEBUYIT_CATEGORY_QUERIES,
)
from pipeline.checkpoint import CheckpointStore
from pipeline.entities import extract_entities
from pipeline.logger import get_logger
from pipeline.ner import all_tickers, extract_ner
from pipeline.sentiment import score_text
from pipeline.storage import Storage
from pipeline.tiktok_api import TikTokResearchAPI
from pipeline.velocity import compute_velocity

log = get_logger(__name__)


@dataclass(frozen=True)
class SignalQuery:
    text: str
    signal_type: str


def _date_to_tiktok(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def _epoch_to_iso(epoch: int | None) -> str | None:
    if not epoch:
        return None
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).replace(microsecond=0).isoformat()


def _engagement(video: dict[str, Any]) -> int:
    return int(video.get("view_count", 0) or 0) + int(video.get("like_count", 0) or 0) + int(video.get("comment_count", 0) or 0)


def _contains_excluded_finance_terms(text: str) -> bool:
    blob = (text or "").lower()
    return any(term in blob for term in EXCLUDED_FINANCE_TERMS)


def build_signal_queries() -> list[SignalQuery]:
    queries: list[SignalQuery] = []
    for phrase in HIGH_PURCHASE_INTENT_PHRASES:
        queries.append(SignalQuery(text=phrase, signal_type="high_purchase_intent"))
    for phrase in TIKTOKMADEMEBUYIT_CATEGORY_QUERIES:
        queries.append(SignalQuery(text=phrase, signal_type="high_purchase_intent"))
    for phrase in NEGATIVE_SENTIMENT_PHRASES:
        queries.append(SignalQuery(text=phrase, signal_type="negative_consumer_sentiment"))
    return queries


def _iter_windows(window_days: int, lookback_months: int) -> Iterable[tuple[str, str]]:
    end = datetime.now(timezone.utc)
    start_limit = end - timedelta(days=lookback_months * 30)

    cursor_end = end
    while cursor_end > start_limit:
        cursor_start = max(start_limit, cursor_end - timedelta(days=window_days - 1))
        yield _date_to_tiktok(cursor_start), _date_to_tiktok(cursor_end)
        cursor_end = cursor_start - timedelta(days=1)


def _base_query(keyword: str, region_codes: list[str]) -> dict[str, Any]:
    return {
        "and": [
            {
                "operation": "IN",
                "field_name": "region_code",
                "field_values": region_codes,
            },
            {
                "operation": "EQ",
                "field_name": "keyword",
                "field_values": [keyword],
            },
        ]
    }


def _fetch_video_pages(
    api: TikTokResearchAPI,
    query_obj: dict[str, Any],
    start_date: str,
    end_date: str,
    target_count: int,
    is_random: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = 0
    search_id: str | None = None

    while len(rows) < target_count:
        data = api.query_videos(
            query=query_obj,
            start_date=start_date,
            end_date=end_date,
            cursor=cursor,
            max_count=min(100, target_count - len(rows)),
            is_random=is_random,
            search_id=search_id,
        )
        batch = data.get("videos", [])
        if not batch:
            break

        rows.extend(batch)
        if not data.get("has_more"):
            break
        cursor = int(data.get("cursor", 0) or 0)
        search_id = data.get("search_id")

    return rows


def _save_video(storage: Storage, raw: dict[str, Any], query: SignalQuery) -> None:
    transcript = raw.get("voice_to_text") or raw.get("video_description") or ""
    if _contains_excluded_finance_terms(transcript):
        return

    full_text = " ".join([transcript] + (raw.get("hashtag_names") or []))
    extraction = extract_entities(full_text)
    sentiment_score, sentiment_label = score_text(transcript)
    ner = extract_ner(full_text)
    collection_ts = _utc_now_iso()
    post_dt = _epoch_to_iso(raw.get("create_time"))
    vel = compute_velocity(
        views=int(raw.get("view_count", 0) or 0),
        likes=int(raw.get("like_count", 0) or 0),
        comments=int(raw.get("comment_count", 0) or 0),
        post_datetime_utc=post_dt,
        collection_timestamp=collection_ts,
    )
    vid = str(raw.get("id") or raw.get("video_id"))
    payload = {
        "video_id": vid,
        "creator_username": raw.get("username"),
        "post_datetime_utc": post_dt,
        "full_transcript": transcript,
        "all_hashtags": raw.get("hashtag_names") or [],
        "number_of_views": int(raw.get("view_count", 0) or 0),
        "number_of_likes": int(raw.get("like_count", 0) or 0),
        "number_of_shares": int(raw.get("share_count", 0) or 0),
        "total_comments": int(raw.get("comment_count", 0) or 0),
        "collection_timestamp": collection_ts,
        "signal_type": query.signal_type,
        "query_text": query.text,
        "engagement_score": _engagement(raw),
        "extracted_brands": extraction.brands,
        "extracted_products": extraction.products,
        "extracted_retailers": extraction.retailers,
        "extracted_categories": extraction.categories,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "cashtag_tickers": ner.cashtag_tickers,
        "company_names": ner.company_names,
        "mapped_tickers": ner.mapped_tickers,
        "views_per_hour": vel.views_per_hour if vel else None,
        "likes_per_hour": vel.likes_per_hour if vel else None,
        "comments_per_hour": vel.comments_per_hour if vel else None,
        "engagement_per_hour": vel.engagement_per_hour if vel else None,
        "hours_since_post": vel.hours_since_post if vel else None,
        "velocity_tier": vel.velocity_tier if vel else None,
    }
    ticker_list = all_tickers(ner)
    storage.insert_mentions("video", vid, "ticker", ticker_list)
    storage.upsert_video(payload)
    storage.insert_mentions("video", vid, "brand", extraction.brands)
    storage.insert_mentions("video", vid, "product", extraction.products)
    storage.insert_mentions("video", vid, "retailer", extraction.retailers)
    storage.insert_mentions("video", vid, "category", extraction.categories)


def _collect_comments_for_video(
    api: TikTokResearchAPI,
    storage: Storage,
    video_id: str,
    comments_per_video_target: int,
    comment_scan_limit: int,
) -> None:
    all_comments: list[dict[str, Any]] = []
    cursor = 0

    while len(all_comments) < comment_scan_limit:
        data = api.query_video_comments(video_id=int(video_id), cursor=cursor, max_count=min(100, comment_scan_limit - len(all_comments)))
        batch = data.get("comments", [])
        if not batch:
            break
        all_comments.extend(batch)
        if not data.get("has_more"):
            break
        cursor = int(data.get("cursor", 0) or 0)

    selected = sorted(all_comments, key=lambda c: int(c.get("like_count", 0) or 0), reverse=True)[:comments_per_video_target]

    for c in selected:
        text = c.get("text") or ""
        extraction = extract_entities(text)
        sentiment_score, sentiment_label = score_text(text)
        ner = extract_ner(text)
        parent_comment_id = str(c.get("parent_comment_id") or "")
        cid = str(c.get("id"))
        payload = {
            "comment_id": cid,
            "video_id": str(c.get("video_id") or video_id),
            "comment_text": text,
            "comment_timestamp": _epoch_to_iso(c.get("create_time")),
            "comment_likes": int(c.get("like_count", 0) or 0),
            "reply_indicator": parent_comment_id not in {"", str(video_id)},
            "parent_comment_id": parent_comment_id or None,
            "extracted_brands": extraction.brands,
            "extracted_products": extraction.products,
            "extracted_retailers": extraction.retailers,
            "extracted_categories": extraction.categories,
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "cashtag_tickers": ner.cashtag_tickers,
            "company_names": ner.company_names,
            "mapped_tickers": ner.mapped_tickers,
        }
        storage.upsert_comment(payload)
        storage.insert_mentions("comment", cid, "brand", extraction.brands)
        storage.insert_mentions("comment", cid, "product", extraction.products)
        storage.insert_mentions("comment", cid, "retailer", extraction.retailers)
        storage.insert_mentions("comment", cid, "category", extraction.categories)
        storage.insert_mentions("comment", cid, "ticker", all_tickers(ner))


def run_collection(settings: Settings) -> None:
    api = TikTokResearchAPI(
        client_key=settings.client_key,
        client_secret=settings.client_secret,
        access_token=settings.access_token,
    )
    storage = Storage(settings.db_path)
    storage.init_schema()
    checkpoints = CheckpointStore(settings.db_path)

    signal_queries = build_signal_queries()
    total_windows = sum(1 for _ in _iter_windows(settings.window_days, settings.lookback_months))
    log.info(
        "collection_started",
        queries=len(signal_queries),
        windows_per_query=total_windows,
        lookback_months=settings.lookback_months,
        db_path=str(settings.db_path),
    )

    try:
        for query in signal_queries:
            for start_date, end_date in _iter_windows(settings.window_days, settings.lookback_months):
                if checkpoints.is_done(query.text, start_date, end_date):
                    log.debug("window_skipped", query=query.text, start=start_date, end=end_date)
                    continue

                log.info("window_started", query=query.text, start=start_date, end=end_date)
                query_obj = _base_query(query.text, settings.region_codes)

                try:
                    ranked_raw = _fetch_video_pages(
                        api=api,
                        query_obj=query_obj,
                        start_date=start_date,
                        end_date=end_date,
                        target_count=max(settings.query_videos_target * 2, 120),
                        is_random=False,
                    )
                    ranked = sorted(ranked_raw, key=_engagement, reverse=True)[: settings.query_videos_target]

                    random_raw = _fetch_video_pages(
                        api=api,
                        query_obj=query_obj,
                        start_date=start_date,
                        end_date=end_date,
                        target_count=max(settings.random_low_engagement_target * 3, 90),
                        is_random=True,
                    )
                except Exception as exc:
                    log.error("window_fetch_failed", query=query.text, start=start_date, end=end_date, exc=str(exc))
                    continue

                ranked_floor = min((_engagement(v) for v in ranked), default=0)
                low_random = [v for v in random_raw if _engagement(v) <= ranked_floor][: settings.random_low_engagement_target]

                batch = ranked + low_random
                deduped: dict[str, dict[str, Any]] = {}
                for v in batch:
                    deduped[str(v.get("id") or v.get("video_id"))] = v

                saved = 0
                for v in deduped.values():
                    _save_video(storage, v, query)
                    saved += 1
                    video_id = str(v.get("id") or v.get("video_id"))
                    try:
                        _collect_comments_for_video(
                            api=api,
                            storage=storage,
                            video_id=video_id,
                            comments_per_video_target=settings.comments_per_video_target,
                            comment_scan_limit=settings.comment_scan_limit,
                        )
                    except Exception as exc:
                        log.warning("comment_fetch_failed", video_id=video_id, exc=str(exc))

                storage.commit()
                checkpoints.mark_done(query.text, start_date, end_date, videos_saved=saved)
                log.info("window_done", query=query.text, start=start_date, end=end_date, videos_saved=saved)

        log.info("collection_finished")
    finally:
        checkpoints.close()
        storage.close()

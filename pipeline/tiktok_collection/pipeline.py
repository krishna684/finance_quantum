from __future__ import annotations

"""Orchestration for the financial TikTok collection pipeline.

Stage sequence
--------------
1. Load keyword universe (keywords_finance.json or TICKER_UNIVERSE_PATH)
2. For each keyword, collect posts via Bright Data (date-windowed, paginated)
   — uses existing CheckpointStore on (keyword, start_date, end_date) windows
3. Pass 1: regex keyword filter
4. Pass 2: LLM topic filter (Anthropic YES/NO)
5. For YES posts: yt-dlp download + Whisper transcription + ffmpeg frames
6. Financial entity / stance extraction (Anthropic JSON extraction)
7. Content classification: INFORMATION vs TRADING (Anthropic)
8. Store all enriched posts + entity rows in tiktok_fin.db
9. Run CLIP frame embeddings (image_embeddings table)
10. Run FinBERT transcript embeddings (text_embeddings table)

Entry point::

    from pipeline.tiktok_collection.pipeline import run_fin_collection
    from pipeline.config import load_settings

    settings = load_settings()
    run_fin_collection(settings)

Or via CLI::

    python run_pipeline.py fin-collect
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pipeline.checkpoint import CheckpointStore
from pipeline.config import Settings
from pipeline.logger import get_logger
from pipeline.tiktok_collection.classifier import ContentClassifier, TopicClassifier
from pipeline.tiktok_collection.collector import make_client
from pipeline.tiktok_collection.entity_extractor import run_entity_extraction
from pipeline.tiktok_collection.filter import build_filter
from pipeline.tiktok_collection.frame_embedder import TikTokFrameEmbedder
from pipeline.tiktok_collection.storage import FinCollectionStorage
from pipeline.tiktok_collection.transcript_embedder import TikTokTranscriptEmbedder
from pipeline.tiktok_collection.transcriber import run_transcription

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Date windowing (mirrors existing collector.py _iter_windows)
# ---------------------------------------------------------------------------

def _date_str(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def _iter_windows(window_days: int, lookback_months: int):
    end = datetime.now(timezone.utc)
    start_limit = end - timedelta(days=lookback_months * 30)
    cursor_end = end
    while cursor_end > start_limit:
        cursor_start = max(start_limit, cursor_end - timedelta(days=window_days - 1))
        yield _date_str(cursor_start), _date_str(cursor_end)
        cursor_end = cursor_start - timedelta(days=1)


# ---------------------------------------------------------------------------
# Keyword loading
# ---------------------------------------------------------------------------

def _load_search_keywords(settings: Settings) -> list[str]:
    """Return Bright Data search keywords from the keyword universe.

    Each ticker generates one keyword query (e.g. "AAPL stock"), plus each
    genre hashtag generates a query.  This keeps total keyword count reasonable
    while covering the full ticker universe.
    """
    kw_path = settings.ticker_universe_path
    default = Path(__file__).parent / "keywords_finance.json"
    path = kw_path if (kw_path and kw_path.exists()) else default

    with open(path, encoding="utf-8") as f:
        kw = json.load(f)

    tickers: list[str] = kw.get("tickers", [])
    hashtags: list[str] = kw.get("hashtags", [])

    keywords: list[str] = []
    # Ticker-based queries: "$AAPL" and "AAPL stock"
    for t in tickers:
        keywords.append(f"${t}")
        keywords.append(f"{t} stock")
    # Hashtag-based queries (unqualified, Bright Data will search in description/tags)
    for h in hashtags[:10]:  # cap at 10 hashtag queries to limit API spend
        keywords.append(f"#{h}")

    return keywords


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post_date(raw_post: dict[str, Any]) -> str:
    """Extract YYYY-MM-DD from a Bright Data post's create_time field."""
    ct = raw_post.get("create_time") or ""
    if "T" in ct:
        return ct[:10]
    if len(ct) >= 8 and ct.isdigit():
        # Unix epoch seconds
        try:
            dt = datetime.fromtimestamp(int(ct), tz=timezone.utc)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass
    return ct[:10] if len(ct) >= 10 else ""


def _fin_db_path(settings: Settings) -> Path:
    if settings.fin_db_path:
        return settings.fin_db_path
    return settings.db_path.parent / "tiktok_fin.db"


def _progress_path(settings: Settings, stage: str) -> Path:
    return _fin_db_path(settings).parent / f"tiktok_fin_{stage}_progress.json"


def _failed_log_path(settings: Settings) -> Path:
    return _fin_db_path(settings).parent / "tiktok_fin_whisper_failed.json"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_fin_collection(settings: Settings) -> None:
    fin_db = _fin_db_path(settings)
    storage = FinCollectionStorage(fin_db)
    storage.init_schema()

    checkpoints = CheckpointStore(fin_db)

    bright_client = make_client(settings.brightdata_api_key)
    pass1_filter = build_filter(settings.ticker_universe_path)

    topic_clf = TopicClassifier(
        anthropic_api_key=settings.anthropic_api_key,
        model=settings.fin_llm_model,
        checkpoint_path=_progress_path(settings, "topic"),
    ) if settings.anthropic_api_key else None

    content_clf = ContentClassifier(
        anthropic_api_key=settings.anthropic_api_key,
        model=settings.fin_llm_model,
        checkpoint_path=_progress_path(settings, "content"),
    ) if settings.anthropic_api_key else None

    keywords = _load_search_keywords(settings)
    log.info("fin_collection_start", keywords=len(keywords), db=str(fin_db))

    try:
        for keyword in keywords:
            for start_date, end_date in _iter_windows(settings.window_days, settings.lookback_months):
                ckpt_key = f"fin:{keyword}"
                if checkpoints.is_done(ckpt_key, start_date, end_date):
                    log.debug("fin_window_skipped", keyword=keyword, start=start_date, end=end_date)
                    continue

                log.info("fin_window_start", keyword=keyword, start=start_date, end=end_date)

                # Stage 1: Collect raw posts from Bright Data
                try:
                    raw_posts = bright_client.search_posts(keyword, limit=500)
                except Exception as exc:
                    log.error("brightdata_fetch_failed", keyword=keyword, exc=str(exc))
                    continue

                if not raw_posts:
                    checkpoints.mark_done(ckpt_key, start_date, end_date)
                    continue

                # Deduplicate by post_id
                seen: dict[str, dict[str, Any]] = {}
                for p in raw_posts:
                    pid = str(p.get("post_id", ""))
                    if pid and pid not in seen:
                        seen[pid] = p
                raw_posts = list(seen.values())

                # Stage 2: Pass 1 filter
                survivors = pass1_filter.apply_batch(raw_posts)
                log.info(
                    "pass1_done",
                    keyword=keyword,
                    raw=len(raw_posts),
                    survived=len(survivors),
                )

                if not survivors:
                    checkpoints.mark_done(ckpt_key, start_date, end_date, videos_saved=0)
                    continue

                # Persist Pass 1 survivors immediately (llm_topic will be set later)
                for post in survivors:
                    payload = {
                        "post_id": str(post.get("post_id", "")),
                        "url": post.get("url"),
                        "description": post.get("description"),
                        "hashtags": post.get("hashtags") or [],
                        "post_date": _post_date(post),
                        "creator_username": post.get("profile_username"),
                        "play_count": post.get("play_count", 0),
                        "digg_count": post.get("digg_count", 0),
                        "share_count": post.get("share_count", 0),
                        "comment_count": post.get("comment_count", 0),
                        "region": post.get("region"),
                        "search_keyword": keyword,
                        "pass1_matched": True,
                        "llm_topic": None,
                    }
                    storage.upsert_post(payload)
                storage.commit()

                # Stage 3: Pass 2 LLM topic filter
                if topic_clf:
                    survivors = topic_clf.classify_batch(survivors)
                else:
                    log.warning("topic_classifier_skipped_no_key")
                    for v in survivors:
                        v["_llm_topic"] = "UNKNOWN"

                yes_posts = [v for v in survivors if v.get("_llm_topic") == "YES"]
                log.info("pass2_done", keyword=keyword, survived=len(survivors), yes=len(yes_posts))

                # Update llm_topic in DB
                for post in survivors:
                    pid = str(post.get("post_id", ""))
                    topic = post.get("_llm_topic", "UNKNOWN")
                    storage.conn.execute(
                        "UPDATE tiktok_fin_posts SET llm_topic=? WHERE post_id=?",
                        (topic, pid),
                    )
                storage.commit()

                if not yes_posts:
                    checkpoints.mark_done(ckpt_key, start_date, end_date, videos_saved=0)
                    continue

                # Stage 4: Transcription + frame extraction
                run_transcription(
                    posts=yes_posts,
                    storage=storage,
                    openai_api_key=settings.openai_api_key,
                    sample_rate=settings.frame_sample_rate,
                    workers=settings.transcription_workers,
                    failed_log_path=_failed_log_path(settings),
                )

                # Reload transcripts from DB so entity extraction sees them
                enriched_yes: list[dict[str, Any]] = []
                for post in yes_posts:
                    pid = str(post.get("post_id", ""))
                    row = storage.get_post(pid)
                    if row:
                        enriched_yes.append({**post, "transcript": row.get("transcript") or ""})

                # Stage 5: Entity extraction
                posts_needing_entities = [
                    p for p in enriched_yes
                    if not storage.entity_rows_exist(str(p.get("post_id", "")))
                ]
                if posts_needing_entities:
                    entity_results = run_entity_extraction(
                        posts=posts_needing_entities,
                        anthropic_api_key=settings.anthropic_api_key,
                        model=settings.fin_llm_model,
                    )
                    for pid, rows in entity_results.items():
                        if rows:
                            storage.insert_entity_rows([
                                {
                                    "post_id": r.post_id,
                                    "post_date": r.post_date,
                                    "ticker": r.ticker,
                                    "company_name": r.company_name,
                                    "stance": r.stance,
                                    "has_position": r.has_position,
                                }
                                for r in rows
                            ])
                            tickers = sorted({r.ticker for r in rows})
                            storage.set_mapped_tickers(pid, tickers)
                    storage.commit()

                # Stage 6: Content classification (INFORMATION vs TRADING)
                if content_clf:
                    content_clf.classify_batch(enriched_yes)
                    for post in enriched_yes:
                        pid = str(post.get("post_id", ""))
                        ctype = post.get("_content_type", "UNKNOWN")
                        storage.set_content_type(pid, ctype)
                    storage.commit()

                checkpoints.mark_done(ckpt_key, start_date, end_date, videos_saved=len(yes_posts))
                log.info("fin_window_done", keyword=keyword, start=start_date, end=end_date,
                         yes_posts=len(yes_posts))

        log.info("fin_collection_finished")

        # Stage 7: Embed frames
        log.info("frame_embedding_start_global")
        frame_embedder = TikTokFrameEmbedder(storage)
        n_frames = frame_embedder.run(batch_size=32)
        log.info("frame_embedding_done", n=n_frames)

        # Stage 8: Embed transcripts
        log.info("transcript_embedding_start_global")
        transcript_embedder = TikTokTranscriptEmbedder(storage)
        n_transcripts = transcript_embedder.run(batch_size=64)
        log.info("transcript_embedding_done", n=n_transcripts)

    finally:
        checkpoints.close()
        storage.close()

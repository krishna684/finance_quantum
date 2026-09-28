from __future__ import annotations

"""Financial entity and stance extraction for TikTok posts.

Two-step approach (cheap-first, LLM-second):

  Step 1 — NER (free): reuse pipeline.ner.extract_ner() to detect cashtags,
            bare tickers, and company names from description + transcript.

  Step 2 — LLM stance (cheap): if tickers were found, call Claude with the
            description and transcript to determine:
              - Stance per ticker: BULLISH | BEARISH | NEUTRAL
              - Whether the creator explicitly states they hold a position

Output rows (one per ticker per post) plug directly into:
  tiktok_entity_rows  (post_id, post_date, ticker, company_name, stance, has_position)
  tiktok_fin_posts.mapped_tickers (JSON list for CRSP merge)

The CRSP merge (CRSPMapper) uses the ticker column from tiktok_entity_rows to
look up the PERMNO, exactly as it does for mapped_tickers in tiktok_videos.
"""

import json
import time
from dataclasses import dataclass
from typing import Any

from pipeline.logger import get_logger
from pipeline.ner import NERResult, all_tickers, extract_ner

log = get_logger(__name__)

_RATE_SLEEP = 0.2
_SAVE_EVERY = 25
_MAX_TOKENS = 400

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_ENTITY_PROMPT = """You are analyzing a TikTok video about stocks or the stock market.

For EACH publicly traded company or stock mentioned in the description or transcript, extract:
1. ticker: the stock ticker symbol (e.g. AAPL, TSLA, NVDA)
2. company: the company name (short form, e.g. Apple, Tesla)
3. stance: the creator's expressed view on the stock — BULLISH, BEARISH, or NEUTRAL
4. has_position: true if the creator explicitly says they own shares or options in this company, false otherwise

Return ONLY valid JSON with no markdown fences:
{{"entities": [{{"ticker": "AAPL", "company": "Apple", "stance": "BULLISH", "has_position": false}}]}}

If no specific publicly traded companies are discussed, return: {{"entities": []}}

---
Description: {description}
Transcript: {transcript}"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EntityRow:
    post_id: str
    post_date: str
    ticker: str
    company_name: str
    stance: str         # BULLISH | BEARISH | NEUTRAL
    has_position: bool


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _ner_tickers(text: str) -> list[str]:
    """Quick free NER pass — returns all detected tickers."""
    ner: NERResult = extract_ner(text)
    return all_tickers(ner)


def _llm_entities(
    client: Any,
    model: str,
    description: str,
    transcript: str,
) -> list[dict[str, Any]]:
    """Call Claude and return list of entity dicts."""
    prompt = _ENTITY_PROMPT.format(
        description=(description or "")[:600],
        transcript=(transcript or "")[:2000],
    )
    resp = client.messages.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    data = json.loads(raw)
    entities = data.get("entities", [])
    if not isinstance(entities, list):
        return []
    return entities


def _coerce_stance(raw: str) -> str:
    u = (raw or "").upper()
    if "BULL" in u:
        return "BULLISH"
    if "BEAR" in u:
        return "BEARISH"
    return "NEUTRAL"


def extract_financial_entities(
    post_id: str,
    post_date: str,
    description: str,
    transcript: str,
    anthropic_client: Any | None,
    model: str,
) -> list[EntityRow]:
    """Extract financial entities from a single post.

    Uses NER for fast ticker detection; LLM for stance if client available.
    Falls back to NEUTRAL stance if LLM unavailable.
    """
    full_text = f"{description or ''} {transcript or ''}"
    ner_tickers = _ner_tickers(full_text)

    if not ner_tickers:
        return []

    if anthropic_client is None:
        # No LLM: return NEUTRAL for all NER-detected tickers
        return [
            EntityRow(
                post_id=post_id,
                post_date=post_date,
                ticker=t,
                company_name="",
                stance="NEUTRAL",
                has_position=False,
            )
            for t in ner_tickers
        ]

    try:
        raw_entities = _llm_entities(anthropic_client, model, description, transcript)
    except Exception as exc:
        log.warning("entity_llm_error", post_id=post_id, exc=str(exc))
        # Degrade gracefully: NEUTRAL for NER-detected tickers
        return [
            EntityRow(
                post_id=post_id,
                post_date=post_date,
                ticker=t,
                company_name="",
                stance="NEUTRAL",
                has_position=False,
            )
            for t in ner_tickers
        ]

    rows: list[EntityRow] = []
    seen: set[str] = set()
    for ent in raw_entities:
        ticker = (ent.get("ticker") or "").upper().strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        rows.append(EntityRow(
            post_id=post_id,
            post_date=post_date,
            ticker=ticker,
            company_name=(ent.get("company") or ""),
            stance=_coerce_stance(ent.get("stance", "")),
            has_position=bool(ent.get("has_position")),
        ))

    # Add any NER-detected tickers not found by the LLM (fallback coverage)
    for t in ner_tickers:
        if t not in seen:
            rows.append(EntityRow(
                post_id=post_id,
                post_date=post_date,
                ticker=t,
                company_name="",
                stance="NEUTRAL",
                has_position=False,
            ))

    return rows


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_entity_extraction(
    posts: list[dict[str, Any]],
    anthropic_api_key: str,
    model: str,
) -> dict[str, list[EntityRow]]:
    """Extract entities for all posts that passed the topic filter.

    Returns {post_id: [EntityRow, ...]} for every post processed.
    Already-extracted posts (tracked internally) are skipped — callers should
    check storage.entity_rows_exist() before passing posts here.
    """
    client: Any | None = None
    if anthropic_api_key:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_api_key)

    results: dict[str, list[EntityRow]] = {}
    api_calls = 0
    errors = 0

    for i, post in enumerate(posts):
        pid = str(post.get("post_id", ""))
        post_date = str(post.get("post_date") or "")[:10]
        description = post.get("description") or ""
        transcript = post.get("transcript") or post.get("_transcript") or ""

        try:
            rows = extract_financial_entities(
                post_id=pid,
                post_date=post_date,
                description=description,
                transcript=transcript,
                anthropic_client=client,
                model=model,
            )
            results[pid] = rows
            if client:
                api_calls += 1
        except Exception as exc:
            log.error("entity_extraction_error", post_id=pid, exc=str(exc))
            results[pid] = []
            errors += 1

        if (api_calls + errors) % _SAVE_EVERY == 0 and (api_calls + errors) > 0:
            log.info("entity_extraction_progress",
                     processed=i + 1, total=len(posts),
                     api_calls=api_calls, errors=errors)

        if client:
            time.sleep(_RATE_SLEEP)

    log.info("entity_extraction_done", total=len(posts), api_calls=api_calls, errors=errors)
    return results

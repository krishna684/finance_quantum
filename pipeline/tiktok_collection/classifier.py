from __future__ import annotations

"""LLM-based classifiers for the financial TikTok collection pipeline.

Pass 2 — Topic filter (TopicClassifier)
    Input: description + transcript
    Output: YES | NO
    Question: Does this video substantively discuss publicly traded companies
              or overall stock market conditions?

Content type — Info vs Trading (ContentClassifier)
    Input: description + transcript
    Output: INFORMATION | TRADING | UNKNOWN
    Question: Is this informational analysis or explicit trading advice?

Both classifiers:
  - Use the Anthropic SDK with the model configured in Settings
  - Write progress to a JSON checkpoint file (keyed by post_id) so runs
    can resume after failure without re-spending LLM budget
  - Batch new posts; already-classified posts are loaded from checkpoint
"""

import json
import time
from pathlib import Path
from typing import Any

from pipeline.logger import get_logger

log = get_logger(__name__)

_RATE_SLEEP = 0.15   # seconds between API calls (avoid 429s)
_SAVE_EVERY = 25     # checkpoint write frequency

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_TOPIC_PROMPT = """Does this TikTok video substantively discuss one or more specific publicly traded companies, their stock, or overall stock market conditions?

YES includes:
- Stock picks, earnings discussion, price predictions for a named company
- Buy/sell opinions on specific stocks
- Portfolio reveals mentioning specific stock tickers
- Market outlook discussions (S&P 500, NASDAQ, overall bull/bear market)
- Short squeeze or technical analysis on a specific ticker
- Options or derivative strategies for a specific stock

NO includes:
- Generic personal-finance advice (budgeting, credit cards, savings rates)
- Crypto-only content with no stock market discussion
- Real estate investment advice only
- Forex or sports betting content only
- Get-rich-quick promotions with no specific stock mentioned
- Platform tutorials (how to use Robinhood/Webull) with no stock analysis
- General motivational / hustle content

Respond with exactly one word: YES or NO

---
Description: {description}
Transcript: {transcript}"""


_CONTENT_PROMPT = """A TikTok video discusses the stock market or specific publicly traded companies. Classify its PRIMARY PURPOSE.

INFORMATION — The video is informational/analytical. The creator reports news, explains earnings, analyses a company, or provides market context. The point is to inform, not to tell people what trades to make.
Examples:
- "Apple beat earnings by $0.15 per share — here's what the numbers mean"
- "Here's my breakdown of Tesla's delivery report"
- "The Fed raised rates by 25bps — what this means for growth stocks"

TRADING — The video gives explicit trading advice. The creator tells viewers to buy, sell, or hold specific positions, shares their own trades, or provides actionable trade setups.
Examples:
- "I'm loading up on NVDA calls before earnings — you should too"
- "Sell your TSLA now, here's my price target"
- "I made $50k on GME this week, here's my exact options strategy"

Respond with exactly one word: INFORMATION or TRADING

---
Description: {description}
Transcript: {transcript}"""


# ---------------------------------------------------------------------------
# Progress checkpoint (JSON file, keyed by post_id)
# ---------------------------------------------------------------------------

class _ProgressCheckpoint:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, str] = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                self._data = json.load(f)
            log.info("checkpoint_loaded", path=str(path), n=len(self._data))

    def get(self, post_id: str) -> str | None:
        return self._data.get(post_id)

    def set(self, post_id: str, value: str) -> None:
        self._data[post_id] = value

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False)

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def _call_anthropic(client: Any, model: str, prompt: str, max_tokens: int = 5) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


# ---------------------------------------------------------------------------
# Topic Classifier (Pass 2)
# ---------------------------------------------------------------------------

class TopicClassifier:
    """Classify videos as YES (financial stock content) or NO."""

    def __init__(self, anthropic_api_key: str, model: str, checkpoint_path: Path) -> None:
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for TopicClassifier")
        import anthropic
        self._client = anthropic.Anthropic(api_key=anthropic_api_key)
        self._model = model
        self._ckpt = _ProgressCheckpoint(checkpoint_path)

    def classify_batch(
        self, videos: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Classify each video; attach _llm_topic field. Returns all videos."""
        new_calls = 0
        errors = 0

        for i, v in enumerate(videos):
            pid = str(v.get("post_id", ""))
            cached = self._ckpt.get(pid)
            if cached is not None:
                v["_llm_topic"] = cached
                continue

            description = (v.get("description") or "")[:600]
            transcript = (v.get("transcript") or v.get("_transcript") or "")[:2000]
            prompt = _TOPIC_PROMPT.format(description=description, transcript=transcript)

            try:
                raw = _call_anthropic(self._client, self._model, prompt, max_tokens=5).upper()
                result = "YES" if "YES" in raw else ("NO" if "NO" in raw else "UNKNOWN")
                new_calls += 1
            except Exception as exc:
                log.warning("topic_classify_error", post_id=pid, exc=str(exc))
                result = "UNKNOWN"
                errors += 1

            v["_llm_topic"] = result
            self._ckpt.set(pid, result)

            if (new_calls + errors) % _SAVE_EVERY == 0 and (new_calls + errors) > 0:
                self._ckpt.save()
                log.info(
                    "topic_classify_progress",
                    processed=i + 1,
                    total=len(videos),
                    api_calls=new_calls,
                    errors=errors,
                )

            time.sleep(_RATE_SLEEP)

        self._ckpt.save()
        yes = sum(1 for v in videos if v.get("_llm_topic") == "YES")
        log.info("topic_classify_done", total=len(videos), yes=yes, api_calls=new_calls, errors=errors)
        return videos


# ---------------------------------------------------------------------------
# Content Classifier (INFORMATION vs TRADING)
# ---------------------------------------------------------------------------

class ContentClassifier:
    """Classify financial videos as INFORMATION or TRADING."""

    def __init__(self, anthropic_api_key: str, model: str, checkpoint_path: Path) -> None:
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for ContentClassifier")
        import anthropic
        self._client = anthropic.Anthropic(api_key=anthropic_api_key)
        self._model = model
        self._ckpt = _ProgressCheckpoint(checkpoint_path)

    def classify_batch(
        self, videos: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Classify YES-labelled videos. Attaches _content_type field."""
        new_calls = 0
        errors = 0

        for i, v in enumerate(videos):
            # Only classify videos that passed topic filter
            if v.get("_llm_topic") != "YES":
                v["_content_type"] = "N/A"
                continue

            pid = str(v.get("post_id", ""))
            cached = self._ckpt.get(pid)
            if cached is not None:
                v["_content_type"] = cached
                continue

            description = (v.get("description") or "")[:600]
            transcript = (v.get("transcript") or v.get("_transcript") or "")[:2000]
            prompt = _CONTENT_PROMPT.format(description=description, transcript=transcript)

            try:
                raw = _call_anthropic(self._client, self._model, prompt, max_tokens=5).upper()
                if "INFORMATION" in raw:
                    result = "INFORMATION"
                elif "TRADING" in raw:
                    result = "TRADING"
                else:
                    result = "UNKNOWN"
                new_calls += 1
            except Exception as exc:
                log.warning("content_classify_error", post_id=pid, exc=str(exc))
                result = "UNKNOWN"
                errors += 1

            v["_content_type"] = result
            self._ckpt.set(pid, result)

            if (new_calls + errors) % _SAVE_EVERY == 0 and (new_calls + errors) > 0:
                self._ckpt.save()
                log.info(
                    "content_classify_progress",
                    processed=i + 1,
                    total=len(videos),
                    api_calls=new_calls,
                    errors=errors,
                )

            time.sleep(_RATE_SLEEP)

        self._ckpt.save()
        info = sum(1 for v in videos if v.get("_content_type") == "INFORMATION")
        trade = sum(1 for v in videos if v.get("_content_type") == "TRADING")
        log.info(
            "content_classify_done",
            total=len(videos),
            information=info,
            trading=trade,
            api_calls=new_calls,
            errors=errors,
        )
        return videos


def parse_yes_no(raw: str) -> str:
    """Parse a raw LLM string into YES | NO | UNKNOWN.  Exported for tests."""
    upper = raw.strip().upper()
    if "YES" in upper:
        return "YES"
    if "NO" in upper:
        return "NO"
    return "UNKNOWN"


def parse_info_trading(raw: str) -> str:
    """Parse INFORMATION | TRADING | UNKNOWN.  Exported for tests."""
    upper = raw.strip().upper()
    if "INFORMATION" in upper:
        return "INFORMATION"
    if "TRADING" in upper:
        return "TRADING"
    return "UNKNOWN"

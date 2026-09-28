from __future__ import annotations

"""Pass 1 regex filter for financial TikTok content.

Applies cheap, LLM-free pattern matching over description, hashtags, and
transcript.  A video survives if it meets at least one of:

  Rule A — Cashtag: contains $TICKER (1-5 uppercase letters)
  Rule B — Ticker/company name: description or transcript contains a known
            stock symbol or company name from the keyword universe
  Rule C — Genre + market: contains a #fintok-style hashtag AND a
            market-condition phrase (e.g. "earnings report", "buy the dip")

The goal is high recall (don't miss real finance content) while keeping
precision high enough to avoid wasting LLM budget in Pass 2.

False-positive guard: dollar amounts like $5, $10, $100 are not matched
because [A-Z]{1,5} requires uppercase letters.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Cashtag regex  — matches $AAPL, $GME, $A (single-letter tickers are valid)
# Does NOT match $5 or $100 because digits != [A-Z]
# ---------------------------------------------------------------------------
_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")


@dataclass(frozen=True)
class FilterResult:
    passed: bool
    matched_rule: str          # "cashtag" | "ticker_name" | "genre_market" | ""
    matched_tickers: list[str]  # cashtags or ticker names found
    matched_hashtags: list[str]
    matched_terms: list[str]


def _load_keywords(universe_path: Path | None = None) -> dict[str, Any]:
    """Load keyword universe, with optional override path."""
    default = Path(__file__).parent / "keywords_finance.json"
    path = universe_path if (universe_path and universe_path.exists()) else default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_filter(universe_path: Path | None = None) -> "Pass1Filter":
    kw = _load_keywords(universe_path)
    return Pass1Filter(
        tickers=set(kw.get("tickers", [])),
        company_names={k.lower(): v for k, v in kw.get("company_names", {}).items()},
        hashtags={h.lower().lstrip("#") for h in kw.get("hashtags", [])},
        market_terms=[t.lower() for t in kw.get("market_terms", [])],
    )


class Pass1Filter:
    def __init__(
        self,
        tickers: set[str],
        company_names: dict[str, str],
        hashtags: set[str],
        market_terms: list[str],
    ) -> None:
        self._tickers = tickers
        self._company_names = company_names  # name_lower → ticker
        self._hashtags = hashtags
        self._market_terms = market_terms
        # Pre-compile bare ticker pattern: whole-word uppercase 1-5 letters
        ticker_alternation = "|".join(re.escape(t) for t in sorted(tickers, key=len, reverse=True))
        self._ticker_re = re.compile(rf"\b({ticker_alternation})\b") if tickers else None
        # Company names sorted longest-first for greedy matching
        self._sorted_companies = sorted(company_names.keys(), key=len, reverse=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, video: dict[str, Any]) -> FilterResult:
        description = (video.get("description") or "").strip()
        hashtags_raw = video.get("hashtags") or []
        if isinstance(hashtags_raw, str):
            # Already a space/comma-separated string or JSON
            try:
                hashtags_raw = json.loads(hashtags_raw)
            except (json.JSONDecodeError, TypeError):
                hashtags_raw = [h.strip() for h in re.split(r"[,\s]+", hashtags_raw) if h.strip()]
        hashtag_str = " ".join(str(h).lower().lstrip("#") for h in hashtags_raw)
        transcript = (video.get("transcript") or video.get("_transcript") or "").strip()

        full_text = f"{description} {transcript}"
        full_text_lower = full_text.lower()

        # Rule A — cashtag
        cashtags = _CASHTAG_RE.findall(full_text)
        if cashtags:
            matched = sorted({c for c in cashtags if c in self._tickers or len(c) >= 2})
            return FilterResult(
                passed=True,
                matched_rule="cashtag",
                matched_tickers=matched or cashtags,
                matched_hashtags=[],
                matched_terms=[],
            )

        # Rule B — bare ticker / company name in text
        ticker_matches: list[str] = []
        if self._ticker_re:
            ticker_matches = self._ticker_re.findall(full_text)
        company_matches: list[str] = []
        for name in self._sorted_companies:
            if name in full_text_lower:
                ticker = self._company_names[name]
                if ticker:
                    company_matches.append(ticker)
        all_ticker_hits = sorted(set(ticker_matches + company_matches))
        if all_ticker_hits:
            return FilterResult(
                passed=True,
                matched_rule="ticker_name",
                matched_tickers=all_ticker_hits,
                matched_hashtags=[],
                matched_terms=[],
            )

        # Rule C — finance genre hashtag AND market condition term
        matched_hashtags = [h for h in self._hashtags if h in hashtag_str]
        if matched_hashtags:
            matched_terms = [t for t in self._market_terms if t in full_text_lower]
            if matched_terms:
                return FilterResult(
                    passed=True,
                    matched_rule="genre_market",
                    matched_tickers=[],
                    matched_hashtags=matched_hashtags,
                    matched_terms=matched_terms,
                )

        return FilterResult(
            passed=False,
            matched_rule="",
            matched_tickers=[],
            matched_hashtags=[],
            matched_terms=[],
        )

    def apply_batch(self, videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply Pass 1 and return videos that survive, with filter metadata attached."""
        survivors = []
        for v in videos:
            result = self.apply(v)
            if result.passed:
                v = dict(v)
                v["_pass1_rule"] = result.matched_rule
                v["_pass1_tickers"] = result.matched_tickers
                v["_pass1_hashtags"] = result.matched_hashtags
                v["_pass1_terms"] = result.matched_terms
                survivors.append(v)
        return survivors

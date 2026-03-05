from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from pipeline.constants import PRODUCT_CATEGORIES, RETAILER_KEYWORDS


@dataclass
class ExtractionResult:
    brands: list[str]
    products: list[str]
    retailers: list[str]
    categories: list[str]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _find_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    blob = _normalize_text(text)
    found = []
    for kw in keywords:
        pattern = rf"\b{re.escape(kw.lower())}\b"
        if re.search(pattern, blob):
            found.append(kw)
    return sorted(set(found))


def _extract_product_like_phrases(text: str) -> list[str]:
    blob = _normalize_text(text)
    phrases = set()
    for match in re.finditer(r"\b([a-z0-9][a-z0-9\-]{2,}(?:\s+[a-z0-9][a-z0-9\-]{2,}){0,3})\b", blob):
        chunk = match.group(1)
        if any(token in chunk for token in ["buy", "link", "ship", "scam", "worth"]):
            continue
        if len(chunk) < 4:
            continue
        phrases.add(chunk)
    return sorted(phrases)[:20]


def extract_entities(text: str) -> ExtractionResult:
    retailers = _find_keywords(text, RETAILER_KEYWORDS)
    categories = _find_keywords(text, PRODUCT_CATEGORIES)

    products = _extract_product_like_phrases(text)
    brands = []
    for token in products:
        parts = token.split()
        if 1 <= len(parts) <= 2 and all(len(p) > 2 for p in parts):
            brands.append(token)

    return ExtractionResult(
        brands=sorted(set(brands))[:20],
        products=products,
        retailers=retailers,
        categories=categories,
    )

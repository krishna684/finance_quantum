from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NewsArticle:
    article_id: str           # UUID or database-assigned ID
    source: str               # "WSJ" | "NYT"
    headline: str
    full_body: str
    publication_date: str     # YYYY-MM-DD
    publication_time: Optional[str]   # HH:MM:SS or None
    timezone: str             # e.g. "America/New_York"
    section: Optional[str]    # "Markets", "Business", etc.
    front_page: bool
    authors: list[str]
    word_count: int
    database_source_id: str   # original DB record ID for audit trail
    url: Optional[str]        # if available from export


@dataclass
class NewsImage:
    image_id: str             # UUID
    article_id: str
    image_binary: Optional[bytes]
    caption: Optional[str]
    placement: Optional[str]  # "lead" | "inline" | "thumbnail"
    image_order: int
    width_px: Optional[int]
    height_px: Optional[int]
    sha256_hash: str
    file_extension: str       # "jpg" | "png"

    @staticmethod
    def compute_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

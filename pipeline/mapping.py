from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pipeline.storage import Storage


@dataclass(frozen=True)
class BrandFirmRecord:
    parent_company_name: str
    ticker: str
    permno: str
    brand_name: str
    product_line: str
    product_category: str
    source_type: str
    source_ref: str
    confidence_score: float
    is_public_company: bool


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_mapping_csv(csv_path: Path, storage: Storage) -> int:
    required = {
        "parent_company_name",
        "ticker",
        "permno",
        "brand_name",
        "product_line",
        "product_category",
        "source_type",
        "source_ref",
        "confidence_score",
        "is_public_company",
    }
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        now = _utc_now_iso()
        count = 0
        for row in reader:
            storage.conn.execute(
                """
                INSERT INTO brand_firm_mapping(
                    parent_company_name, ticker, permno, brand_name, product_line, product_category,
                    source_type, source_ref, confidence_score, is_public_company, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["parent_company_name"].strip(),
                    row["ticker"].strip(),
                    row["permno"].strip(),
                    row["brand_name"].strip(),
                    row["product_line"].strip(),
                    row["product_category"].strip(),
                    row["source_type"].strip(),
                    row["source_ref"].strip(),
                    float(row["confidence_score"] or 0),
                    1 if str(row["is_public_company"]).strip().lower() in {"1", "true", "yes"} else 0,
                    now,
                    now,
                ),
            )
            count += 1
    storage.commit()
    return count

from __future__ import annotations

"""CRSP PERMNO lookup and ticker→identifier bridging.

This module loads a CRSP-style identifier mapping file and provides
time-aware lookups from ticker symbols to PERMNO / NCUSIP / gvkey.

Expected CSV schema (CRSP msenames or custom export)
-----------------------------------------------------
permno, ticker, comnam, ncusip, gvkey, namedt, nameendt
  - namedt / nameendt: effective date range (YYYY-MM-DD)
  - All columns except permno are optional but recommended.

Usage
-----
    from pipeline.crsp import CRSPMapper
    mapper = CRSPMapper.from_csv(Path("data/crsp_identifiers.csv"))
    result = mapper.lookup("AAPL", as_of="2024-01-15")
    print(result.permno, result.ncusip)

If you do not yet have a CRSP file, you can load a minimal stub:
    mapper = CRSPMapper.from_stub({"AAPL": "14593", "TSLA": "93436"})
"""

import csv
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class IdentifierRecord:
    permno: str
    ticker: str
    company_name: str
    ncusip: Optional[str]
    gvkey: Optional[str]
    date_from: Optional[date]   # inclusive
    date_to: Optional[date]     # inclusive, None = present


@dataclass
class LookupResult:
    permno: str
    ticker: str
    company_name: str
    ncusip: Optional[str] = None
    gvkey: Optional[str] = None
    ambiguous: bool = False     # True if multiple records matched


def _parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s or s.lower() in ("", ".", "nan", "none"):
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            from datetime import datetime
            return datetime.strptime(s[:10], fmt).date()
        except ValueError:
            continue
    return None


class CRSPMapper:
    def __init__(self, records: list[IdentifierRecord]) -> None:
        # Index by uppercase ticker for O(1) lookups
        self._by_ticker: dict[str, list[IdentifierRecord]] = {}
        self._by_permno: dict[str, list[IdentifierRecord]] = {}
        for rec in records:
            self._by_ticker.setdefault(rec.ticker.upper(), []).append(rec)
            self._by_permno.setdefault(rec.permno, []).append(rec)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: Path) -> "CRSPMapper":
        """Load from CRSP msenames CSV or similar flat file.

        Accepts any CSV with at least 'permno' and 'ticker' columns.
        """
        records: list[IdentifierRecord] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return cls([])
            # normalize header
            norm_fields = {k.lower().strip() for k in reader.fieldnames}
            for row in reader:
                row_n = {k.lower().strip(): v for k, v in row.items()}
                permno = row_n.get("permno", "").strip()
                ticker = row_n.get("ticker", "").strip().upper()
                if not permno or not ticker:
                    continue
                records.append(
                    IdentifierRecord(
                        permno=permno,
                        ticker=ticker,
                        company_name=row_n.get("comnam", row_n.get("company_name", "")).strip(),
                        ncusip=row_n.get("ncusip", "").strip() or None,
                        gvkey=row_n.get("gvkey", "").strip() or None,
                        date_from=_parse_date(row_n.get("namedt", "")),
                        date_to=_parse_date(row_n.get("nameendt", "")),
                    )
                )
        return cls(records)

    @classmethod
    def from_stub(cls, ticker_permno: dict[str, str]) -> "CRSPMapper":
        """Create a minimal mapper from a {ticker: permno} dict.

        Useful for testing or when CRSP access is not yet available.
        """
        records = [
            IdentifierRecord(
                permno=permno,
                ticker=ticker.upper(),
                company_name="",
                ncusip=None,
                gvkey=None,
                date_from=None,
                date_to=None,
            )
            for ticker, permno in ticker_permno.items()
        ]
        return cls(records)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def lookup(self, ticker: str, as_of: str | date | None = None) -> Optional[LookupResult]:
        """Return identifier record for *ticker* effective on *as_of* date.

        If *as_of* is None, returns the most recent record.
        Returns None if ticker is not found.
        """
        candidates = self._by_ticker.get(ticker.upper(), [])
        if not candidates:
            return None

        if as_of is not None:
            if isinstance(as_of, str):
                as_of_date = _parse_date(as_of) or date.today()
            else:
                as_of_date = as_of
            active = [
                r for r in candidates
                if (r.date_from is None or r.date_from <= as_of_date)
                and (r.date_to is None or r.date_to >= as_of_date)
            ]
            if active:
                candidates = active

        # prefer record with latest start date
        candidates = sorted(candidates, key=lambda r: r.date_from or date.min, reverse=True)
        rec = candidates[0]
        return LookupResult(
            permno=rec.permno,
            ticker=rec.ticker,
            company_name=rec.company_name,
            ncusip=rec.ncusip,
            gvkey=rec.gvkey,
            ambiguous=len(candidates) > 1,
        )

    def lookup_permno(self, permno: str) -> Optional[LookupResult]:
        """Reverse lookup: PERMNO → latest identifier record."""
        candidates = self._by_permno.get(permno, [])
        if not candidates:
            return None
        rec = sorted(candidates, key=lambda r: r.date_from or date.min, reverse=True)[0]
        return LookupResult(
            permno=rec.permno,
            ticker=rec.ticker,
            company_name=rec.company_name,
            ncusip=rec.ncusip,
            gvkey=rec.gvkey,
            ambiguous=len(candidates) > 1,
        )

    def enrich_tickers(
        self, tickers: list[str], as_of: str | date | None = None
    ) -> list[dict]:
        """Map a list of tickers to PERMNO records.

        Returns list of dicts with keys: ticker, permno, ncusip, gvkey, company_name.
        Tickers not found in CRSP are omitted.
        """
        results = []
        seen = set()
        for ticker in tickers:
            if ticker in seen:
                continue
            seen.add(ticker)
            res = self.lookup(ticker, as_of=as_of)
            if res:
                results.append({
                    "ticker": res.ticker,
                    "permno": res.permno,
                    "ncusip": res.ncusip,
                    "gvkey": res.gvkey,
                    "company_name": res.company_name,
                })
        return results

    @property
    def ticker_count(self) -> int:
        return len(self._by_ticker)

from __future__ import annotations

"""CRSP daily returns loader and time-alignment utilities.

Loads CRSP daily stock returns from a flat CSV and provides:
  - Per-PERMNO return lookup by date
  - Trading day calendar (forward-fill weekends/holidays)
  - Time alignment: classifies media timestamps into pre/market/post-market
    windows and assigns the correct trading day for return matching

CRSP DSF (Daily Stock File) expected columns
--------------------------------------------
permno, date, ret, retx, prc, vol, shrout, cfacpr
  - date: YYYY-MM-DD
  - ret: holding period return (may contain 'C' suffix — handle gracefully)
  - prc: absolute value = closing price

Usage
-----
    from pipeline.returns import CRSPReturns, align_timestamp

    returns = CRSPReturns.from_csv(Path("data/crsp_dsf.csv"))
    ret = returns.get_return(permno="14593", date="2024-01-16")
    print(ret)  # 0.0123

    event_date, window = align_timestamp("2024-01-15T18:00:00-05:00")
    # event_date = "2024-01-16", window = "post_market"
"""

import csv
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, time as time_type
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

# Trading day session boundaries (US Eastern time)
_ET = ZoneInfo("America/New_York")

# Pre-market:  4:00 PM previous day → 9:30 AM current day
# Market:      9:30 AM → 4:00 PM
# Post-market: 4:00 PM → midnight
_MARKET_OPEN = time_type(9, 30)
_MARKET_CLOSE = time_type(16, 0)
_PRE_MARKET_START = time_type(4, 0)


# ---------------------------------------------------------------------------
# Time alignment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlignedEvent:
    event_datetime_utc: datetime
    event_date_et: date
    trading_day: date        # date whose open-to-close return this predicts
    window: str              # 'pre_market' | 'market' | 'post_market' | 'weekend'
    next_trading_day: date   # always the next available trading day after event


def _is_us_market_day(d: date) -> bool:
    """True for Mon–Fri. Does not exclude individual holidays (requires calendar library)."""
    return d.weekday() < 5


def _next_trading_day(d: date) -> date:
    """Return next Mon–Fri after *d*."""
    nxt = d + timedelta(days=1)
    while not _is_us_market_day(nxt):
        nxt += timedelta(days=1)
    return nxt


def align_timestamp(ts: str | datetime) -> AlignedEvent:
    """Classify a media event timestamp and return its aligned trading day.

    Classification logic (matching Section IX of the protocol):
      - Pre-market  (4:00 AM–9:30 AM ET) → predicts open-to-close return on same day
      - Market hours (9:30 AM–4:00 PM ET) → predicts remaining intraday or t+1
      - Post-market  (4:00 PM–midnight ET) → assigned to next trading day
      - Weekend/holiday → rolled forward to next trading day
    """
    if isinstance(ts, str):
        # Handle ISO 8601 with or without timezone
        ts_clean = ts.strip()
        try:
            dt = datetime.fromisoformat(ts_clean)
        except ValueError:
            # fallback: treat as UTC date only
            dt = datetime.strptime(ts_clean[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        dt = ts

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    dt_et = dt.astimezone(_ET)
    event_date = dt_et.date()
    event_time = dt_et.time()

    if not _is_us_market_day(event_date):
        window = "weekend"
        trading_day = _next_trading_day(event_date)
    elif event_time < _PRE_MARKET_START:
        # Treat as previous evening (post-market of prior day)
        window = "post_market"
        trading_day = event_date if _is_us_market_day(event_date) else _next_trading_day(event_date)
    elif event_time < _MARKET_OPEN:
        window = "pre_market"
        trading_day = event_date
    elif event_time < _MARKET_CLOSE:
        window = "market"
        trading_day = event_date
    else:
        window = "post_market"
        trading_day = _next_trading_day(event_date)

    return AlignedEvent(
        event_datetime_utc=dt.astimezone(timezone.utc),
        event_date_et=event_date,
        trading_day=trading_day,
        window=window,
        next_trading_day=_next_trading_day(trading_day),
    )


# ---------------------------------------------------------------------------
# CRSP daily returns
# ---------------------------------------------------------------------------

def _clean_ret(raw: str) -> Optional[float]:
    """Parse CRSP return field; returns None for missing/error codes."""
    s = (raw or "").strip()
    if not s or s in (".", "", "C", "B", "T", "P", "S", "nan"):
        return None
    # Remove trailing letter codes (e.g. "0.05C")
    s = re.sub(r"[A-Za-z]+$", "", s)
    try:
        return float(s)
    except ValueError:
        return None


@dataclass
class ReturnRecord:
    permno: str
    date: date
    ret: Optional[float]       # holding period return
    retx: Optional[float]      # ex-dividend return
    prc: Optional[float]       # closing price (abs value)
    vol: Optional[int]         # trading volume
    shrout: Optional[int]      # shares outstanding (thousands)


class CRSPReturns:
    def __init__(self, records: list[ReturnRecord]) -> None:
        # Index: (permno, date_str) → ReturnRecord
        self._idx: dict[tuple[str, str], ReturnRecord] = {
            (r.permno, r.date.isoformat()): r for r in records
        }
        # Trading calendar: sorted unique dates
        all_dates = sorted({r.date for r in records})
        self._trading_days = all_dates

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: Path) -> "CRSPReturns":
        """Load CRSP DSF from flat CSV.

        Handles common column name variants from WRDS exports.
        """
        records: list[ReturnRecord] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return cls([])
            for row in reader:
                row_n = {k.lower().strip(): v for k, v in row.items()}
                permno = row_n.get("permno", "").strip()
                raw_date = row_n.get("date", "").strip()
                if not permno or not raw_date:
                    continue
                from datetime import datetime as _dt
                try:
                    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
                        try:
                            parsed_date = _dt.strptime(raw_date[:10], fmt).date()
                            break
                        except ValueError:
                            continue
                    else:
                        continue
                except Exception:
                    continue

                prc_raw = _clean_ret(row_n.get("prc", ""))

                records.append(ReturnRecord(
                    permno=permno,
                    date=parsed_date,
                    ret=_clean_ret(row_n.get("ret", "")),
                    retx=_clean_ret(row_n.get("retx", "")),
                    prc=abs(prc_raw) if prc_raw is not None else None,
                    vol=int(float(row_n["vol"])) if row_n.get("vol", "").strip() not in ("", ".") else None,
                    shrout=int(float(row_n["shrout"])) if row_n.get("shrout", "").strip() not in ("", ".") else None,
                ))
        return cls(records)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_return(self, permno: str, date_: str | date) -> Optional[float]:
        """Return holding period return for (permno, date). None if missing."""
        if isinstance(date_, date):
            key = (permno, date_.isoformat())
        else:
            key = (permno, date_[:10])
        rec = self._idx.get(key)
        return rec.ret if rec else None

    def get_record(self, permno: str, date_: str | date) -> Optional[ReturnRecord]:
        if isinstance(date_, date):
            key = (permno, date_.isoformat())
        else:
            key = (permno, date_[:10])
        return self._idx.get(key)

    def window_return(
        self,
        permno: str,
        start_date: str | date,
        end_date: str | date,
        use_retx: bool = False,
    ) -> Optional[float]:
        """Cumulative return over a date window (buy-and-hold).

        Returns None if any day in the window has a missing return.
        """
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date[:10])
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date[:10])

        cumret = 1.0
        current = start_date
        found_any = False
        while current <= end_date:
            rec = self._idx.get((permno, current.isoformat()))
            if rec:
                r = rec.retx if use_retx else rec.ret
                if r is None:
                    return None
                cumret *= (1 + r)
                found_any = True
            current += timedelta(days=1)
        return (cumret - 1.0) if found_any else None

    def next_open_to_close(self, permno: str, event_ts: str) -> Optional[float]:
        """Return the open-to-close return on the trading day assigned by align_timestamp."""
        aligned = align_timestamp(event_ts)
        return self.get_return(permno, aligned.trading_day)

    @property
    def permno_count(self) -> int:
        return len({k[0] for k in self._idx})

    @property
    def date_range(self) -> tuple[Optional[date], Optional[date]]:
        if not self._trading_days:
            return None, None
        return self._trading_days[0], self._trading_days[-1]

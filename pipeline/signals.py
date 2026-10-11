from __future__ import annotations

"""Signal construction and first-pass regression framework.

Builds panel datasets linking TikTok/news signals to CRSP returns,
then runs basic OLS regressions to test signal → return relationships.

Requires:  pip install pandas scipy statsmodels

Workflow
--------
    from pipeline.signals import SignalBuilder, run_signal_test

    builder = SignalBuilder(
        tiktok_db="data/tiktok_signals.db",
        news_db="data/news.db",
        crsp_csv="data/crsp_dsf.csv",
        crsp_ids_csv="data/crsp_identifiers.csv",
    )
    panel = builder.build_panel(start_date="2023-01-01", end_date="2024-12-31")
    results = run_signal_test(panel)
    print(results.summary())
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Panel row
# ---------------------------------------------------------------------------

@dataclass
class PanelRow:
    permno: str
    ticker: str
    signal_date: date        # date signal was created / posted
    trading_day: date        # assigned trading day (from align_timestamp)
    window: str              # pre_market | market | post_market | weekend
    source: str              # 'tiktok' | 'wsj' | 'nyt'
    signal_type: str         # 'high_purchase_intent' | 'negative_consumer_sentiment' | 'article'
    sentiment_score: Optional[float]
    sentiment_label: Optional[str]
    engagement_score: Optional[int]
    engagement_per_hour: Optional[float]
    velocity_tier: Optional[str]
    ret_t: Optional[float]     # return on trading_day
    ret_t1: Optional[float]    # return on trading_day + 1
    ret_t5: Optional[float]    # cumulative return over next 5 trading days


# ---------------------------------------------------------------------------
# Signal builder
# ---------------------------------------------------------------------------

class SignalBuilder:
    def __init__(
        self,
        tiktok_db: str | Path,
        crsp_csv: str | Path,
        crsp_ids_csv: str | Path,
        news_db: str | Path | None = None,
    ) -> None:
        self.tiktok_db = Path(tiktok_db)
        self.news_db = Path(news_db) if news_db else None
        self.crsp_csv = Path(crsp_csv)
        self.crsp_ids_csv = Path(crsp_ids_csv)

    def _load_crsp(self):
        raise NotImplementedError(
            "CRSP data is not available in this pipeline. "
            "Portfolio economics are descoped. "
            "Do not call this module."
        )
        from pipeline.crsp import CRSPMapper
        from pipeline.returns import CRSPReturns
        mapper = CRSPMapper.from_csv(self.crsp_ids_csv)
        returns = CRSPReturns.from_csv(self.crsp_csv)
        return mapper, returns

    def _trading_days_window(self, returns, start: date, n: int) -> date:
        """Return the trading day *n* business days after *start*."""
        current = start
        count = 0
        while count < n:
            current += timedelta(days=1)
            if current in {r.date for r in returns._idx.values()}:  # rough check
                count += 1
        return current

    def _next_n_return(self, returns, permno: str, trading_day: date, n: int) -> Optional[float]:
        """Cumulative return over the n trading days starting from trading_day."""
        from pipeline.returns import CRSPReturns
        # Find next n trading days
        days_ahead = []
        current = trading_day
        while len(days_ahead) < n:
            rec = returns.get_record(permno, current)
            if rec and rec.ret is not None:
                days_ahead.append(rec.ret)
            current += timedelta(days=1)
            if (current - trading_day).days > 30:
                break
        if not days_ahead:
            return None
        cumret = 1.0
        for r in days_ahead:
            cumret *= (1 + r)
        return cumret - 1.0

    def build_panel(
        self,
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31",
    ) -> list[PanelRow]:
        """Build panel linking all signals with tickers to CRSP returns."""
        from pipeline.returns import align_timestamp

        mapper, returns = self._load_crsp()
        rows: list[PanelRow] = []

        # --- TikTok signals ---
        if self.tiktok_db.exists():
            conn = sqlite3.connect(self.tiktok_db)
            conn.row_factory = sqlite3.Row
            query = """
            SELECT
                video_id, post_datetime_utc, signal_type,
                sentiment_score, sentiment_label,
                engagement_score, engagement_per_hour, velocity_tier,
                mapped_tickers
            FROM tiktok_videos
            WHERE post_datetime_utc >= ? AND post_datetime_utc <= ?
              AND mapped_tickers IS NOT NULL AND mapped_tickers != '[]'
            """
            cur = conn.execute(query, (start_date, end_date + "T23:59:59"))
            for rec in cur.fetchall():
                tickers = json.loads(rec["mapped_tickers"] or "[]")
                for ticker in tickers:
                    crsp = mapper.lookup(ticker, as_of=rec["post_datetime_utc"][:10] if rec["post_datetime_utc"] else None)
                    if not crsp:
                        continue
                    try:
                        aligned = align_timestamp(rec["post_datetime_utc"])
                    except Exception:
                        continue
                    ret_t = returns.get_return(crsp.permno, aligned.trading_day)
                    ret_t1 = returns.get_return(crsp.permno, aligned.next_trading_day)
                    ret_t5 = self._next_n_return(returns, crsp.permno, aligned.trading_day, 5)
                    rows.append(PanelRow(
                        permno=crsp.permno,
                        ticker=ticker,
                        signal_date=aligned.event_date_et,
                        trading_day=aligned.trading_day,
                        window=aligned.window,
                        source="tiktok",
                        signal_type=rec["signal_type"] or "",
                        sentiment_score=rec["sentiment_score"],
                        sentiment_label=rec["sentiment_label"],
                        engagement_score=rec["engagement_score"],
                        engagement_per_hour=rec["engagement_per_hour"],
                        velocity_tier=rec["velocity_tier"],
                        ret_t=ret_t,
                        ret_t1=ret_t1,
                        ret_t5=ret_t5,
                    ))
            conn.close()

        # --- News signals ---
        if self.news_db and self.news_db.exists():
            conn = sqlite3.connect(self.news_db)
            conn.row_factory = sqlite3.Row
            query = """
            SELECT
                article_id, source, publication_date, publication_time,
                sentiment_score, sentiment_label, mapped_tickers, word_count
            FROM news_articles
            WHERE publication_date >= ? AND publication_date <= ?
              AND mapped_tickers IS NOT NULL AND mapped_tickers != '[]'
            """
            cur = conn.execute(query, (start_date[:10], end_date[:10]))
            for rec in cur.fetchall():
                tickers = json.loads(rec["mapped_tickers"] or "[]")
                ts = f"{rec['publication_date']}T{rec['publication_time'] or '09:00:00'}-05:00"
                for ticker in tickers:
                    crsp = mapper.lookup(ticker, as_of=rec["publication_date"])
                    if not crsp:
                        continue
                    try:
                        aligned = align_timestamp(ts)
                    except Exception:
                        continue
                    ret_t = returns.get_return(crsp.permno, aligned.trading_day)
                    ret_t1 = returns.get_return(crsp.permno, aligned.next_trading_day)
                    ret_t5 = self._next_n_return(returns, crsp.permno, aligned.trading_day, 5)
                    rows.append(PanelRow(
                        permno=crsp.permno,
                        ticker=ticker,
                        signal_date=aligned.event_date_et,
                        trading_day=aligned.trading_day,
                        window=aligned.window,
                        source=rec["source"].lower(),
                        signal_type="article",
                        sentiment_score=rec["sentiment_score"],
                        sentiment_label=rec["sentiment_label"],
                        engagement_score=rec["word_count"],
                        engagement_per_hour=None,
                        velocity_tier=None,
                        ret_t=ret_t,
                        ret_t1=ret_t1,
                        ret_t5=ret_t5,
                    ))
            conn.close()

        return rows


# ---------------------------------------------------------------------------
# Signal test
# ---------------------------------------------------------------------------

@dataclass
class SignalTestResults:
    n_obs: int
    n_firms: int
    signal_name: str
    dep_var: str
    coef: float
    t_stat: float
    p_value: float
    r_squared: float
    mean_signal: float
    std_signal: float
    mean_return: float
    std_return: float

    def summary(self) -> str:
        sig = "***" if self.p_value < 0.01 else ("**" if self.p_value < 0.05 else ("*" if self.p_value < 0.10 else ""))
        return (
            f"\n=== Signal Test: {self.signal_name} → {self.dep_var} ===\n"
            f"  N={self.n_obs:,}  Firms={self.n_firms}\n"
            f"  Coef={self.coef:.6f}  t={self.t_stat:.3f}  p={self.p_value:.4f} {sig}\n"
            f"  R²={self.r_squared:.4f}\n"
            f"  Signal: mean={self.mean_signal:.4f}  std={self.std_signal:.4f}\n"
            f"  Return: mean={self.mean_return:.4f}  std={self.std_return:.4f}\n"
        )


def run_signal_test(
    panel: list[PanelRow],
    signal_col: str = "sentiment_score",
    return_col: str = "ret_t1",
    source_filter: str | None = None,
    signal_type_filter: str | None = None,
) -> SignalTestResults:
    """Run OLS regression: return_col ~ signal_col.

    Requires pandas and statsmodels.
    """
    try:
        import pandas as pd
        import statsmodels.api as sm
    except ImportError as e:
        raise ImportError("pip install pandas statsmodels scipy") from e

    rows = panel
    if source_filter:
        rows = [r for r in rows if r.source == source_filter]
    if signal_type_filter:
        rows = [r for r in rows if r.signal_type == signal_type_filter]

    data = []
    for r in rows:
        sig = getattr(r, signal_col, None)
        ret = getattr(r, return_col, None)
        if sig is not None and ret is not None:
            data.append({"signal": sig, "return": ret, "permno": r.permno})

    if len(data) < 10:
        raise ValueError(f"Insufficient data for regression: {len(data)} rows after filtering")

    df = pd.DataFrame(data)
    X = sm.add_constant(df["signal"])
    y = df["return"]
    model = sm.OLS(y, X).fit(cov_type="HC3")

    return SignalTestResults(
        n_obs=len(df),
        n_firms=df["permno"].nunique(),
        signal_name=signal_col,
        dep_var=return_col,
        coef=float(model.params["signal"]),
        t_stat=float(model.tvalues["signal"]),
        p_value=float(model.pvalues["signal"]),
        r_squared=float(model.rsquared),
        mean_signal=float(df["signal"].mean()),
        std_signal=float(df["signal"].std()),
        mean_return=float(y.mean()),
        std_return=float(y.std()),
    )

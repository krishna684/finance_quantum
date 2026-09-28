#!/usr/bin/env python
"""Build labeled.parquet: N24News finance subset joined with market returns.

Label construction
------------------
  mkt_ret_next  — Ken French daily Mkt-RF + RF, in DECIMAL units,
                  on the next_trading_day following the article date.
  label_up      — int(mkt_ret_next > 0)

Factor source
-------------
  pandas_datareader  F-F_Research_Data_Factors_daily
  Raw download is in PERCENT; empirically verified before conversion.
  Cross-checked against SPY adjusted-close (yfinance); pass threshold: corr > 0.99.

NOTE: data/ff_factors_1857.csv is NOT used here (covers 1857 only, irrelevant
for N24News 2010-2021). It is retained in data/ as it may be used by other
parts of the pipeline that reference historical data.

Usage
-----
    python scripts/build_labels.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = _PROJECT_ROOT / "data" / "index.parquet"
OUTPUT = _PROJECT_ROOT / "data" / "labeled.parquet"

_FF_START = "2009-01-01"
_FF_END   = "2022-01-01"
_SPY_CORR_THRESHOLD = 0.99


# ---------------------------------------------------------------------------
# Ken French factor download
# ---------------------------------------------------------------------------

def _download_ff_factors() -> pd.DataFrame:
    """Download F-F daily factors. Returns DataFrame with date index (date type)
    and decimal-unit columns: Mkt_RF, RF, mkt_ret."""
    print(f"Downloading F-F_Research_Data_Factors_daily ({_FF_START} – {_FF_END}) ...")
    raw = web.DataReader(
        "F-F_Research_Data_Factors_daily", "famafrench",
        start=_FF_START, end=_FF_END,
    )[0]

    # Empirical unit check — MUST be percent (mean |Mkt-RF| ~0.6-0.8 per day)
    mean_abs_mkt = float(np.abs(raw["Mkt-RF"]).mean())
    print(f"  Raw mean(|Mkt-RF|): {mean_abs_mkt:.4f}")
    if mean_abs_mkt < 0.05:
        print("[VALIDATION FAILED] mean(|Mkt-RF|) < 0.05: data appears already in decimal "
              "or is near-zero. Cannot safely divide by 100.")
        sys.exit(1)
    if mean_abs_mkt > 5.0:
        print("[VALIDATION FAILED] mean(|Mkt-RF|) > 5.0: unexpectedly large — "
              "check data source.")
        sys.exit(1)
    print(f"  Unit check PASSED: PERCENT (dividing by 100)")

    # Convert Period index -> date
    ff = raw[["Mkt-RF", "RF"]].copy()
    ff.index = pd.to_datetime([str(p) for p in ff.index]).date
    ff.index.name = "date"

    # Convert PERCENT -> DECIMAL
    ff["Mkt_RF"] = ff["Mkt-RF"] / 100.0
    ff["RF"]     = ff["RF"]     / 100.0
    ff["mkt_ret"] = ff["Mkt_RF"] + ff["RF"]
    ff = ff[["Mkt_RF", "RF", "mkt_ret"]]

    print(f"  FF factors: {len(ff):,} trading days  "
          f"({ff.index.min()} -> {ff.index.max()})")
    return ff


# ---------------------------------------------------------------------------
# SPY cross-check
# ---------------------------------------------------------------------------

def _spy_crosscheck(ff: pd.DataFrame) -> None:
    """Download SPY adjusted close and verify correlation with FF market return."""
    import yfinance as yf

    print("Downloading SPY for cross-check ...")
    spy_raw = yf.download("SPY", start=_FF_START, end=_FF_END,
                          auto_adjust=True, progress=False)
    if spy_raw.empty:
        print("[VALIDATION FAILED] yfinance returned empty SPY data.")
        sys.exit(1)

    # Handle MultiIndex columns (yfinance 0.2+)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)

    spy_ret = spy_raw["Close"].pct_change().dropna()
    spy_ret.index = pd.to_datetime(spy_ret.index).date

    # Align on common dates
    common = sorted(set(ff.index) & set(spy_ret.index))
    if len(common) < 100:
        print(f"[VALIDATION FAILED] Only {len(common)} common dates between FF and SPY.")
        sys.exit(1)

    ff_aligned  = ff.loc[common, "mkt_ret"]
    spy_aligned = spy_ret.loc[common]

    corr = float(np.corrcoef(ff_aligned.values, spy_aligned.values)[0, 1])
    print(f"  SPY vs FF mkt_ret correlation: {corr:.6f}  (threshold > {_SPY_CORR_THRESHOLD})")
    if corr < _SPY_CORR_THRESHOLD:
        print(f"[VALIDATION FAILED] SPY correlation {corr:.4f} < {_SPY_CORR_THRESHOLD}. "
              f"Check FF download or unit conversion.")
        sys.exit(1)
    print(f"  Cross-check PASSED")


# ---------------------------------------------------------------------------
# Join & label
# ---------------------------------------------------------------------------

def _build_labeled(index_df: pd.DataFrame, ff: pd.DataFrame) -> pd.DataFrame:
    ff_dates = set(ff.index)
    index_dates = set(index_df["next_trading_day"])

    in_coverage    = index_df["next_trading_day"].isin(ff_dates)
    out_of_coverage = ~in_coverage

    n_out = int(out_of_coverage.sum())
    if n_out > 0:
        tail_dates = sorted(index_df.loc[out_of_coverage, "next_trading_day"].unique())
        print(f"  Tail-date rows outside FF coverage: {n_out:,}")
        print(f"  Tail dates (unique): {tail_dates[:10]}")
    else:
        print("  All rows within FF coverage.")

    # Match rate check within coverage
    in_cov_df = index_df[in_coverage].copy()
    matched = in_cov_df["next_trading_day"].isin(ff_dates).sum()
    match_rate = matched / len(in_cov_df) if len(in_cov_df) > 0 else 0.0
    print(f"  Match rate within coverage: {matched:,}/{len(in_cov_df):,} = {match_rate:.6f}")
    if match_rate < 1.0:
        print(f"[VALIDATION FAILED] Match rate {match_rate:.4f} < 1.0 within coverage.")
        sys.exit(1)

    # Join
    ff_series = ff["mkt_ret"]
    df = in_cov_df.copy()
    df["mkt_ret_next"] = df["next_trading_day"].map(ff_series)
    df["label_up"]     = (df["mkt_ret_next"] > 0).astype(int)

    return df, n_out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, n_dropped: int) -> None:
    errors: list[str] = []

    null_mkt = int(df["mkt_ret_next"].isna().sum())
    if null_mkt:
        errors.append(f"Null mkt_ret_next: {null_mkt}")

    null_lbl = int(df["label_up"].isna().sum())
    if null_lbl:
        errors.append(f"Null label_up: {null_lbl}")

    if errors:
        print("\n[VALIDATION FAILED]")
        for e in errors:
            print(f"  x {e}")
        sys.exit(1)

    n_up    = int(df["label_up"].sum())
    n_total = len(df)
    pct_up  = n_up / n_total * 100

    print("\n[VALIDATION PASSED]")
    print(f"  + Rows retained:      {n_total:,}")
    if n_dropped:
        print(f"  + Rows dropped (tail-date past FF coverage): {n_dropped:,}")
    print(f"  + Null mkt_ret_next:  0")
    print(f"  + Null label_up:      0")
    print(f"  + label_up=1 (up):    {n_up:,} / {n_total:,} = {pct_up:.1f}%")
    print(f"  + label_up=0 (down):  {n_total - n_up:,} / {n_total:,} = {100 - pct_up:.1f}%")
    print(f"  + mkt_ret_next range: {df['mkt_ret_next'].min():.5f} -> {df['mkt_ret_next'].max():.5f}")
    print(f"  + Date range:         {df['date'].min()} -> {df['date'].max()}")
    print(f"  + date_source counts: {dict(df['date_source'].value_counts())}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading {INDEX_PATH} ...")
    index_df = pd.read_parquet(INDEX_PATH)
    finance_df = index_df[index_df["is_finance"]].copy().reset_index(drop=True)
    print(f"Finance rows: {len(finance_df):,}")

    ff = _download_ff_factors()
    _spy_crosscheck(ff)

    print("\nJoining on next_trading_day ...")
    labeled_df, n_dropped = _build_labeled(finance_df, ff)

    _validate(labeled_df, n_dropped)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_parquet(OUTPUT, index=False)
    print(f"\n  -> {OUTPUT}  ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

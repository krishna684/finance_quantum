"""Tests for pipeline.portfolio — metrics, quintile sort, factor regression."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def _make_series(returns: list[float], start: str = "2021-01-04"):
    """Create a pandas Series with a DatetimeIndex."""
    import pandas as pd

    idx = pd.date_range(start=start, periods=len(returns), freq="B")
    return pd.Series(returns, index=idx, name="ls")


def test_compute_summary_stats_positive_sharpe():
    from pipeline.portfolio import compute_summary_stats

    returns = [0.002] * 252  # constant positive daily return
    s = compute_summary_stats(_make_series(returns), model="test")

    assert s.ann_return > 0
    assert s.ann_vol > 0
    assert s.sharpe > 0
    assert s.max_drawdown == pytest.approx(0.0, abs=1e-9)
    assert s.n_days == 252


def test_compute_summary_stats_all_negative():
    from pipeline.portfolio import compute_summary_stats

    returns = [-0.002] * 252
    s = compute_summary_stats(_make_series(returns), model="test")

    assert s.ann_return < 0
    assert s.sharpe < 0


def test_compute_summary_stats_max_drawdown():
    from pipeline.portfolio import compute_summary_stats

    # Peak at day 50, then a 10% drawdown
    returns = [0.001] * 50 + [-0.002] * 50 + [0.001] * 52
    s = compute_summary_stats(_make_series(returns), model="test")

    assert s.max_drawdown < 0, "Max drawdown should be negative"
    assert s.max_drawdown > -1.0, "Max drawdown should be greater than -100%"


def test_compute_summary_stats_empty():
    from pipeline.portfolio import compute_summary_stats

    import pandas as pd

    s = compute_summary_stats(pd.Series([], dtype=float), model="empty")
    assert s.n_days == 0
    assert s.sharpe == 0.0


# ---------------------------------------------------------------------------
# _build_portfolio_series
# ---------------------------------------------------------------------------

def _make_pred_rows(
    dates: list[str],
    permnos: list[str],
    probs: list[float],
    returns: list[float],
    model: str = "test",
) -> list[dict]:
    rows = []
    for d, p, prob, ret in zip(dates, permnos, probs, returns):
        rows.append({
            "date": d,
            "permno": p,
            "predicted_prob": str(prob),
            "actual_return": str(ret),
            "model_name": model,
        })
    return rows


def test_build_portfolio_series_basic():
    from pipeline.portfolio import _build_portfolio_series

    # 10 stocks on same day: top 2 get positive return, bottom 2 get negative
    date_str = "2021-01-04"
    permnos = [f"P{i}" for i in range(10)]
    probs = [i / 10 for i in range(10)]        # P0=low, P9=high
    returns_val = [0.01 if i >= 8 else -0.01 for i in range(10)]  # top 20% positive

    rows = _make_pred_rows(
        [date_str] * 10, permnos, probs, returns_val
    )
    series = _build_portfolio_series(rows, model_name="test")

    assert len(series.dates) == 1
    assert series.long_returns[0] > 0     # Q5 (high prob) → positive return
    assert series.short_returns[0] < 0   # Q1 (low prob) → negative return
    assert series.long_short_returns[0] > 0


def test_build_portfolio_series_skips_insufficient_stocks():
    from pipeline.portfolio import _build_portfolio_series

    # Only 3 stocks → cannot form quintiles
    rows = _make_pred_rows(
        ["2021-01-04"] * 3,
        ["P0", "P1", "P2"],
        [0.3, 0.5, 0.7],
        [0.01, -0.01, 0.02],
    )
    series = _build_portfolio_series(rows, model_name="test")
    assert len(series.dates) == 0


def test_build_portfolio_series_multiple_articles_same_stock():
    from pipeline.portfolio import _build_portfolio_series

    # Two articles for the same permno on the same day → should average probabilities
    rows = _make_pred_rows(
        ["2021-01-04"] * 2 + ["2021-01-04"] * 8,
        ["P0", "P0"] + [f"P{i}" for i in range(1, 9)],
        [0.9, 0.8] + [float(i) / 10 for i in range(1, 9)],
        [0.01, 0.02] + [0.0] * 8,
    )
    series = _build_portfolio_series(rows, model_name="test")
    assert len(series.dates) == 1
    # P0 merged probability = mean(0.9, 0.8) = 0.85 → should be in Q5
    assert series.long_counts[0] >= 1


def test_build_portfolio_series_multiple_dates():
    from pipeline.portfolio import _build_portfolio_series

    dates = ["2021-01-04", "2021-01-05"]
    rows = []
    for d in dates:
        for i in range(10):
            rows.append({
                "date": d,
                "permno": f"P{i}",
                "predicted_prob": str(i / 10),
                "actual_return": str(0.01 if i >= 8 else -0.01),
            })
    series = _build_portfolio_series(rows, model_name="test")
    assert len(series.dates) == 2


# ---------------------------------------------------------------------------
# Factor regression
# ---------------------------------------------------------------------------

def test_factor_regression_returns_alpha_result():
    from pipeline.portfolio import AlphaResult, factor_regression

    import pandas as pd

    n = 100
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    ls = pd.Series(np.random.randn(n) * 0.01, index=idx)
    factors = pd.DataFrame({
        "Mkt-RF": np.random.randn(n) * 0.01,
        "SMB": np.random.randn(n) * 0.005,
        "HML": np.random.randn(n) * 0.005,
        "RMW": np.random.randn(n) * 0.004,
        "CMA": np.random.randn(n) * 0.004,
        "UMD": np.random.randn(n) * 0.006,
    }, index=idx)

    result = factor_regression(ls, factors, model="test_model", factor_model="ff5_umd")

    assert isinstance(result, AlphaResult)
    assert not math.isnan(result.alpha)
    assert not math.isnan(result.alpha_tstat)
    assert not math.isnan(result.r_squared)
    assert 0.0 <= result.r_squared <= 1.0


def test_factor_regression_insufficient_data():
    from pipeline.portfolio import factor_regression

    import pandas as pd

    ls = pd.Series([0.01, 0.02], index=pd.date_range("2020-01-02", periods=2))
    factors = pd.DataFrame({"Mkt-RF": [0.005, 0.003]}, index=ls.index)

    result = factor_regression(ls, factors, model="m", factor_model="capm")
    assert math.isnan(result.alpha)


# ---------------------------------------------------------------------------
# Spanning test
# ---------------------------------------------------------------------------

def test_spanning_test_returns_dict():
    from pipeline.portfolio import spanning_test

    import pandas as pd

    n = 100
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    classical = pd.Series(np.random.randn(n) * 0.01, index=idx)
    quantum = pd.Series(np.random.randn(n) * 0.01, index=idx)

    result = spanning_test(classical, quantum)
    assert set(result.keys()) >= {"alpha", "alpha_tstat", "alpha_pvalue", "beta", "r_squared"}


def test_spanning_test_insufficient_data():
    from pipeline.portfolio import spanning_test

    import pandas as pd

    r1 = pd.Series([0.01], index=pd.date_range("2020-01-02", periods=1))
    r2 = pd.Series([0.02], index=pd.date_range("2020-01-02", periods=1))
    result = spanning_test(r1, r2)
    assert math.isnan(result["alpha"])


# ---------------------------------------------------------------------------
# PortfolioAnalyzer.run (no real data, just checks CSV outputs)
# ---------------------------------------------------------------------------

def _write_predictions_csv(path: Path, n_stocks: int = 10, n_dates: int = 30) -> None:
    """Write a synthetic predictions CSV for PortfolioAnalyzer to consume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date", "article_id", "permno", "ticker",
        "predicted_prob", "predicted_label", "true_label", "actual_return", "fold",
    ]
    base_date = "2021-01-04"
    import pandas as pd
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(base_date, periods=n_dates)]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, dt in enumerate(dates):
            for j in range(n_stocks):
                prob = j / n_stocks
                ret = 0.005 if j >= n_stocks * 0.8 else -0.003
                writer.writerow({
                    "date": dt,
                    "article_id": f"art_{i}_{j}",
                    "permno": f"P{j}",
                    "ticker": f"TK{j}",
                    "predicted_prob": prob,
                    "predicted_label": 1 if prob > 0.5 else 0,
                    "true_label": 1 if ret > 0 else 0,
                    "actual_return": ret,
                    "fold": i % 3,
                })


def test_portfolio_analyzer_run_no_ff(tmp_path):
    """PortfolioAnalyzer should write portfolio return CSVs even without FF factors."""
    from pipeline.portfolio import PortfolioAnalyzer

    results_dir = tmp_path / "results"
    clf_dir = results_dir / "classifiers"
    pred_path = clf_dir / "predictions_image_xgboost.csv"
    _write_predictions_csv(pred_path, n_stocks=10, n_dates=30)

    analyzer = PortfolioAnalyzer(results_dir=results_dir)
    out = analyzer.run(embedding_type="image")

    assert "summary_stats" in out
    assert len(out["summary_stats"]) == 1

    ret_file = results_dir / "portfolio" / "returns_image_xgboost.csv"
    assert ret_file.exists(), "Portfolio returns CSV not written"
    stats_file = results_dir / "portfolio" / "summary_stats_image.csv"
    assert stats_file.exists(), "Summary stats CSV not written"

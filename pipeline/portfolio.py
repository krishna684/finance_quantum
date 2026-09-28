from __future__ import annotations

"""Portfolio construction and economic evaluation for classifier predictions.

Reads the prediction CSVs produced by ClassificationPipeline, aggregates them
into daily sentiment indices, constructs long-short quintile portfolios, and
runs factor-model regressions to compute risk-adjusted alphas.

Portfolio construction
----------------------
    For each trading date and each model:
      1. Average predicted probability across all (article, permno) observations
         for that permno on that date → daily sentiment score per stock.
      2. Rank stocks into quintiles.
      3. Long Q5 (top 20%), short Q1 (bottom 20%).
      4. Equal-weight within each leg; portfolio return = mean(Q5) − mean(Q1).

Factor models
-------------
    CAPM   : R_ls = α + β·Mkt-RF + ε
    FF3    : R_ls = α + β·Mkt-RF + s·SMB + h·HML + ε
    FF5    : FF3 + r·RMW + c·CMA
    FF5+UMD: FF5 + u·UMD

    All regressions use HC3 (White) heteroskedasticity-robust standard errors.

Spanning test (placeholder)
---------------------------
    spanning_test(classical_returns, quantum_returns) runs a regression of
    the quantum long-short return on the classical long-short return to test
    whether the classical strategy is spanned by the quantum one.

Fama-French factor data
-----------------------
    Loaded in this order:
      1. --ff-factors CSV if provided (local file)
      2. pandas_datareader from Kenneth French data library (requires internet)

Outputs (results/portfolio/)
-----------------------------
    returns_{embedding_type}_{model_name}.csv
        date, long_return, short_return, long_short_return, long_count, short_count

    factor_alphas_{embedding_type}.csv
        model, factor_model, alpha, alpha_tstat, alpha_pvalue, mkt_beta,
        smb_beta, hml_beta, umd_beta, rmw_beta, cma_beta, r_squared

    summary_stats_{embedding_type}.csv
        model, ann_return, ann_vol, sharpe, max_drawdown, n_days

Usage
-----
    from pipeline.portfolio import PortfolioAnalyzer
    from pathlib import Path

    analyzer = PortfolioAnalyzer(
        results_dir=Path("results"),
        crsp_csv=Path("data/crsp_dsf.csv"),
    )
    analyzer.run(embedding_type="image")
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline.logger import get_logger

log = get_logger(__name__)

_TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PortfolioSeries:
    model_name: str
    dates: list[str]            # ISO date strings
    long_returns: list[float]
    short_returns: list[float]
    long_short_returns: list[float]
    long_counts: list[int]
    short_counts: list[int]


@dataclass
class SummaryStats:
    model: str
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    n_days: int


@dataclass
class AlphaResult:
    model: str
    factor_model: str           # "capm" | "ff3" | "ff5" | "ff5_umd"
    alpha: float
    alpha_tstat: float
    alpha_pvalue: float
    mkt_beta: float
    smb_beta: float
    hml_beta: float
    umd_beta: float
    rmw_beta: float
    cma_beta: float
    r_squared: float


# ---------------------------------------------------------------------------
# FF factor loading
# ---------------------------------------------------------------------------

def load_ff_factors(
    start_date: str,
    end_date: str,
    local_path: Optional[Path] = None,
) -> "pd.DataFrame":
    """Return daily FF5 + UMD factors as a DataFrame.

    Columns: Mkt-RF, SMB, HML, RMW, CMA, UMD, RF
    Values are returns (not percentages; divided by 100 automatically).

    Priority: local CSV > pandas_datareader.
    """
    import pandas as pd

    if local_path and local_path.exists():
        log.info("ff_factors_from_local", path=str(local_path))
        df = pd.read_csv(local_path, index_col=0, parse_dates=True)
        df.columns = [c.strip() for c in df.columns]
        # Normalise if values look like percentages (absolute mean > 0.1)
        for col in df.columns:
            if abs(df[col].mean()) > 0.1:
                df[col] = df[col] / 100.0
        return df

    try:
        import pandas_datareader.data as web

        log.info("ff_factors_downloading_from_french")
        ff5 = web.DataReader(
            "F-F_Research_Data_5_Factors_2x3_daily",
            "famafrench",
            start=start_date,
            end=end_date,
        )[0] / 100.0

        umd = web.DataReader(
            "F-F_Momentum_Factor_daily",
            "famafrench",
            start=start_date,
            end=end_date,
        )[0] / 100.0
        umd = umd.rename(columns={umd.columns[0]: "UMD"})

        factors = ff5.join(umd, how="inner")
        log.info("ff_factors_downloaded", rows=len(factors))
        return factors

    except Exception as exc:
        log.error("ff_factors_unavailable", exc=str(exc))
        raise RuntimeError(
            "Fama-French factors unavailable. Provide --ff-factors path to a local CSV "
            "or install pandas-datareader with internet access."
        ) from exc


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------

def compute_summary_stats(returns: "pd.Series", model: str) -> SummaryStats:
    """Compute annualised return, vol, Sharpe, and max drawdown."""
    import numpy as np

    r = returns.dropna()
    n = len(r)
    if n == 0:
        return SummaryStats(model=model, ann_return=0.0, ann_vol=0.0, sharpe=0.0, max_drawdown=0.0, n_days=0)

    ann_ret = float((1 + r.mean()) ** _TRADING_DAYS_PER_YEAR - 1)
    ann_vol = float(r.std() * np.sqrt(_TRADING_DAYS_PER_YEAR))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = float(dd.min())

    return SummaryStats(
        model=model,
        ann_return=ann_ret,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        n_days=n,
    )


# ---------------------------------------------------------------------------
# Factor regressions
# ---------------------------------------------------------------------------

def factor_regression(
    ls_returns: "pd.Series",
    factors: "pd.DataFrame",
    model: str,
    factor_model: str = "ff5_umd",
) -> AlphaResult:
    """Regress long-short returns on Fama-French factors.

    Uses HC3 heteroskedasticity-robust standard errors.
    """
    import pandas as pd
    import statsmodels.api as sm

    cols_map = {
        "capm": ["Mkt-RF"],
        "ff3": ["Mkt-RF", "SMB", "HML"],
        "ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
        "ff5_umd": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"],
    }
    factor_cols = cols_map.get(factor_model, cols_map["ff5_umd"])
    available = [c for c in factor_cols if c in factors.columns]

    # Align index
    data = pd.concat([ls_returns.rename("y"), factors[available]], axis=1).dropna()
    if len(data) < 10:
        log.warning("insufficient_data_for_regression", model=model, rows=len(data))
        return AlphaResult(
            model=model, factor_model=factor_model,
            alpha=float("nan"), alpha_tstat=float("nan"), alpha_pvalue=float("nan"),
            mkt_beta=float("nan"), smb_beta=float("nan"), hml_beta=float("nan"),
            umd_beta=float("nan"), rmw_beta=float("nan"), cma_beta=float("nan"),
            r_squared=float("nan"),
        )

    # Convention note: long-short portfolio returns are NOT further reduced by the
    # risk-free rate before regression.  A self-financing L/S book requires no net
    # capital, so subtracting RF is non-standard (see Fama & French 1993, footnote 5;
    # and Cochrane 2005 §6.2).  The intercept (α) from this regression is therefore
    # the abnormal return relative to the factor model, not relative to T-bills.
    y = data["y"]
    X = sm.add_constant(data[available])
    result = sm.OLS(y, X).fit(cov_type="HC3")

    def _param(col: str) -> float:
        return float(result.params.get(col, float("nan")))

    def _tstat(col: str) -> float:
        return float(result.tvalues.get(col, float("nan")))

    def _pval(col: str) -> float:
        return float(result.pvalues.get(col, float("nan")))

    return AlphaResult(
        model=model,
        factor_model=factor_model,
        alpha=_param("const"),
        alpha_tstat=_tstat("const"),
        alpha_pvalue=_pval("const"),
        mkt_beta=_param("Mkt-RF"),
        smb_beta=_param("SMB"),
        hml_beta=_param("HML"),
        umd_beta=_param("UMD"),
        rmw_beta=_param("RMW"),
        cma_beta=_param("CMA"),
        r_squared=float(result.rsquared),
    )


# ---------------------------------------------------------------------------
# Spanning test (placeholder)
# ---------------------------------------------------------------------------

def spanning_test(
    classical_returns: "pd.Series",
    quantum_returns: "pd.Series",
    label: str = "",
) -> dict:
    """Test whether the quantum strategy spans the classical one.

    Regresses quantum long-short returns on classical long-short returns.
    An insignificant intercept (α ≈ 0) means the quantum strategy is
    fully spanned by the classical one.  A significant intercept means
    the quantum strategy adds incremental value beyond classical.

    This function is a placeholder to be populated once the quantum
    pipeline is built.

    Returns:
        dict with keys: alpha, alpha_tstat, alpha_pvalue, beta, r_squared
    """
    import pandas as pd
    import statsmodels.api as sm

    data = pd.concat(
        [quantum_returns.rename("y"), classical_returns.rename("x")], axis=1
    ).dropna()

    if len(data) < 10:
        log.warning("spanning_test_insufficient_data", n=len(data), label=label)
        return {"alpha": float("nan"), "alpha_tstat": float("nan"),
                "alpha_pvalue": float("nan"), "beta": float("nan"), "r_squared": float("nan")}

    X = sm.add_constant(data["x"])
    res = sm.OLS(data["y"], X).fit(cov_type="HC3")

    return {
        "alpha": float(res.params.get("const", float("nan"))),
        "alpha_tstat": float(res.tvalues.get("const", float("nan"))),
        "alpha_pvalue": float(res.pvalues.get("const", float("nan"))),
        "beta": float(res.params.get("x", float("nan"))),
        "r_squared": float(res.rsquared),
    }


# ---------------------------------------------------------------------------
# Quintile portfolio builder
# ---------------------------------------------------------------------------

def _build_portfolio_series(
    pred_rows: list[dict],
    model_name: str,
) -> PortfolioSeries:
    """Construct a daily long-short quintile portfolio from predictions.

    For each date: rank stocks by mean predicted probability → quintiles →
    long Q5, short Q1.  Returns are the actual_return stored in predictions.
    """
    import collections
    import numpy as np

    # Aggregate: (date, permno) → (mean_prob, actual_return)
    by_date_permno: dict[tuple[str, str], list[tuple[float, float]]] = (
        collections.defaultdict(list)
    )
    for row in pred_rows:
        key = (row["date"], row["permno"])
        by_date_permno[key].append((float(row["predicted_prob"]), float(row["actual_return"])))

    # Collapse to (date, permno) → (mean_prob, mean_return)
    collapsed: dict[str, list[tuple[str, float, float]]] = collections.defaultdict(list)
    for (dt, permno), obs in by_date_permno.items():
        mean_prob = float(np.mean([o[0] for o in obs]))
        mean_ret = float(np.mean([o[1] for o in obs]))
        collapsed[dt].append((permno, mean_prob, mean_ret))

    dates_sorted = sorted(collapsed.keys())
    port = PortfolioSeries(
        model_name=model_name,
        dates=[],
        long_returns=[],
        short_returns=[],
        long_short_returns=[],
        long_counts=[],
        short_counts=[],
    )

    for dt in dates_sorted:
        stocks = collapsed[dt]
        if len(stocks) < 5:
            # Need at least 5 stocks for quintile ranking
            continue

        stocks_sorted = sorted(stocks, key=lambda x: x[1])  # sort by predicted_prob
        n = len(stocks_sorted)
        quintile_size = n // 5

        q1 = stocks_sorted[:quintile_size]          # bottom 20%
        q5 = stocks_sorted[n - quintile_size:]      # top 20%

        long_ret = float(np.mean([s[2] for s in q5]))
        short_ret = float(np.mean([s[2] for s in q1]))

        port.dates.append(dt)
        port.long_returns.append(long_ret)
        port.short_returns.append(short_ret)
        port.long_short_returns.append(long_ret - short_ret)
        port.long_counts.append(len(q5))
        port.short_counts.append(len(q1))

    return port


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class PortfolioAnalyzer:
    def __init__(
        self,
        results_dir: Path,
        crsp_csv: Optional[Path] = None,
        ff_factors_csv: Optional[Path] = None,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.crsp_csv = Path(crsp_csv) if crsp_csv else None
        self.ff_factors_csv = Path(ff_factors_csv) if ff_factors_csv else None

    def run(self, embedding_type: str = "image") -> dict:
        """Build portfolios and run factor regressions for all models.

        Returns:
            dict with keys 'summary_stats' and 'alpha_results'
        """
        import pandas as pd

        clf_dir = self.results_dir / "classifiers"
        pred_files = list(clf_dir.glob(f"predictions_{embedding_type}_*.csv"))

        if not pred_files:
            log.warning("no_prediction_files", embedding_type=embedding_type, dir=str(clf_dir))
            return {"summary_stats": [], "alpha_results": []}

        port_dir = self.results_dir / "portfolio"
        port_dir.mkdir(parents=True, exist_ok=True)

        all_series: list[PortfolioSeries] = []
        for pred_path in sorted(pred_files):
            model_name = pred_path.stem.replace(f"predictions_{embedding_type}_", "")
            log.info("building_portfolio", model=model_name, file=str(pred_path))

            rows = self._load_predictions(pred_path)
            if not rows:
                continue

            series = _build_portfolio_series(rows, model_name)
            all_series.append(series)
            self._save_portfolio_returns(port_dir, embedding_type, series)

        summary_stats = [
            compute_summary_stats(
                pd.Series(s.long_short_returns, index=pd.to_datetime(s.dates), name="ls"),
                model=s.model_name,
            )
            for s in all_series
        ]
        self._save_summary_stats(port_dir, embedding_type, summary_stats)

        # Factor regressions
        all_alphas: list[AlphaResult] = []
        if all_series:
            min_date = min(s.dates[0] for s in all_series if s.dates)
            max_date = max(s.dates[-1] for s in all_series if s.dates)
            try:
                factors = load_ff_factors(min_date, max_date, self.ff_factors_csv)
                for series in all_series:
                    ls = pd.Series(
                        series.long_short_returns,
                        index=pd.to_datetime(series.dates),
                        name="ls",
                    )
                    for fm in ("capm", "ff3", "ff5", "ff5_umd"):
                        ar = factor_regression(ls, factors, series.model_name, fm)
                        all_alphas.append(ar)
                self._save_alphas(port_dir, embedding_type, all_alphas)
            except RuntimeError as exc:
                log.warning("factor_regression_skipped", exc=str(exc))

        log.info(
            "portfolio_analysis_complete",
            embedding_type=embedding_type,
            n_models=len(all_series),
        )
        return {"summary_stats": summary_stats, "alpha_results": all_alphas}

    @staticmethod
    def _load_predictions(path: Path) -> list[dict]:
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    @staticmethod
    def _save_portfolio_returns(port_dir: Path, embedding_type: str, series: PortfolioSeries) -> None:
        out = port_dir / f"returns_{embedding_type}_{series.model_name}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["date", "long_return", "short_return", "long_short_return",
                            "long_count", "short_count"],
            )
            writer.writeheader()
            for i, dt in enumerate(series.dates):
                writer.writerow({
                    "date": dt,
                    "long_return": f"{series.long_returns[i]:.8f}",
                    "short_return": f"{series.short_returns[i]:.8f}",
                    "long_short_return": f"{series.long_short_returns[i]:.8f}",
                    "long_count": series.long_counts[i],
                    "short_count": series.short_counts[i],
                })
        log.info("portfolio_returns_saved", path=str(out), n_days=len(series.dates))

    @staticmethod
    def _save_summary_stats(port_dir: Path, embedding_type: str, stats: list[SummaryStats]) -> None:
        if not stats:
            return
        out = port_dir / f"summary_stats_{embedding_type}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model", "ann_return", "ann_vol", "sharpe", "max_drawdown", "n_days"],
            )
            writer.writeheader()
            for s in stats:
                writer.writerow({
                    "model": s.model,
                    "ann_return": f"{s.ann_return:.6f}",
                    "ann_vol": f"{s.ann_vol:.6f}",
                    "sharpe": f"{s.sharpe:.6f}",
                    "max_drawdown": f"{s.max_drawdown:.6f}",
                    "n_days": s.n_days,
                })
        log.info("summary_stats_saved", path=str(out))

    @staticmethod
    def _save_alphas(port_dir: Path, embedding_type: str, alphas: list[AlphaResult]) -> None:
        if not alphas:
            return
        out = port_dir / f"factor_alphas_{embedding_type}.csv"
        fieldnames = [
            "model", "factor_model", "alpha", "alpha_tstat", "alpha_pvalue",
            "mkt_beta", "smb_beta", "hml_beta", "umd_beta", "rmw_beta", "cma_beta", "r_squared",
        ]
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for a in alphas:
                writer.writerow({
                    "model": a.model,
                    "factor_model": a.factor_model,
                    "alpha": f"{a.alpha:.8f}",
                    "alpha_tstat": f"{a.alpha_tstat:.4f}",
                    "alpha_pvalue": f"{a.alpha_pvalue:.6f}",
                    "mkt_beta": f"{a.mkt_beta:.6f}",
                    "smb_beta": f"{a.smb_beta:.6f}",
                    "hml_beta": f"{a.hml_beta:.6f}",
                    "umd_beta": f"{a.umd_beta:.6f}",
                    "rmw_beta": f"{a.rmw_beta:.6f}",
                    "cma_beta": f"{a.cma_beta:.6f}",
                    "r_squared": f"{a.r_squared:.6f}",
                })
        log.info("alphas_saved", path=str(out))

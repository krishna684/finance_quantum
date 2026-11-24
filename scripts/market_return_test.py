#!/usr/bin/env python
"""Market-return predictive regression test (FAJ revision).

For each of the four arms in the phase-4 comparison, aggregate the daily
out-of-sample predicted sentiment and test whether it predicts next-day and
five-day S&P 500 excess returns.

Methodology
-----------
1. Re-run the same 72-fold rolling walk-forward splits used in Phase 4
   (deterministic: same seed, same records, same harness).
2. For each test record (labeled photos only), attach the arm's
   out-of-sample P(positive) score and the article's trading_day.
3. For each trading_day t with at least one test prediction, compute:
       daily_sentiment[t] = mean(y_prob_i)  for i in {test records on day t}
4. Merge with the COMPLETE S&P 500 excess-return calendar built from
   data/ff_daily_cache.csv (Fama-French market factor, 2012-2020).
   Using the full trading calendar ensures five-day returns compound over
   five consecutive trading days, not five article-publication days.
       ret_t1  = excess market return on the next actual trading day
       ret_t5  = buy-and-hold excess return over days t+1 ... t+5

   Article timing: trading_day is the next NYSE session on or after the
   article's publication date (from labeled.parquet next_trading_day).
   Because publication times are not available, we cannot distinguish
   pre-close from post-close articles. For the TIMING STRATEGY we add
   one extra lag (use ret_t2, the return two actual trading days ahead)
   so the signal is always at least one full trading day old when the
   trade is placed. The PREDICTIVE REGRESSIONS follow Obaid and
   Pukthuanthong (2022) and use ret_t1.

5. Predictive OLS regression with Newey-West HAC standard errors:
       ret_h = alpha + beta*sentiment_t + gamma_1..5*lag_mkt + delta*dvix
               + dow_dummies + epsilon_t
   maxlags = 5 for both horizons.

6. Long-short timing strategy (conservative, ret_t2):
       signal_t = +1 if daily_sentiment[t] > 0.5, else -1
       strategy_ret = signal_t * ret_t2
   Annualized Sharpe = sqrt(252) * mean(strategy_ret) / std(strategy_ret)
   CAPM alpha: regress strategy_ret on ret_t2 (market), HAC 10 lags, x252.

7. Joint regression ("classical and quantum together"):
       ret_t1 = alpha + beta_c*classical8 + beta_q*qkernel + controls + eps
   Reports the quantum index's incremental coefficient.

Sign convention
---------------
y_prob = P(caption sentiment is positive).
A signal > 0.5 is a bullish photo-sentiment day. The regressions test
whether that positivity predicts higher market excess returns.
Scoring covers labeled photos only (3,612 of 10,206 articles; neutral
captions are excluded because the phase-4 harness requires binary labels).

Excess returns
--------------
All market returns are excess returns (S&P 500 raw minus daily risk-free
rate) from the Fama-French data library, expressed as decimals.

Arms
----
  classical_rbf_pca8   Recomputed from scratch each run (~3 s for 72 folds).
  classical_rbf_pca50  Recomputed from scratch each run.
  qkernel_pca8         Re-run or loaded from cache (~2 min if not cached).
  vqc_pca8             Re-run or loaded from cache (~45 min if not cached).

Outputs
-------
  results/market_return/predictions_{arm}.csv
  results/market_return/daily_sentiment.csv
  results/market_return/regression_results.csv
  results/market_return/joint_regression_results.csv
  results/market_return/timing_strategy.csv
  results/market_return/sharpe_by_arm.csv
  results/market_return/capm_alpha_daily.csv

Usage
-----
  python scripts/market_return_test.py --arms classical
  python scripts/market_return_test.py --arms all
  python scripts/market_return_test.py --arms all --use-cache
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
from sklearn.svm import SVC

_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from phase4_harness import (
    load_records,
    rolling_walk_forward_splits,
    assert_no_leakage,
    build_fold_pipeline,
    undersample_train,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METHODS_PATH  = _PROJECT_ROOT / "data" / "methods_label.parquet"
LABELED_PATH  = _PROJECT_ROOT / "data" / "labeled.parquet"
NPZ_PATH      = _PROJECT_ROOT / "data" / "embeddings" / "n24news_finance.npz"
RESULTS_DIR   = _PROJECT_ROOT / "results" / "market_return"
FF_CACHE_PATH = _PROJECT_ROOT / "data" / "ff_daily_cache.csv"
VIX_PATH      = _PROJECT_ROOT / "data" / "vix_daily.csv"

# Fixed settings shared with phase4_run.py
_QS_BASE = dict(
    n_qubits=8,
    max_train_samples=400,
    vqc_layers=3,
    seed=42,
    shots=None,
)
_VQC_REPS  = 2
_QK_REPS   = 1
_NW_LAGS   = 5   # Newey-West lags for both 1-day and 5-day horizons


# ---------------------------------------------------------------------------
# Complete trading-day calendar (Fama-French + VIX)
# ---------------------------------------------------------------------------

def build_complete_calendar(
    ff_path: Path = FF_CACHE_PATH,
    vix_path: Path = VIX_PATH,
) -> pd.DataFrame:
    """Return a complete trading-day calendar from Fama-French data.

    Uses data/ff_daily_cache.csv (Mkt-RF in percentage, RF in percentage).
    Returns DataFrame indexed by date with columns:
        mkt_excess   daily excess market return (decimal)
        ret_t1       excess return on next actual trading day (prediction)
        ret_t2       excess return 2 actual trading days ahead (conservative trading)
        ret_t5       buy-hold excess return days t+1..t+5 (prediction)
        lag1..lag5   lagged mkt_excess
        dvix         change in VIX level (VIX_t - VIX_{t-1})
        dow_Mon..dow_Thu  day-of-week indicators (Friday is baseline)
    """
    # Fama-French complete calendar
    ff = pd.read_csv(ff_path, parse_dates=["Date"])
    ff = ff.sort_values("Date").reset_index(drop=True)
    ff["trading_day"] = ff["Date"].dt.date
    # Excess return: Mkt-RF is already market minus risk-free, in percentage
    ff["mkt_excess"] = ff["Mkt-RF"] / 100.0

    # VIX
    vix = pd.read_csv(vix_path, parse_dates=["Date"])
    vix["trading_day"] = pd.to_datetime(vix["Date"]).dt.date
    vix = vix[["trading_day", "VIX"]].rename(columns={"VIX": "vix"})
    ff = ff.merge(vix, on="trading_day", how="left")
    ff["dvix"] = ff["vix"].diff()

    dates  = list(ff["trading_day"])
    excess = ff["mkt_excess"].values
    dvix   = ff["dvix"].values
    n      = len(dates)

    rows = []
    for i, d in enumerate(dates):
        # Prediction returns (standard: next actual trading day)
        ret_t1 = float(excess[i + 1]) if i + 1 < n else float("nan")
        # Trading return (conservative: 2 actual trading days ahead)
        ret_t2 = float(excess[i + 2]) if i + 2 < n else float("nan")
        # 5-day prediction: compound t+1 ... t+5
        if i + 5 < n:
            ret_t5 = float(np.prod([1.0 + excess[i + j] for j in range(1, 6)]) - 1.0)
        else:
            ret_t5 = float("nan")
        rows.append({
            "trading_day": d,
            "mkt_excess":  float(excess[i]),
            "ret_t1":      ret_t1,
            "ret_t2":      ret_t2,
            "ret_t5":      ret_t5,
            "lag1": float(excess[i - 1]) if i >= 1 else float("nan"),
            "lag2": float(excess[i - 2]) if i >= 2 else float("nan"),
            "lag3": float(excess[i - 3]) if i >= 3 else float("nan"),
            "lag4": float(excess[i - 4]) if i >= 4 else float("nan"),
            "lag5": float(excess[i - 5]) if i >= 5 else float("nan"),
            "dvix":        float(dvix[i]) if not np.isnan(dvix[i]) else float("nan"),
        })

    cal = pd.DataFrame(rows).set_index("trading_day")

    # Day-of-week dummies: Monday=0 ... Friday=4; Friday is baseline
    cal.index = pd.to_datetime(cal.index)
    for name, num in [("Mon", 0), ("Tue", 1), ("Wed", 2), ("Thu", 3)]:
        cal[f"dow_{name}"] = (cal.index.dayofweek == num).astype(int)
    cal.index = cal.index.date

    return cal


_CONTROL_COLS = ["lag1", "lag2", "lag3", "lag4", "lag5", "dvix",
                 "dow_Mon", "dow_Tue", "dow_Wed", "dow_Thu"]


# ---------------------------------------------------------------------------
# PassThroughReducer (same as phase4_run.py)
# ---------------------------------------------------------------------------

class PassThroughReducer:
    n_components: int = 8
    fitted: bool = True

    def fit(self, X: np.ndarray) -> "PassThroughReducer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)


# ---------------------------------------------------------------------------
# Arm runners (identical to phase4_run.py)
# ---------------------------------------------------------------------------

def run_classical(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    svc = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    svc.fit(X_train, y_train)
    return svc.predict_proba(X_test)[:, 1]


def run_qkernel(
    X_train_angle: np.ndarray, y_train: np.ndarray,
    X_test_angle: np.ndarray, reps: int,
) -> np.ndarray:
    from pipeline.quantum.config import QuantumSettings
    from pipeline.quantum.kernel import QuantumKernel

    qs = QuantumSettings(**{**_QS_BASE, "feature_map_reps": reps})
    qk = QuantumKernel(
        n_qubits=qs.n_qubits,
        feature_map=qs.feature_map,
        reps=reps,
        shots=qs.shots,
        seed=qs.seed,
    )
    assert qk.shots is None, f"Expected shots=None, got {qk.shots}"
    K_train = qk.train_matrix(X_train_angle)
    svc = SVC(kernel="precomputed", C=qs.svm_C, probability=True, random_state=qs.seed)
    svc.fit(K_train, y_train)
    K_test = qk.cross_matrix(X_test_angle, X_train_angle)
    return svc.predict_proba(K_test)[:, 1]


def run_vqc(
    X_train_angle: np.ndarray, y_train: np.ndarray,
    X_test_angle: np.ndarray,
) -> np.ndarray:
    from pipeline.quantum.config import QuantumSettings
    from pipeline.quantum.vqc import VariationalQuantumClassifier

    qs = VariationalQuantumClassifier(
        QuantumSettings(**{**_QS_BASE, "feature_map_reps": _VQC_REPS})
    )
    qs.reducer = PassThroughReducer()
    qs.fit(X_train_angle, y_train)
    return qs.predict_proba(X_test_angle)[:, 1]


# ---------------------------------------------------------------------------
# Fold runner: produce per-sample predictions for one or more arms
# ---------------------------------------------------------------------------

def generate_predictions(
    records, splits, arms_to_run: list[str]
) -> dict[str, list[dict]]:
    """Run specified arms over all folds; return {arm: [row_dict, ...]}.

    Each row dict: {image_id, trading_day, fold, y_prob}.
    """
    arm_rows: dict[str, list[dict]] = {arm: [] for arm in arms_to_run}
    n_folds = len(splits)

    for fold_idx, (train_recs, test_recs) in enumerate(splits):
        t0 = time.time()

        X_train_full = np.stack([r.embedding for r in train_recs]).astype(np.float64)
        y_train_full = np.array([r.finbert_bin for r in train_recs])
        X_test_full  = np.stack([r.embedding for r in test_recs]).astype(np.float64)

        pipeline = build_fold_pipeline(X_train_full)
        us       = undersample_train(train_recs)
        bal_idx  = us.train_indices
        X_bal    = X_train_full[bal_idx]
        y_bal    = y_train_full[bal_idx]

        needs_pca8   = any(a in arms_to_run for a in ("classical_rbf_pca8", "qkernel_pca8", "vqc_pca8"))
        needs_pca50  = "classical_rbf_pca50" in arms_to_run
        needs_angle  = any(a in arms_to_run for a in ("qkernel_pca8", "vqc_pca8"))

        X_tr_pca8  = pipeline.transform_classical8(X_bal)      if needs_pca8  else None
        X_te_pca8  = pipeline.transform_classical8(X_test_full) if needs_pca8  else None
        X_tr_pca50 = pipeline.transform_classical50(X_bal)      if needs_pca50 else None
        X_te_pca50 = pipeline.transform_classical50(X_test_full) if needs_pca50 else None
        X_tr_angle = pipeline.transform_quantum(X_bal)           if needs_angle else None
        X_te_angle = pipeline.transform_quantum(X_test_full)     if needs_angle else None

        probs: dict[str, np.ndarray] = {}

        if "classical_rbf_pca8" in arms_to_run:
            probs["classical_rbf_pca8"] = run_classical(X_tr_pca8, y_bal, X_te_pca8)

        if "classical_rbf_pca50" in arms_to_run:
            probs["classical_rbf_pca50"] = run_classical(X_tr_pca50, y_bal, X_te_pca50)

        if "qkernel_pca8" in arms_to_run:
            probs["qkernel_pca8"] = run_qkernel(X_tr_angle, y_bal, X_te_angle, _QK_REPS)

        if "vqc_pca8" in arms_to_run:
            probs["vqc_pca8"] = run_vqc(X_tr_angle, y_bal, X_te_angle)

        for i, rec in enumerate(test_recs):
            for arm, yp in probs.items():
                arm_rows[arm].append({
                    "fold":        fold_idx,
                    "image_id":    rec.image_id,
                    "trading_day": rec.trading_day,
                    "y_prob":      float(yp[i]),
                })

        if fold_idx % 10 == 0 or fold_idx == n_folds - 1:
            elapsed = time.time() - t0
            c8_str = (f"c8={probs['classical_rbf_pca8'].mean():.3f}"
                      if "classical_rbf_pca8" in probs else "")
            print(f"  fold {fold_idx:3d}/{n_folds}  n_test={len(test_recs)}  "
                  f"{c8_str}  ({elapsed:.1f}s)")

    return arm_rows


# ---------------------------------------------------------------------------
# Newey-West predictive regression with controls
# ---------------------------------------------------------------------------

def nw_regression_with_controls(
    merged: pd.DataFrame,
    horizon: str,
    maxlags: int,
) -> dict:
    """OLS of excess market return on sentiment plus controls, Newey-West SEs.

    Controls: five lags of market excess return, change in VIX (dvix),
    day-of-week indicators (Mon-Thu; Friday is baseline).
    Uses excess returns throughout.
    """
    try:
        import statsmodels.api as sm
    except ImportError as e:
        raise ImportError("pip install statsmodels") from e

    ret_col  = "ret_t1" if horizon == "1d" else "ret_t5"
    needed   = [ret_col, "sentiment"] + _CONTROL_COLS
    sub      = merged.dropna(subset=needed).copy()

    if len(sub) < 30:
        return {}

    X = sm.add_constant(sub[["sentiment"] + _CONTROL_COLS])
    res = sm.OLS(sub[ret_col], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    return {
        "coef":  float(res.params["sentiment"]),
        "t_nw":  float(res.tvalues["sentiment"]),
        "p_nw":  float(res.pvalues["sentiment"]),
        "r2":    float(res.rsquared),
        "n_obs": int(res.nobs),
    }


# ---------------------------------------------------------------------------
# Joint regression: classical-8 + quantum-kernel in one equation
# ---------------------------------------------------------------------------

def joint_regression(
    daily_c: pd.DataFrame,
    daily_q: pd.DataFrame,
    horizon: str,
    maxlags: int,
) -> dict:
    """Joint regression with classical-8 and quantum-kernel indices.

    Returns the quantum index's incremental coefficient (and its t-stat)
    when the classical-8 index is included as a control.
    """
    try:
        import statsmodels.api as sm
    except ImportError as e:
        raise ImportError("pip install statsmodels") from e

    ret_col = "ret_t1" if horizon == "1d" else "ret_t5"

    # Align on trading_day index
    c_cols = ["sentiment", ret_col] + _CONTROL_COLS
    q_cols = ["sentiment"]

    daily_c2 = daily_c.set_index("trading_day")[c_cols].rename(
        columns={"sentiment": "sent_c"}
    )
    daily_q2 = daily_q.set_index("trading_day")[q_cols].rename(
        columns={"sentiment": "sent_q"}
    )

    merged = daily_c2.join(daily_q2, how="inner")
    needed = [ret_col, "sent_c", "sent_q"] + _CONTROL_COLS
    sub    = merged.dropna(subset=needed).copy()

    if len(sub) < 30:
        return {}

    X = sm.add_constant(sub[["sent_c", "sent_q"] + _CONTROL_COLS])
    res = sm.OLS(sub[ret_col], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    return {
        "coef_c": float(res.params["sent_c"]),
        "t_c":    float(res.tvalues["sent_c"]),
        "coef_q": float(res.params["sent_q"]),
        "t_q":    float(res.tvalues["sent_q"]),
        "p_q":    float(res.pvalues["sent_q"]),
        "n_obs":  int(res.nobs),
    }


# ---------------------------------------------------------------------------
# Daily CAPM alpha (strategy return on day t+2 ~ alpha + beta * mkt_t+2)
# ---------------------------------------------------------------------------

def daily_capm_alpha(strategy_ret: np.ndarray, market_ret: np.ndarray,
                     maxlags: int = 10) -> dict:
    """CAPM intercept from regressing daily strategy return on daily market
    excess return; annualised by x252.

    Both strategy_ret and market_ret should be aligned day-by-day returns
    (strategy uses ret_t2, market uses the same ret_t2).
    """
    try:
        import statsmodels.api as sm
    except ImportError as e:
        raise ImportError("pip install statsmodels") from e

    df = pd.DataFrame({"strat": strategy_ret, "mkt": market_ret})
    X = sm.add_constant(df[["mkt"]])
    res = sm.OLS(df["strat"], X).fit(cov_type="HAC",
                                     cov_kwds={"maxlags": maxlags})
    alpha_daily = float(res.params["const"])
    return {
        "alpha_daily":    alpha_daily,
        "alpha_ann_pct":  round(alpha_daily * 252 * 100, 2),
        "t_alpha":        round(float(res.tvalues["const"]), 3),
        "p_alpha":        round(float(res.pvalues["const"]), 4),
        "beta":           round(float(res.params["mkt"]), 4),
        "n_obs":          int(res.nobs),
        "hac_maxlags":    maxlags,
    }


# ---------------------------------------------------------------------------
# Timing strategy Sharpe ratio (daily only)
# ---------------------------------------------------------------------------

def timing_sharpe(signals: np.ndarray, returns: np.ndarray,
                  threshold: float = 0.5) -> dict:
    """Long-short timing strategy: +1 if signal > threshold, -1 otherwise.

    Uses ret_t2 (two-actual-trading-days ahead) for conservative trading
    since publication times are not available.
    Returns annualized Sharpe, mean return in basis points, N days.
    """
    direction = np.where(signals > threshold, 1.0, -1.0)
    strat_ret = direction * returns
    if strat_ret.std() < 1e-12:
        sharpe = float("nan")
    else:
        sharpe = float(np.sqrt(252) * strat_ret.mean() / strat_ret.std())
    return {
        "sharpe_annualized": round(sharpe, 4),
        "mean_ret_bps":      round(float(strat_ret.mean() * 10_000), 2),
        "n_days":            int(len(strat_ret)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--arms",
        choices=["classical", "all"],
        default="classical",
        help="'classical' = PCA-8 + PCA-50 only (fast). "
             "'all' adds quantum kernel (~2 min) and VQC (~45 min).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load per-arm prediction CSVs if they exist instead of re-running.",
    )
    parser.add_argument(
        "--min-articles",
        type=int,
        default=1,
        metavar="N",
        help="Drop test days with fewer than N articles (default: 1 = keep all).",
    )
    args = parser.parse_args()

    ARMS_CLASSICAL = ["classical_rbf_pca8", "classical_rbf_pca50"]
    ARMS_QUANTUM   = ["qkernel_pca8", "vqc_pca8"]
    ALL_ARMS = ARMS_CLASSICAL + ARMS_QUANTUM if args.arms == "all" else ARMS_CLASSICAL

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_wall = time.time()

    # ------------------------------------------------------------------
    # Complete trading-day calendar (Fama-French + VIX)
    # ------------------------------------------------------------------
    print("Building complete trading-day calendar from Fama-French data ...")
    cal = build_complete_calendar(FF_CACHE_PATH, VIX_PATH)
    print(f"  Trading days: {len(cal):,}  "
          f"({cal.index.min()} -> {cal.index.max()})")
    missing_vix = cal["dvix"].isna().sum()
    if missing_vix:
        print(f"  Warning: {missing_vix} days with missing dvix (will be dropped from regressions).")

    # ------------------------------------------------------------------
    # Load records and splits
    # ------------------------------------------------------------------
    print(f"\nLoading records from {METHODS_PATH} ...")
    records = load_records(METHODS_PATH, NPZ_PATH)
    print(f"\nGenerating rolling splits ...")
    splits, _ = rolling_walk_forward_splits(records)
    print(f"  Folds: {len(splits)}")
    assert_no_leakage(splits)

    # ------------------------------------------------------------------
    # Generate or load predictions per arm
    # ------------------------------------------------------------------
    all_pred_dfs: dict[str, pd.DataFrame] = {}

    for arm in ALL_ARMS:
        cache_path = RESULTS_DIR / f"predictions_{arm}.csv"
        if args.use_cache and cache_path.exists():
            print(f"\n[cache] Loading {cache_path.name} ...")
            df = pd.read_csv(cache_path, parse_dates=["trading_day"])
            df["trading_day"] = df["trading_day"].dt.date
            all_pred_dfs[arm] = df
            print(f"  Rows: {len(df):,}")
            continue

        print(f"\nRunning arm: {arm} ...")
        t_arm = time.time()
        raw = generate_predictions(records, splits, [arm])
        df = pd.DataFrame(raw[arm])
        df.to_csv(cache_path, index=False)
        all_pred_dfs[arm] = df
        print(f"  Done in {time.time() - t_arm:.1f}s  Rows: {len(df):,}  -> {cache_path}")

    # ------------------------------------------------------------------
    # Build daily sentiment per arm and merge with complete calendar
    # ------------------------------------------------------------------
    print("\nAggregating daily sentiment and merging with market calendar ...")
    cal_reset = cal.reset_index().rename(columns={"index": "trading_day"})
    # Ensure trading_day column is in date form
    if "trading_day" not in cal_reset.columns:
        cal_reset = cal.reset_index()
        cal_reset.columns = ["trading_day"] + list(cal.columns)

    daily_rows = []
    for arm, df in all_pred_dfs.items():
        agg = (
            df.groupby("trading_day")["y_prob"]
              .agg(sentiment="mean", n_articles="count")
              .reset_index()
        )
        agg["arm"] = arm
        daily_rows.append(agg)

    daily = pd.concat(daily_rows, ignore_index=True)
    daily["trading_day"] = pd.to_datetime(daily["trading_day"]).dt.date

    # Merge with complete calendar (left join: keep only signal days)
    cal_df = cal.copy().reset_index()
    # reset_index() names the index column after the index name; rename to 'trading_day'
    if cal_df.columns[0] != "trading_day":
        cal_df = cal_df.rename(columns={cal_df.columns[0]: "trading_day"})
    cal_df["trading_day"] = pd.to_datetime(cal_df["trading_day"]).dt.date
    daily = daily.merge(cal_df, on="trading_day", how="left")

    n_missing_ret = daily["ret_t1"].isna().sum()
    if n_missing_ret > 0:
        print(f"  Warning: {n_missing_ret} rows missing ret_t1 (calendar boundary); dropping.")
    daily = daily.dropna(subset=["ret_t1"])

    if args.min_articles > 1:
        before = len(daily)
        daily = daily[daily["n_articles"] >= args.min_articles]
        print(f"  --min-articles={args.min_articles}: dropped {before - len(daily)} rows.")

    daily.to_csv(RESULTS_DIR / "daily_sentiment.csv", index=False)
    print(f"  Daily sentiment rows: {len(daily):,}  "
          f"Unique signal days per arm: {daily.groupby('arm')['trading_day'].nunique().to_dict()}")

    # ------------------------------------------------------------------
    # Predictive regressions with Newey-West SEs and controls
    # Controls: 5 lags of market excess return, dvix, day-of-week dummies
    # ------------------------------------------------------------------
    print(f"\nRunning Newey-West predictive regressions (NW lags={_NW_LAGS}) ...")
    print("  Controls: lag1-lag5 of mkt excess return, dvix, day-of-week (Mon-Thu)")

    reg_rows = []
    arm_daily: dict[str, pd.DataFrame] = {}  # keep for joint regression

    for arm in ALL_ARMS:
        arm_df = daily[daily["arm"] == arm].sort_values("trading_day").copy()
        arm_daily[arm] = arm_df

        for horizon, ret_col in [("1d", "ret_t1"), ("5d", "ret_t5")]:
            if arm_df.dropna(subset=[ret_col, "sentiment"]).shape[0] < 30:
                print(f"  [SKIP] {arm} {horizon}: too few obs")
                continue

            res = nw_regression_with_controls(arm_df, horizon, maxlags=_NW_LAGS)
            if not res:
                continue

            stars = (
                "***" if res["p_nw"] < 0.01 else
                "**"  if res["p_nw"] < 0.05 else
                "*"   if res["p_nw"] < 0.10 else ""
            )
            print(
                f"  {arm:<26}  {horizon}  "
                f"coef={res['coef']:+.6f}  t_NW={res['t_nw']:+.3f}  "
                f"p={res['p_nw']:.4f}{stars}  R2={res['r2']:.5f}  "
                f"N={res['n_obs']}"
            )
            reg_rows.append({"arm": arm, "horizon": horizon, **res, "significance": stars})

    pd.DataFrame(reg_rows).to_csv(RESULTS_DIR / "regression_results.csv", index=False)

    # ------------------------------------------------------------------
    # Joint regression: classical-8 + quantum-kernel
    # ------------------------------------------------------------------
    joint_rows = []
    if "classical_rbf_pca8" in arm_daily and "qkernel_pca8" in arm_daily:
        print("\nRunning joint regression (classical-8 + quantum-kernel) ...")
        for horizon in ["1d", "5d"]:
            jres = joint_regression(
                arm_daily["classical_rbf_pca8"],
                arm_daily["qkernel_pca8"],
                horizon,
                maxlags=_NW_LAGS,
            )
            if jres:
                print(
                    f"  joint  {horizon}  "
                    f"coef_c={jres['coef_c']:+.6f} (t={jres['t_c']:+.3f})  "
                    f"coef_q={jres['coef_q']:+.6f} (t={jres['t_q']:+.3f})  "
                    f"p_q={jres['p_q']:.4f}  N={jres['n_obs']}"
                )
                joint_rows.append({"horizon": horizon, **jres})
    pd.DataFrame(joint_rows).to_csv(RESULTS_DIR / "joint_regression_results.csv", index=False)

    # ------------------------------------------------------------------
    # Timing strategy (daily Sharpe only, using ret_t2 — conservative)
    # ------------------------------------------------------------------
    print("\nComputing timing strategy Sharpe ratios (daily, ret_t2, threshold=0.5) ...")
    print("  ret_t2 = return 2 actual trading days ahead (conservative: no publication times)")

    sharpe_rows  = []
    timing_parts = []

    for arm in ALL_ARMS:
        arm_df = daily[daily["arm"] == arm].sort_values("trading_day").copy()
        arm_df["signal"]       = np.where(arm_df["sentiment"] > 0.5, 1, -1)
        arm_df["strategy_ret"] = arm_df["signal"] * arm_df["ret_t2"]

        timing_parts.append(arm_df[[
            "trading_day", "arm", "sentiment", "signal",
            "ret_t1", "ret_t2", "ret_t5", "strategy_ret", "n_articles"
        ]])

        sub = arm_df.dropna(subset=["ret_t2", "sentiment"])
        if len(sub) < 30:
            continue

        result = timing_sharpe(sub["sentiment"].values, sub["ret_t2"].values, threshold=0.5)
        print(
            f"  {arm:<26}  "
            f"Sharpe={result['sharpe_annualized']:+.4f}  "
            f"mean={result['mean_ret_bps']:+.1f}bps  "
            f"N={result['n_days']}"
        )
        sharpe_rows.append({"arm": arm, **result})

    timing_df = pd.concat(timing_parts, ignore_index=True)
    timing_df.to_csv(RESULTS_DIR / "timing_strategy.csv", index=False)
    pd.DataFrame(sharpe_rows).to_csv(RESULTS_DIR / "sharpe_by_arm.csv", index=False)

    # ------------------------------------------------------------------
    # Daily CAPM alpha (strategy_ret = signal * ret_t2 ~ alpha + beta * ret_t2)
    # Annualised by x252.
    # ------------------------------------------------------------------
    print("\nComputing daily CAPM alpha (strategy_ret ~ alpha + beta * ret_t2) ...")
    capm_rows = []
    for arm in ALL_ARMS:
        arm_df = timing_df[timing_df["arm"] == arm].dropna(
            subset=["strategy_ret", "ret_t2"]
        ).sort_values("trading_day")
        if len(arm_df) < 30:
            print(f"  [SKIP] {arm}: only {len(arm_df)} obs")
            continue
        result = daily_capm_alpha(
            arm_df["strategy_ret"].values,
            arm_df["ret_t2"].values,
            maxlags=10,
        )
        print(
            f"  {arm:<26}  alpha={result['alpha_ann_pct']:+.2f}%  "
            f"t={result['t_alpha']:+.3f}  p={result['p_alpha']:.4f}  "
            f"beta={result['beta']:.4f}  N={result['n_obs']}"
        )
        capm_rows.append({"arm": arm, **result})
    pd.DataFrame(capm_rows).to_csv(RESULTS_DIR / "capm_alpha_daily.csv", index=False)

    # ------------------------------------------------------------------
    # Summary table (Table 5)
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("SUMMARY: Table 5 — Photo Positivity and Market Returns")
    print("=" * 80)
    reg_df    = pd.DataFrame(reg_rows)
    shr_df    = pd.DataFrame(sharpe_rows)
    capm_df   = pd.DataFrame(capm_rows)

    arm_labels = {
        "classical_rbf_pca8":  "Classical SVM, 8 comp.",
        "qkernel_pca8":        "Quantum kernel",
        "vqc_pca8":            "VQC",
        "classical_rbf_pca50": "Classical SVM, 50 comp.",
    }
    for arm in ALL_ARMS:
        label = arm_labels.get(arm, arm)
        r1d = reg_df[(reg_df["arm"] == arm) & (reg_df["horizon"] == "1d")]
        r5d = reg_df[(reg_df["arm"] == arm) & (reg_df["horizon"] == "5d")]
        shr = shr_df[shr_df["arm"] == arm]
        cap = capm_df[capm_df["arm"] == arm]

        coef1  = f"{r1d['coef'].values[0]:+.4f} ({r1d['t_nw'].values[0]:+.2f})" if len(r1d) else "—"
        coef5  = f"{r5d['coef'].values[0]:+.4f} ({r5d['t_nw'].values[0]:+.2f})" if len(r5d) else "—"
        sharpe = f"{shr['sharpe_annualized'].values[0]:+.2f}" if len(shr) else "—"
        alpha  = (f"{cap['alpha_ann_pct'].values[0]:+.1f}% "
                  f"({cap['t_alpha'].values[0]:+.2f})" if len(cap) else "—")
        n1d    = int(r1d["n_obs"].values[0]) if len(r1d) else 0
        n5d    = int(r5d["n_obs"].values[0]) if len(r5d) else 0
        print(f"  {label:<28}  1d: {coef1:<22}  5d: {coef5:<22}  Sharpe: {sharpe:<8}  alpha: {alpha}  N(1d)={n1d}  N(5d)={n5d}")

    if joint_rows:
        print("\n  Joint (classical-8 + quantum-kernel):")
        for jr in joint_rows:
            print(f"    {jr['horizon']}  Q coef={jr['coef_q']:+.6f} t={jr['t_q']:+.3f} p={jr['p_q']:.4f}  N={jr['n_obs']}")

    print(f"\nTotal wall time: {(time.time() - t_wall):.1f}s")
    print(f"Outputs written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

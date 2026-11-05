"""Single source of truth for model features (feature_version 2).

Training and live inference MUST both call these functions. The Nov-2026 audit
(results/model_audit_2026-11.md) found that two separate implementations -- a
DuckDB SQL path for training and a pandas path for live -- disagreed enough
that ~3 of the 10 names bought live would not have been picked by the model as
trained. Version 2 fixes that by construction and also fixes the feature
definitions themselves:

* scale-free, point-in-time features only (returns / ratios, never price levels):
  volatility of *returns* instead of the standard deviation of price, PPO
  (MACD / price) instead of raw MACD;
* ``log_dollar_volume_20d`` (PIT liquidity/size) replaces ``log_market_cap``,
  which came from a static snapshot taken after the training window (leak);
* no features that are constant within a date (macro, FX, calendar): after
  per-date z-scoring they carry no cross-sectional information, and trees used
  their tiny per-date differences as date identifiers;
* one eligibility rule (price >= $1 and >= MIN_HISTORY bars) applied before every
  cross-sectional statistic, in both paths;
* a robust per-date z-score (winsorized, float64, constant-column guard).

Input panel: one row per (ticker, Date) with ``company_close`` and
``company_volume`` (``Open``/``High``/``Low`` optional), computed on the FULL,
unfiltered history so rolling windows are warm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_VERSION = 2
MIN_PRICE = 1.0
MIN_HISTORY = 80  # bars; covers return_50d and lets the 26-span EMA converge

FEATURE_COLUMNS = [
    "daily_return", "return_5d", "return_10d", "return_20d", "return_50d",
    "price_vs_ma5", "price_vs_ma20", "stoch_20d",
    "volume_ratio", "volume_zscore_20d",
    "ret_vol_5d", "ret_vol_10d", "ret_vol_20d", "vol_adjusted_return",
    "rsi_14", "ppo", "ppo_signal", "bb_width", "log_dollar_volume_20d",
    "sector_daily_return", "sector_return_5d",
    "rank_daily_return", "rank_return_5d", "rank_return_20d",
    "rank_volatility", "rank_volume_ratio",
    "daily_return_lag_1", "daily_return_lag_2", "daily_return_lag_5",
    "volume_ratio_lag_1", "volume_ratio_lag_2", "volume_ratio_lag_5",
]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b
    return out.where(np.isfinite(out) & (b != 0))


def compute_features(panel: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute feature_version-2 features for every (ticker, Date) row.

    Returns the panel (sorted by ticker, Date) with ``FEATURE_COLUMNS``,
    ``Sector``, ``n_bars``, ``eligible`` and ``return_volatility_20d`` (an alias
    of ``ret_vol_20d`` used for position sizing). Cross-sectional features are
    computed over eligible rows only and are NaN elsewhere.
    """
    need = {"ticker", "Date", "company_close", "company_volume"}
    missing = need - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    df = panel.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = (df.drop_duplicates(["ticker", "Date"], keep="last")
            .sort_values(["ticker", "Date"]).reset_index(drop=True))
    c = df["company_close"].astype("float64")
    v = df["company_volume"].astype("float64")
    df["company_close"], df["company_volume"] = c, v

    g = df.groupby("ticker", sort=False)

    def roll(series: pd.Series, n: int, fn: str) -> pd.Series:
        r = series.groupby(df["ticker"], sort=False).rolling(n, min_periods=n)
        return getattr(r, fn)().reset_index(level=0, drop=True)

    def lag(series: pd.Series, k: int) -> pd.Series:
        return series.groupby(df["ticker"], sort=False).shift(k)

    def ewm(series: pd.Series, **kw) -> pd.Series:
        return series.groupby(df["ticker"], sort=False).transform(lambda s: s.ewm(adjust=False, **kw).mean())

    # Returns (simple, over bars).
    df["daily_return"] = _safe_div(c, lag(c, 1)) - 1.0
    for k in (5, 10, 20, 50):
        df[f"return_{k}d"] = _safe_div(c, lag(c, k)) - 1.0

    # Trend / range position.
    df["price_vs_ma5"] = _safe_div(c, roll(c, 5, "mean")) - 1.0
    df["price_vs_ma20"] = _safe_div(c, roll(c, 20, "mean")) - 1.0
    lo, hi = roll(c, 20, "min"), roll(c, 20, "max")
    rng = hi - lo
    df["stoch_20d"] = ((c - lo) / rng).where(rng > 0, 0.5)

    # Volume.
    df["volume_ratio"] = _safe_div(v, roll(v, 5, "mean"))
    vm, vs = roll(v, 20, "mean"), roll(v, 20, "std")
    df["volume_zscore_20d"] = ((v - vm) / vs).where(vs > 0, 0.0)

    # Volatility of returns (not of price) and risk-adjusted return.
    r1 = df["daily_return"]
    for k in (5, 10, 20):
        df[f"ret_vol_{k}d"] = roll(r1, k, "std")
    df["vol_adjusted_return"] = (r1 / df["ret_vol_10d"]).where(df["ret_vol_10d"] > 0)
    df["return_volatility_20d"] = df["ret_vol_20d"]  # sizing alias

    # Oscillators (scale-free).
    delta = c.groupby(df["ticker"], sort=False).diff()
    gain = ewm(delta.clip(lower=0), alpha=1 / 14, min_periods=14)
    loss = ewm(-delta.clip(upper=0), alpha=1 / 14, min_periods=14)
    df["rsi_14"] = (100 - 100 / (1 + gain / loss)).where(loss > 0, 100.0).where(gain.notna())
    ema12, ema26 = ewm(c, span=12), ewm(c, span=26)
    df["ppo"] = _safe_div(ema12 - ema26, ema26)
    df["ppo_signal"] = ewm(df["ppo"], span=9)
    df["bb_width"] = _safe_div(4.0 * roll(c, 20, "std"), roll(c, 20, "mean"))
    dv = roll(c * v, 20, "mean")
    df["log_dollar_volume_20d"] = np.log(dv.where(dv > 0))

    # Lags.
    for k in (1, 2, 5):
        df[f"daily_return_lag_{k}"] = lag(r1, k)
        df[f"volume_ratio_lag_{k}"] = lag(df["volume_ratio"], k)

    # Eligibility: one rule for training rows, live scoring and every
    # cross-sectional statistic below.
    df["n_bars"] = g.cumcount() + 1
    df["eligible"] = (c >= MIN_PRICE) & (df["n_bars"] >= MIN_HISTORY) & c.notna()

    # Sector.
    if meta is not None and "Sector" in meta.columns:
        sec = meta.drop_duplicates("ticker").set_index("ticker")["Sector"]
        df["Sector"] = df["ticker"].map(sec)
    elif "Sector" not in df.columns:
        df["Sector"] = np.nan
    df["Sector"] = df["Sector"].fillna("Unknown")

    # Cross-sectional features over eligible rows only (NaN-skipping means).
    cs_cols = ["sector_daily_return", "sector_return_5d", "rank_daily_return",
               "rank_return_5d", "rank_return_20d", "rank_volatility", "rank_volume_ratio"]
    for col in cs_cols:
        df[col] = np.nan
    el = df.loc[df["eligible"]]
    if len(el):
        by_sd = el.groupby(["Sector", "Date"])
        df.loc[el.index, "sector_daily_return"] = by_sd["daily_return"].transform("mean")
        df.loc[el.index, "sector_return_5d"] = by_sd["return_5d"].transform("mean")
        by_d = el.groupby("Date")
        for src, dst in (("daily_return", "rank_daily_return"), ("return_5d", "rank_return_5d"),
                         ("return_20d", "rank_return_20d"), ("ret_vol_10d", "rank_volatility"),
                         ("volume_ratio", "rank_volume_ratio")):
            df.loc[el.index, dst] = by_d[src].rank(pct=True)

    return df


def decision_rows(features: pd.DataFrame, decision_date: str | None = None) -> pd.DataFrame:
    """Eligible rows on ``decision_date`` (default: the latest date in the panel).

    Tickers without a bar on the decision date are excluded, so stale names (last
    bar weeks/months ago) can never be ranked or bought.
    """
    d = decision_date or features["Date"].max()
    out = features[(features["Date"] == d) & features["eligible"]]
    return out.reset_index(drop=True)


def cross_sectional_zscore(df: pd.DataFrame, cols: list, date_col: str = "Date",
                           winsor=(0.01, 0.99), clip: float = 6.0) -> pd.DataFrame:
    """Robust per-date z-score shared by training and live.

    Winsorizes each column at the per-date ``winsor`` quantiles, standardizes in
    float64, returns 0 where a column is (numerically) constant within the date
    or the value is missing, and clips to +/- ``clip``.
    """
    X = df[cols].astype("float64")
    dates = df[date_col].to_numpy()
    # Inward (non-interpolated) quantiles: with linear interpolation a single
    # extreme value drags its own cap far out and escapes winsorization.
    lo = X.groupby(dates).quantile(winsor[0], interpolation="higher").reindex(dates).to_numpy()
    hi = X.groupby(dates).quantile(winsor[1], interpolation="lower").reindex(dates).to_numpy()
    Xw = pd.DataFrame(np.clip(X.to_numpy(), lo, hi), columns=cols, index=X.index)
    grp = Xw.groupby(dates)
    mu = grp.transform("mean")
    sd = grp.transform("std", ddof=0)
    tiny = sd < 1e-9 * (1.0 + mu.abs())
    Z = (Xw - mu) / sd.mask(tiny)
    Z = Z.mask(tiny, 0.0).fillna(0.0).clip(-clip, clip)
    return Z.astype("float32")

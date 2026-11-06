"""Signal and performance statistics for the v3 walk-forward (WS3).

Self-contained re-implementations of the helpers in
``scripts/backtests/candidate_significance.py`` (that module pulls in the v1
backtest engine and network code at import): stationary block bootstrap,
probabilistic / deflated Sharpe ratio, plus Newey-West t-stats and the IC
family used by the promotion gate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PPY = 252


# --------------------------------------------------------------------------- #
# Signal metrics
# --------------------------------------------------------------------------- #
def newey_west_t(x, lags: int) -> float:
    """t-stat of the mean of ``x`` with a Newey-West (Bartlett) HAC variance."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, min(lags, n - 1) + 1):
        s += 2.0 * (1.0 - L / (lags + 1.0)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else float("nan")


def _spearman(a, b) -> float:
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def per_date_signal(df: pd.DataFrame, pred: str = "pred", ret: str = "fwd_ret_30",
                    sector: str = "Sector", min_sector_names: int = 20) -> pd.DataFrame:
    """Per date: IC (Spearman), top-minus-bottom decile mean return, and
    sector-neutral IC (within-sector Spearman averaged over sectors with at
    least ``min_sector_names`` names; 'Unknown' excluded)."""
    rows = []
    d = df[df[ret].notna() & df[pred].notna()]
    for date, g in d.groupby("Date", sort=True):
        if len(g) < 30:
            continue
        ic = _spearman(g[pred], g[ret])
        q = pd.qcut(g[pred].rank(method="first"), 10, labels=False)
        spread = g.loc[q == 9, ret].mean() - g.loc[q == 0, ret].mean()
        sics = []
        if sector in g.columns:
            for s, gs in g.groupby(sector):
                if s == "Unknown" or len(gs) < min_sector_names:
                    continue
                v = _spearman(gs[pred], gs[ret])
                if np.isfinite(v):
                    sics.append(v)
        rows.append({"Date": date, "ic": ic, "decile_spread": spread,
                     "sector_neutral_ic": np.mean(sics) if sics else np.nan, "n": len(g)})
    return pd.DataFrame(rows)


def signal_summary(per_date: pd.DataFrame, nw_lags: int = 6) -> pd.DataFrame:
    """Per-year and overall mean IC, NW t, IC>0 share, decile spread, SN-IC."""
    p = per_date.assign(year=per_date["Date"].str[:4])

    def agg(g):
        return pd.Series({"n_dates": len(g), "mean_ic": g["ic"].mean(),
                          "nw_t": newey_west_t(g["ic"], nw_lags),
                          "pct_ic_pos": (g["ic"] > 0).mean(),
                          "decile_spread": g["decile_spread"].mean(),
                          "sector_neutral_ic": g["sector_neutral_ic"].mean(),
                          "avg_names": g["n"].mean()})
    by = p.groupby("year").apply(agg, include_groups=False)
    by.loc["all"] = agg(p)
    return by


# --------------------------------------------------------------------------- #
# Performance metrics
# --------------------------------------------------------------------------- #
def max_drawdown(r) -> float:
    nav = np.cumprod(1.0 + np.asarray(r, dtype=float))
    peak = np.maximum.accumulate(np.r_[1.0, nav])[1:]
    return float((nav / peak - 1.0).min()) if len(nav) else float("nan")


def sharpe(r, ppy=PPY) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1) if len(r) > 1 else np.nan
    return float(np.sqrt(ppy) * r.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan")


def info_ratio(r, b, ppy=PPY) -> float:
    return sharpe(np.asarray(r, dtype=float) - np.asarray(b, dtype=float), ppy)


def cagr(r, ppy=PPY) -> float:
    r = np.asarray(r, dtype=float)
    if not len(r):
        return float("nan")
    return float(np.prod(1.0 + r) ** (ppy / len(r)) - 1.0)


def perf_metrics(r: pd.Series, spy: pd.Series, ew: pd.Series, turnover: pd.Series | None = None) -> dict:
    """``r``, ``spy``, ``ew``: daily returns on the same index ('YYYY-MM-DD')."""
    idx = r.index
    s, e = spy.reindex(idx).fillna(0.0), ew.reindex(idx).fillna(0.0)
    m = (1 + r).groupby(idx.str[:7]).prod() - 1
    me = (1 + e).groupby(idx.str[:7]).prod() - 1
    out = {"cagr": cagr(r), "vol": float(r.std(ddof=1) * np.sqrt(PPY)), "sharpe": sharpe(r),
           "max_dd": max_drawdown(r), "ir_spy": info_ratio(r, s), "ir_ew": info_ratio(r, e),
           "hit_rate_m": float((m > 0).mean()), "hit_vs_ew_m": float((m > me).mean()),
           "n_days": int(len(r))}
    if turnover is not None and len(turnover):
        t = turnover[(turnover.index >= idx.min()) & (turnover.index <= idx.max())]
        out["turnover_m"] = float(t.mean()) if len(t) else float("nan")  # one-way per rebalance
    return out


def perf_by_year(r, spy, ew, turnover=None) -> pd.DataFrame:
    rows = {}
    for y, g in r.groupby(r.index.str[:4]):
        rows[y] = perf_metrics(g, spy, ew, turnover)
    rows["all"] = perf_metrics(r, spy, ew, turnover)
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Bootstrap and deflated Sharpe
# --------------------------------------------------------------------------- #
def stationary_bootstrap_indices(n: int, expected_block: float, rng) -> np.ndarray:
    """One Politis-Romano stationary-bootstrap resample (vectorized)."""
    p = 1.0 / max(expected_block, 1.0)
    new = rng.random(n) < p
    new[0] = True
    starts = rng.integers(0, n, n)
    block = np.cumsum(new) - 1
    first_t = np.flatnonzero(new)
    offset = np.arange(n) - first_t[block]
    return (starts[first_t][block] + offset) % n


def bootstrap_ci(series: dict, fn, n_resamples: int = 2000, expected_block: float = 21.0,
                 seed: int = 12345, alpha: float = 0.05) -> dict:
    """CI of ``fn(*resampled series)`` with a shared index draw per resample
    (paired). ``series``: name -> aligned np arrays; returns (point, lo, hi)."""
    arrs = [np.asarray(v, dtype=float) for v in series.values()]
    n = len(arrs[0])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_resamples):
        i = stationary_bootstrap_indices(n, expected_block, rng)
        vals.append(fn(*[a[i] for a in arrs]))
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(fn(*arrs)), "lo": float(lo), "hi": float(hi)}


def probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=0.0) -> float:
    """PSR with per-period SRs and non-excess kurtosis (Bailey & Lopez de Prado)."""
    if n < 2 or not np.isfinite(sr):
        return float("nan")
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)))


def expected_max_sharpe(n_trials: int, var_trials_sr: float) -> float:
    if n_trials < 2 or var_trials_sr <= 0:
        return 0.0
    g = 0.5772156649
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_trials_sr) * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe(r, trial_srs_pp, n_trials: int | None = None) -> dict:
    """DSR of daily returns ``r`` given the per-period Sharpe ratios of ALL
    trials (configurations) evaluated, including this one. ``n_trials`` overrides
    the count when some trials' Sharpe ratios are not available (their dispersion
    is then estimated from the ones given)."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    sd = r.std(ddof=1)
    sr = r.mean() / sd if sd > 0 else float("nan")
    sk = float(stats.skew(r, bias=False))
    ku = float(stats.kurtosis(r, fisher=False, bias=False))
    trial_srs_pp = np.asarray([x for x in trial_srs_pp if np.isfinite(x)], dtype=float)
    n_trials = int(n_trials) if n_trials is not None else len(trial_srs_pp)
    var_t = float(trial_srs_pp.var(ddof=1)) if n_trials > 1 else 0.0
    sr_star = expected_max_sharpe(n_trials, var_t)
    return {"sr_pp": float(sr), "sr_ann": float(sr * np.sqrt(PPY)), "n": n, "skew": sk, "kurt": ku,
            "n_trials": n_trials, "var_trial_sr_pp": var_t, "sr_star_pp": sr_star,
            "sr_star_ann": float(sr_star * np.sqrt(PPY)),
            "psr0": probabilistic_sharpe_ratio(sr, n, sk, ku, 0.0),
            "dsr": probabilistic_sharpe_ratio(sr, n, sk, ku, sr_star)}

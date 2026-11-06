"""v3 modelling dataset (WS3): PIT features, eligibility, sectors and labels.

One code path for training and live scoring
--------------------------------------------
``compute_v3_features(panel, meta)`` computes, for a self-contained price panel,
exactly what the offline builder (``build_dataset``) writes to disk; the live
bot (feature_version 3) calls it on freshly fetched bars. The offline builder
runs the same building blocks in chunks so the ~15M-row daily panel never has
to sit in memory as several float64 copies:

  stage A (per ticker chunk, full history per ticker)
      ``core.time_series_features``  -- per-ticker features (PIT by construction)
      ``universe.eligibility_details`` -- v3 rule: raw close >= $3, 20-session
                                          ADV >= $2M, >= 120 bars
      ``forward_returns``            -- labels (see below)
  proxy sectors (whole panel, monthly)
      ``proxy_sector_assignments``
  stage B (per calendar year, whole cross-section)
      effective sector, ``core.add_cross_sectional_features`` over the v3
      eligible population, per-date rank targets.

Sectors
-------
Without EDGAR, a sector label exists only for the ~1.5k names of the legacy
DuckDB metadata (``bot.fetch_company_metadata``; 352 of its 1,890 rows are
'None'). Every other ticker gets a **proxy sector**: the known sector whose
market-residual equal-weight daily return it correlates with most over the 120
sessions ending at the last session of the *previous* calendar month (>= 60
return observations required; otherwise 'Unknown'). The assignment is
recomputed monthly, uses only past data, and is identical offline and live.
``sector_source`` records meta / proxy / none.

Labels
------
``fwd_ret_H`` = close(t+H sessions) / close(t) - 1 on adjusted closes, where
"sessions" are market sessions (the union calendar of the store), and
close(t+H) is the ticker's last close at or before session t+H: a stock that
stops trading inside the horizon (delisting, acquisition, bankruptcy) gets the
return to its last available close -- no survivorship in the labels. NaN only
when session t+H is beyond the end of the data. Returns are winsorized to
``LABEL_CLIP`` (adjustment-error jumps remain in the store). Targets are
per-date percentile ranks over eligible rows: ``target_raw_H`` (raw return)
and ``target_sn_H`` (return minus its effective-sector mean on that date).
"""
from __future__ import annotations

import gc
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from DataAnalysisPipeline2.features import core

from . import universe as uni
from ._alpaca import data_dir

FEATURE_VERSION = 3
FEATURE_COLUMNS = list(core.FEATURE_COLUMNS)
HORIZONS = (21, 30)
LABEL_CLIP = (-0.95, 3.0)
ELIGIBILITY_RULE = {"min_price": 3.0, "min_adv_usd": 2e6, "min_bars": 120}
PROXY_WINDOW = 120
PROXY_MIN_OBS = 60
UNKNOWN = "Unknown"
DECISION_START = "2017-01-01"

_TS_KEEP = (["ticker", "Date", "company_close", "close_raw", "return_volatility_20d", "n_bars",
             "adv_usd", "eligible"] + core.TS_COLUMNS)


def v3_dir(*parts: str) -> str:
    return data_dir("v3model", *parts)


# --------------------------------------------------------------------------- #
# Calendar helpers
# --------------------------------------------------------------------------- #
def session_calendar(dates) -> np.ndarray:
    return np.array(sorted(pd.unique(pd.Series(dates).astype(str))))


def weekly_decision_dates(calendar, start: str = DECISION_START, end: str | None = None) -> list[str]:
    """Last session of each (Mon-Sun) week, from ``start``."""
    cal = pd.to_datetime(pd.Series(calendar))
    wk = cal.dt.to_period("W-SUN")
    last = cal.groupby(wk).max().dt.strftime("%Y-%m-%d").tolist()
    return [d for d in last if d >= start and (end is None or d <= end)]


def first_sessions_of_month(calendar) -> list[str]:
    cal = pd.Series(sorted(calendar))
    return cal.groupby(cal.str[:7]).min().tolist()


def _month_window(calendar: np.ndarray, month: str, window: int = PROXY_WINDOW) -> np.ndarray:
    """The ``window`` sessions ending at the last session before ``month``
    ('YYYY-MM') starts."""
    prior = calendar[calendar < f"{month}-01"]
    return prior[-window:]


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def forward_returns(df: pd.DataFrame, calendar, horizons=HORIZONS, clip=LABEL_CLIP) -> pd.DataFrame:
    """Forward H-session returns for each row of ``df`` (ticker, Date,
    company_close), aligned to ``df.index``. See the module docstring for the
    delisting rule."""
    cal = np.asarray(calendar)
    n = len(cal)
    k = np.searchsorted(cal, df["Date"].to_numpy().astype(str))
    if (k >= n).any() or (cal[np.minimum(k, n - 1)] != df["Date"].to_numpy().astype(str)).any():
        raise ValueError("dates missing from the session calendar")
    code = pd.factorize(df["ticker"])[0].astype(np.int64)
    span = n + max(horizons) + 1
    key = code * span + k
    order = np.argsort(key, kind="mergesort")
    skey = key[order]
    c = df["company_close"].to_numpy(dtype=float)[order]
    out = pd.DataFrame(index=df.index)
    for h in horizons:
        j = np.searchsorted(skey, skey + h, side="right") - 1   # own last bar at or before t+H
        r = c[j] / c - 1.0
        r = np.where((k[order] + h) < n, r, np.nan)              # horizon beyond the data
        col = np.empty(len(df))
        col[order] = r
        out[f"fwd_ret_{h}"] = np.clip(col, *clip)
    return out


# --------------------------------------------------------------------------- #
# Sectors
# --------------------------------------------------------------------------- #
def known_sector_map(meta: pd.DataFrame | None) -> pd.Series:
    """ticker -> sector for names with a real label (None/'None'/''/'Unknown' dropped)."""
    if meta is None or "Sector" not in meta.columns or not len(meta):
        return pd.Series(dtype=object)
    m = meta.drop_duplicates("ticker").set_index("ticker")["Sector"]
    bad = m.isna() | m.astype(str).str.strip().str.lower().isin({"", "none", "nan", "unknown", "n/a"})
    return m[~bad].astype(str)


def proxy_sectors_for_window(rets: pd.DataFrame, ref_members: dict, targets) -> pd.Series:
    """Assign ``targets`` to the known sector they co-move with most.

    ``rets``: dates x tickers daily returns over the window (NaN = no bar).
    ``ref_members``: sector -> list of reference tickers (known sector,
    eligible at the window end). Correlations are on market-residual returns
    (market = mean over all reference members), pairwise complete, and need
    >= PROXY_MIN_OBS observations. Returns ticker -> sector (or UNKNOWN)."""
    targets = [t for t in targets if t in rets.columns]
    if not targets or not ref_members:
        return pd.Series(UNKNOWN, index=targets, dtype=object)
    secs = sorted(s for s, m in ref_members.items() if len(m))
    all_ref = sorted({t for s in secs for t in ref_members[s]})
    mkt = rets[all_ref].mean(axis=1)
    S = pd.concat({s: rets[ref_members[s]].mean(axis=1) for s in secs}, axis=1).sub(mkt, axis=0)
    S = S.fillna(0.0).to_numpy(dtype=float)                     # dates x sectors
    X = rets[targets].sub(mkt, axis=0).to_numpy(dtype=float)    # dates x targets
    M = np.isfinite(X)
    nobs = M.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = np.where(M, X, 0.0).sum(axis=0) / np.maximum(nobs, 1)
        Xc = np.where(M, X - xm, 0.0)
        Mf = M.astype(float)
        s1 = Mf.T @ S                        # targets x sectors: sum of s over valid
        s2 = Mf.T @ (S ** 2)
        n = np.maximum(nobs, 1)[:, None]
        var_s = s2 / n - (s1 / n) ** 2
        cov = (Xc.T @ S) / n                 # sum x_c * s (x_c sums to 0 over valid)
        var_x = (Xc ** 2).sum(axis=0)[:, None] / n
        corr = cov / np.sqrt(var_x * var_s)
    corr = np.where(np.isfinite(corr), corr, -np.inf)
    best = np.array(secs, dtype=object)[corr.argmax(axis=1)]
    ok = (nobs >= PROXY_MIN_OBS) & np.isfinite(corr.max(axis=1))
    return pd.Series(np.where(ok, best, UNKNOWN), index=targets, dtype=object)


def proxy_sector_assignments(long: pd.DataFrame, known: pd.Series, calendar, months=None) -> pd.DataFrame:
    """Monthly proxy-sector table [month, ticker, proxy_sector].

    ``long``: ticker, Date, daily_return, eligible for (at least) the windows
    needed. For month M only data in the 120 sessions ending at the last
    session before M is used. Tickers present in the window without a known
    sector are assigned."""
    cal = np.asarray(calendar)
    dates = long["Date"].to_numpy().astype(str)
    if months is None:
        months = sorted(pd.unique(pd.Series(dates).str[:7]))
    order = np.argsort(dates, kind="mergesort")
    sdates = dates[order]
    known_set = set(known.index)
    rows = []
    for m in months:
        win = _month_window(cal, m)
        if len(win) == 0:
            continue
        lo, hi = np.searchsorted(sdates, win[0], "left"), np.searchsorted(sdates, win[-1], "right")
        if hi <= lo:
            continue
        w = long.iloc[order[lo:hi]][["ticker", "Date", "daily_return", "eligible"]]
        w = w.assign(ticker=w["ticker"].astype(str))
        rets = w.pivot(index="Date", columns="ticker", values="daily_return").reindex(win).astype(float)
        end_el = w.loc[(w["Date"] == win[-1]) & w["eligible"].astype(bool), "ticker"]
        ref = end_el[end_el.isin(known_set)]
        ref_members = {s: g.tolist() for s, g in ref.groupby(ref.map(known))}
        targets = [t for t in rets.columns if t not in known_set]
        a = proxy_sectors_for_window(rets, ref_members, targets)
        rows.append(pd.DataFrame({"month": m, "ticker": a.index.astype(str), "proxy_sector": a.values}))
    if not rows:
        return pd.DataFrame(columns=["month", "ticker", "proxy_sector"])
    return pd.concat(rows, ignore_index=True)


def effective_sector(df: pd.DataFrame, known: pd.Series, assignments: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(Sector, sector_source) for rows of ``df`` (ticker, Date)."""
    sec = df["ticker"].map(known)
    src = pd.Series(np.where(sec.notna(), "meta", "none"), index=df.index, dtype=object)
    miss = sec.isna()
    if miss.any() and assignments is not None and len(assignments):
        key = df.loc[miss, ["ticker", "Date"]].assign(month=df.loc[miss, "Date"].str[:7])
        mp = key.merge(assignments, on=["month", "ticker"], how="left")["proxy_sector"].to_numpy()
        mp = pd.Series(mp, index=key.index)
        good = mp.notna() & (mp != UNKNOWN)
        sec.loc[good[good].index] = mp[good]
        src.loc[good[good].index] = "proxy"
    return sec.fillna(UNKNOWN).astype(str), src


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def add_targets(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Per-date percentile-rank targets over eligible rows with a label."""
    for h in horizons:
        y = df[f"fwd_ret_{h}"].astype(float)
        ok = df["eligible"].astype(bool) & y.notna()
        sub = df.loc[ok, ["Date", "Sector"]].assign(y=y[ok])
        df[f"target_raw_{h}"] = np.nan
        df[f"target_sn_{h}"] = np.nan
        if len(sub):
            df.loc[ok, f"target_raw_{h}"] = sub.groupby("Date")["y"].rank(pct=True)
            dm = sub["y"] - sub.groupby(["Date", "Sector"])["y"].transform("mean")
            df.loc[ok, f"target_sn_{h}"] = dm.groupby(sub["Date"]).rank(pct=True)
    return df


# --------------------------------------------------------------------------- #
# Self-contained computation (live path and tests)
# --------------------------------------------------------------------------- #
def stage_a(panel: pd.DataFrame, calendar=None, labels: bool = True) -> pd.DataFrame:
    """Per-ticker part: time-series features + v3 eligibility (+ labels)."""
    ts = core.time_series_features(panel)
    el = uni.eligibility_details(ts, **ELIGIBILITY_RULE)
    ts["adv_usd"] = el["adv_usd"].to_numpy()
    ts["eligible"] = el["eligible"].astype(bool).to_numpy()
    if "close_raw" not in ts.columns:
        ts["close_raw"] = np.nan
    keep = [c for c in _TS_KEEP if c in ts.columns]
    out = ts[keep].copy()
    if labels:
        cal = session_calendar(panel["Date"]) if calendar is None else calendar
        out = pd.concat([out, forward_returns(out, cal)], axis=1)
    return out


def stage_b(df: pd.DataFrame, known: pd.Series, assignments: pd.DataFrame,
            targets: bool = True) -> pd.DataFrame:
    """Cross-sectional part (needs the full cross-section of each date)."""
    df["Sector"], df["sector_source"] = effective_sector(df, known, assignments)
    core.add_cross_sectional_features(df, "eligible")
    if targets and f"fwd_ret_{HORIZONS[0]}" in df.columns:
        add_targets(df)
    return df


def compute_v3_features(panel: pd.DataFrame, meta: pd.DataFrame | None, labels: bool = False,
                        decision_date: str | None = None) -> pd.DataFrame:
    """All v3 features for a self-contained panel (every ticker's history in
    ``panel``). ``panel`` needs ticker, Date, company_close, company_volume and
    ``close_raw`` (raw printed close for the $3 rule; falls back to the
    adjusted close if absent). With ``decision_date`` only the months needed
    for that date's proxy sectors are computed."""
    cal = session_calendar(panel["Date"])
    a = stage_a(panel, cal, labels=labels)
    known = known_sector_map(meta)
    months = [decision_date[:7]] if decision_date else None
    assign = proxy_sector_assignments(a[["ticker", "Date", "daily_return", "eligible"]], known, cal, months)
    return stage_b(a, known, assign, targets=labels)


def decision_rows(features: pd.DataFrame, decision_date: str | None = None) -> pd.DataFrame:
    """Eligible rows with a bar on ``decision_date`` (default: the last date)."""
    d = decision_date or features["Date"].max()
    return features[(features["Date"] == d) & features["eligible"].astype(bool)].reset_index(drop=True)


def zscore(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """The v3 model input: ``core.cross_sectional_zscore`` per date over the
    rows given (callers pass eligible rows only)."""
    return core.cross_sectional_zscore(df, list(cols or FEATURE_COLUMNS), date_col="Date")


# --------------------------------------------------------------------------- #
# Offline, chunked builder
# --------------------------------------------------------------------------- #
def _f32(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    return df


def load_sector_meta(refresh: bool = False) -> pd.DataFrame:
    """Legacy sector metadata (``bot.fetch_company_metadata``), cached."""
    p = os.path.join(v3_dir(), "sector_meta.parquet")
    if os.path.exists(p) and not refresh:
        return pd.read_parquet(p)
    from DataAnalysisPipeline2.trading_agent.bot import fetch_company_metadata
    m = fetch_company_metadata()[["ticker", "Sector"]]
    m.to_parquet(p, index=False)
    return m


def build_dataset(prices: pd.DataFrame | None = None, meta: pd.DataFrame | None = None,
                  out_dir: str | None = None, chunk_size: int = 800, log=print) -> dict:
    """Build the daily v3 dataset under ``out_dir`` (default ``<data>/v3model/dataset``):

    ``stage_a/c#####.parquet`` (per ticker chunk), ``proxy_sectors.parquet``,
    ``daily/year=YYYY.parquet`` (features, eligibility, sector, labels,
    targets; float32) and ``meta.json``. ``prices`` defaults to the cleaned WS1
    store (``prices.load_prices``)."""
    t0 = time.time()
    out_dir = out_dir or v3_dir("dataset")
    for sub in ("stage_a", "daily"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        for f in glob.glob(os.path.join(out_dir, sub, "*.parquet")):
            os.remove(f)
    if prices is None:
        from .prices import load_prices
        prices = load_prices(columns=["company_close", "company_volume"], with_raw=True)
        prices = prices[["ticker", "Date", "company_close", "company_volume", "close_raw"]]
        log(f"[dataset] loaded {len(prices):,} price rows, {prices['ticker'].nunique():,} tickers "
            f"({time.time() - t0:.0f}s)")
    meta = load_sector_meta() if meta is None else meta
    known = known_sector_map(meta)
    cal = session_calendar(prices["Date"])
    tickers = sorted(prices["ticker"].unique())
    prices = prices.sort_values(["ticker", "Date"], kind="mergesort").reset_index(drop=True)
    bounds = prices.groupby("ticker", sort=True).indices
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    for k, ch in enumerate(chunks):
        idx = np.concatenate([bounds[t] for t in ch])
        a = _f32(stage_a(prices.iloc[idx], cal, labels=True))
        a.to_parquet(os.path.join(out_dir, "stage_a", f"c{k:05d}.parquet"), index=False)
        log(f"[dataset] stage A chunk {k + 1}/{len(chunks)}: {len(a):,} rows ({time.time() - t0:.0f}s)")
        del a
        gc.collect()
    del prices, bounds
    gc.collect()

    import pyarrow.dataset as pads
    dset = pads.dataset(os.path.join(out_dir, "stage_a"), format="parquet")
    long = dset.to_table(columns=["ticker", "Date", "daily_return", "eligible"]).to_pandas()
    long["ticker"] = long["ticker"].astype("category")
    assign = proxy_sector_assignments(long, known, cal)
    assign.to_parquet(os.path.join(out_dir, "proxy_sectors.parquet"), index=False)
    log(f"[dataset] proxy sectors: {len(assign):,} assignments over "
        f"{assign['month'].nunique()} months ({time.time() - t0:.0f}s)")
    del long
    gc.collect()

    years = sorted({d[:4] for d in cal})
    stats = {}
    for y in years:
        tab = dset.to_table(filter=(pads.field("Date") >= f"{y}-01-01") & (pads.field("Date") <= f"{y}-12-31"))
        df = tab.to_pandas()
        df = stage_b(df, known, assign[assign["month"].str[:4] == y], targets=True)
        df = _f32(df.sort_values(["Date", "ticker"]).reset_index(drop=True))
        df.to_parquet(os.path.join(out_dir, "daily", f"year={y}.parquet"), index=False)
        el = df[df["eligible"]]
        stats[y] = {"rows": int(len(df)), "eligible_rows": int(len(el)),
                    "sector_source_eligible": el["sector_source"].value_counts(normalize=True).round(4).to_dict()}
        log(f"[dataset] year {y}: {len(df):,} rows, {len(el):,} eligible ({time.time() - t0:.0f}s)")
        del df, el, tab
        gc.collect()
    info = {"feature_version": FEATURE_VERSION, "feature_columns": FEATURE_COLUMNS,
            "horizons": list(HORIZONS), "label_clip": list(LABEL_CLIP),
            "eligibility_rule": ELIGIBILITY_RULE, "proxy_window": PROXY_WINDOW,
            "proxy_min_obs": PROXY_MIN_OBS, "calendar_start": str(cal[0]), "calendar_end": str(cal[-1]),
            "n_known_sector_tickers": int(len(known)), "years": stats,
            "seconds": round(time.time() - t0)}
    json.dump(info, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return info


def load_daily(years=None, columns=None, out_dir: str | None = None, dates=None) -> pd.DataFrame:
    out_dir = out_dir or v3_dir("dataset")
    files = sorted(glob.glob(os.path.join(out_dir, "daily", "year=*.parquet")))
    if years is not None:
        ys = {str(y) for y in years}
        files = [f for f in files if os.path.basename(f)[5:9] in ys]
    filt = [("Date", "in", list(dates))] if dates is not None else None
    frames = [pd.read_parquet(f, columns=columns, filters=filt) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def calendar_from_dataset(out_dir: str | None = None) -> np.ndarray:
    out_dir = out_dir or v3_dir("dataset")
    files = sorted(glob.glob(os.path.join(out_dir, "daily", "year=*.parquet")))
    return session_calendar(pd.concat([pd.read_parquet(f, columns=["Date"])["Date"] for f in files]))


if __name__ == "__main__":
    build_dataset()

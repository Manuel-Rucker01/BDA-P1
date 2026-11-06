"""Round-4 signal blocks (pre-registered in ``v3/gate.py``, round 4).

* ``sue_events``            -- EDGAR earnings surprise (SUE) and fundamental
                               momentum, as an event table per (cik, asof_date)
                               that ``dataset.FundamentalsBlock(extra_events=...)``
                               joins PIT and hot-deck-imputes jointly with the
                               round-2 fundamentals.
* ``residual_momentum``     -- residual 12-1 / 6-1 momentum, 1-month residual
                               reversal, industry momentum, market beta,
                               idiosyncratic vol (rolling OLS through rolling
                               sums, fully vectorised).
* ``lvq_composite``         -- low-vol/quality composite rank.
* ``build_r4_dataset``      -- the dataset_r4 store (dataset_edgar + all of the above).
"""
from __future__ import annotations

import gc
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from . import dataset as ds

SUE_COLUMNS = ["sue", "sue_chg", "days_since_report", "d_gross_margin", "d_roa", "d_accruals", "asset_growth"]
RESMOM_COLUMNS = ["resmom_12_1", "resmom_6_1", "strev_1m", "ind_mom_12_1"]
LVQ_COLUMNS = ["beta_mkt", "idio_vol", "lvq_composite"]
BLOCK_COLUMNS = {"resmom": RESMOM_COLUMNS, "sue": SUE_COLUMNS, "lvq": LVQ_COLUMNS}
R4_DATASET = "dataset_r4"
REG_WINDOW, REG_MIN = 504, 252
RET_CLIP = (-0.5, 1.0)


# --------------------------------------------------------------------------- #
# SUE + fundamental momentum (EDGAR)
# --------------------------------------------------------------------------- #
def sue_events(fund_asof: pd.DataFrame) -> pd.DataFrame:
    """One row per (cik, asof_date) of ``fund_asof`` with the SUE / fundamental
    momentum values as known at that availability date. ``report_date`` = the
    availability date of the latest *new* net-income period (for
    ``days_since_report``)."""
    f = fund_asof.sort_values(["cik", "asof_date"], kind="mergesort")
    out = []
    for cik, g in f.groupby("cik", sort=False):
        ends, ttms, ds_hist = [], [], []
        a_hist = {}                      # assets period end -> value (as known so far)
        sue_prev = np.nan
        last = {"sue": np.nan, "sue_chg": np.nan, "report_date": pd.NaT}
        for r in g.itertuples(index=False):
            if pd.notna(r.assets_end) and pd.notna(r.assets):
                a_hist[pd.Timestamp(r.assets_end)] = float(r.assets)
            E, ttm = r.net_income_end, r.net_income_ttm
            if pd.notna(E) and pd.notna(ttm) and (not ends or E > ends[-1]):
                d = np.nan
                if ends and 75 <= (E - ends[-1]).days <= 105:
                    d = float(ttm) - ttms[-1]
                ends.append(E)
                ttms.append(float(ttm))
                ds_hist.append(d)
                hist = np.array([x for x in ds_hist[-8:] if np.isfinite(x)])
                sue = d / hist.std(ddof=1) if np.isfinite(d) and len(hist) >= 4 and hist.std(ddof=1) > 0 else np.nan
                last = {"sue": sue, "sue_chg": sue - sue_prev if np.isfinite(sue) else np.nan,
                        "report_date": r.asof_date}
                if np.isfinite(sue):
                    sue_prev = sue
            elif pd.notna(E) and ends and E == ends[-1] and pd.notna(ttm):
                ttms[-1] = float(ttm)    # restatement of the latest period (as of this filing)
            A = r.assets if pd.notna(r.assets) and r.assets > 0 else np.nan
            a1 = np.nan
            if pd.notna(r.assets_end):
                Ae = pd.Timestamp(r.assets_end)
                cand = [k for k in a_hist if 320 <= (Ae - k).days <= 410]
                if cand:
                    a1 = a_hist[max(cand)]
            a1 = a1 if np.isfinite(a1) and a1 > 0 else np.nan
            rev, revp = r.revenue_ttm, r.revenue_ttm_prev
            gm = r.gross_profit_ttm / rev if pd.notna(rev) and rev > 0 else np.nan
            gmp = r.gross_profit_ttm_prev / revp if pd.notna(revp) and revp > 0 else np.nan
            ni, nip, cfo, cfop = r.net_income_ttm, r.net_income_ttm_prev, r.cfo_ttm, r.cfo_ttm_prev
            out.append((cik, r.asof_date, last["sue"], last["sue_chg"], last["report_date"], gm - gmp,
                        ni / A - nip / a1, (ni - cfo) / A - (nip - cfop) / a1, A / a1 - 1.0))
    ev = pd.DataFrame(out, columns=["cik", "asof_date", "sue", "sue_chg", "report_date", "d_gross_margin",
                                    "d_roa", "d_accruals", "asset_growth"])
    for c in ("d_gross_margin", "d_roa", "d_accruals", "asset_growth"):
        ev[c] = pd.to_numeric(ev[c], errors="coerce").astype(float)
    return ev


# --------------------------------------------------------------------------- #
# Residual momentum (rolling OLS via rolling sums)
# --------------------------------------------------------------------------- #
def _window_sum(cs: np.ndarray, start: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum over rows [i-hi, i-lo] of the ticker of row i, from per-ticker-reset
    cumulative sums ``cs`` (cs[i] = sum of rows start..i). Returns (sum, ok)."""
    i = np.arange(len(cs))
    a, b = i - hi, i - lo                      # inclusive bounds
    ok = a >= start
    a_c = np.clip(a - 1, 0, None)
    b_c = np.clip(b, 0, None)
    left = np.where(a - 1 >= start, cs[a_c], 0.0)
    return np.where(ok, cs[b_c] - left, np.nan), ok


def residual_momentum(long: pd.DataFrame, calendar) -> pd.DataFrame:
    """``long``: ticker, Date, daily_return, Sector, eligible for the FULL
    history of every ticker. Returns RESMOM_COLUMNS + beta_mkt, idio_vol aligned
    to ``long.index``. Everything at row t uses bars <= t only."""
    L = long[["ticker", "Date", "daily_return", "Sector", "eligible"]].copy()
    L["y"] = L["daily_return"].astype(float).clip(*RET_CLIP)
    el = L[L["eligible"].astype(bool) & L["y"].notna()]
    mkt = el.groupby("Date")["y"].mean()
    sec = el.groupby(["Sector", "Date"])["y"].mean()
    L["x1"] = L["Date"].map(mkt).astype(float)
    sk = pd.MultiIndex.from_arrays([L["Sector"], L["Date"]])
    L["x2"] = pd.Series(sec.reindex(sk).to_numpy(), index=L.index).fillna(L["x1"])
    order = np.lexsort((L["Date"].to_numpy(), L["ticker"].to_numpy()))
    S = L.iloc[order]
    tick = S["ticker"].to_numpy()
    first = np.r_[True, tick[1:] != tick[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(S)), 0))
    v = S["y"].notna().to_numpy() & S["x1"].notna().to_numpy()
    y = np.where(v, S["y"].to_numpy(), 0.0)
    x1 = np.where(v, S["x1"].to_numpy(), 0.0)
    x2 = np.where(v, S["x2"].to_numpy(), 0.0)

    def csum(a):
        c = np.cumsum(a)
        base = np.where(start > 0, c[np.clip(start - 1, 0, None)], 0.0)
        return c - base

    terms = {"n": v.astype(float), "y": y, "a": x1, "b": x2, "aa": x1 * x1, "bb": x2 * x2, "ab": x1 * x2,
             "ay": x1 * y, "by": x2 * y, "yy": y * y}
    W = {}
    for k, arr in terms.items():
        W[k], ok = _window_sum(csum(arr), start, 0, REG_WINDOW - 1)
        # shorter history: use everything since the first bar
        full = csum(arr)
        W[k] = np.where(ok, W[k], full)
    n = W["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb, my = W["a"] / n, W["b"] / n, W["y"] / n
        vaa, vbb, vab = W["aa"] / n - ma * ma, W["bb"] / n - mb * mb, W["ab"] / n - ma * mb
        cay, cby, vyy = W["ay"] / n - ma * my, W["by"] / n - mb * my, W["yy"] / n - my * my
        det = vaa * vbb - vab * vab
        good = (n >= REG_MIN) & (det > 1e-14)
        b1 = np.where(good, (vbb * cay - vab * cby) / det, np.nan)
        b2 = np.where(good, (vaa * cby - vab * cay) / det, np.nan)
        s2 = vyy - b1 * cay - b2 * cby
        sig = np.sqrt(np.where(s2 > 0, s2, np.nan))
        beta_mkt = np.where((n >= REG_MIN) & (vaa > 0), cay / vaa, np.nan)
        out = {}
        for name, lo, hi in (("resmom_12_1", 21, 251), ("resmom_6_1", 21, 125), ("strev_1m", 0, 20)):
            sy, ok1 = _window_sum(csum(y), start, lo, hi)
            sa, _ = _window_sum(csum(x1), start, lo, hi)
            sb, _ = _window_sum(csum(x2), start, lo, hi)
            m, _ = _window_sum(csum(v.astype(float)), start, lo, hi)
            e = sy - b1 * sa - b2 * sb
            out[name] = np.where(ok1 & (m >= 0.8 * (hi - lo + 1)), e / (sig * np.sqrt(np.maximum(m, 1))), np.nan)
    out["beta_mkt"] = beta_mkt
    out["idio_vol"] = sig * np.sqrt(252.0)
    res = pd.DataFrame(out, index=S.index)
    # industry momentum: cumulative sector log return over sessions t-251..t-21
    cal = pd.Index(np.asarray(calendar))
    secw = sec.unstack(0).reindex(cal)
    cl = np.log1p(secw.fillna(0.0)).cumsum()
    im = (cl.shift(21) - cl.shift(252))
    im_long = im.stack().rename("ind_mom_12_1")
    key = pd.MultiIndex.from_arrays([L["Date"], L["Sector"]])
    res = res.reindex(L.index)
    res["ind_mom_12_1"] = im_long.reindex(key).to_numpy()
    return res[RESMOM_COLUMNS + ["beta_mkt", "idio_vol"]]


def lvq_composite(df: pd.DataFrame) -> pd.Series:
    """Mean of per-date percentile ranks of (-beta_mkt, -idio_vol, gpa) over eligible rows."""
    el = df["eligible"].astype(bool)
    r = pd.DataFrame(index=df.index)
    g = df.loc[el].groupby("Date")
    r.loc[el, "b"] = g["beta_mkt"].rank(pct=True, ascending=False)
    r.loc[el, "v"] = g["idio_vol"].rank(pct=True, ascending=False)
    r.loc[el, "q"] = g["gpa"].rank(pct=True) if "gpa" in df.columns else np.nan
    return r[["b", "v", "q"]].mean(axis=1).where(el)


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def r4_block(fund_dir: str | None = None) -> "ds.FundamentalsBlock":
    fund_dir = fund_dir or ds.data_dir("fundamentals")
    fa = pd.read_parquet(os.path.join(fund_dir, "fundamentals_asof.parquet"))
    return ds.FundamentalsBlock(fa, pd.read_parquet(os.path.join(fund_dir, "cik_map.parquet")),
                                extra_events=sue_events(fa))


def build_r4_dataset(log=print) -> dict:
    """dataset_edgar + SUE/fundamental-momentum (jointly imputed) + residual
    momentum + low-vol/quality, under ``<data>/v3model/dataset_r4``."""
    t0 = time.time()
    from .prices import load_prices
    out_dir = ds.v3_dir(R4_DATASET)
    prices = load_prices(columns=["company_close", "company_volume"], with_raw=True)
    prices = prices[["ticker", "Date", "company_close", "company_volume", "close_raw"]]
    meta = ds.edgar_meta_for_prices(prices)
    block = r4_block()
    log(f"[r4] SUE events built ({time.time() - t0:.0f}s)")
    info = ds.build_dataset(prices=prices, meta=meta, out_dir=out_dir, blocks=[block], log=log)
    del prices
    gc.collect()
    files = sorted(glob.glob(os.path.join(out_dir, "daily", "year=*.parquet")))
    long = pd.concat([pd.read_parquet(f, columns=["ticker", "Date", "daily_return", "Sector", "eligible"])
                      for f in files], ignore_index=True)
    cal = ds.session_calendar(long["Date"])
    rm = residual_momentum(long, cal)
    rm["ticker"], rm["Date"] = long["ticker"].to_numpy(), long["Date"].to_numpy()
    del long
    gc.collect()
    log(f"[r4] residual momentum done ({time.time() - t0:.0f}s)")
    for f in files:
        d = pd.read_parquet(f)
        y = os.path.basename(f)[5:9]
        part = rm[rm["Date"].str[:4] == y]
        d = d.merge(part, on=["ticker", "Date"], how="left")
        d["lvq_composite"] = lvq_composite(d)
        for c in RESMOM_COLUMNS + LVQ_COLUMNS:
            d[c] = d[c].astype("float32")
        d.to_parquet(f, index=False)
    info["r4_columns"] = BLOCK_COLUMNS
    info["seconds_total"] = round(time.time() - t0)
    json.dump(info, open(os.path.join(out_dir, "meta.json"), "w"), indent=1, default=str)
    log(f"[r4] dataset_r4 written ({time.time() - t0:.0f}s)")
    return info


if __name__ == "__main__":
    build_r4_dataset()

"""Round 14 ``spill`` block: firm-to-firm graph and return-spillover features.

Definitions are fixed in ``results/round14_addendum_spill.md``; the
pre-registration is ``results/round14_preregistration.md``.

* **E1 ``tnic``**: Hoberg-Phillips TNIC-3 text-product peers, keyed by gvkey.
  gvkey maps to CIK through the free ``farr`` ``gvkey_ciks`` table. Year Y is
  usable from the first session on or after July 1 of Y+1. Top-10 by score.
* **E2 ``own``**: common institutional ownership from the Round-9 13F store.
  The weight is the TF-IDF cosine of binary holder vectors. Quarter P is usable
  from ``f13_avail(P)``. Top-10 neighbours.
* **E3 ``sic``**: same 4-digit SIC, equal weights. Control only.

Features per edge set e:

* ``spill_{e}_r{5,21,63}``: weight-normalised mean of the neighbours' trailing
  returns, excluding the stock itself.
* ``spill_{e}_x{k}``: ``spill_{e}_r{k}`` minus the stock's own return.
* ``spill_{e}_n``: neighbour count.

The panel is weekly, on exactly the rows of Round 8's ``alt_weekly.parquet``.
Output files:

* ``$BDA_DATA_V3/v3model/alt14_spill_weekly.parquet``
* point-in-time edges in ``$BDA_DATA_V3/spill/``

Commands:

    python -m DataAnalysisPipeline2.v3.altdata_r14_spill edges features kg checks
"""
from __future__ import annotations

import glob
import os
import time
import urllib.parse

import numpy as np
import pandas as pd

from . import dataset as ds
from ._alpaca import data_dir

K_TOP = 10
WINDOWS = (5, 21, 63)
RET_CLIP = (-0.9, 2.0)
TNIC_FIRST_YEAR = 2015                    # year 2015 -> usable 2016-07-01 (covers 2017 decision dates)
TNIC_LAG_MONTH_DAY = (7, 1)               # year t usable from July 1 of t+1
OWN_MIN_HOLDERS = 5
OWN_MIN_COMMON = 3
EDGE_SETS = ("tnic", "own", "sic")
BLOCK_SETS = ("tnic", "own")              # the trial block; sic = control


def cols_for(e: str) -> list:
    return [f"spill_{e}_r{k}" for k in WINDOWS] + [f"spill_{e}_x{k}" for k in WINDOWS] + [f"spill_{e}_n"]


SPILL_COLUMNS = [c for e in BLOCK_SETS for c in cols_for(e)]
CONTROL_COLUMNS = cols_for("sic")
ALL_COLUMNS = [c for e in EDGE_SETS for c in cols_for(e)]
HAS_COLUMNS = [f"spill_{e}_has" for e in EDGE_SETS]
REF_COLUMNS = [f"ref_own_r{k}" for k in WINDOWS]       # the stock's own trailing returns (reference, not block)
BLOCKS = {"spill": SPILL_COLUMNS}


def spill_dir(*parts: str) -> str:
    return data_dir("spill", *parts)


def out_path() -> str:
    return os.path.join(ds.v3_dir(), "alt14_spill_weekly.parquet")


def edges_path(kind: str) -> str:
    return os.path.join(spill_dir(), f"edges_{kind}.parquet")


# --------------------------------------------------------------------------- #
# Generic helpers (pure; unit-tested)
# --------------------------------------------------------------------------- #
def first_session_on_or_after(cal: np.ndarray, day) -> str | None:
    cal = np.asarray(cal).astype(str)
    i = np.searchsorted(cal, pd.Timestamp(day).strftime("%Y-%m-%d"), side="left")
    return str(cal[i]) if i < len(cal) else None


def tnic_valid_from(year: int, cal: np.ndarray) -> str | None:
    m, d = TNIC_LAG_MONTH_DAY
    return first_session_on_or_after(cal, pd.Timestamp(year + 1, m, d))


def topk(df: pd.DataFrame, src: str, dst: str, w: str, k: int = K_TOP) -> pd.DataFrame:
    """Top-k rows per ``src`` by ``w`` descending, ties broken by ``dst`` ascending."""
    return (df.sort_values([src, w, dst], ascending=[True, False, True], kind="mergesort")
            .groupby(src, sort=False).head(k).reset_index(drop=True))


def map_gvkey_cik(links: pd.DataFrame, year: int) -> pd.Series:
    """gvkey -> CIK for TNIC year Y: links whose [first, last] dates overlap
    [Y-01-01, Y+1-12-31]; several -> latest last_date, then smallest CIK."""
    lo, hi = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 12, 31)
    f = links[(pd.to_datetime(links["first_date"]) <= hi) & (pd.to_datetime(links["last_date"]) >= lo)]
    f = f.assign(_l=pd.to_datetime(f["last_date"])).sort_values(["gvkey", "_l", "cik"], ascending=[True, False, True])
    f = f.drop_duplicates("gvkey")
    return f.set_index("gvkey")["cik"].astype("int64")


def trailing_returns(close: np.ndarray, pos: int, k: int) -> np.ndarray:
    """close[pos] / close[pos - k] - 1 per column (NaN if either missing), clipped."""
    if pos - k < 0:
        return np.full(close.shape[1], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = close[pos] / close[pos - k] - 1.0
    r[~np.isfinite(r)] = np.nan
    return np.clip(r, *RET_CLIP)


def neighbour_means(src: np.ndarray, w: np.ndarray, rdst: np.ndarray, n_src: int) -> tuple[np.ndarray, np.ndarray]:
    """Per source node s in [0, n_src): weight-normalised mean of ``rdst`` over
    its edges with finite ``rdst`` (``rdst``: E x W, one column per window) and
    the count of edges with a finite value in column 1 (the 21-session window).
    -> (means n_src x W, count n_src); means NaN where no finite neighbour."""
    W = rdst.shape[1]
    out = np.full((n_src, W), np.nan)
    for j in range(W):
        ok = np.isfinite(rdst[:, j]) & np.isfinite(w) & (w > 0)
        num = np.bincount(src[ok], weights=w[ok] * rdst[ok, j], minlength=n_src)
        den = np.bincount(src[ok], weights=w[ok], minlength=n_src)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, j] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    j21 = 1 if W > 1 else 0
    cnt = np.bincount(src[np.isfinite(rdst[:, j21])], minlength=n_src).astype(float)
    return out, cnt


def idf_cosine_edges(h: pd.DataFrame, k: int = K_TOP, min_holders: int = OWN_MIN_HOLDERS,
                     min_common: int = OWN_MIN_COMMON, same_cik: dict | None = None,
                     block: int = 512) -> pd.DataFrame:
    """Top-k TF-IDF-cosine common-ownership neighbours.
    ``h``: [ticker, cik] positions (filer ``cik`` holds ``ticker``) of one quarter.
    ``same_cik``: ticker -> issuer CIK; pairs with the same known issuer CIK are dropped.
    -> [src, dst, weight, n_common]."""
    from scipy import sparse
    h = h[["ticker", "cik"]].drop_duplicates()
    nh = h.groupby("ticker")["cik"].size()
    tick = np.array(sorted(nh.index[nh >= min_holders]))
    h = h[h["ticker"].isin(set(tick))]
    if len(tick) < 2:
        return pd.DataFrame(columns=["src", "dst", "weight", "n_common"])
    filers = np.array(sorted(h["cik"].unique()))
    ti = np.searchsorted(tick, h["ticker"].to_numpy())
    fi = np.searchsorted(filers, h["cik"].to_numpy())
    B = sparse.csr_matrix((np.ones(len(h), np.float32), (fi, ti)), shape=(len(filers), len(tick)))
    n_f = np.asarray(B.sum(axis=1)).ravel()
    idf = np.log(len(tick) / np.maximum(n_f, 1)).astype(np.float32)
    X = sparse.diags(idf) @ B
    norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
    Xc, Bc = X.tocsc(), B.tocsc()
    XT, BT = X.T.tocsr(), B.T.tocsr()
    issuer = np.array([same_cik.get(t, -1) if same_cik else -1 for t in tick], dtype=np.int64)
    out = []
    for s in range(0, len(tick), block):
        e = min(s + block, len(tick))
        G = (XT[s:e] @ Xc).toarray()                                                # (blk, T) idf^2 dot products
        C = (BT[s:e] @ Bc).toarray()                                                # common holder counts
        with np.errstate(invalid="ignore", divide="ignore"):
            S = G / (norm[s:e, None] * norm[None, :])
        S[~np.isfinite(S)] = 0.0
        S[C < min_common] = 0.0
        S[np.arange(e - s), np.arange(s, e)] = 0.0                                  # self
        if same_cik:
            ii = issuer[s:e]
            m = (ii[:, None] == issuer[None, :]) & (ii[:, None] >= 0)
            S[m] = 0.0
        kk = min(3 * k, S.shape[1] - 1)
        cand = np.argpartition(-S, kk, axis=1)[:, :kk]
        for r in range(e - s):
            c = cand[r]
            v = S[r, c]
            keep = v > 0
            c, v = c[keep], v[keep]
            if not len(c):
                continue
            o = np.lexsort((tick[c], -v))[:k]
            out.append(pd.DataFrame({"src": tick[s + r], "dst": tick[c[o]], "weight": v[o].astype(np.float32),
                                     "n_common": C[r, c[o]].astype(np.int32)}))
    if not out:
        return pd.DataFrame(columns=["src", "dst", "weight", "n_common"])
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def weekly_rows() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(ds.v3_dir(), "alt_weekly.parquet"),
                           columns=["ticker", "Date", "eligible", "fwd_ret_30", "cik"])


def price_store(log=print):
    """(calendar, close matrix sessions x tickers (float64), ticker->col,
    n_bars on weekly rows [ticker, Date, n_bars])."""
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(ds.v3_dir(ds.EDGAR_DATASET), "stage_a", "*.parquet")))
    daily = pd.concat([pd.read_parquet(f, columns=["ticker", "Date", "company_close", "n_bars"]) for f in files],
                      ignore_index=True)
    cal = ds.session_calendar(daily["Date"])
    tick = np.array(sorted(daily["ticker"].unique()))
    col = {t: k for k, t in enumerate(tick)}
    ri = np.searchsorted(cal, daily["Date"].to_numpy().astype(str))
    ci = daily["ticker"].map(col).to_numpy()
    close = np.full((len(cal), len(tick)), np.nan, dtype=np.float64)
    close[ri, ci] = daily["company_close"].to_numpy(dtype=float)
    nb = daily[["ticker", "Date", "n_bars"]]
    del daily
    log(f"[r14s] price store {close.shape} ({time.time() - t0:.0f}s)")
    return cal, close, col, tick, nb


def primary_tickers(rt: pd.DataFrame) -> pd.DataFrame:
    """[cik, ticker] on one date: eligible first, then most bars, then alphabetical."""
    m = rt[rt["cik"].notna()].assign(_e=lambda x: x["eligible"].astype(bool).astype(int),
                                     _b=lambda x: x["n_bars"].fillna(0))
    m = m.sort_values(["cik", "_e", "_b", "ticker"], ascending=[True, False, False, True], kind="mergesort")
    m = m.drop_duplicates("cik")
    return m[["cik", "ticker"]].assign(cik=lambda x: x["cik"].astype("int64"))


# --------------------------------------------------------------------------- #
# Edge builders
# --------------------------------------------------------------------------- #
def build_tnic_edges(rows: pd.DataFrame, cal: np.ndarray, log=print) -> pd.DataFrame:
    tn = pd.read_parquet(os.path.join(data_dir("tnic"), "tnic3.parquet"))
    links = pd.read_parquet(os.path.join(data_dir("tnic"), "gvkey_ciks.parquet"))
    links["gvkey"] = links["gvkey"].astype(str).str.zfill(6)
    tn = tn[tn["year"] >= TNIC_FIRST_YEAR]
    tn["gvkey1"] = tn["gvkey1"].astype(str).str.zfill(6)
    tn["gvkey2"] = tn["gvkey2"].astype(str).str.zfill(6)
    years = sorted(tn["year"].unique())
    vf = {y: tnic_valid_from(int(y), cal) for y in years}
    rd = rows.dropna(subset=["cik"])
    out = []
    for i, y in enumerate(years):
        if vf[y] is None:
            continue
        vt = vf.get(years[i + 1]) if i + 1 < len(years) else None
        win = rd[(rd["Date"] >= vf[y]) & ((rd["Date"] < vt) if vt else True)]
        ciks = set(win["cik"].astype("int64"))
        mp = map_gvkey_cik(links, int(y))
        t = tn[tn["year"] == y]
        e = pd.DataFrame({"src_cik": t["gvkey1"].map(mp), "dst_cik": t["gvkey2"].map(mp),
                          "weight": t["score"].astype(float)}).dropna()
        e = e.astype({"src_cik": "int64", "dst_cik": "int64"})
        e = e[(e["src_cik"] != e["dst_cik"]) & e["dst_cik"].isin(ciks)]
        e = e.sort_values("weight", ascending=False).drop_duplicates(["src_cik", "dst_cik"])
        e = topk(e, "src_cik", "dst_cik", "weight")
        e["tnic_year"], e["valid_from"], e["valid_to"] = int(y), vf[y], vt
        out.append(e)
        log(f"[r14s] tnic {y}: {len(t):,} pairs -> {len(e):,} edges, {e['src_cik'].nunique():,} source CIKs, "
            f"valid {vf[y]} .. {vt}")
    e = pd.concat(out, ignore_index=True)
    e["weight"] = e["weight"].astype("float32")
    return e


def build_own_edges(rows: pd.DataFrame, log=print) -> pd.DataFrame:
    from . import altdata_r9 as a9
    t0 = time.time()
    ftd = a9.load_ftd()
    filings, hold = a9.load_13f(set(ftd["cusip"].unique()), log=log)
    files = sorted(glob.glob(os.path.join(ds.v3_dir(ds.EDGAR_DATASET), "stage_a", "*.parquet")))
    first = pd.concat([pd.read_parquet(f, columns=["ticker", "Date"]).groupby("ticker")["Date"].min() for f in files])
    tickers = first.groupby(level=0).min().rename("first_date").reset_index()
    periods = sorted(filings["period"].unique())
    q = a9.quarter_holders(filings, hold, ftd, tickers, periods)
    del hold
    log(f"[r14s] 13F quarters {len(q)} ({time.time() - t0:.0f}s)")
    ps = sorted(q)
    avail = {P: pd.Timestamp(q[P][3]).strftime("%Y-%m-%d") for P in ps}
    rd = rows.dropna(subset=["cik"])
    dates = np.array(sorted(rows["Date"].unique()))
    out = []
    for i, P in enumerate(ps):
        A = avail[P]
        vt = avail[ps[i + 1]] if i + 1 < len(ps) else None
        j = np.searchsorted(dates, A)
        same = {}
        if j < len(dates):
            r0 = rd[rd["Date"] == dates[j]]
            same = dict(zip(r0["ticker"], r0["cik"].astype("int64")))
        h = q[P][1]
        h = h[h["shares"] > 0]
        e = idf_cosine_edges(h, same_cik=same)
        e["period"], e["valid_from"], e["valid_to"] = pd.Timestamp(P), A, vt
        out.append(e)
        log(f"[r14s] 13F {pd.Timestamp(P).date()}: {h['ticker'].nunique():,} held tickers -> {len(e):,} edges "
            f"({e['src'].nunique():,} sources), valid {A} .. {vt} ({time.time() - t0:.0f}s)")
    return pd.concat(out, ignore_index=True)


def build_edges(log=print) -> dict:
    rows = weekly_rows()
    cal = ds.calendar_from_dataset(ds.v3_dir(ds.EDGAR_DATASET))
    et = build_tnic_edges(rows, cal, log=log)
    et.to_parquet(edges_path("tnic"), index=False)
    eo = build_own_edges(rows, log=log)
    eo.to_parquet(edges_path("own"), index=False)
    return {"tnic": et, "own": eo}


def _version_for(edges: pd.DataFrame, date: str) -> pd.DataFrame:
    vf = edges["valid_from"].astype(str)
    vt = edges["valid_to"].fillna("9999-12-31").astype(str)
    return edges[(vf <= date) & (vt > date)]


def edges_asof(date: str, kind: str, edges: pd.DataFrame | None = None,
               rows: pd.DataFrame | None = None, nbars: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ticker-space edges [src, dst, weight] used on decision date ``date``.
    ``tnic``: CIKs -> every ticker of the source CIK on ``date``, the primary
    ticker of the neighbour CIK (as in the features)."""
    edges = pd.read_parquet(edges_path(kind)) if edges is None else edges
    e = _version_for(edges, date)
    if kind == "own":
        return e[["src", "dst", "weight"]].reset_index(drop=True)
    rows = weekly_rows() if rows is None else rows
    rt = rows[rows["Date"] == date]
    if nbars is not None:
        rt = rt.merge(nbars[nbars["Date"] == date], on=["ticker", "Date"], how="left")
    else:
        rt = rt.assign(n_bars=np.nan)
    prim = primary_tickers(rt).set_index("cik")["ticker"]
    allt = rt.dropna(subset=["cik"]).assign(cik=lambda x: x["cik"].astype("int64"))[["cik", "ticker"]]
    m = e.merge(allt.rename(columns={"cik": "src_cik", "ticker": "src"}), on="src_cik")
    m["dst"] = m["dst_cik"].map(prim)
    return m.dropna(subset=["dst"])[["src", "dst", "weight"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Weekly features
# --------------------------------------------------------------------------- #
def _assign(out: np.ndarray, idx: np.ndarray, means: np.ndarray, cnt: np.ndarray, own: np.ndarray, base: int):
    """Write r (3), x (3), n into ``out[idx, base:base+7]``."""
    nw = len(WINDOWS)
    means = np.where(cnt[:, None] > 0, means, np.nan)
    out[idx, base:base + nw] = means
    out[idx, base + nw:base + 2 * nw] = means - own
    out[idx, base + 2 * nw] = cnt


def date_features(rt: pd.DataFrame, R: np.ndarray, col: dict, e_tnic: pd.DataFrame, e_own: pd.DataFrame,
                  sic: pd.Series, Rn: np.ndarray | None = None) -> np.ndarray:
    """ALL_COLUMNS for the rows ``rt`` (ticker, cik, eligible, n_bars) of one
    date. ``R``: (n_tickers_store, W) trailing returns on that date (own);
    ``Rn``: the returns used for neighbours (== R except in the placebo)."""
    Rn = R if Rn is None else Rn
    n, W = len(rt), len(WINDOWS)
    out = np.full((n, 7 * len(EDGE_SETS)), np.nan)
    tcol = rt["ticker"].map(col).to_numpy()
    okc = ~pd.isna(tcol)
    own = np.full((n, W), np.nan)
    own[okc] = R[tcol[okc].astype(np.int64)]
    # ---- E2 own (ticker nodes)
    if len(e_own):
        srcs = np.array(sorted(e_own["src"].unique()))
        si = np.searchsorted(srcs, e_own["src"].to_numpy())
        dc = e_own["dst"].map(col).to_numpy()
        rd = np.full((len(e_own), W), np.nan)
        okd = ~pd.isna(dc)
        rd[okd] = Rn[dc[okd].astype(np.int64)]
        means, cnt = neighbour_means(si, e_own["weight"].to_numpy(float), rd, len(srcs))
        pos = pd.Series(np.arange(len(srcs)), index=srcs)
        rp = rt["ticker"].map(pos).to_numpy()
        m = ~pd.isna(rp)
        idx = np.where(m)[0]
        g = rp[m].astype(np.int64)
        _assign(out, idx, means[g], cnt[g], own[idx], 7 * EDGE_SETS.index("own"))
    # ---- CIK nodes (E1, E3)
    prim = primary_tickers(rt)
    pc = prim["ticker"].map(col).to_numpy()
    cik_ret = np.full((len(prim), W), np.nan)
    okp = ~pd.isna(pc)
    cik_ret[okp] = Rn[pc[okp].astype(np.int64)]
    cik_pos = pd.Series(np.arange(len(prim)), index=prim["cik"].to_numpy())
    row_cik = rt["cik"].to_numpy()
    rc = pd.Series(row_cik).map(lambda c: cik_pos.get(int(c), np.nan) if pd.notna(c) else np.nan).to_numpy()
    if len(e_tnic):
        srcs = np.array(sorted(e_tnic["src_cik"].unique()))
        si = np.searchsorted(srcs, e_tnic["src_cik"].to_numpy())
        dp = e_tnic["dst_cik"].map(cik_pos).to_numpy()
        rd = np.full((len(e_tnic), W), np.nan)
        okd = ~pd.isna(dp)
        rd[okd] = cik_ret[dp[okd].astype(np.int64)]
        means, cnt = neighbour_means(si, e_tnic["weight"].to_numpy(float), rd, len(srcs))
        pos = pd.Series(np.arange(len(srcs)), index=srcs)
        rp = pd.Series(row_cik).map(lambda c: pos.get(int(c), np.nan) if pd.notna(c) else np.nan).to_numpy()
        m = ~pd.isna(rp)
        idx = np.where(m)[0]
        g = rp[m].astype(np.int64)
        _assign(out, idx, means[g], cnt[g], own[idx], 7 * EDGE_SETS.index("tnic"))
    # ---- E3 sic: equal-weight mean over other CIKs with the same SIC-4
    s = prim["cik"].map(sic).to_numpy()
    oks = ~pd.isna(s) & (np.nan_to_num(s.astype(float)) > 0)
    if oks.any():
        codes, inv = np.unique(s[oks].astype(np.int64), return_inverse=True)
        cr = cik_ret[oks]
        nw = len(WINDOWS)
        fin = np.isfinite(cr)
        sums = np.stack([np.bincount(inv, weights=np.where(fin[:, j], cr[:, j], 0.0), minlength=len(codes))
                         for j in range(nw)], 1)
        cnts = np.stack([np.bincount(inv, weights=fin[:, j].astype(float), minlength=len(codes))
                         for j in range(nw)], 1)
        osum = sums[inv] - np.where(fin, cr, 0.0)
        ocnt = cnts[inv] - fin
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(ocnt > 0, osum / np.where(ocnt > 0, ocnt, 1), np.nan)
        n21 = ocnt[:, 1]
        full_m = np.full((len(prim), nw), np.nan)
        full_n = np.full(len(prim), np.nan)
        full_m[oks], full_n[oks] = mean, n21
        m = ~pd.isna(rc)
        idx = np.where(m)[0]
        g = rc[m].astype(np.int64)
        has = ~np.isnan(full_n[g])
        _assign(out, idx[has], full_m[g][has], full_n[g][has], own[idx[has]], 7 * EDGE_SETS.index("sic"))
    return out


def weekly_features(log=print, placebo_seed: int | None = None, write: bool = True) -> pd.DataFrame:
    """ALL_COLUMNS (+ has-flags) on alt_weekly's rows. ``placebo_seed``: permute
    neighbour returns across the store within each date (shuffled-neighbour placebo)."""
    t0 = time.time()
    rows = weekly_rows()
    cal, close, col, tick, nb = price_store(log=log)
    nbw = nb[nb["Date"].isin(set(rows["Date"]))]
    rows = rows.merge(nbw, on=["ticker", "Date"], how="left")
    et = pd.read_parquet(edges_path("tnic"))
    eo = pd.read_parquet(edges_path("own"))
    sic = pd.read_parquet(os.path.join(data_dir("fundamentals"), "cik_map.parquet"), columns=["cik", "sic"])
    sic = sic.dropna().drop_duplicates("cik").set_index("cik")["sic"].astype("int64")
    rng = np.random.default_rng(placebo_seed) if placebo_seed is not None else None
    vals = np.full((len(rows), len(ALL_COLUMNS)), np.nan)
    ref = np.full((len(rows), len(WINDOWS)), np.nan)
    audit = []
    for d, g in rows.groupby("Date", sort=True):
        p = int(np.searchsorted(cal, d))
        assert cal[p] == d
        R = np.stack([trailing_returns(close, p, k) for k in WINDOWS], 1)
        Rn = R
        if rng is not None:
            ok = np.isfinite(R[:, 1])
            perm = np.where(ok)[0]
            Rn = R.copy()
            Rn[perm] = R[rng.permutation(perm)]
        e1, e2 = _version_for(et, d), _version_for(eo, d)
        audit.append({"Date": d, "tnic_year": int(e1["tnic_year"].max()) if len(e1) else None,
                      "tnic_valid_from": e1["valid_from"].max() if len(e1) else None,
                      "own_period": str(pd.Timestamp(e2["period"].max()).date()) if len(e2) else None,
                      "own_valid_from": e2["valid_from"].max() if len(e2) else None,
                      "ret_end_session": d, "ret_end_pos": p})
        vals[g.index.to_numpy()] = date_features(g, R, col, e1, e2, sic, Rn)
        tc = g["ticker"].map(col).to_numpy()
        ok = ~pd.isna(tc)
        ref[g.index.to_numpy()[ok]] = R[tc[ok].astype(np.int64)]
    out = rows[["ticker", "Date", "eligible", "fwd_ret_30", "cik"]].copy()
    for j, c in enumerate(ALL_COLUMNS):
        out[c] = vals[:, j].astype("float32")
    for j, c in enumerate(REF_COLUMNS):
        out[c] = ref[:, j].astype("float32")
    for e in EDGE_SETS:
        out[f"spill_{e}_has"] = (out[f"spill_{e}_n"] >= 1).astype("int8")
    log(f"[r14s] features done ({time.time() - t0:.0f}s); has: "
        + ", ".join(f"{e} {out[f'spill_{e}_has'].mean():.1%}" for e in EDGE_SETS))
    if write and placebo_seed is None:
        out.to_parquet(out_path(), index=False)
        pd.DataFrame(audit).to_parquet(os.path.join(spill_dir(), "asof_audit.parquet"), index=False)
    return out


# --------------------------------------------------------------------------- #
# Knowledge-graph export (Turtle; ontology of ExploitationZone/graph_generation.py)
# --------------------------------------------------------------------------- #
ONTO = "http://bda.upc.edu/finance/ontology#"
ENT = "http://bda.upc.edu/finance/resource/"
REL = {"tnic": ("productPeerOf", "ProductPeerLink", "TNIC-3"), "own": ("sharesOwnersWith", "CommonOwnershipLink", "13F")}


def _iri(local: str) -> str:
    return f"<{ENT}{urllib.parse.quote(str(local), safe='._-~')}>"


def ttl_lines(e: pd.DataFrame, kind: str, valid_from: str, valid_to: str | None, version: str):
    """Turtle lines for ticker-space edges ``e`` [src, dst, weight] of one version:
    a direct triple ``src onto:<rel> dst`` and a reified link node with weight
    and validity dates."""
    rel, cls, srcname = REL[kind]
    for s, d, w in zip(e["src"], e["dst"], e["weight"]):
        link = _iri(f"Link_{kind}_{version}_{s}_{d}")
        yield f"{_iri(s)} onto:{rel} {_iri(d)} ."
        yield (f"{link} a onto:{cls} ; onto:linkSource {_iri(s)} ; onto:linkTarget {_iri(d)} ; "
               f"onto:linkWeight \"{float(w):.6g}\"^^xsd:float ; onto:linkSourceData \"{srcname}\" ; "
               f"onto:validFrom \"{valid_from}\"^^xsd:date"
               + (f" ; onto:validTo \"{valid_to}\"^^xsd:date" if valid_to else "") + " .")


TTL_HEADER = (f"@prefix onto: <{ONTO}> .\n@prefix ent: <{ENT}> .\n"
              "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
              "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n"
              "# Firm-to-firm links (Round 14 spill block). Generated by DataAnalysisPipeline2/v3/altdata_r14_spill.py\n"
              "# onto:productPeerOf = Hoberg-Phillips TNIC-3 top-10 peers; onto:sharesOwnersWith = top-10 TF-IDF cosine\n"
              "# common 13F ownership. Each link node carries onto:validFrom (first decision date the link is usable,\n"
              "# point in time) and onto:validTo (exclusive). Company IRIs = ent:<ticker id of the v3 store>.\n\n"
              "onto:productPeerOf a rdf:Property .\nonto:sharesOwnersWith a rdf:Property .\n"
              "onto:ProductPeerLink a rdf:Class .\nonto:CommonOwnershipLink a rdf:Class .\n\n")


def export_kg(full_path: str | None = None, sample_path: str | None = None, sample_date: str | None = None,
              sample_tickers: set | None = None, log=print) -> dict:
    """Full history (every edge version, ticker space as of its first decision
    date) -> gzip Turtle in the data store; a snapshot sample (edges valid on
    ``sample_date``, sources in ``sample_tickers``) -> ``sample_path``."""
    import gzip
    rows = weekly_rows()
    _, _, _, _, nb = price_store(log=log)
    dates = np.array(sorted(rows["Date"].unique()))
    full_path = full_path or os.path.join(spill_dir(), "firm_links_graph_full.ttl.gz")
    n = {"tnic": 0, "own": 0}
    with gzip.open(full_path, "wt") as fh:
        fh.write(TTL_HEADER)
        for kind in ("tnic", "own"):
            e = pd.read_parquet(edges_path(kind))
            for (vf, vt), g in e.groupby(["valid_from", e["valid_to"].fillna("")], sort=True):
                j = np.searchsorted(dates, vf)
                if j >= len(dates):
                    continue
                d = dates[j]
                te = edges_asof(d, kind, edges=g, rows=rows, nbars=nb)
                ver = str(g["tnic_year"].iloc[0]) if kind == "tnic" else str(pd.Timestamp(g["period"].iloc[0]).date())
                for line in ttl_lines(te, kind, vf, vt or None, ver):
                    fh.write(line + "\n")
                n[kind] += len(te)
    log(f"[r14s] KG full: {n} links -> {full_path} ({os.path.getsize(full_path) / 1e6:.0f} MB gz)")
    if sample_path:
        sample_date = sample_date or dates[-1]
        with open(sample_path, "w") as fh:
            fh.write(TTL_HEADER)
            fh.write(f"# SAMPLE: links valid on {sample_date}"
                     + (f", sources restricted to the {len(sample_tickers)} largest names" if sample_tickers else "")
                     + f". Full history: $BDA_DATA_V3/spill/{os.path.basename(full_path)}\n\n")
            for kind in ("tnic", "own"):
                e = pd.read_parquet(edges_path(kind))
                g = _version_for(e, sample_date)
                te = edges_asof(sample_date, kind, edges=g, rows=rows, nbars=nb)
                if sample_tickers:
                    te = te[te["src"].isin(sample_tickers)]
                vf = str(g["valid_from"].iloc[0])
                vt = g["valid_to"].iloc[0]
                ver = str(g["tnic_year"].iloc[0]) if kind == "tnic" else str(pd.Timestamp(g["period"].iloc[0]).date())
                for line in ttl_lines(te, kind, vf, vt if isinstance(vt, str) and vt else None, ver):
                    fh.write(line + "\n")
                n[f"sample_{kind}"] = len(te)
        log(f"[r14s] KG sample: {n} -> {sample_path} ({os.path.getsize(sample_path) / 1e6:.1f} MB)")
    return n


if __name__ == "__main__":
    import sys
    if "edges" in sys.argv:
        build_edges()
    if "features" in sys.argv:
        weekly_features()
    if "kg" in sys.argv:
        top = pd.read_parquet(os.path.join(ds.v3_dir(), "weekly_top1000.parquet"))
        last = top["Date"].max()
        here = os.path.dirname(os.path.abspath(__file__))
        export_kg(sample_path=os.path.join(here, "..", "..", "ExploitationZone", "firm_links_graph.ttl"),
                  sample_date=last, sample_tickers=set(top.loc[top["Date"] == last, "ticker"]))
    if "checks" in sys.argv:
        from . import round14_spill_checks as chk
        chk.run_all()

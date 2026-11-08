"""Round 10 analyst-free earnings-surprise blocks (pre-registered in ``v3/gate.py``,
round 10). Everything comes from the EDGAR bulk files already in the store
(``$BDA_DATA_V3/edgar/companyfacts.zip`` and ``submissions.zip``) plus the
survivorship-free price store -- no new download.

* **SUE block** (``sue``) -- seasonal-random-walk earnings surprise on quarterly
  diluted EPS from XBRL companyfacts:
  ``d_q = EPS_q - EPS_{q-4}`` with both values as known at q's filing (the
  q filing restates the comparative for splits), ``sue_eps = d_q / std(the
  previous <= 8 recorded d, min 4)``; ``sue_eps_chg`` (vs the previous
  quarter's SUE), ``sue_streak`` (signed run of same-sign surprises, |.| <= 8),
  ``surge`` (the same on quarterly revenue) and ``sue_days`` (calendar days
  since the filing became available). Quarter EPS: a reported 3-month fact >
  YTD differencing > Q4 = FY - (Q1+Q2+Q3) (``fundamentals._FlowState``); if no
  EPS concept reports the quarter, net income / weighted diluted shares of the
  same period. Available the NYSE business day after the 10-Q/10-K ``filed``
  date (the round-2 fundamentals rule).
* **EAR block** (``ear``) -- earnings-announcement return. Announcement = an
  original 8-K whose items include 2.02 (Results of Operations), at its EDGAR
  acceptance timestamp; a 10-Q/10-K is used instead (``ear_src10q`` = 1) only if
  no 8-K 2.02 was accepted in the 75 days up to it. Event session ``day0`` =
  the acceptance date if it is a session and acceptance < 16:00 New York time,
  else the next session. ``ear`` = stock minus EW-eligible-market return over
  sessions [day0-1, day0+1] (adjusted closes); ``ear_abvol`` = log(mean volume
  [day0-1, day0+1] / mean volume [day0-60, day0-11]); ``ear_prev`` = the same
  for the CIK's previous event; ``ear_days`` = calendar days since day0. An event
  is usable from session day0+2 (the business day after its window closes) and
  holds until the next one.
* Both blocks are set missing when the latest event is older than
  ``MAX_AGE_DAYS`` (400) calendar days.

    python -m DataAnalysisPipeline2.v3.altdata_r10 filings | sue | features
"""
from __future__ import annotations

import glob
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

try:
    import orjson as _json
    _loads = _json.loads
except ImportError:  # pragma: no cover
    import json as _json
    _loads = _json.loads

from . import dataset as ds
from . import fundamentals as fu
from ._alpaca import data_dir

SUE_COLUMNS = ["sue_eps", "sue_eps_chg", "sue_streak", "surge", "sue_days"]
EAR_COLUMNS = ["ear", "ear_prev", "ear_abvol", "ear_days", "ear_src10q"]
BLOCKS = {"sue": SUE_COLUMNS, "ear": EAR_COLUMNS}
HAS_COL = {"sue": "sue_eps", "ear": "ear"}

EPS_CONCEPTS = [("us-gaap", "EarningsPerShareDiluted"), ("us-gaap", "EarningsPerShareBasicAndDiluted"),
                ("us-gaap", "EarningsPerShareBasic")]
SUE_HIST, SUE_MIN_HIST = 8, 4
SUE_HIST_MAX_DAYS = 1100                  # d older than ~3 years never enter the std
STREAK_CAP = 8
EAR_FORMS = {"8-K"}
FALLBACK_FORMS = {"10-Q", "10-K", "10-QT", "10-KT"}
FALLBACK_LOOKBACK_DAYS = 75
MARKET_CLOSE_ET = 16                      # hour, America/New_York
EAR_LAG_SESSIONS = 2                      # usable from day0 + 2
VOL_BASE = (60, 11)                       # baseline volume window [day0-60, day0-11]
VOL_BASE_MIN = 20
RET_CLIP = (-0.5, 1.0)
MAX_AGE_DAYS = 400                        # older surprise / announcement -> missing (stopped filing;
                                          # fundamentals' fund_staleness_days already carries that)


def r10_dir(*parts: str) -> str:
    return data_dir("sec_earn", *parts)


# --------------------------------------------------------------------------- #
# Filings index (submissions.zip)
# --------------------------------------------------------------------------- #
def _filing_rows(block: dict, cik: int) -> list[tuple]:
    out = []
    forms = block.get("form") or []
    items = block.get("items") or [""] * len(forms)
    for i, f in enumerate(forms):
        if f in EAR_FORMS or f in FALLBACK_FORMS:
            out.append((cik, block["accessionNumber"][i], f, block["filingDate"][i],
                        block["acceptanceDateTime"][i], items[i] or ""))
    return out


def parse_submissions(zpath: str | None = None, ciks=None) -> pd.DataFrame:
    """8-K / 10-Q / 10-K filings (originals only) of ``ciks`` with EDGAR
    acceptance timestamp (UTC) and 8-K items, from the bulk submissions zip
    (the main ``CIK##########.json`` and its ``-submissions-NNN.json`` pages)."""
    zpath = zpath or os.path.join(data_dir("edgar"), "submissions.zip")
    rows = []
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        for cik in sorted({int(c) for c in ciks}):
            main = f"CIK{cik:010d}.json"
            if main not in names:
                continue
            d = _loads(z.read(main))
            fl = d.get("filings", {}) or {}
            rows += _filing_rows(fl.get("recent", {}) or {}, cik)
            for extra in fl.get("files", []) or []:
                if extra.get("name") in names:
                    rows += _filing_rows(_loads(z.read(extra["name"])), cik)
    f = pd.DataFrame(rows, columns=["cik", "accn", "form", "filing_date", "acceptance", "items"])
    f["acceptance"] = pd.to_datetime(f["acceptance"], utc=True, errors="coerce")
    return f.drop_duplicates("accn").sort_values(["cik", "acceptance"]).reset_index(drop=True)


def filings_path() -> str:
    return os.path.join(r10_dir(), "filings.parquet")


# --------------------------------------------------------------------------- #
# Announcement events
# --------------------------------------------------------------------------- #
def event_day0(acceptance: pd.Series, calendar) -> np.ndarray:
    """Session index (into ``calendar``, sorted 'YYYY-MM-DD') of the first
    session whose close follows the acceptance time: the acceptance date if it
    is a session and New York time < 16:00, else the next session.
    len(calendar) if beyond the calendar."""
    cal = np.asarray(calendar).astype(str)
    et = pd.DatetimeIndex(acceptance).tz_convert("America/New_York")
    day = np.asarray(et.strftime("%Y-%m-%d")).astype(str)
    before_close = np.asarray(et.hour < MARKET_CLOSE_ET)
    pos = np.searchsorted(cal, day, side="left")
    is_session = (pos < len(cal)) & (cal[np.minimum(pos, len(cal) - 1)] == day)
    return np.where(is_session & before_close, pos, np.searchsorted(cal, day, side="right"))


def announcement_events(filings: pd.DataFrame, calendar) -> pd.DataFrame:
    """One row per (cik, day0): 8-K 2.02 events, plus 10-Q/10-K fallbacks with
    no 8-K 2.02 accepted in the preceding FALLBACK_LOOKBACK_DAYS (past only).
    Columns: cik, day0 (session index), src10q, acceptance, avail_pos
    (= day0 + EAR_LAG_SESSIONS), prev_day0 (the CIK's previous event)."""
    f = filings.dropna(subset=["acceptance"]).copy()
    is_ea = (f["form"].isin(EAR_FORMS)) & f["items"].astype(str).str.split(",").apply(
        lambda xs: any(x.strip() == "2.02" for x in xs))
    ea = f[is_ea].assign(src10q=0)
    fb = f[f["form"].isin(FALLBACK_FORMS)].assign(src10q=1)
    if len(fb) and len(ea):
        e2 = ea[["cik", "acceptance"]].rename(columns={"acceptance": "last_ea"}).sort_values("last_ea")
        fb = pd.merge_asof(fb.sort_values("acceptance"), e2, left_on="acceptance", right_on="last_ea",
                           by="cik", direction="backward")
        recent = (fb["acceptance"] - fb["last_ea"]) <= pd.Timedelta(days=FALLBACK_LOOKBACK_DAYS)
        fb = fb[~recent.fillna(False).to_numpy()].drop(columns="last_ea")
    ev = pd.concat([ea, fb], ignore_index=True)
    ev["day0"] = event_day0(ev["acceptance"], calendar)
    ev = ev.sort_values(["cik", "day0", "src10q", "acceptance"]).drop_duplicates(["cik", "day0"])
    ev["avail_pos"] = ev["day0"] + EAR_LAG_SESSIONS
    ev["prev_day0"] = ev.groupby("cik")["day0"].shift(1)
    return ev[["cik", "day0", "src10q", "acceptance", "avail_pos", "prev_day0"]].reset_index(drop=True)


def market_index(daily: pd.DataFrame, calendar) -> np.ndarray:
    """Cumulative EW-eligible market index (log) per session of ``calendar``."""
    el = daily[daily["eligible"].astype(bool)]
    m = el.assign(r=el["daily_return"].astype(float).clip(*RET_CLIP)).groupby("Date")["r"].mean()
    m = m.reindex(np.asarray(calendar)).fillna(0.0).to_numpy()
    return np.cumsum(np.log1p(m))


def ear_values(pairs: pd.DataFrame, close: np.ndarray, vol: np.ndarray, col: dict, mkt_cum: np.ndarray) -> pd.DataFrame:
    """EAR / abnormal volume for (ticker, day0) ``pairs``. ``close`` / ``vol``:
    session x ticker matrices (NaN where no bar), ``col``: ticker -> column."""
    T = close.shape[0]
    j = pairs["ticker"].map(col).to_numpy()
    p = pairs["day0"].to_numpy().astype(np.int64)
    ok = pd.notna(j) & (p - 2 >= 0) & (p + 1 < T)
    jj = np.where(ok, j, 0).astype(np.int64)
    a, b = np.clip(p - 2, 0, T - 1), np.clip(p + 1, 0, T - 1)
    c0, c1 = close[a, jj], close[b, jj]
    with np.errstate(invalid="ignore", divide="ignore"):
        ear = np.log(c1 / c0) - (mkt_cum[b] - mkt_cum[a])
    ear = np.where(ok & np.isfinite(ear), ear, np.nan)
    v = np.nan_to_num(vol)
    n = np.isfinite(vol).astype(float)
    cv = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)])
    cn = np.vstack([np.zeros((1, n.shape[1])), np.cumsum(n, axis=0)])

    def wmean(lo, hi):                     # sessions [lo, hi] inclusive
        lo_c, hi_c = np.clip(lo, 0, T), np.clip(hi + 1, 0, T)
        s, k = cv[hi_c, jj] - cv[lo_c, jj], cn[hi_c, jj] - cn[lo_c, jj]
        return s, k
    se, ke = wmean(p - 1, p + 1)
    sb, kb = wmean(p - VOL_BASE[0], p - VOL_BASE[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        ab = np.log((se / ke) / (sb / kb))
    ab = np.where(ok & (ke >= 2) & (kb >= VOL_BASE_MIN) & (p - VOL_BASE[0] >= 0) & np.isfinite(ab), ab, np.nan)
    return pd.DataFrame({"ear": ear, "ear_abvol": ab}, index=pairs.index)


# --------------------------------------------------------------------------- #
# SUE from companyfacts
# --------------------------------------------------------------------------- #
_EPS_UNIT = "USD/shares"


def _extract_r10(cf: dict) -> list[tuple]:
    """(avail_ord, filed_ord, item, rank, start, end, val) for EPS, net income,
    revenue (flows) and weighted diluted shares."""
    facts = cf.get("facts", {}) or {}
    spec = [("eps", EPS_CONCEPTS, _EPS_UNIT), ("ni", fu.FLOW_ITEMS["net_income"], "USD"),
            ("rev", fu.FLOW_ITEMS["revenue"], "USD"), ("wdil", fu.FLOW_ITEMS["wdil"], "shares")]
    out = []
    for item, concepts, unit in spec:
        for rank, (tax, concept) in enumerate(concepts):
            node = facts.get(tax, {}).get(concept)
            if not node:
                continue
            for f in (node.get("units", {}) or {}).get(unit, ()) or ():
                if f.get("form") not in fu.ALLOWED_FORMS or f.get("val") is None or not f.get("filed") \
                        or not f.get("start"):
                    continue
                v = float(f["val"])
                if item == "wdil" and v <= 0:
                    continue
                out.append((fu._avail_ord(f["filed"]), fu._ord(f["filed"]), item, rank,
                            fu._ord(f["start"]), fu._ord(f["end"]), v))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def _shares_at(st: "fu._FlowState", e: int) -> float | None:
    starts = st.by_end.get(e)
    if not starts:
        return None
    best = min(starts.items(), key=lambda kv: (0 if fu.Q_MIN <= e - kv[0] <= fu.Q_MAX else 1, e - kv[0]))
    return best[1] if best[1] > 0 else None


def _q_eps(st: dict, e: int) -> tuple[float | None, int]:
    """Quarter EPS ending at ``e`` as known now: (value, source) with source 0 =
    an EPS concept, 1 = net income / weighted diluted shares."""
    v = st["eps"].quarter(e)
    if v is not None:
        return v, 0
    ni = st["ni"].quarter(e)
    sh = _shares_at(st["wdil"], e)
    if ni is not None and sh:
        return ni / sh, 1
    return None, -1


def _prior_year_end(ends: list, e: int, ok) -> int | None:
    cand = [p for p in ends if 350 <= e - p <= 380 and ok(p)]
    return min(cand, key=lambda p: abs(e - 364 - p)) if cand else None


def _std_prev(hist: list, e: int) -> float:
    xs = [d for (end, d) in hist if end < e and e - end <= SUE_HIST_MAX_DAYS][-SUE_HIST:]
    if len(xs) < SUE_MIN_HIST:
        return np.nan
    s = float(np.std(xs, ddof=1))
    return s if s > 0 else np.nan


def build_company_sue(cf: dict, cik: int | None = None) -> pd.DataFrame:
    """SUE event table for one companyfacts document: one row per newly
    available quarter (cik, asof_date, qend, sue_eps, sue_eps_chg, sue_streak,
    surge, eps_q, eps_src). Every value uses only facts with avail <= asof_date."""
    from datetime import date
    cik = int(cik if cik is not None else cf.get("cik"))
    facts = _extract_r10(cf)
    st = {k: fu._FlowState() for k in ("eps", "ni", "rev", "wdil")}
    hist_d, hist_r = [], []
    last_e, sue_prev, streak, prev_sign = None, np.nan, 0, 0
    rows, i, n = [], 0, len(facts)
    while i < n:
        avail = facts[i][0]
        while i < n and facts[i][0] == avail:
            _, _, item, rank, s, e, v = facts[i]
            st[item].update(rank, s, e, v)
            i += 1
        ends = sorted(set(st["eps"].ends) | set(st["ni"].ends))
        cand = [e for e in ends if (last_e is None or e > last_e) and _q_eps(st, e)[0] is not None]
        if not cand:
            continue
        e = cand[-1]
        q, src = _q_eps(st, e)
        e4 = _prior_year_end(ends, e, lambda p: _q_eps(st, p)[0] is not None)
        d = q - _q_eps(st, e4)[0] if e4 is not None else np.nan
        sd = _std_prev(hist_d, e)
        sue = d / sd if np.isfinite(d) and np.isfinite(sd) else np.nan
        # revenue surprise
        rq = st["rev"].quarter(e)
        r4e = _prior_year_end(st["rev"].ends, e, lambda p: st["rev"].quarter(p) is not None)
        rd = rq - st["rev"].quarter(r4e) if rq is not None and r4e is not None else np.nan
        rsd = _std_prev(hist_r, e)
        surge = rd / rsd if np.isfinite(rd) and np.isfinite(rsd) else np.nan
        # streak of same-sign surprises over consecutive quarters
        consecutive = last_e is not None and fu.Q_MIN <= e - last_e <= fu.Q_MAX
        sign = int(np.sign(d)) if np.isfinite(d) else 0
        if sign != 0 and consecutive and sign == prev_sign:
            streak = int(np.clip(streak + sign, -STREAK_CAP, STREAK_CAP))
        else:
            streak = sign
        rows.append((cik, avail, e, sue, sue - sue_prev if np.isfinite(sue) else np.nan, float(streak),
                     surge, q, src))
        if np.isfinite(d):
            hist_d.append((e, d))
        if np.isfinite(rd):
            hist_r.append((e, rd))
        if np.isfinite(sue):
            sue_prev = sue
        elif not consecutive:
            sue_prev = np.nan
        prev_sign, last_e = sign, e
    df = pd.DataFrame(rows, columns=["cik", "asof_date", "qend", "sue_eps", "sue_eps_chg", "sue_streak",
                                     "surge", "eps_q", "eps_src"])
    for c in ("asof_date", "qend"):
        df[c] = pd.to_datetime([date.fromordinal(int(x)) for x in df[c]]) if len(df) else pd.Series(dtype="datetime64[ns]")
    return df


def _sue_chunk(args) -> pd.DataFrame:
    zpath, names = args
    parts = []
    with zipfile.ZipFile(zpath) as z:
        for nm in names:
            try:
                cf = _loads(z.read(nm))
                df = build_company_sue(cf, int(nm[3:13]))
                if len(df):
                    parts.append(df)
            except Exception as e:  # noqa: BLE001 -- keep going on a malformed company
                print(f"[r10] skip {nm}: {e}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_sue_events(ciks, zpath: str | None = None, workers: int = 8, log=print) -> pd.DataFrame:
    zpath = zpath or os.path.join(data_dir("edgar"), "companyfacts.zip")
    with zipfile.ZipFile(zpath) as z:
        have = set(z.namelist())
    names = sorted(n for n in (f"CIK{int(c):010d}.json" for c in set(ciks)) if n in have)
    chunks = [(zpath, names[i:i + 100]) for i in range(0, len(names), 100)]
    parts = []
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k, df in enumerate(ex.map(_sue_chunk, chunks), 1):
            if len(df):
                parts.append(df)
            if k % 10 == 0:
                log(f"[r10] SUE {k}/{len(chunks)} chunks ({time.time() - t0:.0f}s)")
    out = pd.concat(parts, ignore_index=True).sort_values(["cik", "asof_date"]).reset_index(drop=True)
    out.to_parquet(os.path.join(r10_dir(), "sue_events.parquet"), index=False)
    log(f"[r10] SUE events: {len(out):,} rows, {out['cik'].nunique():,} CIKs, sue finite {out['sue_eps'].notna().mean():.1%}")
    return out


# --------------------------------------------------------------------------- #
# Weekly features
# --------------------------------------------------------------------------- #
def sue_weekly(rows: pd.DataFrame, sue: pd.DataFrame) -> pd.DataFrame:
    """Latest SUE event with asof_date <= Date for each (cik, Date) row."""
    r = rows[["cik", "Date"]].copy()
    r["_i"] = np.arange(len(r))
    r["_d"] = pd.to_datetime(r["Date"]).astype("datetime64[ns]")
    m = r[r["cik"].notna()].assign(cik=lambda x: x["cik"].astype("int64")).sort_values("_d")
    e = sue.assign(_a=pd.to_datetime(sue["asof_date"]).astype("datetime64[ns]"),
                   cik=sue["cik"].astype("int64")).sort_values("_a")
    j = pd.merge_asof(m, e[["cik", "_a"] + SUE_COLUMNS[:-1]], left_on="_d", right_on="_a", by="cik",
                      direction="backward")
    j["sue_days"] = (j["_d"] - j["_a"]).dt.days.astype(float)
    j.loc[j["sue_days"] > MAX_AGE_DAYS, SUE_COLUMNS] = np.nan
    out = pd.DataFrame(np.nan, index=np.arange(len(r)), columns=SUE_COLUMNS)
    out.loc[j["_i"].to_numpy(), SUE_COLUMNS] = j[SUE_COLUMNS].to_numpy(dtype=float)
    return out


def ear_weekly(rows: pd.DataFrame, events: pd.DataFrame, calendar, close, vol, col, mkt_cum) -> pd.DataFrame:
    """Latest usable announcement (avail_pos <= Date's session) per (ticker,
    cik, Date) row -> EAR_COLUMNS."""
    cal = np.asarray(calendar).astype(str)
    r = rows[["ticker", "cik", "Date"]].copy()
    r["_i"] = np.arange(len(r))
    r["_p"] = np.searchsorted(cal, r["Date"].astype(str).to_numpy(), side="right") - 1
    m = r[r["cik"].notna()].assign(cik=lambda x: x["cik"].astype("int64")).sort_values("_p")
    e = events.assign(cik=events["cik"].astype("int64")).sort_values("avail_pos")
    e = e[e["avail_pos"] < len(cal)]
    j = pd.merge_asof(m, e[["cik", "avail_pos", "day0", "prev_day0", "src10q"]], left_on="_p",
                      right_on="avail_pos", by="cik", direction="backward")
    j = j[j["day0"].notna()]
    cur = ear_values(j[["ticker"]].assign(day0=j["day0"].astype(np.int64)), close, vol, col, mkt_cum)
    pv = j["prev_day0"].notna()
    prev = pd.Series(np.nan, index=j.index)
    if pv.any():
        prev[pv] = ear_values(j.loc[pv, ["ticker"]].assign(day0=j.loc[pv, "prev_day0"].astype(np.int64)),
                              close, vol, col, mkt_cum)["ear"]
    d0 = pd.to_datetime(pd.Series(cal[j["day0"].astype(np.int64).to_numpy()], index=j.index))
    days = (pd.to_datetime(j["Date"]) - d0).dt.days.astype(float)
    out = pd.DataFrame(np.nan, index=np.arange(len(r)), columns=EAR_COLUMNS)
    vals = np.column_stack([cur["ear"], prev, cur["ear_abvol"], days, j["src10q"].astype(float)])
    vals[~np.isfinite(vals[:, 0]) | (vals[:, 3] > MAX_AGE_DAYS), :] = np.nan   # no EAR / stale -> block missing
    out.loc[j["_i"].to_numpy(), EAR_COLUMNS] = vals
    return out


def price_matrices(log=print):
    """(calendar, close, volume, ticker->col, market cum log index) over the
    dataset_edgar store (same ticker ids as the weekly frame)."""
    from .prices import load_prices
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(ds.v3_dir(ds.EDGAR_DATASET), "stage_a", "*.parquet")))
    daily = pd.concat([pd.read_parquet(f, columns=["ticker", "Date", "company_close", "daily_return", "eligible"])
                       for f in files], ignore_index=True)
    cal = ds.session_calendar(daily["Date"])
    mkt = market_index(daily, cal)
    tick = np.array(sorted(daily["ticker"].unique()))
    col = {t: k for k, t in enumerate(tick)}
    ri = np.searchsorted(cal, daily["Date"].to_numpy().astype(str))
    ci = daily["ticker"].map(col).to_numpy()
    close = np.full((len(cal), len(tick)), np.nan, dtype=np.float64)
    close[ri, ci] = daily["company_close"].to_numpy(dtype=float)
    del daily
    pr = load_prices(columns=["company_volume"], with_raw=False)
    pr = pr[pr["ticker"].isin(col)]
    vol = np.full_like(close, np.nan, dtype=np.float32)
    pos = np.searchsorted(cal, pr["Date"].astype(str).to_numpy())
    okp = (pos < len(cal)) & (cal[np.minimum(pos, len(cal) - 1)] == pr["Date"].astype(str).to_numpy())
    vol[pos[okp], pr["ticker"].map(col).to_numpy()[okp]] = pr["company_volume"].to_numpy(dtype=float)[okp]
    log(f"[r10] price matrices {close.shape}, volume matched {np.isfinite(vol).sum() / np.isfinite(close).sum():.1%} "
        f"({time.time() - t0:.0f}s)")
    return cal, close, vol, col, mkt


def weekly_features(log=print, out_path: str | None = None) -> pd.DataFrame:
    """Raw SUE + EAR features for every (ticker, weekly date) row of round 8's
    ``alt_weekly.parquet`` -> ``alt10_weekly.parquet``."""
    t0 = time.time()
    rows = pd.read_parquet(os.path.join(ds.v3_dir(), "alt_weekly.parquet"),
                           columns=["ticker", "Date", "eligible", "fwd_ret_30", "cik"])
    sue = pd.read_parquet(os.path.join(r10_dir(), "sue_events.parquet"))
    rows[SUE_COLUMNS] = sue_weekly(rows, sue).to_numpy()
    log(f"[r10] SUE joined: sue_eps finite {rows['sue_eps'].notna().mean():.1%} ({time.time() - t0:.0f}s)")
    cal, close, vol, col, mkt = price_matrices(log=log)
    ev = announcement_events(pd.read_parquet(filings_path()), cal)
    ev.to_parquet(os.path.join(r10_dir(), "events.parquet"), index=False)
    rows[EAR_COLUMNS] = ear_weekly(rows, ev, cal, close, vol, col, mkt).to_numpy()
    log(f"[r10] EAR joined: ear finite {rows['ear'].notna().mean():.1%} ({time.time() - t0:.0f}s)")
    for c in SUE_COLUMNS + EAR_COLUMNS:
        rows[c] = rows[c].astype("float32")
    rows.to_parquet(out_path or os.path.join(ds.v3_dir(), "alt10_weekly.parquet"), index=False)
    return rows


def build_filings(log=print) -> pd.DataFrame:
    cm = pd.read_parquet(os.path.join(data_dir("fundamentals"), "cik_map.parquet"))
    t0 = time.time()
    f = parse_submissions(ciks=cm["cik"].dropna().unique())
    f.to_parquet(filings_path(), index=False)
    log(f"[r10] filings: {len(f):,} rows for {f['cik'].nunique():,} CIKs; 8-K 2.02 "
        f"{(f['items'].str.contains('2.02', regex=False) & (f['form'] == '8-K')).sum():,} ({time.time() - t0:.0f}s)")
    return f


if __name__ == "__main__":
    import sys
    if "filings" in sys.argv:
        build_filings()
    if "sue" in sys.argv:
        cm = pd.read_parquet(os.path.join(data_dir("fundamentals"), "cik_map.parquet"))
        build_sue_events(cm["cik"].dropna().unique())
    if "features" in sys.argv:
        weekly_features()

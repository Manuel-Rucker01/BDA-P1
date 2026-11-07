"""Round 9 free SEC alternative data (pre-registered in ``v3/gate.py``, round 9).

* **Form 13F institutional holdings** -- SEC "Form 13F Data Sets" (bulk zips:
  calendar quarters 2013Q2-2023Q4, then three-month filing windows). Original
  13F-HR holdings reports only (amendments and notices excluded), common-share
  positions (``SSHPRNAMTTYPE == SH``, no PUT/CALL). For report quarter P only
  filings made **on or before the 45-day deadline** count, and the whole
  quarter becomes **available the business day after the deadline** -- so a
  feature never mixes in late filers and never depends on the decision date
  beyond which quarter is the latest available.
* **Fails-to-deliver** -- SEC CNS fails files (semi-monthly ``cnsfailsYYYYMM{a,b}``,
  settlement date / CUSIP / symbol / quantity). A half-month file becomes
  **available ``FTD_LAG_DAYS`` calendar days after its last settlement day, next
  business day** (calibrated on the HTTP Last-Modified dates of the 2021-2026
  postings, see ``download_ftd``).
* **CUSIP -> ticker** (13F carries no ticker): the FTD files themselves pair
  CUSIP and trading symbol on each settlement date, so the link for quarter P
  uses only FTD records that were published by the quarter's availability date
  (point in time, symmetric for survivors and delisted names; no current-day
  ticker list is involved).

Raw files live under ``$BDA_DATA_V3/sec_13f/raw`` and ``$BDA_DATA_V3/sec_ftd/raw``.

    python -m DataAnalysisPipeline2.v3.altdata_r9 download
    python -m DataAnalysisPipeline2.v3.altdata_r9 parse
    python -m DataAnalysisPipeline2.v3.altdata_r9 features
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re
import time
import zipfile

import numpy as np
import pandas as pd

from . import dataset as ds
from ._alpaca import data_dir
from .altdata import _sec_headers, next_bday

F13_INDEX = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
FTD_INDEX = "https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data"
F13_FIRST_FILING_QUARTER = "2016q2"        # covers report quarters from 2016Q1 (decision dates from 2017)
FTD_FIRST_MONTH = "201601"
F13_DEADLINE_DAYS = 45
FTD_LAG_DAYS = 21                          # see download_ftd / report (max observed posting lag 2021-2026)
FTD_WINDOW_FILES = 6                       # trailing 3 months of half-month files
MAP_LOOKBACK_DAYS = 365
MIN_INTERVAL_S = 0.15                      # <= ~6.7 requests/s (SEC limit 10, project limit 8)

INST_COLUMNS = ["inst_breadth", "inst_dbreadth", "inst_io", "inst_dio", "inst_new_n", "inst_exit_n"]
FTD_COLUMNS = ["ftd_ratio", "ftd_ratio_3m", "ftd_persist_3m"]
BLOCKS = {"inst13f": INST_COLUMNS, "ftd": FTD_COLUMNS}


# --------------------------------------------------------------------------- #
# Downloads (public files only; polite rate, retries)
# --------------------------------------------------------------------------- #
class _Rate:
    def __init__(self, min_interval: float = MIN_INTERVAL_S):
        self.min_interval, self.t = min_interval, 0.0

    def wait(self):
        d = time.time() - self.t
        if d < self.min_interval:
            time.sleep(self.min_interval - d)
        self.t = time.time()


def _get(session, url: str, headers: dict, rate: _Rate, method: str = "get", tries: int = 5, **kw):
    for k in range(tries):
        rate.wait()
        try:
            r = getattr(session, method)(url, headers=headers, timeout=300, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                raise IOError(f"HTTP {r.status_code}")
            return r
        except Exception:                        # noqa: BLE001 -- retried with backoff
            if k == tries - 1:
                raise
            time.sleep(2.0 * 2 ** k)


def _index_links(session, url: str, pattern: str, headers: dict, rate: _Rate) -> dict:
    """{file name: absolute url} of the zips linked from an SEC index page."""
    r = _get(session, url, headers, rate)
    r.raise_for_status()
    out = {}
    for href in re.findall(r'href="([^"]+\.zip)"', r.text):
        name = href.rsplit("/", 1)[-1]
        if re.fullmatch(pattern, name):
            out[name] = href if href.startswith("http") else "https://www.sec.gov" + href
    return out


def f13_window(name: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Filing-date window covered by a 13F data-set zip name."""
    m = re.fullmatch(r"(\d{4})q([1-4])_form13f\.zip", name)
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        s = pd.Timestamp(y, 3 * q - 2, 1)
        return s, s + pd.offsets.QuarterEnd(0)
    m = re.fullmatch(r"(\d{2}[a-z]{3}\d{4})-(\d{2}[a-z]{3}\d{4})_form13f\.zip", name)
    if m:
        return pd.to_datetime(m.group(1), format="%d%b%Y"), pd.to_datetime(m.group(2), format="%d%b%Y")
    return None


def ftd_period(name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Calendar half-month covered by ``cnsfailsYYYYMM{a,b}.zip``."""
    m = re.fullmatch(r"cnsfails(\d{4})(\d{2})([ab])\.zip", name)
    y, mo, h = int(m.group(1)), int(m.group(2)), m.group(3)
    s = pd.Timestamp(y, mo, 1)
    return (s, pd.Timestamp(y, mo, 15)) if h == "a" else (pd.Timestamp(y, mo, 16), s + pd.offsets.MonthEnd(0))


def ftd_avail(name: str, posted: str | None = None, lag_days: int = FTD_LAG_DAYS) -> pd.Timestamp:
    """Business day after max(last day of the half-month + ``lag_days``, the
    file's observed posting date when that is a plausible first posting
    (< 90 days after the period; older Last-Modified stamps are re-uploads))."""
    end = ftd_period(name)[1]
    d = end + pd.Timedelta(days=lag_days)
    if posted:
        p = pd.to_datetime(posted, format="%a, %d %b %Y %H:%M:%S GMT", errors="coerce")
        if pd.notna(p) and 0 < (p.normalize() - end).days < 90:
            d = max(d, p.normalize())
    return pd.Timestamp(next_bday(pd.Series([d]), 1).iloc[0])


def download_13f(log=print) -> list[str]:
    import requests
    out, h, rate, s = data_dir("sec_13f", "raw"), _sec_headers(), _Rate(), requests.Session()
    links = _index_links(s, F13_INDEX, r".+_form13f\.zip", h, rate)
    first = pd.Period(F13_FIRST_FILING_QUARTER.replace("q", "Q"), freq="Q").start_time
    got = []
    for name, url in sorted(links.items(), key=lambda kv: f13_window(kv[0])[0] if f13_window(kv[0]) else pd.Timestamp(0)):
        w = f13_window(name)
        if w is None or w[1] < first:
            continue
        p = os.path.join(out, name)
        if os.path.exists(p) and os.path.getsize(p) > 1e6:
            got.append(p)
            continue
        r = _get(s, url, h, rate)
        if r.status_code == 404:
            log(f"[13f] {name}: 404")
            continue
        r.raise_for_status()
        open(p + ".part", "wb").write(r.content)
        zipfile.ZipFile(p + ".part").testzip()
        os.replace(p + ".part", p)
        got.append(p)
        log(f"[13f] {name}: {len(r.content) / 1e6:.0f} MB")
    return got


def download_ftd(log=print) -> list[str]:
    """All half-month files from FTD_FIRST_MONTH; records each file's HTTP
    Last-Modified (the posting date for files not re-uploaded) in
    ``sec_ftd/posted.json`` -- used to calibrate FTD_LAG_DAYS."""
    import requests
    out, h, rate, s = data_dir("sec_ftd", "raw"), _sec_headers(), _Rate(), requests.Session()
    links = _index_links(s, FTD_INDEX, r"cnsfails\d{6}[ab]\.zip", h, rate)
    meta_p = os.path.join(data_dir("sec_ftd"), "posted.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    got = []
    for name in sorted(links):
        if name[8:14] < FTD_FIRST_MONTH:
            continue
        p = os.path.join(out, name)
        if not (os.path.exists(p) and os.path.getsize(p) > 1e4):
            r = _get(s, links[name], h, rate)
            if r.status_code == 404:
                log(f"[ftd] {name}: 404")
                continue
            r.raise_for_status()
            open(p + ".part", "wb").write(r.content)
            zipfile.ZipFile(p + ".part").testzip()
            os.replace(p + ".part", p)
            meta[name] = r.headers.get("Last-Modified")
        elif name not in meta:
            meta[name] = _get(s, links[name], h, rate, method="head").headers.get("Last-Modified")
        got.append(p)
    json.dump(meta, open(meta_p, "w"), indent=1)
    log(f"[ftd] {len(got)} files")
    return got


def posting_lags(meta: dict, since: str = "2021-01") -> pd.Series:
    """Calendar days from a half-month's last day to its HTTP Last-Modified date,
    for files whose Last-Modified is a plausible first posting (< 90 days)."""
    out = {}
    for name, lm in meta.items():
        if not lm or name[8:12] + "-" + name[12:14] < since:
            continue
        posted = pd.to_datetime(lm, format="%a, %d %b %Y %H:%M:%S GMT", errors="coerce")
        if pd.isna(posted):
            continue
        lag = (posted.normalize() - ftd_period(name)[1]).days
        if 0 < lag < 90:
            out[name] = lag
    return pd.Series(out, dtype=float).sort_index()


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def parse_ftd_zip(path_or_buf, name: str | None = None) -> pd.DataFrame:
    """One half-month CNS fails file -> [settlement_date, cusip, symbol, qty,
    price, file]. Trailer lines and malformed rows dropped."""
    z = zipfile.ZipFile(path_or_buf)
    inner = [n for n in z.namelist() if not n.endswith("/")][0]
    df = pd.read_csv(z.open(inner), sep="|", dtype=str, encoding="latin-1", on_bad_lines="skip", quoting=3)
    df.columns = [c.strip().upper() for c in df.columns]
    qcol = [c for c in df.columns if c.startswith("QUANTITY")][0]
    out = pd.DataFrame({
        "settlement_date": pd.to_datetime(df["SETTLEMENT DATE"].str.strip(), format="%Y%m%d", errors="coerce"),
        "cusip": df["CUSIP"].str.strip().str.upper(),
        "symbol": df["SYMBOL"].fillna("").str.strip().str.upper().str.replace("/", ".", regex=False)
                    .str.replace("-", ".", regex=False),
        "qty": pd.to_numeric(df[qcol], errors="coerce"),
        "price": pd.to_numeric(df.get("PRICE"), errors="coerce"),
    }).dropna(subset=["settlement_date", "qty"])
    out = out[out["cusip"].str.len() == 9]
    out["file"] = name or os.path.basename(str(path_or_buf))
    return out.reset_index(drop=True)


def f13_deadline(period_end) -> pd.Series:
    """Filing deadline (45 days after quarter end, rolled forward to a business day)."""
    d = pd.to_datetime(pd.Series(period_end)) + pd.Timedelta(days=F13_DEADLINE_DAYS)
    return next_bday(d - pd.Timedelta(days=1), 1)       # on-or-after: roll forward from the day before


def f13_avail(period_end) -> pd.Series:
    """The quarter's holdings become usable the business day after the deadline."""
    return next_bday(f13_deadline(period_end), 1)


def parse_13f_zip(path_or_buf, cusips: set | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (filings [accession, cik, filing_date, period, is_original],
           holdings [accession, cusip, shares]) of one 13F data-set zip.
    ``cusips`` restricts holdings (memory); shares summed per (accession, cusip)."""
    z = zipfile.ZipFile(path_or_buf)
    names = {n.rsplit("/", 1)[-1].upper(): n for n in z.namelist()}      # some zips nest a folder
    rd = lambda n, cols: pd.read_csv(z.open(names[n.upper()]), sep="\t", dtype=str, usecols=lambda c: c in cols, quoting=3,
                                     on_bad_lines="skip", encoding="latin-1")
    sub = rd("SUBMISSION.tsv", {"ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"})
    cov = rd("COVERPAGE.tsv", {"ACCESSION_NUMBER", "ISAMENDMENT", "REPORTTYPE"})
    f = sub.merge(cov, on="ACCESSION_NUMBER", how="left")
    rt = f["REPORTTYPE"].fillna("").str.upper()
    f = pd.DataFrame({
        "accession": f["ACCESSION_NUMBER"],
        "cik": pd.to_numeric(f["CIK"], errors="coerce").astype("Int64"),
        "filing_date": pd.to_datetime(f["FILING_DATE"], format="%d-%b-%Y", errors="coerce"),
        "period": pd.to_datetime(f["PERIODOFREPORT"], format="%d-%b-%Y", errors="coerce"),
        "is_original": (f["SUBMISSIONTYPE"].str.strip() == "13F-HR")
                       & (f["ISAMENDMENT"].fillna("N").str.upper().str[:1] != "Y")
                       & (rt.str.contains("HOLDINGS") | rt.str.contains("COMBINATION")),
    }).dropna(subset=["cik", "filing_date", "period"])
    info = rd("INFOTABLE.tsv", {"ACCESSION_NUMBER", "CUSIP", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"})
    info = info[info["ACCESSION_NUMBER"].isin(set(f.loc[f["is_original"], "accession"]))]
    info = info[(info["SSHPRNAMTTYPE"].fillna("").str.strip().str.upper() == "SH")
                & (info["PUTCALL"].fillna("").str.strip() == "")]
    info["cusip"] = info["CUSIP"].fillna("").str.strip().str.upper()
    if cusips is not None:
        info = info[info["cusip"].isin(cusips)]
    info["shares"] = pd.to_numeric(info["SSHPRNAMT"], errors="coerce")
    h = (info.dropna(subset=["shares"]).groupby(["ACCESSION_NUMBER", "cusip"], as_index=False)["shares"].sum()
         .rename(columns={"ACCESSION_NUMBER": "accession"}))
    return f.reset_index(drop=True), h


# --------------------------------------------------------------------------- #
# Cached parsed stores
# --------------------------------------------------------------------------- #
def load_ftd(refresh: bool = False) -> pd.DataFrame:
    """All parsed FTD records with each file's availability date (cached)."""
    p = os.path.join(data_dir("sec_ftd"), "ftd.parquet")
    if os.path.exists(p) and not refresh:
        return pd.read_parquet(p)
    mp = os.path.join(data_dir("sec_ftd"), "posted.json")
    posted = json.load(open(mp)) if os.path.exists(mp) else {}
    parts = []
    for f in sorted(glob.glob(os.path.join(data_dir("sec_ftd", "raw"), "cnsfails*.zip"))):
        name = os.path.basename(f)
        x = parse_ftd_zip(f, name)
        x["avail_date"] = ftd_avail(name, posted.get(name))
        parts.append(x)
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["settlement_date", "cusip", "symbol"], keep="last")
    out.to_parquet(p, index=False)
    return out


def load_13f(cusips: set | None = None, refresh: bool = False, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(filings, holdings) over every raw zip; holdings restricted to the
    deadline-filtered original filings (cached)."""
    pf, ph = os.path.join(data_dir("sec_13f"), "filings.parquet"), os.path.join(data_dir("sec_13f"), "holdings.parquet")
    if os.path.exists(pf) and os.path.exists(ph) and not refresh:
        return pd.read_parquet(pf), pd.read_parquet(ph)
    fs, hs = [], []
    for f in sorted(glob.glob(os.path.join(data_dir("sec_13f", "raw"), "*_form13f.zip"))):
        t0 = time.time()
        a, b = parse_13f_zip(f, cusips)
        fs.append(a)
        hs.append(b)
        log(f"[13f] {os.path.basename(f)}: {len(a):,} filings, {len(b):,} holdings ({time.time() - t0:.0f}s)")
    filings = pd.concat(fs, ignore_index=True).drop_duplicates("accession")
    filings = select_filings(filings)
    hold = pd.concat(hs, ignore_index=True).drop_duplicates(["accession", "cusip"])
    hold = hold[hold["accession"].isin(set(filings["accession"]))]
    hold = hold.astype({"accession": "category", "cusip": "category"}).reset_index(drop=True)
    filings.to_parquet(pf, index=False)
    hold.to_parquet(ph, index=False)
    return filings, hold


def select_filings(filings: pd.DataFrame) -> pd.DataFrame:
    """Original holdings reports filed on or before the quarter's deadline; one
    per (filer, quarter): the earliest (ties: smallest accession)."""
    f = filings[filings["is_original"].astype(bool)].copy()
    f = f[f["period"] == f["period"] + pd.offsets.QuarterEnd(0)]       # calendar quarter ends only
    f["deadline"] = f13_deadline(f["period"]).to_numpy()
    f = f[f["filing_date"] <= f["deadline"]]
    f = f.sort_values(["cik", "period", "filing_date", "accession"]).drop_duplicates(["cik", "period"], keep="first")
    f["avail_date"] = f13_avail(f["period"]).to_numpy()
    return f.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# CUSIP <-> ticker (point in time, from FTD records)
# --------------------------------------------------------------------------- #
def sym_key(s: pd.Series) -> pd.Series:
    """Symbol match key: FTD writes share classes without the dot (BRKB), the
    store with it (BRK.B); ticker ids drop the ``~n`` reuse suffix."""
    return s.astype(str).str.split("~").str[0].str.upper().str.replace(".", "", regex=False)


def cusip_links(ftd: pd.DataFrame, tickers: pd.DataFrame, as_of: pd.Timestamp, ref: pd.Timestamp,
                lookback_days: int = MAP_LOOKBACK_DAYS) -> pd.DataFrame:
    """[ticker, cusip] for ticker ids (``tickers``: ticker, first_date) using
    FTD records published by ``as_of`` with settlement in (ref - lookback,
    as_of] and on/after the ticker id's first bar. Per ticker the CUSIP whose
    latest record is closest to ``ref`` (latest settlement <= ref, else the
    earliest after it); a CUSIP claimed by several tickers keeps the one with
    the most records in the window (ties: dropped)."""
    f = ftd[(ftd["avail_date"] <= as_of) & (ftd["settlement_date"] > ref - pd.Timedelta(days=lookback_days))
            & (ftd["settlement_date"] <= as_of)]
    t = tickers.assign(skey=sym_key(tickers["ticker"]))
    m = f.assign(skey=sym_key(f["symbol"])).merge(t[["ticker", "skey", "first_date"]], on="skey")
    m = m[m["settlement_date"] >= pd.to_datetime(m["first_date"])]
    if m.empty:
        return pd.DataFrame(columns=["ticker", "cusip"])
    m = m.assign(_before=m["settlement_date"].where(m["settlement_date"] <= ref))
    g = m.groupby(["ticker", "cusip"]).agg(last_before=("_before", "max"), first=("settlement_date", "min"),
                                           n=("settlement_date", "size")).reset_index()
    before = g["last_before"].notna()
    g["key"] = np.where(before, (ref - g["last_before"]).dt.days, 10_000 + (g["first"] - ref).dt.days)
    g = g.sort_values(["ticker", "key", "cusip"]).drop_duplicates("ticker", keep="first")
    g = g.sort_values(["cusip", "n"], ascending=[True, False])
    dup = g.duplicated("cusip", keep=False)
    tie = dup & g.groupby("cusip")["n"].transform(lambda s: (s == s.max()).sum() > 1)
    g = g[~tie].drop_duplicates("cusip", keep="first")
    return g[["ticker", "cusip"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def quarter_holders(filings: pd.DataFrame, hold: pd.DataFrame, ftd: pd.DataFrame, tickers: pd.DataFrame,
                    periods) -> dict:
    """{period: (filer set, DataFrame[ticker, cik, shares])}: deadline-filtered
    holdings of each quarter mapped to ticker ids with the CUSIP links known at
    the quarter's availability date."""
    out = {}
    acc = pd.Index(filings["accession"].astype(str))
    ha = hold["accession"].astype("category")
    code = acc.get_indexer(ha.cat.categories.astype(str))[ha.cat.codes.to_numpy()]
    keep = code >= 0
    hm = pd.DataFrame({"cusip": hold["cusip"].astype("category").to_numpy()[keep],
                       "shares": hold["shares"].to_numpy()[keep],
                       "cik": filings["cik"].astype("int64").to_numpy()[code[keep]],
                       "period": filings["period"].to_numpy()[code[keep]]})
    by_p = {p: g for p, g in hm.groupby("period")}
    for P in periods:
        P = pd.Timestamp(P)
        fl = filings[filings["period"] == P]
        if fl.empty or P not in by_p:
            continue
        A = pd.Timestamp(fl["avail_date"].iloc[0])
        links = cusip_links(ftd, tickers, as_of=A, ref=P)
        h = by_p[P].merge(links, on="cusip")
        h = h.groupby(["ticker", "cik"], as_index=False)["shares"].sum()
        out[P] = (set(fl["cik"].astype(int)), h, set(links["ticker"]), A)
    return out


def inst_features_quarter(cur, prev, shares_cur: pd.Series, shares_prev: pd.Series) -> pd.DataFrame:
    """Per-ticker 13F features for quarter P (``cur``) given P-1 (``prev``);
    ``shares_*``: shares outstanding per ticker at each quarter end.
    Tickers linked at P but with no holder get breadth 0 (no institution holds
    them); tickers not linked are absent (missing -> hot-deck downstream)."""
    filers, h, linked, _ = cur
    idx = pd.Index(sorted(linked), name="ticker")
    n_f = max(len(filers), 1)
    br = h.groupby("ticker")["cik"].nunique().reindex(idx, fill_value=0)
    sh = h.groupby("ticker")["shares"].sum().reindex(idx, fill_value=0.0)
    so = shares_cur.reindex(idx)
    out = pd.DataFrame(index=idx)
    out["inst_breadth"] = br / n_f
    out["inst_io"] = (sh / so.where(so > 0)).clip(upper=2.0)
    out["inst_dbreadth"] = np.nan
    out["inst_dio"] = np.nan
    out["inst_new_n"] = np.nan
    out["inst_exit_n"] = np.nan
    if prev is not None:
        pfilers, ph, plinked, _ = prev
        cont = filers & pfilers
        both = idx.intersection(pd.Index(sorted(plinked)))
        hc = h[h["cik"].isin(cont)]
        pc = ph[ph["cik"].isin(cont)]
        cur_pairs = set(zip(hc["ticker"], hc["cik"]))
        prev_pairs = set(zip(pc["ticker"], pc["cik"]))
        new = pd.Series([t for t, _ in cur_pairs - prev_pairs], dtype=object).value_counts()
        ex = pd.Series([t for t, _ in prev_pairs - cur_pairs], dtype=object).value_counts()
        nb = hc.groupby("ticker")["cik"].nunique()
        pb = pc.groupby("ticker")["cik"].nunique()
        nc = max(len(cont), 1)
        out.loc[both, "inst_dbreadth"] = ((nb.reindex(both, fill_value=0) - pb.reindex(both, fill_value=0)) / nc).to_numpy()
        out.loc[both, "inst_new_n"] = np.log1p(new.reindex(both, fill_value=0)).to_numpy()
        out.loc[both, "inst_exit_n"] = np.log1p(ex.reindex(both, fill_value=0)).to_numpy()
        psh = ph.groupby("ticker")["shares"].sum().reindex(both, fill_value=0.0)
        pso = shares_prev.reindex(both)
        pio = (psh / pso.where(pso > 0)).clip(upper=2.0)
        out.loc[both, "inst_dio"] = (out.loc[both, "inst_io"] - pio).to_numpy()
    return out.reset_index()


def inst_weekly(rows: pd.DataFrame, filings: pd.DataFrame, hold: pd.DataFrame, ftd: pd.DataFrame,
                log=print) -> pd.DataFrame:
    """INST_COLUMNS for ``rows`` (ticker, Date, first_date, shares) + ``inst_mapped``:
    on date t the latest quarter P with avail(P) <= t (and P-1)."""
    tickers = rows.groupby("ticker", as_index=False)["first_date"].first()
    periods = sorted(filings["period"].unique())
    q = quarter_holders(filings, hold, ftd, tickers, periods)
    sh = rows[["ticker", "Date", "shares"]].assign(_d=pd.to_datetime(rows["Date"]).astype("datetime64[ns]"))
    sh = sh.dropna(subset=["shares"]).sort_values("_d")

    def shares_at(P):
        x = sh[sh["_d"] <= P]
        x = x[x["_d"] > P - pd.Timedelta(days=21)]
        return x.drop_duplicates("ticker", keep="last").set_index("ticker")["shares"]

    feats = []
    ps = [P for P in periods if P in q]
    for i, P in enumerate(ps):
        prev_p = P - pd.offsets.QuarterEnd(1)
        prev = q.get(prev_p)
        f = inst_features_quarter(q[P], prev, shares_at(P), shares_at(prev_p) if prev else pd.Series(dtype=float))
        f["period"], f["avail_date"] = P, q[P][3]
        feats.append(f)
    qf = pd.concat(feats, ignore_index=True)
    log(f"[13f] {len(ps)} quarters, {len(qf):,} ticker-quarters linked")
    avail = qf[["period", "avail_date"]].drop_duplicates().sort_values("avail_date")
    r = rows[["ticker", "Date"]].copy()
    r["_i"] = np.arange(len(r))
    r["_d"] = pd.to_datetime(r["Date"]).astype("datetime64[ns]")
    r = pd.merge_asof(r.sort_values("_d"), avail.assign(avail_date=avail["avail_date"].astype("datetime64[ns]")),
                      left_on="_d", right_on="avail_date", direction="backward")
    r = r.merge(qf.drop(columns="avail_date"), on=["ticker", "period"], how="left")
    out = pd.DataFrame(np.nan, index=np.arange(len(rows)), columns=INST_COLUMNS + ["inst_mapped", "inst_period"])
    out.loc[r["_i"].to_numpy(), INST_COLUMNS] = r[INST_COLUMNS].to_numpy()
    out.loc[r["_i"].to_numpy(), "inst_mapped"] = r["inst_breadth"].notna().to_numpy().astype(float)
    out["inst_period"] = pd.NaT
    out.loc[r["_i"].to_numpy(), "inst_period"] = r["period"].to_numpy()
    out.index = rows.index
    return out


def ftd_weekly(rows: pd.DataFrame, ftd: pd.DataFrame, n_files: int = FTD_WINDOW_FILES) -> pd.DataFrame:
    """FTD_COLUMNS + ``ftd_mapped`` for rows (ticker, Date, first_date, shares):
    on date t, the latest ``n_files`` half-month files available by t.
      ftd_ratio      mean daily fails (shares) over the latest file / shares out
      ftd_ratio_3m   same over the ``n_files`` latest files
      ftd_persist_3m share of settlement days in those files with a fail record
    Settlement days without a record count as zero fails. ``ftd_mapped`` = the
    symbol has any record (on/after the ticker id's first bar) in files
    available by t with settlement in the last MAP_LOOKBACK_DAYS days; rows not
    mapped get missing features (hot-deck downstream)."""
    files = (ftd.groupby("file").agg(avail=("avail_date", "first"), ndays=("settlement_date", "nunique"))
             .reset_index().sort_values("avail"))
    files["k"] = np.arange(len(files))
    f = ftd.merge(files[["file", "k"]], on="file")
    f = f[f["symbol"] != ""].assign(symbol=lambda x: sym_key(x["symbol"]))
    per = f.groupby(["symbol", "k"]).agg(q=("qty", "sum"), d=("settlement_date", "nunique"),
                                         first=("settlement_date", "min"), last=("settlement_date", "max")).reset_index()
    r = rows[["ticker", "Date", "first_date", "shares"]].copy()
    r["_i"] = np.arange(len(r))
    r["symbol"] = sym_key(r["ticker"])
    r["_d"] = pd.to_datetime(r["Date"]).astype("datetime64[ns]")
    fa = files["avail"].astype("datetime64[ns]").to_numpy()
    r["kmax"] = np.searchsorted(fa, r["_d"].to_numpy(), side="right") - 1
    out = pd.DataFrame(np.nan, index=np.arange(len(r)), columns=FTD_COLUMNS + ["ftd_mapped"])
    nd = files.set_index("k")["ndays"]
    per_by = per.set_index(["symbol", "k"])
    # mapping window (files with last settlement within lookback of t)
    last_settle = ftd.groupby("file")["settlement_date"].max()
    files["last_settle"] = files["file"].map(last_settle)
    for kmax, g in r.groupby("kmax"):
        if kmax < 0:
            continue
        ks = list(range(max(0, kmax - n_files + 1), kmax + 1))
        days_all = float(nd.reindex(ks).sum())
        t_min = g["_d"].min()
        kmap = files.loc[(files["k"] <= kmax)
                         & (files["last_settle"] > t_min - pd.Timedelta(days=MAP_LOOKBACK_DAYS)), "k"].tolist()
        sub = per[per["k"].isin(kmap) & per["symbol"].isin(set(g["symbol"]))]
        gg = g.merge(sub, on="symbol", how="left")
        gg = gg[gg["last"] >= pd.to_datetime(gg["first_date"])]
        mapped = set(gg["_i"])
        w3 = gg[gg["k"].isin(ks)]
        q3 = w3.groupby("_i")["q"].sum()
        d3 = w3.groupby("_i")["d"].sum()
        q1 = w3[w3["k"] == kmax].groupby("_i")["q"].sum()
        idx = g["_i"].to_numpy()
        m = np.array([i in mapped for i in idx])
        sh = g["shares"].to_numpy(dtype=float)
        sh = np.where(sh > 0, sh, np.nan)
        out.loc[idx, "ftd_mapped"] = m.astype(float)
        mi = idx[m]
        out.loc[mi, "ftd_ratio"] = q1.reindex(mi, fill_value=0.0).to_numpy() / max(float(nd.get(kmax, 1)), 1) / sh[m]
        out.loc[mi, "ftd_ratio_3m"] = q3.reindex(mi, fill_value=0.0).to_numpy() / max(days_all, 1) / sh[m]
        out.loc[mi, "ftd_persist_3m"] = d3.reindex(mi, fill_value=0.0).to_numpy() / max(days_all, 1)
    out.index = rows.index
    return out


def weekly_features(log=print, out_path: str | None = None) -> pd.DataFrame:
    """Raw 13F + FTD features for every (ticker, weekly date) row of the store
    (eligible or not) -> ``alt9_weekly.parquet`` (same row set as round 8's
    ``alt_weekly.parquet``)."""
    t0 = time.time()
    alt8 = pd.read_parquet(os.path.join(ds.v3_dir(), "alt_weekly.parquet"), columns=["ticker", "Date", "eligible",
                                                                                      "fwd_ret_30", "cik"])
    rows = _weekly_rows()
    rows = rows.merge(alt8, on=["ticker", "Date"], how="left", suffixes=("", "_8"))
    log(f"[r9] {len(rows):,} weekly rows ({time.time() - t0:.0f}s)")
    ftd = load_ftd()
    filings, hold = load_13f(set(ftd["cusip"].unique()), log=log)
    log(f"[r9] FTD {len(ftd):,} records, 13F {len(filings):,} filings / {len(hold):,} holdings ({time.time() - t0:.0f}s)")
    rows[FTD_COLUMNS + ["ftd_mapped"]] = ftd_weekly(rows, ftd).to_numpy()
    log(f"[r9] FTD mapped {rows['ftd_mapped'].mean():.1%} ({time.time() - t0:.0f}s)")
    inst = inst_weekly(rows, filings, hold, ftd, log=log)
    rows[INST_COLUMNS + ["inst_mapped"]] = inst[INST_COLUMNS + ["inst_mapped"]].to_numpy(dtype=float)
    rows["inst_period"] = inst["inst_period"].to_numpy()
    log(f"[r9] 13F mapped {rows['inst_mapped'].mean():.1%} ({time.time() - t0:.0f}s)")
    keep = ["ticker", "Date", "eligible", "fwd_ret_30", "cik", "inst_mapped", "ftd_mapped", "inst_period"] \
        + INST_COLUMNS + FTD_COLUMNS
    out = rows[keep].copy()
    for c in INST_COLUMNS + FTD_COLUMNS:
        out[c] = out[c].astype("float32")
    out.to_parquet(out_path or os.path.join(ds.v3_dir(), "alt9_weekly.parquet"), index=False)
    return out


def _weekly_rows() -> pd.DataFrame:
    """(ticker, Date, first_date, shares) on weekly decision dates -- the
    round-8 construction (shares = EDGAR mcap / raw close)."""
    cal = ds.calendar_from_dataset(ds.v3_dir(ds.EDGAR_DATASET))
    weekly = ds.weekly_decision_dates(cal)
    files = sorted(glob.glob(os.path.join(ds.v3_dir(ds.EDGAR_DATASET), "stage_a", "*.parquet")))
    rows = pd.concat([pd.read_parquet(f, columns=["ticker", "Date", "close_raw", "company_close"],
                                      filters=[("Date", "in", weekly)]) for f in files], ignore_index=True)
    first = pd.concat([pd.read_parquet(f, columns=["ticker", "Date"]).groupby("ticker")["Date"].min() for f in files])
    first = first.groupby(level=0).min()
    rows["first_date"] = rows["ticker"].map(first)
    mc = pd.concat([ds.load_daily(years=[y], dates=[d for d in weekly if d[:4] == str(y)],
                                  out_dir=ds.v3_dir(ds.EDGAR_DATASET), columns=["ticker", "Date", "log_mcap"])
                    for y in sorted({int(d[:4]) for d in weekly})], ignore_index=True)
    rows = rows.merge(mc, on=["ticker", "Date"], how="left")
    rows["shares"] = np.exp(rows["log_mcap"]) / rows["close_raw"].fillna(rows["company_close"])
    return rows[["ticker", "Date", "first_date", "shares"]]


if __name__ == "__main__":
    import sys
    if "download" in sys.argv:
        download_ftd()
        download_13f()
    if "parse" in sys.argv:
        f = load_ftd(refresh=True)
        load_13f(set(f["cusip"].unique()), refresh=True)
    if "features" in sys.argv:
        weekly_features()

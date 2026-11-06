"""Round 8 free alternative data (pre-registered in ``v3/gate.py``, round 8).

* **SEC insider transactions** -- quarterly "Insider Transactions Data Sets"
  (Forms 3/4/5) from sec.gov (bulk zips, User-Agent from ``SEC_USER_AGENT``,
  <= 2 requests/s). Open-market purchases (P) and sales (S) from
  NONDERIV_TRANS, keyed by ISSUERCIK; **available the business day after the
  filing date**.
* **FINRA short interest** -- bi-monthly files ``shrtYYYYMMDD.csv`` (settlement
  date) from FINRA's public CDN; exchange-listed rows only; **available
  settlement date + 8 business days**.

Raw files live under ``$BDA_DATA_V3/sec_insider/raw`` and ``$BDA_DATA_V3/finra_si/raw``.
Features are built on weekly decision dates (``weekly_features``).

    python -m DataAnalysisPipeline2.v3.altdata download
    python -m DataAnalysisPipeline2.v3.altdata features
"""
from __future__ import annotations

import datetime as dt
import glob
import io
import os
import time
import zipfile

import numpy as np
import pandas as pd

from . import dataset as ds
from ._alpaca import data_dir

SEC_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
FINRA_URL = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt{d}.csv"
INS_WINDOWS = (30, 90, 180)
SI_LAG_BDAYS = 8
INSIDER_COLUMNS = (["ins_buy_n_30", "ins_buy_n_90", "ins_buy_n_180", "ins_sell_n_30", "ins_sell_n_90",
                    "ins_sell_n_180", "ins_buy_val_90", "ins_buy_val_180", "ins_sell_val_90", "ins_sell_val_180",
                    "ins_net_ratio_90", "ins_net_ratio_180", "ins_n_buyers_90", "ins_od_buy_90", "ins_cluster_30"])
SI_COLUMNS = ["si_ratio", "si_dtc", "si_chg", "si_age"]
BLOCKS = {"insider": INSIDER_COLUMNS, "si": SI_COLUMNS}
EXCHANGE_CLASSES = {"NYSE", "NNM", "SC", "AMEX", "ARCA", "BZX", "IEX", "NGM", "NCM"}


def _sec_headers() -> dict:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        from dotenv import dotenv_values
        ua = dotenv_values(os.path.join(os.path.dirname(__file__), "..", "trading_agent", ".env")).get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError("SEC_USER_AGENT not configured")
    return {"User-Agent": ua}


def next_bday(d: pd.Series, n: int = 1) -> pd.Series:
    """n NYSE business days after ``d`` (weekends + NYSE holidays skipped)."""
    from .fundamentals import nyse_holidays
    hol = nyse_holidays()
    out = np.busday_offset(pd.to_datetime(d).to_numpy().astype("datetime64[D]"), n, roll="backward",
                           holidays=hol)
    return pd.Series(pd.to_datetime(out), index=getattr(d, "index", None))


# --------------------------------------------------------------------------- #
# Downloads (public files only; polite rate)
# --------------------------------------------------------------------------- #
def quarters(start: str = "2018q1", end: dt.date | None = None) -> list[str]:
    end = end or dt.date.today()
    y, q = int(start[:4]), int(start[-1])
    out = []
    while (y, q) <= (end.year, (end.month - 1) // 3 + 1):
        out.append(f"{y}q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def download_insider(log=print) -> list[str]:
    import requests
    out = data_dir("sec_insider", "raw")
    h = _sec_headers()
    got = []
    for q in quarters():
        p = os.path.join(out, f"{q}_form345.zip")
        if os.path.exists(p) and os.path.getsize(p) > 1e5:
            got.append(p)
            continue
        r = requests.get(SEC_URL.format(q=q), headers=h, timeout=180)
        time.sleep(0.6)
        if r.status_code == 404:
            log(f"[insider] {q}: not published")
            continue
        r.raise_for_status()
        open(p, "wb").write(r.content)
        got.append(p)
        log(f"[insider] {q}: {len(r.content) / 1e6:.1f} MB")
    return got


def _settlement_candidates(year: int, month: int) -> list[list[str]]:
    """Two candidate lists per month (mid-month and month-end settlement),
    most likely date first."""
    last = (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0))
    mid = [pd.Timestamp(year, month, d) for d in range(15, 8, -1)]
    end = [last - pd.Timedelta(days=k) for k in range(0, 8)]
    return [[x.strftime("%Y%m%d") for x in mid if x.weekday() < 5],
            [x.strftime("%Y%m%d") for x in end if x.weekday() < 5]]


def is_si_payload(text: str) -> bool:
    """FINRA files start with the header row; from 2026-09 the fields are quoted."""
    return text.lstrip('"').startswith("accountingYearMonthNumber")


def download_si(start: str = "2018-06", log=print) -> list[str]:
    import requests
    out = data_dir("finra_si", "raw")
    got = []
    for m in pd.period_range(start, pd.Timestamp.today().to_period("M"), freq="M"):
        for cands in _settlement_candidates(m.year, m.month):
            if any(os.path.exists(os.path.join(out, f"shrt{d}.csv")) for d in cands):
                got += [os.path.join(out, f"shrt{d}.csv") for d in cands
                        if os.path.exists(os.path.join(out, f"shrt{d}.csv"))]
                continue
            for d in cands:
                r = requests.get(FINRA_URL.format(d=d), timeout=120)
                time.sleep(0.4)
                if r.status_code == 200 and is_si_payload(r.text):
                    p = os.path.join(out, f"shrt{d}.csv")
                    open(p, "w").write(r.text)
                    got.append(p)
                    break
        log(f"[si] {m}: {len(got)} files so far")
    return got


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def parse_insider_zip(path_or_buf) -> pd.DataFrame:
    """Open-market P/S lines of one quarterly zip: one row per (accession,
    transaction) with issuer_cik, filing_date, avail_date, owner_cik, is_od
    (officer or director), code, shares, price, value."""
    z = zipfile.ZipFile(path_or_buf)
    rd = lambda n, cols: pd.read_csv(z.open(n), sep="\t", dtype=str, usecols=cols, quoting=3,
                                     on_bad_lines="skip", encoding="latin-1")
    sub = rd("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK", "DOCUMENT_TYPE"])
    sub = sub[sub["DOCUMENT_TYPE"].isin(["4", "5"])]               # amendments excluded (duplicates)
    own = rd("REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
    own = own.drop_duplicates("ACCESSION_NUMBER", keep="first")    # primary reporting owner
    tr_ = rd("NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                                    "TRANS_ACQUIRED_DISP_CD"])
    tr_ = tr_[tr_["TRANS_CODE"].isin(["P", "S"])]
    m = tr_.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")
    rel = m["RPTOWNER_RELATIONSHIP"].fillna("").str.lower()
    out = pd.DataFrame({
        "accession": m["ACCESSION_NUMBER"],
        "issuer_cik": pd.to_numeric(m["ISSUERCIK"], errors="coerce").astype("Int64"),
        "filing_date": pd.to_datetime(m["FILING_DATE"], format="%d-%b-%Y", errors="coerce"),
        "owner_cik": pd.to_numeric(m["RPTOWNERCIK"], errors="coerce").astype("Int64"),
        "is_od": rel.str.contains("officer") | rel.str.contains("director"),
        "code": m["TRANS_CODE"],
        "shares": pd.to_numeric(m["TRANS_SHARES"], errors="coerce"),
        "price": pd.to_numeric(m["TRANS_PRICEPERSHARE"], errors="coerce"),
    })
    out = out.dropna(subset=["issuer_cik", "filing_date"])
    out["value"] = (out["shares"] * out["price"]).where(lambda v: np.isfinite(v) & (v >= 0), 0.0)
    out["avail_date"] = next_bday(out["filing_date"], 1).to_numpy()
    return out.reset_index(drop=True)


def parse_si_csv(path_or_buf) -> pd.DataFrame:
    df = pd.read_csv(path_or_buf, sep="|", dtype=str, on_bad_lines="skip")
    df = df[df["marketClassCode"].isin(EXCHANGE_CLASSES)]
    out = pd.DataFrame({
        "symbol": df["symbolCode"].str.strip().str.upper().str.replace("/", ".", regex=False)
                   .str.replace("-", ".", regex=False),
        "settlement_date": pd.to_datetime(df["settlementDate"], errors="coerce"),
        "si": pd.to_numeric(df["currentShortPositionQuantity"], errors="coerce"),
        "si_prev": pd.to_numeric(df["previousShortPositionQuantity"], errors="coerce"),
        "si_dtc": pd.to_numeric(df["daysToCoverQuantity"], errors="coerce"),
    }).dropna(subset=["settlement_date", "si"])
    out["avail_date"] = next_bday(out["settlement_date"], SI_LAG_BDAYS).to_numpy()
    return out.reset_index(drop=True)


def load_insider_events() -> pd.DataFrame:
    p = os.path.join(data_dir("sec_insider"), "events.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    ev = pd.concat([parse_insider_zip(f) for f in sorted(glob.glob(os.path.join(data_dir("sec_insider", "raw"),
                                                                                 "*_form345.zip")))],
                   ignore_index=True).drop_duplicates(["accession", "code", "shares", "price"])
    ev.to_parquet(p, index=False)
    return ev


def load_si() -> pd.DataFrame:
    p = os.path.join(data_dir("finra_si"), "si.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    si = pd.concat([parse_si_csv(f) for f in sorted(glob.glob(os.path.join(data_dir("finra_si", "raw"),
                                                                            "shrt*.csv")))], ignore_index=True)
    si = si.drop_duplicates(["symbol", "settlement_date"], keep="last")
    si.to_parquet(p, index=False)
    return si


# --------------------------------------------------------------------------- #
# Features on decision dates
# --------------------------------------------------------------------------- #
def insider_features(events: pd.DataFrame, dates) -> pd.DataFrame:
    """[issuer_cik, Date] + INSIDER_COLUMNS for every issuer with any event in
    a window ending at each date (others: zeros = no activity). Uses only
    events with avail_date <= Date."""
    e = events.sort_values("avail_date").reset_index(drop=True)
    av = e["avail_date"].to_numpy().astype("datetime64[D]")
    out = []
    for d in sorted(dates):
        t = np.datetime64(pd.Timestamp(d).date())
        hi = np.searchsorted(av, t, side="right")
        res = None
        for W in INS_WINDOWS:
            lo = np.searchsorted(av, t - np.timedelta64(W, "D"), side="right")
            s = e.iloc[lo:hi]
            if s.empty:
                continue
            b, sl = s[s["code"] == "P"], s[s["code"] == "S"]
            g = pd.DataFrame({
                f"ins_buy_n_{W}": b.groupby("issuer_cik")["accession"].nunique(),
                f"ins_sell_n_{W}": sl.groupby("issuer_cik")["accession"].nunique()})
            if W in (90, 180):
                bv, sv = b.groupby("issuer_cik")["value"].sum(), sl.groupby("issuer_cik")["value"].sum()
                g[f"ins_buy_val_{W}"] = bv
                g[f"ins_sell_val_{W}"] = sv
            if W == 90:
                g["ins_n_buyers_90"] = b.groupby("issuer_cik")["owner_cik"].nunique()
                g["ins_od_buy_90"] = b[b["is_od"]].groupby("issuer_cik").size().gt(0).astype(float)
            if W == 30:
                g["ins_cluster_30"] = b.groupby("issuer_cik")["owner_cik"].nunique().ge(3).astype(float)
            res = g if res is None else res.join(g, how="outer")
        if res is None:
            continue
        res = res.fillna(0.0)
        for W in (90, 180):
            bv, sv = res.get(f"ins_buy_val_{W}", 0.0), res.get(f"ins_sell_val_{W}", 0.0)
            tot = bv + sv
            res[f"ins_net_ratio_{W}"] = np.where(tot > 0, (bv - sv) / np.where(tot > 0, tot, 1), 0.0)
            res[f"ins_buy_val_{W}"] = np.log1p(bv / 1e3)
            res[f"ins_sell_val_{W}"] = np.log1p(sv / 1e3)
        res = res.reset_index().rename(columns={"index": "issuer_cik"})
        res["Date"] = d
        out.append(res)
    f = pd.concat(out, ignore_index=True)
    for c in INSIDER_COLUMNS:
        f[c] = f[c].fillna(0.0) if c in f.columns else 0.0
    return f[["issuer_cik", "Date"] + INSIDER_COLUMNS]


def si_features(si: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """For rows (ticker, Date, first_date, shares): the latest FINRA record of
    the ticker's base symbol with avail_date <= Date and settlement_date >=
    the ticker id's first bar (reused symbols never inherit an old company's
    record). Returns SI_COLUMNS aligned to ``rows``."""
    r = rows.copy()
    r["_i"] = np.arange(len(r))
    r["symbol"] = r["ticker"].astype(str).str.split("~").str[0]
    r["_d"] = pd.to_datetime(r["Date"]).astype("datetime64[ns]")
    s = si.assign(avail_date=pd.to_datetime(si["avail_date"]).astype("datetime64[ns]"),
                  settlement_date=pd.to_datetime(si["settlement_date"]).astype("datetime64[ns]")).sort_values("avail_date")
    m = pd.merge_asof(r.sort_values("_d"), s, left_on="_d", right_on="avail_date", by="symbol",
                      direction="backward")
    m = m[m["settlement_date"] >= pd.to_datetime(m["first_date"])]
    out = pd.DataFrame(np.nan, index=np.arange(len(r)), columns=SI_COLUMNS)
    sh = m["shares"].astype(float)
    out.loc[m["_i"].to_numpy(), "si_ratio"] = (m["si"] / sh.where(sh > 0)).to_numpy()
    out.loc[m["_i"].to_numpy(), "si_dtc"] = m["si_dtc"].to_numpy()
    out.loc[m["_i"].to_numpy(), "si_chg"] = ((m["si"] - m["si_prev"]) / m["si_prev"].where(m["si_prev"] > 0)).to_numpy()
    out.loc[m["_i"].to_numpy(), "si_age"] = (m["_d"] - m["avail_date"]).dt.days.to_numpy()
    out.index = rows.index
    return out


def weekly_features(log=print) -> pd.DataFrame:
    """Raw insider + SI features for every (ticker, weekly date) row of the
    store (eligible or not) with the 30-session label -> ``alt_weekly.parquet``."""
    from . import fundamentals as fu
    t0 = time.time()
    cal = ds.calendar_from_dataset(ds.v3_dir(ds.EDGAR_DATASET))
    weekly = ds.weekly_decision_dates(cal)
    rows = pd.concat([pd.read_parquet(f, columns=["ticker", "Date", "close_raw", "company_close", "eligible",
                                                    "fwd_ret_30"], filters=[("Date", "in", weekly)])
                      for f in sorted(glob.glob(os.path.join(ds.v3_dir(ds.EDGAR_DATASET), "stage_a", "*.parquet")))],
                     ignore_index=True)
    first = pd.concat([pd.read_parquet(f, columns=["ticker", "Date"]).groupby("ticker")["Date"].min()
                       for f in sorted(glob.glob(os.path.join(ds.v3_dir(ds.EDGAR_DATASET), "stage_a", "*.parquet")))])
    rows["first_date"] = rows["ticker"].map(first)
    mc = pd.concat([ds.load_daily(years=[y], dates=[d for d in weekly if d[:4] == str(y)],
                                  out_dir=ds.v3_dir(ds.EDGAR_DATASET), columns=["ticker", "Date", "log_mcap"])
                    for y in sorted({int(d[:4]) for d in weekly})], ignore_index=True)
    rows = rows.merge(mc, on=["ticker", "Date"], how="left")
    rows["shares"] = np.exp(rows["log_mcap"]) / rows["close_raw"].fillna(rows["company_close"])
    log(f"[alt] {len(rows):,} weekly rows ({time.time() - t0:.0f}s)")
    # insider via the CIK map
    cm = pd.read_parquet(os.path.join(data_dir("fundamentals"), "cik_map.parquet"))
    keys = pd.DataFrame({"tkey": rows["ticker"].str.split("~").str[0].map(fu.norm_ticker),
                         "Date": pd.to_datetime(rows["Date"]).astype("datetime64[ns]")})
    rows["cik"] = fu.resolve_cik(keys, cm).to_numpy()
    ev = load_insider_events()
    ins = insider_features(ev, weekly)
    ins = ins.rename(columns={"issuer_cik": "cik"})
    ins["cik"] = ins["cik"].astype("Int64")
    rows = rows.merge(ins, on=["cik", "Date"], how="left")
    rows[INSIDER_COLUMNS] = rows[INSIDER_COLUMNS].fillna(0.0)
    # after the last published quarter the data simply stops: insider features are
    # missing (not "no activity") for decision dates beyond the last filing date
    ins_end = pd.Timestamp(ev["filing_date"].max())
    rows.loc[pd.to_datetime(rows["Date"]) > ins_end, INSIDER_COLUMNS] = np.nan
    log(f"[alt] insider data ends {ins_end.date()}: features NaN after it")
    log(f"[alt] insider features: {len(ev):,} P/S lines; rows with any 180d activity "
        f"{((rows['ins_buy_n_180'] + rows['ins_sell_n_180']) > 0).mean():.1%} ({time.time() - t0:.0f}s)")
    si = load_si()
    rows[SI_COLUMNS] = si_features(si, rows[["ticker", "Date", "first_date", "shares"]]).to_numpy()
    log(f"[alt] SI coverage {rows['si_ratio'].notna().mean():.1%} ({time.time() - t0:.0f}s)")
    keep = ["ticker", "Date", "eligible", "fwd_ret_30", "cik"] + INSIDER_COLUMNS + SI_COLUMNS
    out = rows[keep]
    for c in INSIDER_COLUMNS + SI_COLUMNS:
        out[c] = out[c].astype("float32")
    out.to_parquet(os.path.join(ds.v3_dir(), "alt_weekly.parquet"), index=False)
    return out


if __name__ == "__main__":
    import sys
    if "download" in sys.argv:
        download_insider()
        download_si()
    if "features" in sys.argv:
        weekly_features()

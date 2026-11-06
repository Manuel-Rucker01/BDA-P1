"""Daily EDGAR Form 4/5 ingestion (free) -- the live counterpart of the
quarterly "Insider Transactions Data Sets" used in round 8.

* ``daily_index(date)``: EDGAR ``master.YYYYMMDD.idx`` (Archives/edgar/daily-index),
  Form 4 and 5 only (amendments excluded, as in the bulk parser).
* ``parse_form4(text, accession, filing_date)``: one filing's ownership XML ->
  the exact schema of ``altdata.parse_insider_zip`` (open-market P/S
  non-derivative lines; primary = first reporting owner; officer/director flag).
* ``ingest_day``/``refresh``: incremental, resumable per-day Parquet cache under
  ``$BDA_DATA_V3/sec_insider/daily/``; <= 8 requests/s (shared limiter), retries
  with backoff; User-Agent from ``SEC_USER_AGENT`` (env or trading_agent/.env).
* ``load_events()``: bulk quarters + daily feed, de-duplicated by accession
  (bulk wins where both exist).

    python -m DataAnalysisPipeline2.v3.form4 refresh [--start 2026-04-01] [--end YYYY-MM-DD]
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import re
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import altdata as ad
from ._alpaca import data_dir

ARCHIVES = "https://www.sec.gov/Archives/"
INDEX_URL = ARCHIVES + "edgar/daily-index/{y}/QTR{q}/master.{d}.idx"
FORMS = ("4", "5")
MAX_RPS = 8.0
EVENT_COLUMNS = ["accession", "issuer_cik", "filing_date", "owner_cik", "is_od", "code", "shares", "price",
                 "value", "avail_date"]


class Limiter:
    """Thread-safe minimum spacing between requests (<= ``rps`` per second)."""

    def __init__(self, rps: float = MAX_RPS, clock=time.monotonic, sleep=time.sleep):
        self.dt, self.clock, self.sleep = 1.0 / rps, clock, sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = self.clock()
            t = max(now, self._next)
            self._next = t + self.dt
        if t > now:
            self.sleep(t - now)


class SECClient:
    def __init__(self, session=None, limiter: Limiter | None = None, retries: int = 5, sleep=time.sleep):
        if session is None:
            import requests
            session = requests.Session()
            session.headers.update(ad._sec_headers())
        self.s, self.lim, self.retries, self.sleep = session, limiter or Limiter(), retries, sleep
        self.n = 0

    def get(self, url: str) -> str | None:
        for k in range(self.retries):
            self.lim.wait()
            self.n += 1
            try:
                r = self.s.get(url, timeout=60)
            except Exception:
                self.sleep(2 ** k)
                continue
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                return None
            self.sleep(2 ** k * (5 if r.status_code in (403, 429) else 1))
        raise RuntimeError(f"SEC GET failed after {self.retries} tries: {url}")


# --------------------------------------------------------------------------- #
# Index + parser
# --------------------------------------------------------------------------- #
def parse_index(text: str) -> pd.DataFrame:
    rows = [l.split("|") for l in text.splitlines() if l.count("|") == 4 and not l.startswith("CIK|")]
    df = pd.DataFrame(rows, columns=["cik", "company", "form", "date_filed", "filename"])
    df = df[df["form"].isin(FORMS)].copy()
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    df["accession"] = df["filename"].str.extract(r"/([0-9]{10}-[0-9]{2}-[0-9]{6})\.txt$")[0]
    df["date_filed"] = pd.to_datetime(df["date_filed"], format="%Y%m%d", errors="coerce")
    return df.dropna(subset=["accession"])


def daily_index(date, client: SECClient) -> pd.DataFrame | None:
    d = pd.Timestamp(date)
    txt = client.get(INDEX_URL.format(y=d.year, q=(d.month - 1) // 3 + 1, d=d.strftime("%Y%m%d")))
    return None if txt is None else parse_index(txt)


def select_filings(idx: pd.DataFrame, issuer_ciks: set | None) -> pd.DataFrame:
    """One row per accession. The index lists a filing under the issuer AND each
    reporting owner; with ``issuer_ciks`` keep accessions listed under one of them."""
    if issuer_ciks is not None:
        keep = set(idx.loc[idx["cik"].isin(issuer_ciks), "accession"])
        idx = idx[idx["accession"].isin(keep)]
    return idx.drop_duplicates("accession")[["accession", "filename", "date_filed", "form"]]


def _t(el, path):
    x = el.find(path)
    return (x.text or "").strip() if x is not None and x.text is not None else ""


def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return np.nan


def _true(s):
    return str(s).strip().lower() in ("1", "true")


def parse_form4(text: str, accession: str, filing_date) -> pd.DataFrame:
    """Open-market P/S lines of one filing (``altdata.parse_insider_zip`` schema)."""
    m = re.search(r"<XML>\s*(.*?)\s*</XML>", text, re.S | re.I)
    xml = m.group(1) if m else text
    try:
        root = ET.fromstring(xml.strip().encode("utf-8", "replace"))
    except ET.ParseError:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    if root.tag != "ownershipDocument" or _t(root, "documentType") not in FORMS:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    issuer = _num(_t(root, "issuer/issuerCik"))
    own = root.find("reportingOwner")
    owner = _num(_t(own, "reportingOwnerId/rptOwnerCik")) if own is not None else np.nan
    rel = own.find("reportingOwnerRelationship") if own is not None else None
    is_od = bool(rel is not None and (_true(_t(rel, "isDirector")) or _true(_t(rel, "isOfficer"))))
    rows = []
    for tr in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = _t(tr, "transactionCoding/transactionCode")
        if code not in ("P", "S"):
            continue
        sh = _num(_t(tr, "transactionAmounts/transactionShares/value"))
        px = _num(_t(tr, "transactionAmounts/transactionPricePerShare/value"))
        rows.append({"accession": accession, "issuer_cik": issuer, "owner_cik": owner, "is_od": is_od,
                     "code": code, "shares": sh, "price": px})
    out = pd.DataFrame(rows, columns=["accession", "issuer_cik", "owner_cik", "is_od", "code", "shares", "price"])
    if out.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    out["issuer_cik"] = out["issuer_cik"].astype("Int64")
    out["owner_cik"] = out["owner_cik"].astype("Int64")
    out["filing_date"] = pd.Timestamp(filing_date)
    out["avail_date"] = ad.next_bday(out["filing_date"], 1).to_numpy()
    return normalize(out)


def normalize(ev: pd.DataFrame) -> pd.DataFrame:
    """Bulk-data conventions: the SEC data sets round TRANS_PRICEPERSHARE and
    TRANS_SHARES to 2 decimals, half up (the XML keeps full precision); value =
    shares x price (0 if unknown)."""
    ev = ev.copy()
    half_up = lambda x: np.floor(x.astype(float) * 100.0 + 0.5 + 1e-7) / 100.0
    ev["price"] = half_up(ev["price"])         # the SEC data sets round price and shares to 2 decimals, half up
    ev["shares"] = half_up(ev["shares"])
    v = ev["shares"] * ev["price"]
    ev["value"] = v.where(np.isfinite(v) & (v >= 0), 0.0)
    return ev[EVENT_COLUMNS]


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def cache_dir() -> str:
    return data_dir("sec_insider", "daily")


def universe_issuer_ciks() -> set:
    cm = pd.read_parquet(os.path.join(data_dir("fundamentals"), "cik_map.parquet"))
    return set(cm["cik"].dropna().astype("int64"))


def ingest_day(date, client: SECClient, issuer_ciks: set | None, workers: int = 4, log=print) -> int:
    """Fetch and parse every Form 4/5 of ``date`` into ``YYYY-MM-DD.parquet``.

    A finished day is never re-fetched. Filings whose fetch failed persistently
    are listed in ``YYYY-MM-DD.failed.json``; the next call for that day
    retries only those and merges them in (the day counts as complete once the
    list is empty)."""
    import json
    d = pd.Timestamp(date).strftime("%Y-%m-%d")
    out = os.path.join(cache_dir(), f"{d}.parquet")
    failed_p = os.path.join(cache_dir(), f"{d}.failed.json")
    if os.path.exists(out) and not os.path.exists(failed_p):
        return -1
    if os.path.exists(out):
        sel = pd.DataFrame(json.load(open(failed_p)))
        if len(sel):
            sel["date_filed"] = pd.to_datetime(sel["date_filed"])
        prev = pd.read_parquet(out)
    else:
        try:
            idx = daily_index(d, client)
        except RuntimeError as e:         # persistent error: not cached, retried on the next refresh
            log(f"[form4] {d}: index unavailable ({e}); will retry next refresh")
            return -3
        if idx is None:                   # weekend / holiday / not yet published
            return -2
        sel = select_filings(idx, issuer_ciks)
        prev = None
    from concurrent.futures import ThreadPoolExecutor

    def one(r):
        try:
            txt = client.get(ARCHIVES + r.filename)
        except RuntimeError:
            return r, None
        return r, (parse_form4(txt, r.accession, r.date_filed) if txt else pd.DataFrame(columns=EVENT_COLUMNS))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(one, sel.itertuples(index=False)))
    parts = [p for _, p in res if p is not None and len(p)]
    failed = [r for r, p in res if p is None]
    if prev is not None and len(prev):
        parts = [prev] + parts
    ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=EVENT_COLUMNS)
    ev.to_parquet(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)
    if failed:
        json.dump([{"accession": r.accession, "filename": r.filename, "date_filed": str(r.date_filed),
                    "form": r.form} for r in failed], open(failed_p, "w"))
    elif os.path.exists(failed_p):
        os.remove(failed_p)
    log(f"[form4] {d}: {len(sel):,} filings -> {len(ev):,} P/S lines"
        f"{f', {len(failed)} failed (retry next refresh)' if failed else ''} ({client.n:,} requests so far)")
    return len(ev)


def refresh(start: str | None = None, end: str | None = None, issuer_filter: bool = True, log=print) -> dict:
    """Incremental backfill / daily refresh, from the day after the last bulk
    quarter (or ``start``) through ``end`` (default: yesterday)."""
    t0 = time.time()
    start = start or bulk_end_date()
    end = end or (dt.date.today() - dt.timedelta(days=1)).isoformat()
    client = SECClient()
    ciks = universe_issuer_ciks() if issuer_filter else None
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from .fundamentals import nyse_holidays
    # EDGAR publishes no daily index on NYSE holidays nor on federal holidays (e.g. Columbus Day)
    hol = set(pd.to_datetime(nyse_holidays())) | set(USFederalHolidayCalendar().holidays(start, end))
    days = [d for d in pd.bdate_range(start, end) if d not in hol]
    n_rows = 0
    for d in days:
        n = ingest_day(d, client, ciks, log=log)
        n_rows += max(n, 0)
    info = {"start": str(start), "end": str(end), "days": len(days), "rows": n_rows, "requests": client.n,
            "seconds": round(time.time() - t0)}
    log(f"[form4] refresh done: {info}")
    return info


def bulk_end_date() -> str:
    files = sorted(glob.glob(os.path.join(data_dir("sec_insider", "raw"), "*_form345.zip")))
    q = os.path.basename(files[-1])[:6]
    y, k = int(q[:4]), int(q[-1])
    return (pd.Timestamp(y, 3 * k, 1) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def load_daily_events() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(cache_dir(), "*.parquet")))
    parts = [pd.read_parquet(f) for f in files]
    parts = [normalize(p) for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    # identical lines collapsed exactly as in altdata.load_insider_events
    return pd.concat(parts, ignore_index=True).drop_duplicates(["accession", "code", "shares", "price"])


def load_events() -> pd.DataFrame:
    """Bulk quarters + daily feed (accessions already in the bulk files dropped)."""
    bulk = ad.load_insider_events()
    daily = load_daily_events()
    if len(daily):
        daily = daily[~daily["accession"].isin(set(bulk["accession"]))]
    ev = pd.concat([bulk, daily], ignore_index=True)
    ev["issuer_cik"] = ev["issuer_cik"].astype("Int64")
    return ev


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["refresh"])
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--all-issuers", action="store_true")
    a = p.parse_args()
    refresh(a.start, a.end, issuer_filter=not a.all_issuers)

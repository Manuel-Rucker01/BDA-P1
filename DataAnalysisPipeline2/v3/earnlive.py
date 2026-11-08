"""Live 8-K item 2.02 (earnings release) events for the Round-11 earnings
blackout (``round11.retained_events`` / ``expected_dates`` / ``blackout_window``).

Offline source (Round 10/11): ``$BDA_DATA_V3/sec_earn/events.parquet`` built by
``altdata_r10`` from the bulk EDGAR ``submissions.zip`` (original 8-K whose
items include 2.02, EDGAR acceptance timestamp). That store only reaches the
day the bulk zip was downloaded; this module tops it up incrementally:

* the EDGAR daily master index (``master.YYYYMMDD.idx``, ``form4.INDEX_URL``)
  of every business day after the covered date lists the original 8-Ks filed
  that day; for every universe CIK among them
* the per-CIK submissions JSON (``data.sec.gov/submissions/CIK##########.json``,
  the per-company version of the same bulk file) gives items + acceptance time,
  parsed with ``altdata_r10._filing_rows`` (identical fields to the bulk path).

Requests go through ``form4.SECClient`` (<= 8 req/s, retries, ``SEC_USER_AGENT``).
The result is cached in ``$BDA_DATA_V3/sec_earn/live_8k.parquet`` with
``live_8k_manifest.json`` (``as_of`` = last day whose index and every needed
submissions file were fetched). ``raw_events()`` = bulk-store 8-K 2.02 events
+ the live cache, de-duplicated, in the ``round11.retained_events`` input schema.

    python -m DataAnalysisPipeline2.v3.earnlive refresh [--start YYYY-MM-DD] [--end YYYY-MM-DD]

(also run by ``python -m DataAnalysisPipeline2.v3.altlive refresh``).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time

import pandas as pd

from ._alpaca import data_dir

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
LIVE_FILE = "live_8k.parquet"
LIVE_MANIFEST = "live_8k_manifest.json"
FILING_COLS = ["cik", "accn", "form", "filing_date", "acceptance", "items"]
FORM = "8-K"


def earn_dir() -> str:
    return data_dir("sec_earn")


# --------------------------------------------------------------------------- #
# Parsing (pure)
# --------------------------------------------------------------------------- #
def parse_index_8k(text: str) -> pd.DataFrame:
    """EDGAR daily master index -> original 8-K rows (cik, date_filed, accession)."""
    rows = [l.split("|") for l in text.splitlines() if l.count("|") == 4 and not l.startswith("CIK|")]
    df = pd.DataFrame(rows, columns=["cik", "company", "form", "date_filed", "filename"])
    df = df[df["form"] == FORM].copy()
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    df["date_filed"] = pd.to_datetime(df["date_filed"], format="%Y%m%d", errors="coerce")
    df["accession"] = df["filename"].str.extract(r"/([0-9]{10}-[0-9]{2}-[0-9]{6})\.txt$")[0]
    return df.dropna(subset=["cik", "accession"])[["cik", "date_filed", "accession"]]


def parse_submissions_json(doc: dict, cik: int) -> pd.DataFrame:
    """Per-CIK submissions JSON -> 8-K / 10-Q / 10-K rows of ``FILING_COLS``
    from ``filings.recent`` (same parser as the bulk zip)."""
    from .altdata_r10 import _filing_rows
    rows = _filing_rows((doc.get("filings", {}) or {}).get("recent", {}) or {}, int(cik))
    f = pd.DataFrame(rows, columns=FILING_COLS)
    f["acceptance"] = pd.to_datetime(f["acceptance"], utc=True, errors="coerce")
    return f


def is_202(items: pd.Series) -> pd.Series:
    return items.fillna("").astype(str).str.split(",").apply(lambda xs: any(x.strip() == "2.02" for x in xs))


def to_raw(filings: pd.DataFrame) -> pd.DataFrame:
    """Filings -> ``retained_events`` input rows (original 8-K with item 2.02):
    cik, acceptance (UTC), src10q = 0."""
    f = filings.dropna(subset=["acceptance"])
    f = f[(f["form"] == FORM) & is_202(f["items"])]
    return pd.DataFrame({"cik": f["cik"].astype("int64").to_numpy(),
                         "acceptance": pd.to_datetime(f["acceptance"], utc=True).to_numpy(), "src10q": 0})


# --------------------------------------------------------------------------- #
# Local store + live cache
# --------------------------------------------------------------------------- #
def store_as_of(path: str | None = None) -> pd.Timestamp | None:
    """New York date of the newest acceptance in the bulk-built filings table."""
    p = path or os.path.join(earn_dir(), "filings.parquet")
    if not os.path.exists(p):
        return None
    a = pd.read_parquet(p, columns=["acceptance"])["acceptance"]
    return pd.Timestamp(pd.to_datetime(a, utc=True).max().tz_convert("America/New_York").date())


def live_as_of() -> pd.Timestamp | None:
    p = os.path.join(earn_dir(), LIVE_MANIFEST)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return pd.Timestamp(json.load(f)["as_of"])


def coverage_as_of() -> pd.Timestamp | None:
    ds = [d for d in (store_as_of(), live_as_of()) if d is not None]
    return max(ds) if ds else None


def load_live() -> pd.DataFrame:
    p = os.path.join(earn_dir(), LIVE_FILE)
    if not os.path.exists(p):
        return pd.DataFrame(columns=FILING_COLS)
    return pd.read_parquet(p)


def raw_events() -> pd.DataFrame:
    """Bulk-store 8-K 2.02 events + live cache, one row per (cik, acceptance)."""
    ev = pd.read_parquet(os.path.join(earn_dir(), "events.parquet"), columns=["cik", "acceptance", "src10q"])
    ev = ev[ev["src10q"] == 0]
    parts = [pd.DataFrame({"cik": ev["cik"].astype("int64").to_numpy(),
                           "acceptance": pd.to_datetime(ev["acceptance"], utc=True).to_numpy(), "src10q": 0})]
    live = load_live()
    if len(live):
        parts.append(to_raw(live))
    out = pd.concat(parts, ignore_index=True)
    out["acceptance"] = pd.to_datetime(out["acceptance"], utc=True)
    return out.drop_duplicates(["cik", "acceptance"]).sort_values(["cik", "acceptance"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Incremental refresh from EDGAR
# --------------------------------------------------------------------------- #
def _business_days(start, end) -> list:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from .fundamentals import nyse_holidays
    hol = set(pd.to_datetime(nyse_holidays())) | set(USFederalHolidayCalendar().holidays(start, end))
    return [d for d in pd.bdate_range(start, end) if d not in hol]


def fetch(start, end, ciks: set | None, client=None, log=print) -> tuple[pd.DataFrame, pd.Timestamp | None, dict]:
    """(filings rows filed in [start, end] for 8-K filers among ``ciks``, last
    day fully covered (None if none), info). Never writes."""
    from . import form4
    client = client or form4.SECClient()
    t0 = time.time()
    days = _business_days(start, end)
    filers, done_days = [], []
    for d in days:
        try:
            txt = client.get(form4.INDEX_URL.format(y=d.year, q=(d.month - 1) // 3 + 1, d=d.strftime("%Y%m%d")))
        except RuntimeError as e:
            log(f"[earnlive] {d.date()}: index unavailable ({e}); stopping")
            break
        if txt is None:            # not published yet: later days cannot be complete either
            log(f"[earnlive] {d.date()}: no daily index (not yet published); stopping")
            break
        idx = parse_index_8k(txt)
        if ciks is not None:
            idx = idx[idx["cik"].isin(ciks)]
        filers.append(idx.assign(day=d))
        done_days.append(d)
    if not done_days:
        return pd.DataFrame(columns=FILING_COLS), None, {"days": 0, "requests": client.n}
    fl = pd.concat(filers, ignore_index=True)
    first_day = fl.groupby("cik")["day"].min()
    parts, failed_from = [], None
    for cik, d0 in first_day.sort_index().items():
        try:
            txt = client.get(SUBMISSIONS_URL.format(cik=int(cik)))
            if txt is None:
                raise RuntimeError("404")
            parts.append(parse_submissions_json(json.loads(txt), int(cik)))
        except Exception as e:     # noqa: BLE001 -- the covered date stops before this CIK's first 8-K day
            log(f"[earnlive] CIK {int(cik)}: submissions fetch failed ({type(e).__name__}: {e})")
            failed_from = d0 if failed_from is None else min(failed_from, d0)
    covered = done_days[-1]
    if failed_from is not None:
        prev = [d for d in done_days if d < failed_from]
        covered = prev[-1] if prev else None
    f = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=FILING_COLS)
    fd = pd.to_datetime(f["filing_date"], errors="coerce")
    f = f[(f["form"] == FORM) & (fd >= pd.Timestamp(start)) & (fd <= done_days[-1])].reset_index(drop=True)
    info = {"start": str(pd.Timestamp(start).date()), "end": str(pd.Timestamp(end).date()), "days": len(done_days),
            "filers": int(len(first_day)), "rows_8k": int(len(f)), "rows_202": int(is_202(f["items"]).sum()),
            "covered": str(covered.date()) if covered is not None else None, "requests": client.n,
            "seconds": round(time.time() - t0)}
    return f, covered, info


def refresh(start: str | None = None, end: str | None = None, ciks: set | None = None, client=None,
            log=print) -> dict:
    """Top up the live cache from the day after the covered date (store / cache)
    through ``end`` (default yesterday). Universe = the fundamentals cik_map."""
    cov = coverage_as_of()
    start = pd.Timestamp(start) if start else (cov + pd.Timedelta(days=1) if cov is not None else None)
    if start is None:
        raise RuntimeError("no sec_earn store / live cache to extend; pass --start")
    end = pd.Timestamp(end or (dt.date.today() - dt.timedelta(days=1)))
    if start > end:
        info = {"start": str(start.date()), "end": str(end.date()), "days": 0, "covered": str(cov.date())}
        log(f"[earnlive] up to date: {info}")
        return info
    if ciks is None:
        cm = pd.read_parquet(os.path.join(data_dir("fundamentals"), "cik_map.parquet"), columns=["cik"])
        ciks = set(cm["cik"].dropna().astype("int64"))
    f, covered, info = fetch(start, end, ciks, client=client, log=log)
    if covered is None:
        log(f"[earnlive] nothing covered: {info}")
        return info
    old = load_live()
    new = pd.concat([old, f], ignore_index=True) if len(old) else f
    new["acceptance"] = pd.to_datetime(new["acceptance"], utc=True)
    new = new.drop_duplicates("accn", keep="last").sort_values(["cik", "acceptance"]).reset_index(drop=True)
    p = os.path.join(earn_dir(), LIVE_FILE)
    new.to_parquet(p + ".tmp", index=False)
    os.replace(p + ".tmp", p)
    as_of = max(covered, live_as_of()) if live_as_of() is not None else covered
    with open(os.path.join(earn_dir(), LIVE_MANIFEST), "w") as fh:
        json.dump({"as_of": str(as_of.date()), "updated": str(dt.date.today()), "rows": int(len(new)),
                   "last_refresh": info}, fh, indent=1)
    log(f"[earnlive] refresh: {info}")
    return info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["refresh", "status"])
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    if a.cmd == "refresh":
        print(refresh(a.start, a.end))
    else:
        print({"store_as_of": str(store_as_of()), "live_as_of": str(live_as_of()), "coverage": str(coverage_as_of())})

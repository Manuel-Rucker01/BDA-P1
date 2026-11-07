"""Live inputs for the Round-8 signal ``sn30_altall_hd`` (``BDA_SIGNAL=altall``).

PAPER opt-in only: the signal FAILS the pre-registered gate (DSR 0.863 at
N = 92, IR vs SPY -0.13); the production model stays ``best_model.pkl``.

Model input row (``model_input``) = exactly the Round-8 training frame's:

* v3 core features (``dataset.compute_v3_features``: eligibility, EDGAR FF-12
  sectors with proxy fallback, cross-sectional features over eligible names);
* EDGAR fundamentals (``dataset.FundamentalsBlock``: as-of join on filing
  availability, hot-deck for names without any EDGAR record) on the
  decision-date cross-section -- the block is per date, so computing it on the
  decision date only is identical to the offline full-panel computation;
* core + fundamentals z-scored per date over eligible names (``dataset.zscore``);
* insider (hot-decked) + short-interest block from ``v3.altlive.compute`` (same
  point-in-time rules as training), z-scored over the same rows.

Data sources in the cloud (no ``$BDA_DATA_V3``):

* fundamentals: ``trading_agent/data/fundamentals_snapshot`` (``optimized.resolve_fund_dir``;
  it must carry every ``fundamental_features`` column, see ``optimized.build_snapshot``);
* alternative data: ``trading_agent/data/altdata_snapshot`` -- Form 4/5 open-market
  P/S lines with availability in the last ``SNAPSHOT_DAYS`` days + recent FINRA
  files, with ``manifest.json`` (``insider_as_of`` = last EDGAR day ingested,
  ``si_last_settlement``). Built locally with

      python -m DataAnalysisPipeline2.v3.altlive refresh           # local caches
      python -m DataAnalysisPipeline2.trading_agent.optimized snapshot
      python -m DataAnalysisPipeline2.trading_agent.altall snapshot

Freshness (``load_sources``): the insider feed must reach within
``BDA_ALT_INSIDER_MAX_AGE_DAYS`` (default 14) calendar days of the decision
date, and the snapshot must hold the latest FINRA settlement that is surely
public by the decision date. Optional in-runner top-ups: FINRA files from the
public CDN (a handful of requests) and, only when ``SEC_USER_AGENT`` is set
(e.g. a repository secret), the EDGAR daily Form 4 feed for the missing days
(<= 8 req/s, time-boxed by ``BDA_ALT_REFRESH_BUDGET_S``). Anything missing or
stale raises ``AltDataUnavailable``; the bot then falls back to the
production model (``bot.BDATradingAgent.use_production``).
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import tempfile
import time

import numpy as np
import pandas as pd

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(AGENT_DIR, "data", "altdata_snapshot")
EVENTS_FILE = "insider_events.parquet"
SI_FILE = "finra_si.parquet"
SNAPSHOT_DAYS = 400          # >= the 180-day window + availability lag + max age, with room for checks
SI_SNAPSHOT_DAYS = 130       # a few bi-monthly files (si_prev is inside each record)
SI_LAG_BDAYS = 8


class AltDataUnavailable(RuntimeError):
    """The altall model input cannot be built safely; the caller must fall back."""


def _v3(name):
    import importlib
    return importlib.import_module(f"DataAnalysisPipeline2.v3.{name}")


# --------------------------------------------------------------------------- #
# Snapshot
# --------------------------------------------------------------------------- #
def daily_feed_as_of(cache_dir: str | None = None) -> pd.Timestamp | None:
    """Last day of the local Form 4 cache that is complete (no ``.failed.json``)
    with every earlier business day in the cache also complete."""
    import glob
    cache_dir = cache_dir or _v3("form4").cache_dir()
    days = sorted(os.path.basename(f)[:10] for f in glob.glob(os.path.join(cache_dir, "????-??-??.parquet")))
    failed = sorted(os.path.basename(f)[:10] for f in glob.glob(os.path.join(cache_dir, "*.failed.json")))
    if failed:
        days = [d for d in days if d < failed[0]]
    return pd.Timestamp(days[-1]) if days else None


def build_snapshot(out: str = SNAPSHOT_DIR, days: int = SNAPSHOT_DAYS, events: pd.DataFrame | None = None,
                   si: pd.DataFrame | None = None, insider_as_of=None, ciks: set | None = None,
                   today=None) -> dict:
    """Trim the local caches to what ``model_input`` needs for a live decision."""
    ad, form4 = _v3("altdata"), _v3("form4")
    today = pd.Timestamp(today or dt.date.today())
    events = form4.load_events() if events is None else events
    si = ad.load_si(refresh_cache=True) if si is None else si
    as_of = pd.Timestamp(insider_as_of) if insider_as_of is not None else daily_feed_as_of()
    last_filing = pd.Timestamp(events["filing_date"].max())
    as_of = max(as_of, last_filing) if as_of is not None else last_filing
    ciks = form4.universe_issuer_ciks() if ciks is None else ciks
    ev = events[pd.to_datetime(events["avail_date"]) >= as_of - pd.Timedelta(days=days)]
    ev = ev[ev["issuer_cik"].isin(ciks)].sort_values(["avail_date", "accession"]).reset_index(drop=True)
    ev["issuer_cik"] = ev["issuer_cik"].astype("Int64")
    ev["owner_cik"] = ev["owner_cik"].astype("Int64")
    s_last = pd.Timestamp(si["settlement_date"].max())
    s = si[pd.to_datetime(si["settlement_date"]) >= s_last - pd.Timedelta(days=SI_SNAPSHOT_DAYS)]
    s = s.sort_values(["settlement_date", "symbol"]).reset_index(drop=True)
    os.makedirs(out, exist_ok=True)
    ev.to_parquet(os.path.join(out, EVENTS_FILE), index=False, compression="zstd")
    s.to_parquet(os.path.join(out, SI_FILE), index=False, compression="zstd")
    man = {"insider_as_of": str(as_of.date()), "insider_last_filing": str(last_filing.date()),
           "insider_window_start": str((as_of - pd.Timedelta(days=days)).date()),
           "si_last_settlement": str(s_last.date()),
           "si_settlements": sorted({str(pd.Timestamp(x).date()) for x in s["settlement_date"].unique()}),
           "built": str(today.date()),
           "rows": {"insider_events": int(len(ev)), "finra_si": int(len(s))},
           "bytes": {f: os.path.getsize(os.path.join(out, f)) for f in (EVENTS_FILE, SI_FILE)},
           "source": "local caches: SEC bulk quarters + daily EDGAR Form 4 feed ($BDA_DATA_V3/sec_insider), "
                     "FINRA bi-monthly files ($BDA_DATA_V3/finra_si)",
           "note": "Refresh before each monthly rebalance: `python -m DataAnalysisPipeline2.v3.altlive refresh` "
                   "then `python -m DataAnalysisPipeline2.trading_agent.altall snapshot`; the bot rejects it "
                   "when the insider feed is older than BDA_ALT_INSIDER_MAX_AGE_DAYS or a FINRA file that "
                   "is already public is missing."}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man


def read_snapshot(path: str = SNAPSHOT_DIR) -> dict:
    files = [os.path.join(path, f) for f in (EVENTS_FILE, SI_FILE, "manifest.json")]
    miss = [f for f in files if not os.path.exists(f)]
    if miss:
        raise AltDataUnavailable(f"alt-data snapshot incomplete: missing {miss}")
    try:
        man = json.load(open(files[2]))
        ev = pd.read_parquet(files[0])
        si = pd.read_parquet(files[1])
    except Exception as e:
        raise AltDataUnavailable(f"alt-data snapshot unreadable ({path}): {e}") from e
    if not len(ev) or not len(si):
        raise AltDataUnavailable("alt-data snapshot is empty")
    return {"events": ev, "si": si, "manifest": man, "insider_as_of": pd.Timestamp(man["insider_as_of"]),
            "source": "snapshot", "path": path}


# --------------------------------------------------------------------------- #
# Freshness
# --------------------------------------------------------------------------- #
def expected_si_floor(decision_date) -> pd.Timestamp:
    """Lower bound of the latest FINRA settlement date that is SURELY public on
    ``decision_date`` (settlement + 8 NYSE business days, taking the latest
    possible settlement day of each half-month); the snapshot must reach it."""
    ad = _v3("altdata")
    d = pd.Timestamp(decision_date)
    for m in pd.period_range(d.to_period("M") - 3, d.to_period("M"), freq="M")[::-1]:
        mid, end = ad._settlement_candidates(m.year, m.month)
        for cands in (end, mid):
            latest = pd.Timestamp(max(cands))
            if pd.Timestamp(ad.next_bday(pd.Series([latest]), SI_LAG_BDAYS).iloc[0]) <= d:
                return pd.Timestamp(min(cands))
    raise ValueError(f"no FINRA settlement found before {d.date()}")


def check_fresh(src: dict, decision_date, insider_max_age_days: int) -> dict:
    """Raise ``AltDataUnavailable`` if ``src`` is too old for ``decision_date``."""
    d = pd.Timestamp(decision_date).normalize()
    ins_age = int((d - src["insider_as_of"].normalize()).days)
    si_last = pd.Timestamp(pd.to_datetime(src["si"]["settlement_date"]).max())
    floor = expected_si_floor(d)
    problems = []
    if ins_age > insider_max_age_days:
        problems.append(f"insider feed ends {src['insider_as_of'].date()} ({ins_age} d before {d.date()} "
                        f"> {insider_max_age_days} d)")
    if si_last < floor:
        problems.append(f"FINRA short interest ends {si_last.date()} but the settlement of >= {floor.date()} "
                        f"is public by {d.date()}")
    if problems:
        raise AltDataUnavailable("stale alt data: " + "; ".join(problems))
    return {"insider_as_of": str(src["insider_as_of"].date()), "insider_age_days": ins_age,
            "si_last_settlement": str(si_last.date()), "si_required_floor": str(floor.date())}


# --------------------------------------------------------------------------- #
# Optional in-runner top-ups
# --------------------------------------------------------------------------- #
def refresh_si(si: pd.DataFrame, decision_date, log=print, session=None) -> pd.DataFrame:
    """Download FINRA files settled after ``si``'s last settlement that are
    public by ``decision_date`` (public CDN, a few requests). Returns the
    combined frame (unchanged on any network failure)."""
    ad = _v3("altdata")
    d = pd.Timestamp(decision_date)
    last = pd.Timestamp(pd.to_datetime(si["settlement_date"]).max())
    try:
        import requests
        get = (session or requests).get
        new = []
        for m in pd.period_range(last.to_period("M"), d.to_period("M"), freq="M"):
            for cands in ad._settlement_candidates(m.year, m.month):
                if pd.Timestamp(max(cands)) <= last:
                    continue
                for c in cands:
                    cd = pd.Timestamp(c)
                    if cd <= last or pd.Timestamp(ad.next_bday(pd.Series([cd]), SI_LAG_BDAYS).iloc[0]) > d:
                        continue
                    r = get(ad.FINRA_URL.format(d=c), timeout=60)
                    if r.status_code == 200 and ad.is_si_payload(r.text):
                        new.append(ad.parse_si_csv(io.StringIO(r.text)))
                        log(f"[altall] FINRA file {c} downloaded ({len(new[-1]):,} rows)")
                        break
        if new:
            si = pd.concat([si] + new, ignore_index=True).drop_duplicates(["symbol", "settlement_date"], keep="last")
    except Exception as e:
        log(f"[altall] FINRA refresh failed ({type(e).__name__}: {e}); using the snapshot")
    return si


def refresh_insider(src: dict, end, ciks: set, budget_s: float, log=print) -> dict:
    """Top up the snapshot with the EDGAR daily Form 4 feed from the day after
    ``insider_as_of`` through ``end`` (needs ``SEC_USER_AGENT``). Time-boxed:
    stops at the first incomplete day or when ``budget_s`` is spent; the new
    ``insider_as_of`` is the last complete day. Never raises."""
    form4 = _v3("form4")
    t0 = time.time()
    start = src["insider_as_of"] + pd.Timedelta(days=1)
    end = pd.Timestamp(end)
    if start > end:
        return src
    old = os.environ.get("BDA_DATA_V3")
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar
        from DataAnalysisPipeline2.v3.fundamentals import nyse_holidays
        hol = set(pd.to_datetime(nyse_holidays())) | set(USFederalHolidayCalendar().holidays(start, end))
        days = [x for x in pd.bdate_range(start, end) if x not in hol]
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["BDA_DATA_V3"] = tmp
            client = form4.SECClient()
            done = None
            for day in days:
                if time.time() - t0 > budget_s:
                    log(f"[altall] Form 4 refresh: time budget {budget_s:.0f}s spent before {day.date()}")
                    break
                n = form4.ingest_day(day, client, ciks, log=log)
                if n == -3 or os.path.exists(os.path.join(form4.cache_dir(), f"{day:%Y-%m-%d}.failed.json")):
                    log(f"[altall] Form 4 refresh: {day.date()} incomplete; stopping there")
                    break
                if n == -2:                  # no index (not published): stop, later days cannot be complete
                    break
                done = day
            if done is None:
                return src
            new = form4.load_daily_events()
        ev = src["events"]
        new = new[~new["accession"].isin(set(ev["accession"]))]
        new = new.assign(issuer_cik=new["issuer_cik"].astype("Int64"), owner_cik=new["owner_cik"].astype("Int64"))
        log(f"[altall] Form 4 refresh: {len(new):,} new P/S lines through {done.date()} "
            f"({client.n} requests, {time.time() - t0:.0f}s)")
        return {**src, "events": pd.concat([ev, new], ignore_index=True), "insider_as_of": done,
                "source": src["source"] + "+edgar_daily"}
    except Exception as e:
        log(f"[altall] Form 4 refresh failed ({type(e).__name__}: {e}); using the snapshot")
        return src
    finally:
        if old is None:
            os.environ.pop("BDA_DATA_V3", None)
        else:
            os.environ["BDA_DATA_V3"] = old


def load_sources(decision_date, *, snapshot_dir: str = SNAPSHOT_DIR, insider_max_age_days: int = 14,
                 refresh: bool = True, refresh_budget_s: float = 900.0, ciks: set | None = None,
                 log=print) -> dict:
    """Snapshot (+ optional top-ups) checked for freshness at ``decision_date``.
    Raises ``AltDataUnavailable``."""
    src = read_snapshot(snapshot_dir)
    d = pd.Timestamp(decision_date)
    if refresh:
        if pd.to_datetime(src["si"]["settlement_date"]).max() < expected_si_floor(d):
            src["si"] = refresh_si(src["si"], d, log=log)
        if (d - src["insider_as_of"]).days > 1 and os.environ.get("SEC_USER_AGENT"):
            src = refresh_insider(src, d - pd.Timedelta(days=1), ciks if ciks is not None
                                  else set(src["events"]["issuer_cik"].dropna().astype("int64")),
                                  refresh_budget_s, log=log)
    src["fresh"] = check_fresh(src, d, insider_max_age_days)
    return src


# --------------------------------------------------------------------------- #
# Model input
# --------------------------------------------------------------------------- #
def read_fundamentals(fund_dir: str) -> dict:
    rd = lambda n: pd.read_parquet(os.path.join(fund_dir, f"{n}.parquet"))
    fa, cm, secs = rd("fundamentals_asof"), rd("cik_map"), rd("sectors")
    need = ["cik", "asof_date", "last_filed", "revenue_ttm", "revenue_ttm_prev", "net_income_ttm",
            "net_income_ttm_prev", "gross_profit_ttm", "cfo_ttm", "assets", "liabilities", "equity",
            "shares", "wdil_yoy"]
    miss = [c for c in need if c not in fa.columns]
    if miss:
        raise AltDataUnavailable(f"fundamentals at {fund_dir} lack {miss} (rebuild the snapshot with "
                                 "`python -m DataAnalysisPipeline2.trading_agent.optimized snapshot`)")
    return {"fund_asof": fa, "cik_map": cm, "sectors": secs}


def model_input(price_history_df: pd.DataFrame, tabular_cols: list, fund: dict, events: pd.DataFrame,
                si: pd.DataFrame, decision_date: str | None = None, meta: pd.DataFrame | None = None,
                first_dates: pd.Series | None = None, log=print) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(eligible decision-date rows with raw features, model input X aligned to
    them with columns ``tabular_cols``, info).

    ``price_history_df``: ticker, Date, company_close, company_volume, close_raw.
    ``meta`` (ticker, Sector) defaults to the EDGAR FF-12 sector of each ticker's
    CIK at its last bar; ``first_dates`` (ticker -> first bar) defaults to the
    first bar in ``price_history_df`` (only used by the SI reused-symbol guard,
    which no >= 120-bar name can trip on a recent settlement)."""
    ds, altlive = _v3("dataset"), _v3("altlive")
    t0 = time.time()
    panel = price_history_df[["ticker", "Date", "company_close", "company_volume", "close_raw"]].copy()
    panel["Date"] = panel["Date"].astype(str).str[:10]
    decision_date = decision_date or str(panel["Date"].max())
    panel = panel[panel["Date"] <= decision_date]
    cm, secs = fund["cik_map"], fund["sectors"]
    if meta is None:
        last = panel.groupby("ticker")["Date"].max()
        meta = ds.edgar_sector_map(last.index.astype(str), last.to_numpy(), cm, secs)
    feats = ds.compute_v3_features(panel, meta, decision_date=decision_date)
    day = feats[feats["Date"] == decision_date].reset_index(drop=True)
    if not day["eligible"].astype(bool).any():
        raise AltDataUnavailable(f"no v3-eligible tickers on {decision_date}")
    block = ds.FundamentalsBlock(fund["fund_asof"], cm)
    fvals, _ = block(day)
    for c in fvals.columns:
        day[c] = fvals[c].to_numpy(dtype=bool) if fvals[c].dtype == bool else fvals[c].to_numpy(dtype=float)
    latest = day[day["eligible"].astype(bool)].reset_index(drop=True)
    core_fund = [c for c in tabular_cols if c in set(ds.FEATURE_COLUMNS) | set(block.columns)]
    Z = ds.zscore(latest, core_fund).reset_index(drop=True)
    alt_cols = [c for c in tabular_cols if c in set(altlive.MODEL_COLUMNS)]
    info = {"decision_date": decision_date, "n_eligible": int(len(latest)),
            "fund_imputed_share": float(latest["fund_imputed"].mean()),
            "sector_source": latest["sector_source"].value_counts().to_dict()}
    if alt_cols:
        cf = latest[["ticker", "Date", "eligible", "close_raw", "company_close", "log_mcap"]].copy()
        fd = first_dates if first_dates is not None else panel.groupby("ticker")["Date"].min()
        cf["first_date"] = cf["ticker"].map(fd)
        for c in altlive.HD_MATCH:
            cf[c] = Z[c].to_numpy() if c in Z else ds.zscore(latest, [c])[c].to_numpy()
        alt = altlive.compute(cf, events, si, cm)
        alt = alt.set_index("ticker").reindex(latest["ticker"])
        for c in alt_cols:
            Z[c] = alt[c].to_numpy(dtype=np.float32)
        d = pd.Timestamp(decision_date)
        av = pd.to_datetime(events["avail_date"])
        active = set(events.loc[(av <= d) & (av > d - pd.Timedelta(days=180)), "issuer_cik"].dropna().astype("int64"))
        s_av, s_set = pd.to_datetime(si["avail_date"]), pd.to_datetime(si["settlement_date"])
        si_syms = set(si.loc[(s_av <= d) & (s_set >= d - pd.Timedelta(days=45)), "symbol"])
        info.update({"insider_mapped_share": float(alt["cik"].notna().mean()),
                     "insider_active_180d_share": float(alt["cik"].dropna().astype("int64").isin(active).sum()
                                                        / max(len(alt), 1)),
                     "si_recent_coverage": float(latest["ticker"].astype(str).str.split("~").str[0]
                                                 .isin(si_syms).mean())})
    missing = [c for c in tabular_cols if c not in Z.columns]
    if missing:
        raise AltDataUnavailable(f"model columns not produced by the live path: {missing}")
    X = Z[list(tabular_cols)].astype(np.float32)
    info["seconds"] = round(time.time() - t0, 1)
    return latest, X, info


# --------------------------------------------------------------------------- #
# Historical live==offline check (local store; used by the tests)
# --------------------------------------------------------------------------- #
def offline_vs_live(date: str, history_days: int = 420, frame_path: str | None = None):
    """(live X, offline X) for every eligible row of the final training frame on
    ``date``: the live path is run on the local price store's window of
    ``history_days`` calendar days before ``date`` with the local fundamentals
    store and alt-data caches."""
    import glob
    ds, fa_mod, form4, ad = _v3("dataset"), _v3("final_altall"), _v3("form4"), _v3("altdata")
    from DataAnalysisPipeline2.v3._alpaca import data_dir
    from DataAnalysisPipeline2.v3.prices import load_prices
    feats = fa_mod.feature_columns()
    off = pd.read_parquet(frame_path or fa_mod.frame_path(), columns=["ticker", "Date"] + feats,
                          filters=[("Date", "==", date)]).sort_values("ticker").reset_index(drop=True)
    start = (pd.Timestamp(date) - pd.Timedelta(days=history_days)).strftime("%Y-%m-%d")
    px = load_prices(start=start, end=date, columns=["company_close", "company_volume"], with_raw=True)
    px = px[["ticker", "Date", "company_close", "company_volume", "close_raw"]]
    sa = sorted(glob.glob(os.path.join(ds.v3_dir(ds.EDGAR_DATASET), "stage_a", "*.parquet")))
    span = pd.concat([pd.read_parquet(f, columns=["ticker", "Date"]).groupby("ticker")["Date"].agg(["min", "max"])
                      for f in sa])
    fund = read_fundamentals(data_dir("fundamentals"))
    meta = ds.edgar_sector_map(span.index.astype(str), span["max"].to_numpy(), fund["cik_map"], fund["sectors"])
    latest, X, info = model_input(px, feats, fund, form4.load_events(), ad.load_si(), decision_date=date,
                                  meta=meta, first_dates=span["min"])
    live = pd.concat([latest[["ticker"]], X], axis=1).sort_values("ticker").reset_index(drop=True)
    return live, off, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="altall (sn30_altall_hd) live alt-data snapshot")
    ap.add_argument("cmd", choices=["snapshot", "check"])
    ap.add_argument("--out", default=SNAPSHOT_DIR)
    ap.add_argument("--days", type=int, default=SNAPSHOT_DAYS)
    ap.add_argument("--date", default=None, help="decision date for 'check' (default today)")
    a = ap.parse_args(argv)
    if a.cmd == "snapshot":
        print(json.dumps(build_snapshot(a.out, a.days), indent=1))
    else:
        src = read_snapshot(a.out)
        try:
            print(json.dumps(check_fresh(src, a.date or dt.date.today(), 14), indent=1))
        except AltDataUnavailable as e:
            print(f"STALE: {e}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

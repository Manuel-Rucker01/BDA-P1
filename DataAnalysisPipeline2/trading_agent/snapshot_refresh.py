"""Incremental refresh of the committed live-data snapshots, without ``$BDA_DATA_V3``.

The cloud counterpart (``.github/workflows/refresh_snapshots.yml``) of the local
monthly task

    python -m DataAnalysisPipeline2.v3.altlive refresh
    python -m DataAnalysisPipeline2.trading_agent.optimized snapshot
    python -m DataAnalysisPipeline2.trading_agent.altall snapshot

which needs the local caches (SEC bulk quarters, FINRA files, the EDGAR
fundamentals store). Here each snapshot is topped up from its own committed
state plus public sources only, giving the same files the local path writes:

* ``altdata`` (``trading_agent/data/altdata_snapshot/{insider_events,finra_si}.parquet``):
  the EDGAR daily Form 4/5 feed from the day after ``insider_as_of`` through
  ``end`` (``altall.refresh_insider``: per-day index + one request per filing,
  <= 8 req/s), plus FINRA short-interest files that are public by today
  (``altall.refresh_si``, public CDN). It is then trimmed and written by
  ``altall.build_snapshot``. The 400-day insider window and the 130-day SI window only
  need the snapshot itself, so the top-up is exact.
* ``earnings`` (``earnings_events.parquet``): the 8-K item 2.02 events filed after
  the manifest's ``as_of`` (``earnlive.fetch``: daily index + per-CIK submissions
  JSON), merged with the snapshot's retained events and re-compacted. Exact,
  because the retained-event filter only depends on the previous retained event
  (``earnings.compact``).
* ``fundamentals`` (``fundamentals_snapshot/fundamentals_asof.parquet``): SEC's
  nightly ``companyfacts.zip`` (one ~1.4 GB download), rebuilt with
  ``fundamentals.build_company_asof`` for the snapshot's CIKs (~8k, a couple of
  minutes) and trimmed by ``optimized.build_snapshot``. This is the same
  computation as the local ``v3.fundamentals build`` + ``optimized snapshot``.
  ``cik_map`` / ``sectors`` are kept as committed. They come from
  ``company_tickers_exchange.json`` + ``submissions.zip`` and change slowly,
  so they are refreshed by the local / quarterly path only.

Every part writes its files only after it succeeds, so a failed part leaves the
committed snapshot as it was. Freshness is checked after the writes (the
bot's own rules: insider feed <= ``BDA_ALT_INSIDER_MAX_AGE_DAYS``, the latest
public FINRA settlement present, earnings ``as_of`` <= 14 d). The CLI exits 1
if any part failed, and 2 if ``SEC_USER_AGENT`` is not configured.

    python -m DataAnalysisPipeline2.trading_agent.snapshot_refresh [--parts altdata,earnings,fundamentals]
        [--end YYYY-MM-DD] [--insider-budget-s 7200] [--companyfacts PATH]
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import tempfile
import time

import numpy as np
import pandas as pd

from . import altall, earnings, optimized

PARTS = ("altdata", "earnings", "fundamentals")
ALT_DIR = altall.SNAPSHOT_DIR
FUND_DIR = optimized.SNAPSHOT_DIR
MAX_LAG_DAYS = 4                # the insider feed must reach within this many days of ``end``
CI_SOURCE = "incremental top-up from the committed snapshot (trading_agent/snapshot_refresh.py)"


class RefreshFailed(RuntimeError):
    pass


@contextlib.contextmanager
def _temp_data_dir():
    """Point ``$BDA_DATA_V3`` at a throw-away directory (nothing local is read or written)."""
    old = os.environ.get("BDA_DATA_V3")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["BDA_DATA_V3"] = tmp
        try:
            yield tmp
        finally:
            if old is None:
                os.environ.pop("BDA_DATA_V3", None)
            else:
                os.environ["BDA_DATA_V3"] = old


def require_user_agent() -> str:
    """The SEC User-Agent (env or trading_agent/.env); raises if missing/invalid. Never logged."""
    from DataAnalysisPipeline2.v3 import edgar
    return edgar.sec_user_agent()


def _patch_manifest(path: str, **fields) -> dict:
    with open(path) as f:
        man = json.load(f)
    man.update(fields)
    with open(path, "w") as f:
        json.dump(man, f, indent=1)
    return man


def universe_ciks(fund_dir: str = FUND_DIR) -> set:
    cm = pd.read_parquet(os.path.join(fund_dir, "cik_map.parquet"), columns=["cik"])
    return set(cm["cik"].dropna().astype("int64"))


def last_session_on_or_before(d) -> pd.Timestamp:
    from DataAnalysisPipeline2.v3.earnlive import _business_days
    d = pd.Timestamp(d)
    days = _business_days(d - pd.Timedelta(days=10), d)
    return pd.Timestamp(days[-1])


# --------------------------------------------------------------------------- #
# alt data (Form 4 + FINRA)
# --------------------------------------------------------------------------- #
def refresh_altdata(end, *, out: str = ALT_DIR, fund_dir: str = FUND_DIR, today=None,
                    budget_s: float = 7200.0, refresh_insider=None, refresh_si=None, log=print) -> dict:
    """Top up the alt-data snapshot through ``end`` and rewrite it. Raises
    ``RefreshFailed`` (files untouched) if the feeds do not reach ``end``."""
    today = pd.Timestamp(today or dt.date.today())
    end = pd.Timestamp(end)
    refresh_insider = refresh_insider or altall.refresh_insider
    refresh_si = refresh_si or altall.refresh_si
    src = altall.read_snapshot(out)
    old_as_of = src["insider_as_of"]
    ciks = universe_ciks(fund_dir) | set(src["events"]["issuer_cik"].dropna().astype("int64"))
    si = refresh_si(src["si"], today, log=log)
    floor = altall.expected_si_floor(today)
    si_last = pd.Timestamp(pd.to_datetime(si["settlement_date"]).max())
    if si_last < floor:
        raise RefreshFailed(f"FINRA short interest ends {si_last.date()} but the settlement of >= {floor.date()} "
                            f"is public by {today.date()} (download failed?)")
    with _temp_data_dir():
        new = refresh_insider(src, end, ciks, budget_s, log=log)
    target = last_session_on_or_before(end)
    lag = int((target - pd.Timestamp(new["insider_as_of"])).days)
    if lag > MAX_LAG_DAYS:
        raise RefreshFailed(f"Form 4 feed only reached {pd.Timestamp(new['insider_as_of']).date()} "
                            f"(target {target.date()}, started from {old_as_of.date()}); see the log above")
    man = altall.build_snapshot(out, altall.SNAPSHOT_DAYS, events=new["events"], si=si,
                                insider_as_of=new["insider_as_of"], ciks=ciks, today=today)
    man = _patch_manifest(os.path.join(out, "manifest.json"), source=CI_SOURCE + ": EDGAR daily Form 4/5 feed "
                          "+ FINRA bi-monthly files on top of the previous snapshot",
                          refreshed_from=str(old_as_of.date()))
    return {"insider_as_of": man["insider_as_of"], "from": str(old_as_of.date()),
            "si_last_settlement": man["si_last_settlement"], "rows": man["rows"],
            "new_events": int(len(new["events"]) - len(src["events"]))}


# --------------------------------------------------------------------------- #
# earnings (8-K item 2.02)
# --------------------------------------------------------------------------- #
def kept_to_raw(kept: pd.DataFrame) -> pd.DataFrame:
    """Snapshot rows (cik, adate) -> ``retained_events`` input rows whose New
    York acceptance date is ``adate`` (noon NY)."""
    acc = (pd.to_datetime(kept["adate"]) + pd.Timedelta(hours=12)).dt.tz_localize("America/New_York")
    return pd.DataFrame({"cik": kept["cik"].astype("int64").to_numpy(),
                         "acceptance": acc.dt.tz_convert("UTC").to_numpy(), "src10q": 0})


def refresh_earnings(end, *, out: str = ALT_DIR, fund_dir: str = FUND_DIR, today=None, client=None,
                     fetch=None, log=print) -> dict:
    from DataAnalysisPipeline2.v3 import earnlive
    today = pd.Timestamp(today or dt.date.today())
    end = pd.Timestamp(end)
    fetch = fetch or earnlive.fetch
    snap = earnings.read_snapshot(out)
    as_of = pd.Timestamp(snap["as_of"])
    start = as_of + pd.Timedelta(days=1)
    if start > end:
        return {"as_of": str(as_of.date()), "up_to_date": True}
    ciks = universe_ciks(fund_dir) | set(snap["kept"]["cik"].astype("int64"))
    f, covered, info = fetch(start, end, ciks, client=client, log=log)
    if covered is None:
        raise RefreshFailed(f"8-K 2.02 refresh covered no day from {start.date()} ({info})")
    raw_new = earnlive.to_raw(f)
    raw = pd.concat([kept_to_raw(snap["kept"]), raw_new], ignore_index=True)
    raw["acceptance"] = pd.to_datetime(raw["acceptance"], utc=True)
    raw = raw.drop_duplicates(["cik", "acceptance"]).sort_values(["cik", "acceptance"]).reset_index(drop=True)
    man = earnings.build_snapshot(out, raw=raw, as_of=covered, today=today)
    man = _patch_manifest(os.path.join(out, earnings.MANIFEST_FILE),
                          source=CI_SOURCE + ": EDGAR daily index + per-CIK submissions JSON (v3/earnlive.fetch) "
                                 "on top of the previous snapshot's retained events",
                          raw_8k_202_events_added=int(len(raw_new)), refreshed_from=str(as_of.date()))
    man.pop("raw_8k_202_events", None)
    with open(os.path.join(out, earnings.MANIFEST_FILE), "w") as fh:
        json.dump(man, fh, indent=1)
    return {"as_of": man["as_of"], "from": str(as_of.date()), "new_202_events": int(len(raw_new)),
            "rows": man["rows"], "fetch": info}


# --------------------------------------------------------------------------- #
# fundamentals (companyfacts.zip)
# --------------------------------------------------------------------------- #
def download_companyfacts(dest_dir: str, log=print) -> str:
    """SEC's nightly companyfacts.zip (one resumable request, <= 8 req/s limiter)."""
    from pathlib import Path

    from DataAnalysisPipeline2.v3 import edgar
    with _temp_data_dir():          # edgar.download records a manifest under $BDA_DATA_V3/edgar
        t0 = time.time()
        p = edgar.download(edgar.COMPANYFACTS_URL, dest=Path(dest_dir) / "companyfacts.zip", force=True)
    log(f"[snapshot_refresh] companyfacts.zip: {os.path.getsize(p) / 1e9:.2f} GB in {time.time() - t0:.0f}s")
    return str(p)


def build_fund_asof(zip_path: str, ciks, log=print) -> pd.DataFrame:
    """``fundamentals_asof`` rows for ``ciks`` from a companyfacts zip (same per-company
    builder as ``fundamentals.build_fundamentals_asof``)."""
    import zipfile

    from DataAnalysisPipeline2.v3 import fundamentals as fu
    parts, n_missing, n_bad = [], 0, 0
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for k, c in enumerate(sorted(int(x) for x in ciks), 1):
            nm = f"CIK{c:010d}.json"
            if nm not in names:
                n_missing += 1
                continue
            try:
                df = fu.build_company_asof(fu._loads(z.read(nm)), c)
            except Exception as e:  # noqa: BLE001 -- as in the bulk build: skip a malformed company
                n_bad += 1
                log(f"[snapshot_refresh] skip CIK {c}: {type(e).__name__}: {e}")
                continue
            if not df.empty:
                parts.append(df)
            if k % 2000 == 0:
                log(f"[snapshot_refresh] companyfacts: {k:,} CIKs ({time.time() - t0:.0f}s)")
    if not parts:
        raise RefreshFailed("companyfacts.zip produced no fundamentals rows for the snapshot CIKs")
    res = pd.concat(parts, ignore_index=True).sort_values(["cik", "asof_date"]).reset_index(drop=True)
    log(f"[snapshot_refresh] fundamentals_asof: {len(res):,} rows, {res['cik'].nunique():,} CIKs "
        f"({n_missing} without companyfacts, {n_bad} malformed, {time.time() - t0:.0f}s)")
    return res


def refresh_fundamentals(*, out: str = FUND_DIR, today=None, zip_path: str | None = None,
                         workdir: str | None = None, log=print) -> dict:
    today = pd.Timestamp(today or dt.date.today())
    old = optimized.fund_as_of(out)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        zp = zip_path or download_companyfacts(tmp, log=log)
        cm = pd.read_parquet(os.path.join(out, "cik_map.parquet"))
        secs = pd.read_parquet(os.path.join(out, "sectors.parquet"))
        fa = build_fund_asof(zp, set(cm["cik"].dropna().astype("int64")), log=log)
        new_as_of = pd.Timestamp(fa["last_filed"].max())
        if new_as_of < old:
            raise RefreshFailed(f"companyfacts newest filing {new_as_of.date()} is older than the snapshot's "
                                f"{old.date()}")
        src = os.path.join(tmp, "store")
        os.makedirs(src)
        cm.to_parquet(os.path.join(src, "cik_map.parquet"), index=False)
        secs.to_parquet(os.path.join(src, "sectors.parquet"), index=False)
        fa.to_parquet(os.path.join(src, "fundamentals_asof.parquet"), index=False)
        stage = os.path.join(tmp, "out")      # write to a staging dir, then move into place
        man = optimized.build_snapshot(src=src, out=stage, today=today)
        for f in optimized.FUND_FILES + ("manifest.json",):
            os.replace(os.path.join(stage, f), os.path.join(out, f))
    man = _patch_manifest(os.path.join(out, "manifest.json"),
                          source=CI_SOURCE + ": SEC companyfacts.zip rebuilt per CIK "
                                 "(fundamentals.build_company_asof); cik_map/sectors kept from the previous "
                                 "snapshot", refreshed_from=str(old.date()))
    return {"as_of": man["as_of"], "from": str(old.date()), "rows": man["rows"]}


# --------------------------------------------------------------------------- #
# Verification (the bot's own readers / freshness rules)
# --------------------------------------------------------------------------- #
def verify(today=None, *, alt_dir: str = ALT_DIR, fund_dir: str = FUND_DIR,
           insider_max_age_days: int | None = None) -> dict:
    today = pd.Timestamp(today or dt.date.today()).normalize()
    age = int(insider_max_age_days if insider_max_age_days is not None
              else os.environ.get("BDA_ALT_INSIDER_MAX_AGE_DAYS", 14))
    out, problems = {}, []
    try:
        out["altdata"] = altall.check_fresh(altall.read_snapshot(alt_dir), today, age)
    except Exception as e:  # noqa: BLE001
        problems.append(f"altdata: {e}")
    try:
        out["earnings"] = {k: str(v) for k, v in earnings.resolve_events(
            today, 14, candidates=[("snapshot", lambda: earnings.read_snapshot(alt_dir))]).items()
            if k in ("as_of", "age_days")}
    except Exception as e:  # noqa: BLE001
        problems.append(f"earnings: {e}")
    try:
        r = optimized.resolve_fund_dir(max_age_days=int(os.environ.get("BDA_OPT_FUND_MAX_AGE_DAYS", 120)),
                                       today=today, candidates=[("snapshot", fund_dir)])
        out["fundamentals"] = {"as_of": r["as_of"], "age_days": r["age_days"]}
        altall.read_fundamentals(fund_dir)
    except Exception as e:  # noqa: BLE001
        problems.append(f"fundamentals: {e}")
    out["problems"] = problems
    return out


def run(parts=PARTS, end=None, today=None, insider_budget_s: float = 7200.0, companyfacts: str | None = None,
        log=print) -> dict:
    today = pd.Timestamp(today or dt.date.today())
    end = pd.Timestamp(end) if end else today - pd.Timedelta(days=1)
    res = {"today": str(today.date()), "end": str(end.date()), "parts": {}, "failed": []}
    for part in parts:
        t0 = time.time()
        try:
            if part == "altdata":
                info = refresh_altdata(end, today=today, budget_s=insider_budget_s, log=log)
            elif part == "earnings":
                info = refresh_earnings(end, today=today, log=log)
            elif part == "fundamentals":
                info = refresh_fundamentals(today=today, zip_path=companyfacts, log=log)
            else:
                raise ValueError(f"unknown part {part!r}")
            info["seconds"] = round(time.time() - t0)
            res["parts"][part] = {"ok": True, **info}
        except Exception as e:  # noqa: BLE001 -- one part failing must not block the others
            log(f"[snapshot_refresh] {part} FAILED: {type(e).__name__}: {e}")
            res["parts"][part] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            res["failed"].append(part)
    res["verify"] = verify(today)
    if res["verify"]["problems"]:
        res["failed"].append("verify")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Incremental refresh of the committed live-data snapshots")
    ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--end", default=None, help="last EDGAR day to ingest (default yesterday)")
    ap.add_argument("--insider-budget-s", type=float, default=float(os.environ.get("BDA_SNAPSHOT_BUDGET_S", 7200)))
    ap.add_argument("--companyfacts", default=None, help="use a local companyfacts.zip instead of downloading")
    a = ap.parse_args(argv)
    try:
        require_user_agent()
    except Exception as e:  # noqa: BLE001
        print(f"SEC_USER_AGENT missing or invalid: {e}")
        return 2
    parts = [p.strip() for p in a.parts.split(",") if p.strip()]
    res = run(parts, a.end, insider_budget_s=a.insider_budget_s, companyfacts=a.companyfacts)
    print("REFRESH_JSON: " + json.dumps(res, default=str))
    return 1 if res["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

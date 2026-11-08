"""Round-11 ``earn_blackout`` rule for the live ``optimized_v3`` book (PAPER).

Rule (``v3/round11.py``, pre-registered in ``v3/gate.py`` R11_*): a name whose
expected next earnings announcement falls in [R - 10 calendar days, the 5th
session after R] (R = the rebalance session: the first session after the decision
date and on/after the run date) may not be bought or increased: its optimizer upper bound is its current
drifted weight (0 if not held). Expected date = the CIK's last retained 8-K
item 2.02 event accepted on/before the decision date (New York date) + 91
days; an event is retained only if >= 60 days after the CIK's previous
retained event. Same functions as the backtest (``round11.expected_dates``,
``blackout_window``, ``blackout_tickers``).

Live caps: the optimizer builds a fully-invested book that the bot scales to
``OPT_TARGET_EXPOSURE``; a blacked-out name is capped at
``current weight / target exposure`` in the optimizer so that the traded
weight never exceeds the current weight (fraction of equity, Alpaca positions).

Event data:

* local: ``$BDA_DATA_V3/sec_earn`` bulk store + the live 8-K cache
  (``v3/earnlive.py``, refreshed by ``python -m DataAnalysisPipeline2.v3.altlive refresh``);
* cloud: committed snapshot ``trading_agent/data/altdata_snapshot/earnings_events.parquet``
  (the last ``KEEP_PER_CIK`` retained events of every CIK -- the retained
  filter only depends on the previous retained event, so this is exact for any
  decision after the snapshot's events) + ``earnings_manifest.json`` (``as_of``
  = last EDGAR day covered). Rebuilt by ``python -m DataAnalysisPipeline2.trading_agent.altall snapshot``
  (or ``python -m DataAnalysisPipeline2.trading_agent.earnings snapshot``).

Staleness: a source whose ``as_of`` is more than ``BDA_EARN_MAX_AGE_DAYS``
(14) days before the decision date is not used; with no usable source the
blackout is SKIPPED (logged) -- it never fails the rebalance.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(AGENT_DIR, "data", "altdata_snapshot")
EVENTS_FILE = "earnings_events.parquet"
MANIFEST_FILE = "earnings_manifest.json"
KEEP_PER_CIK = 3


class EarningsUnavailable(RuntimeError):
    """No usable (fresh) earnings-event source: the blackout is skipped."""


def _r11():
    from DataAnalysisPipeline2.v3 import round11
    return round11


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def compact(raw: pd.DataFrame, keep: int = KEEP_PER_CIK) -> pd.DataFrame:
    """``round11.retained_events`` over the full history, then the last ``keep``
    retained events per CIK (cik int64, adate 'YYYY-MM-DD')."""
    kept = _r11().retained_events(raw)
    kept = kept.sort_values(["cik", "adate"]).groupby("cik").tail(keep).reset_index(drop=True)
    kept["cik"] = kept["cik"].astype("int64")
    return kept


def build_snapshot(out: str = SNAPSHOT_DIR, raw: pd.DataFrame | None = None, as_of=None,
                   keep: int = KEEP_PER_CIK, today=None) -> dict:
    """Write the compact retained-event table + manifest from the local store
    (bulk ``sec_earn`` events + live 8-K cache)."""
    from DataAnalysisPipeline2.v3 import earnlive
    today = pd.Timestamp(today or dt.date.today())
    raw = earnlive.raw_events() if raw is None else raw
    as_of = pd.Timestamp(as_of) if as_of is not None else earnlive.coverage_as_of()
    if as_of is None:
        raise EarningsUnavailable("no coverage date for the earnings events")
    kept = compact(raw, keep)
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, EVENTS_FILE)
    kept.to_parquet(p, index=False, compression="zstd")
    man = {"as_of": str(as_of.date()), "built": str(today.date()), "keep_per_cik": keep,
           "rows": int(len(kept)), "ciks": int(kept["cik"].nunique()), "last_event": str(kept["adate"].max()),
           "raw_8k_202_events": int(len(raw)), "bytes": os.path.getsize(p),
           "source": "EDGAR 8-K item 2.02: bulk submissions.zip ($BDA_DATA_V3/sec_earn/events.parquet) + "
                     "daily index / per-CIK submissions JSON top-up (v3/earnlive.py)",
           "note": "Round-11 earn_blackout input. Refresh: `python -m DataAnalysisPipeline2.v3.altlive refresh` then "
                   "`python -m DataAnalysisPipeline2.trading_agent.altall snapshot`; the bot skips the blackout "
                   "when as_of is older than BDA_EARN_MAX_AGE_DAYS before the decision date."}
    with open(os.path.join(out, MANIFEST_FILE), "w") as f:
        json.dump(man, f, indent=1)
    return man


def read_snapshot(path: str = SNAPSHOT_DIR) -> dict:
    ev, mf = os.path.join(path, EVENTS_FILE), os.path.join(path, MANIFEST_FILE)
    if not (os.path.exists(ev) and os.path.exists(mf)):
        raise EarningsUnavailable(f"earnings snapshot missing in {path}")
    with open(mf) as f:
        man = json.load(f)
    kept = pd.read_parquet(ev)
    if not len(kept):
        raise EarningsUnavailable("earnings snapshot is empty")
    return {"kept": kept, "as_of": pd.Timestamp(man["as_of"]), "source": "snapshot", "path": path}


def read_store() -> dict:
    from DataAnalysisPipeline2.v3 import earnlive
    if not os.path.exists(os.path.join(earnlive.earn_dir(), "events.parquet")):
        raise EarningsUnavailable("no local sec_earn store")
    as_of = earnlive.coverage_as_of()
    return {"kept": compact(earnlive.raw_events()), "as_of": as_of, "source": "store", "path": earnlive.earn_dir()}


def _store_exists() -> bool:
    from DataAnalysisPipeline2.v3._alpaca import DEFAULT_DATA_DIR
    root = os.environ.get("BDA_DATA_V3", DEFAULT_DATA_DIR)
    return os.path.exists(os.path.join(root, "sec_earn", "events.parquet"))


def resolve_events(decision_date, max_age_days: int = 14, snapshot_dir: str = SNAPSHOT_DIR,
                   candidates: list | None = None) -> dict:
    """First source (local store if present, then the snapshot) whose ``as_of``
    is within ``max_age_days`` of ``decision_date``. ``candidates``: list of
    (name, loader) for tests. Raises ``EarningsUnavailable``."""
    d = pd.Timestamp(decision_date).normalize()
    if candidates is None:
        candidates = ([("store", read_store)] if _store_exists() else []) + \
            [("snapshot", lambda: read_snapshot(snapshot_dir))]
    reasons = []
    for name, load in candidates:
        try:
            src = load()
        except Exception as e:      # noqa: BLE001 -- unreadable / missing source
            reasons.append(f"{name}: {type(e).__name__}: {e}")
            continue
        if src.get("as_of") is None:
            reasons.append(f"{name}: no as_of")
            continue
        age = int((d - pd.Timestamp(src["as_of"]).normalize()).days)
        if age > max_age_days:
            reasons.append(f"{name}: stale (events through {pd.Timestamp(src['as_of']).date()}, {age} d before "
                           f"{d.date()} > {max_age_days} d)")
            continue
        return {**src, "age_days": age}
    raise EarningsUnavailable("no usable earnings events: " + "; ".join(reasons))


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #
def session_calendar(decision_date, days_after: int = 40, days_before: int = 40) -> np.ndarray:
    """NYSE sessions (weekdays minus NYSE holidays) around ``decision_date``."""
    from DataAnalysisPipeline2.v3.fundamentals import nyse_holidays
    d = pd.Timestamp(decision_date)
    hol = set(pd.to_datetime(nyse_holidays()))
    days = [x for x in pd.bdate_range(d - pd.Timedelta(days=days_before), d + pd.Timedelta(days=days_after))
            if x not in hol]
    return np.array([x.strftime("%Y-%m-%d") for x in days])


def rebalance_session(decision_date, calendar, today=None) -> str:
    """The session the book is traded on: the first session after the decision
    date and on/after ``today`` (the backtest's R = the plan's trade session)."""
    cal = np.asarray(calendar).astype(str)
    k = int(np.searchsorted(cal, str(pd.Timestamp(decision_date).date()), side="right"))
    if today is not None:
        k = max(k, int(np.searchsorted(cal, str(pd.Timestamp(today).date()), side="left")))
    return str(cal[k])


def cik_lookup(tickers, decision_date, cik_map: pd.DataFrame) -> dict:
    """ticker -> CIK on the decision date (``altlive._cik`` = the alt-data rule)."""
    from DataAnalysisPipeline2.v3 import altlive
    rows = pd.DataFrame({"ticker": np.asarray(tickers).astype(str), "Date": str(decision_date)})
    c = altlive._cik(rows, cik_map)
    return {t: int(v) for t, v in zip(rows["ticker"], c) if pd.notna(v)}


def blackout_caps(tickers, decision_date, kept: pd.DataFrame, cik_of: dict, w_cur: dict,
                  target_exposure: float = 1.0, calendar=None, today=None) -> tuple[dict, dict]:
    """(ticker -> optimizer cap for blacked-out names, info). Cap = max(current
    weight, 0) / ``target_exposure``. Rebalance session R: ``rebalance_session``."""
    r11 = _r11()
    tickers = np.asarray(tickers).astype(str)
    cal = session_calendar(decision_date) if calendar is None else np.asarray(calendar).astype(str)
    dd = str(pd.Timestamp(decision_date).date())
    reb = rebalance_session(dd, cal, today)
    exp = r11.expected_dates(kept, dd)
    win = r11.blackout_window(reb, cal)
    mask = r11.blackout_tickers(tickers, cik_of, exp, win)
    cur = np.array([max(float((w_cur or {}).get(t, 0.0) or 0.0), 0.0) for t in tickers])
    te = float(target_exposure)
    caps = {t: float(c) / te for t, c, m in zip(tickers, cur, mask) if m}
    names = sorted(tickers[mask].tolist())
    info = {"applied": True, "decision_date": dd, "rebalance_session": reb, "window": list(win),
            "n_universe": int(len(tickers)), "n_mapped": int(sum(1 for t in tickers if t in cik_of)),
            "n_blackout": int(mask.sum()), "blackout_frac": float(mask.mean()) if len(mask) else 0.0,
            "n_blackout_held": int((cur[mask] > 0).sum()), "blackout_held_w": float(cur[mask].sum()),
            "held": {t: round(float(c), 5) for t, c, m in zip(tickers, cur, mask) if m and c > 0},
            "names": names}
    return caps, info


def live_caps(tickers, decision_date, w_cur: dict, target_exposure: float, fund_dir: str, *,
              max_age_days: int = 14, snapshot_dir: str = SNAPSHOT_DIR, today=None,
              log=print) -> tuple[dict, dict]:
    """Bot entry point: resolve a fresh event source and compute the caps.
    Raises (``EarningsUnavailable`` when stale / missing) -- the caller skips."""
    src = resolve_events(decision_date, max_age_days, snapshot_dir)
    cm = pd.read_parquet(os.path.join(fund_dir, "cik_map.parquet"))
    cik_of = cik_lookup(tickers, decision_date, cm)
    today = pd.Timestamp(today or dt.date.today())
    caps, info = blackout_caps(tickers, decision_date, src["kept"], cik_of, w_cur, target_exposure, today=today)
    info.update({"events_source": src["source"], "events_as_of": str(pd.Timestamp(src["as_of"]).date()),
                 "events_age_days": src["age_days"]})
    log(f"[earn_blackout] events {src['source']} through {info['events_as_of']} ({src['age_days']} d); "
        f"rebalance {info['rebalance_session']}, window {win_str(info['window'])}: {info['n_blackout']} of "
        f"{info['n_universe']} names blacked out ({info['n_blackout_held']} held, "
        f"{info['blackout_held_w']:.2%} of equity)")
    return caps, info


def win_str(w) -> str:
    return f"[{w[0]} .. {w[1]}]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="earn_blackout event snapshot")
    ap.add_argument("cmd", choices=["snapshot", "check"])
    ap.add_argument("--out", default=SNAPSHOT_DIR)
    ap.add_argument("--date", default=None, help="decision date for 'check' (default today)")
    a = ap.parse_args(argv)
    if a.cmd == "snapshot":
        print(json.dumps(build_snapshot(a.out), indent=1))
        return 0
    try:
        src = resolve_events(a.date or dt.date.today(), 14, a.out, [("snapshot", lambda: read_snapshot(a.out))])
        print(json.dumps({"as_of": str(src["as_of"].date()), "age_days": src["age_days"], "rows": len(src["kept"])}))
    except EarningsUnavailable as e:
        print(f"STALE: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

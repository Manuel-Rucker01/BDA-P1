"""Analyst-estimate adapter (consensus EPS, revisions, dispersion, coverage).

There is no paid estimates data in this project. Two providers implement the
same interface:

* ``YahooEstimates`` (default, free, no key): Yahoo Finance ``quoteSummary``
  module ``earningsTrend`` via ``yfinance`` (pinned 1.2.0). One request per
  ticker returns, for the periods ``0q`` (current quarter = next to report),
  ``+1q``, ``0y`` (current fiscal year) and ``+1y``, what yfinance exposes as
  ``Ticker.earnings_estimate`` (avg / low / high / numberOfAnalysts /
  yearAgoEps / growth), ``Ticker.eps_trend`` (consensus now and 7 / 30 / 60 / 90
  days ago), ``Ticker.eps_revisions`` (# analysts revising up / down over the
  last 7 / 30 days) and ``Ticker.revenue_estimate``, plus the period
  ``endDate`` and currency (only in the raw payload, read from the same
  cached response -- no extra request).
* ``FMPEstimates``: Financial Modeling Prep, **disabled unless
  ``FMP_API_KEY``** is set (process env or ``trading_agent/.env``).

Point-in-time design
--------------------
Vendors serve the *current* consensus, not its history. Using today's
consensus for past dates would be look-ahead, so the adapter is
snapshot-based: each daily run stores what the provider returns with
``asof_date`` = the UTC fetch date and ``available_date`` = the next weekday
(the GitHub workflow fetches at 22:30 UTC, after the US close, so the data is
usable from the next session on). ``estimate_features`` only uses snapshots
with ``available_date <= Date``. **History starts on the first snapshot
date; nothing can be backfilled.** Own revisions over 1 / 3 months become
available 1 / 3 months after the first snapshot.

Two kinds of revision features
------------------------------
* ``est_*`` (true PIT): computed from *our own* accumulated snapshots
  (consensus in the latest usable snapshot vs the latest snapshot >= 30 / 91
  calendar days older, same ticker / horizon / fiscal period end).
* ``estv_*`` (vendor-reported, NOT verified PIT): computed from Yahoo's own
  "N days ago" consensus and up/down revision counts inside a single
  snapshot. Usable from the first snapshot, but we cannot verify that Yahoo's
  "90 days ago" value is what was actually published then (restatements,
  analyst-set changes). Treat them as a separate block in any test.

Features per (ticker, Date):
  est_eps             consensus mean EPS, current quarter (``0q``; for FMP rows:
                      the earliest fiscal quarter ending after the snapshot)
  est_rev_1m/3m       (eps_now - eps_then) / max(|eps_then|, 0.01), quarter target
  est_rev_fy_1m/3m    same for the current fiscal year (``0y``)
  est_dispersion      (high - low) / max(|mean|, 0.01)
  est_n_analysts      number of analysts (quarter)
  est_age_days        Date - asof_date of the snapshot used
  estv_rev_7d/30d/90d (current - N days ago) / max(|N days ago|, 0.01), ``0q``
  estv_rev_fy_30d/90d same for ``0y``
  estv_breadth_7d/30d (up - down) / max(n_analysts, 1), ``0q``
  estv_breadth_fy_30d same for ``0y``

Storage
-------
* ``write_daily`` / ``load_dir``: one Parquet per fetch day,
  ``<root>/estimates/YYYY/YYYY-MM-DD.parquet`` (also reads the legacy flat
  ``snapshot_<date>.parquet`` files of ``SnapshotStore``).
* The GitHub workflow ``.github/workflows/estimates_snapshot.yml`` commits the
  daily file to the orphan branch ``data/estimates``; ``load_branch_snapshots``
  reads every file from that branch with plain ``git`` (no checkout).

CLI::

    python -m DataAnalysisPipeline2.v3.estimates universe            # refresh committed ticker list (local data)
    python -m DataAnalysisPipeline2.v3.estimates snapshot --out DIR  # fetch today's snapshot
    python -m DataAnalysisPipeline2.v3.estimates load                # summary of data/estimates
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import io
import json
import logging
import os
import random
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UNIVERSE_FILE = os.path.join(REPO_ROOT, "DataAnalysisPipeline2", "trading_agent", "data", "estimates_universe.txt")
PREDICTIONS_GLOB = os.path.join(REPO_ROOT, "ops", "predictions", "*", "*.csv.gz")
DATA_BRANCH = "data/estimates"

# The first seven columns are the original (FMP) schema; the rest were added for
# the Yahoo provider. Older files are reindexed to this list on load (NaN fill).
BASE_COLUMNS = ["ticker", "asof_date", "period_end", "eps_avg", "eps_high", "eps_low", "n_analysts"]
EXTRA_COLUMNS = [
    "source", "period", "available_date", "fetched_at", "provider_symbol",
    "eps_currency", "eps_year_ago", "eps_growth",
    "eps_trend_current", "eps_trend_7d", "eps_trend_30d", "eps_trend_60d", "eps_trend_90d",
    "eps_up_7d", "eps_up_30d", "eps_down_7d", "eps_down_30d",
    "revenue_avg", "revenue_low", "revenue_high", "revenue_n_analysts", "revenue_year_ago",
    "revenue_growth", "revenue_currency",
]
SNAPSHOT_COLUMNS = BASE_COLUMNS + EXTRA_COLUMNS
STRING_COLUMNS = ["ticker", "asof_date", "period_end", "source", "period", "available_date", "fetched_at",
                  "provider_symbol", "eps_currency", "revenue_currency"]
NUMERIC_COLUMNS = [c for c in SNAPSHOT_COLUMNS if c not in STRING_COLUMNS]

FEATURE_COLUMNS = ["est_eps", "est_rev_1m", "est_rev_3m", "est_dispersion", "est_n_analysts"]
PIT_FEATURE_COLUMNS = FEATURE_COLUMNS + ["est_rev_fy_1m", "est_rev_fy_3m", "est_age_days"]
VENDOR_FEATURE_COLUMNS = ["estv_rev_7d", "estv_rev_30d", "estv_rev_90d", "estv_rev_fy_30d", "estv_rev_fy_90d",
                          "estv_breadth_7d", "estv_breadth_30d", "estv_breadth_fy_30d"]
ALL_FEATURE_COLUMNS = PIT_FEATURE_COLUMNS + VENDOR_FEATURE_COLUMNS

FMP_URL = "https://financialmodelingprep.com/api/v3/analyst-estimates/{symbol}"
REV_DAYS = {"est_rev_1m": 30, "est_rev_3m": 91}
FY_REV_DAYS = {"est_rev_fy_1m": 30, "est_rev_fy_3m": 91}
YAHOO_PERIODS = ("0q", "+1q", "0y", "+1y")
_PERIOD_ORDER = {"0q": 0, "+1q": 1, "0y": 0, "+1y": 1}

# Yahoo earningsTrend sub-dict -> {yahoo key: our column}
_YAHOO_FIELDS = {
    "earningsEstimate": {"avg": "eps_avg", "low": "eps_low", "high": "eps_high", "numberOfAnalysts": "n_analysts",
                         "yearAgoEps": "eps_year_ago", "growth": "eps_growth"},
    "epsTrend": {"current": "eps_trend_current", "7daysAgo": "eps_trend_7d", "30daysAgo": "eps_trend_30d",
                 "60daysAgo": "eps_trend_60d", "90daysAgo": "eps_trend_90d"},
    "epsRevisions": {"upLast7days": "eps_up_7d", "upLast30days": "eps_up_30d",
                     "downLast7Days": "eps_down_7d", "downLast7days": "eps_down_7d",
                     "downLast30days": "eps_down_30d", "downLast30Days": "eps_down_30d"},
    "revenueEstimate": {"avg": "revenue_avg", "low": "revenue_low", "high": "revenue_high",
                        "numberOfAnalysts": "revenue_n_analysts", "yearAgoRevenue": "revenue_year_ago",
                        "growth": "revenue_growth"},
}
_VALUE_COLUMNS = [c for d in _YAHOO_FIELDS.values() for c in d.values() if c != "n_analysts"
                  and c != "revenue_n_analysts"]


class EstimatesProvider(Protocol):
    enabled: bool

    def current_consensus(self, symbols: list[str]) -> pd.DataFrame:
        """Current consensus rows: SNAPSHOT_COLUMNS minus asof_date / available_date."""


def _empty(cols=None) -> pd.DataFrame:
    return pd.DataFrame(columns=cols or [c for c in SNAPSHOT_COLUMNS if c not in ("asof_date", "available_date")])


def _env_key(name: str = "FMP_API_KEY") -> str | None:
    v = os.environ.get(name)
    if v:
        return v
    env = os.path.join(os.path.dirname(__file__), "..", "trading_agent", ".env")
    if os.path.exists(env):
        try:
            from dotenv import dotenv_values
            v = dotenv_values(env).get(name)
        except Exception:
            v = None
    return v or None


# =========================================================================== #
# FMP (disabled without a key)
# =========================================================================== #
class FMPEstimates:
    """FMP ``analyst-estimates`` (quarterly). Read-only GETs; disabled without a key."""

    def __init__(self, api_key: str | None = None, session=None, timeout: float = 30.0, log=print):
        self.api_key = api_key if api_key is not None else _env_key()
        self.enabled = bool(self.api_key)
        self.session, self.timeout, self.log = session, timeout, log
        if not self.enabled:
            self.log("[estimates] FMP_API_KEY not set -- analyst-estimate features disabled")

    def _get(self, symbol: str):
        if self.session is None:
            import requests
            self.session = requests.Session()
        r = self.session.get(FMP_URL.format(symbol=symbol),
                             params={"period": "quarter", "apikey": self.api_key}, timeout=self.timeout)
        if getattr(r, "status_code", 200) != 200:
            raise RuntimeError(f"FMP HTTP {r.status_code} for {symbol}")
        return r.json()

    def current_consensus(self, symbols: list[str]) -> pd.DataFrame:
        cols = [c for c in BASE_COLUMNS if c != "asof_date"] + ["source"]
        if not self.enabled:
            return pd.DataFrame(columns=cols)
        rows = []
        for s in symbols:
            try:
                data = self._get(s)
            except Exception as e:   # a failed symbol never breaks the batch
                self.log(f"[estimates] {s}: {e}")
                continue
            for d in data or []:
                rows.append({"ticker": s, "period_end": d.get("date"),
                             "eps_avg": d.get("estimatedEpsAvg"), "eps_high": d.get("estimatedEpsHigh"),
                             "eps_low": d.get("estimatedEpsLow"),
                             "n_analysts": d.get("numberAnalystsEstimatedEps"), "source": "fmp"})
        out = pd.DataFrame(rows, columns=cols)
        for c in ("eps_avg", "eps_high", "eps_low", "n_analysts"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out


# =========================================================================== #
# Yahoo via yfinance (free, no key)
# =========================================================================== #
def yahoo_symbol(ticker: str) -> str:
    """Our (Alpaca) symbols use '.' for share classes; Yahoo uses '-' (BRK.B -> BRK-B)."""
    return ticker.replace(".", "-").replace("/", "-")


def _raw(v):
    """Yahoo values come as {'raw': x, 'fmt': ...}, {} (missing) or bare numbers."""
    if isinstance(v, dict):
        return v.get("raw", np.nan) if v else np.nan
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    return np.nan


def parse_yahoo_trend(ticker: str, trend: list[dict], provider_symbol: str | None = None) -> list[dict]:
    """Rows (one per period 0q/+1q/0y/+1y) from the raw ``earningsTrend.trend``
    list of a quoteSummary response. Periods with no estimate at all (stale
    shells such as a 2011 ``endDate`` with empty fields) are dropped."""
    rows = []
    for item in trend or []:
        period = item.get("period")
        if period not in YAHOO_PERIODS:
            continue
        row = {"ticker": ticker, "source": "yahoo", "period": period, "period_end": item.get("endDate"),
               "provider_symbol": provider_symbol or yahoo_symbol(ticker)}
        for block, fields in _YAHOO_FIELDS.items():
            sub = item.get(block) or {}
            for k, col in fields.items():
                if k in sub:
                    row[col] = _raw(sub[k])
        row["eps_currency"] = (item.get("earningsEstimate") or {}).get("earningsCurrency") \
            or (item.get("epsTrend") or {}).get("epsTrendCurrency")
        row["revenue_currency"] = (item.get("revenueEstimate") or {}).get("revenueCurrency")
        # Yahoo reports 0 (not missing) for avg/low/high when no analyst covers a period
        for n_col, cols in (("n_analysts", ("eps_avg", "eps_low", "eps_high")),
                            ("revenue_n_analysts", ("revenue_avg", "revenue_low", "revenue_high"))):
            if row.get(n_col) == 0:
                row.update({c: np.nan for c in cols})
        if all(pd.isna(row.get(c, np.nan)) for c in _VALUE_COLUMNS):
            continue
        rows.append(row)
    return rows


class _RateLimiter:
    """Shared minimum interval between request starts (+ global cool-down)."""

    def __init__(self, rate_per_s: float):
        self.interval = 1.0 / rate_per_s if rate_per_s > 0 else 0.0
        self.next_t = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        if t > now:
            time.sleep(t - now)

    def cool_down(self, seconds: float):
        with self.lock:
            self.next_t = max(self.next_t, time.monotonic() + seconds)


def _yf_fetch_trend(symbol: str) -> list[dict]:
    """One quoteSummary(earningsTrend) request through the public yfinance API
    (``Ticker.eps_trend`` fills the analysis cache that ``earnings_estimate``,
    ``eps_revisions`` and ``revenue_estimate`` share), then the raw cached
    trend list, which also carries ``endDate`` and currency. yfinance hides
    404s ("no fundamentals") as an empty result; rate limits raise."""
    import yfinance as yf
    t = yf.Ticker(symbol)
    t.eps_trend
    return list(getattr(t._analysis, "_earnings_trend", None) or [])


class YahooEstimates:
    """Polite, fault-tolerant Yahoo consensus snapshot (no key, read-only)."""

    def __init__(self, rate_per_s: float = 4.0, workers: int = 4, retries: int = 3, backoff_s: float = 5.0,
                 deadline: float | None = None, fetch=None, log=print, progress_every: int = 500):
        self.enabled = True
        self.workers, self.retries, self.backoff_s = max(1, workers), retries, backoff_s
        self.deadline = deadline                  # time.monotonic() after which no new ticker starts
        self.fetch = fetch or _yf_fetch_trend
        self.log, self.progress_every = log, progress_every
        self.limiter = _RateLimiter(rate_per_s)
        self.stats: dict = {}
        if fetch is None:
            logging.getLogger("yfinance").setLevel(logging.CRITICAL)   # 404 noise for uncovered names

    def _one(self, ticker: str):
        if self.deadline is not None and time.monotonic() > self.deadline:
            return ticker, None, "skipped_deadline"
        sym = yahoo_symbol(ticker)
        err = None
        for attempt in range(self.retries + 1):
            self.limiter.wait()
            try:
                trend = self.fetch(sym)
                rows = parse_yahoo_trend(ticker, trend, sym)
                return ticker, rows, ("ok" if rows else "empty")
            except Exception as e:            # rate limit / network: back off and retry
                err = f"{type(e).__name__}: {e}"
                limited = "RateLimit" in type(e).__name__ or "429" in str(e)
                wait = self.backoff_s * (3 ** attempt) * (2.0 if limited else 1.0) * (1 + 0.2 * random.random())
                if limited:
                    self.limiter.cool_down(wait)
                if attempt < self.retries:
                    time.sleep(wait)
        return ticker, None, f"error: {err}"

    def current_consensus(self, symbols: list[str]) -> pd.DataFrame:
        t0 = time.monotonic()
        rows, status = [], {}
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = [ex.submit(self._one, s) for s in symbols]
            for n, f in enumerate(as_completed(futs), 1):
                tkr, r, st = f.result()
                status[tkr] = st
                if r:
                    rows.extend(r)
                if self.progress_every and n % self.progress_every == 0:
                    self.log(f"[estimates] {n}/{len(symbols)} tickers, {time.monotonic() - t0:.0f}s")
        kinds = pd.Series([("error" if v.startswith("error") else v) for v in status.values()], dtype=object)
        self.stats = {"requested": len(symbols), "elapsed_s": round(time.monotonic() - t0, 1),
                      **{k: int(v) for k, v in kinds.value_counts().items()},
                      "errors_sample": [f"{k}: {v}" for k, v in status.items() if v.startswith("error")][:10]}
        out = pd.DataFrame(rows).reindex(columns=_empty().columns)
        for c in NUMERIC_COLUMNS:
            if c in out:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out


# =========================================================================== #
# Storage
# =========================================================================== #
def next_weekday(d: str | dt.date) -> str:
    d = pd.Timestamp(d).date() + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d.isoformat()


def finalize_snapshot(cur: pd.DataFrame, asof: str, fetched_at: str | None = None) -> pd.DataFrame:
    """Stamp provider rows with asof_date / available_date and the full schema."""
    out = cur.copy()
    out["asof_date"] = asof
    out["available_date"] = next_weekday(asof)
    out["fetched_at"] = fetched_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return normalize(out)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to SNAPSHOT_COLUMNS with stable dtypes (old files -> NaN for new columns)."""
    out = df.reindex(columns=SNAPSHOT_COLUMNS)
    for c in NUMERIC_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    for c in STRING_COLUMNS:
        out[c] = out[c].astype(object).where(out[c].notna(), None)
        out[c] = out[c].map(lambda v: v if v is None else (v.date().isoformat()
                                                             if isinstance(v, pd.Timestamp) and c != "fetched_at"
                                                             else str(v)))
    return out


def daily_path(root: str, asof: str) -> str:
    return os.path.join(root, "estimates", asof[:4], f"{asof}.parquet")


def write_daily(df: pd.DataFrame, root: str, asof: str) -> str:
    path = daily_path(root, asof)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    normalize(df).to_parquet(path, index=False)
    return path


_DAILY_RE = re.compile(r"(?:^|/)(\d{4}-\d{2}-\d{2})\.parquet$")


def load_dir(root: str) -> pd.DataFrame:
    """All snapshots under ``root``: ``estimates/YYYY/YYYY-MM-DD.parquet`` (any
    depth) and legacy flat ``snapshot_<date>.parquet``."""
    files = sorted(set(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)))
    files = [f for f in files if _DAILY_RE.search(f.replace(os.sep, "/"))
             or os.path.basename(f).startswith("snapshot_")]
    if not files:
        return normalize(pd.DataFrame(columns=SNAPSHOT_COLUMNS))
    return normalize(pd.concat([normalize(pd.read_parquet(f)) for f in files], ignore_index=True))


class SnapshotStore:
    """Append-only Parquet store of consensus snapshots under ``root`` (legacy
    flat layout ``snapshot_<date>.parquet``; ``load`` also reads the daily layout)."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def record(self, provider: EstimatesProvider, symbols: list[str], asof: str | None = None) -> int:
        if not getattr(provider, "enabled", False):
            return 0
        asof = asof or dt.datetime.now(dt.timezone.utc).date().isoformat()
        cur = provider.current_consensus(symbols)
        if cur.empty:
            return 0
        finalize_snapshot(cur, asof).to_parquet(os.path.join(self.root, f"snapshot_{asof}.parquet"), index=False)
        return len(cur)

    def load(self) -> pd.DataFrame:
        return load_dir(self.root)


def _git(repo: str, *args, binary: bool = False):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode()


def load_branch_snapshots(repo: str = REPO_ROOT, branch: str = DATA_BRANCH, remote: str = "origin",
                          fetch: bool = True, since: str | None = None) -> pd.DataFrame:
    """Read every daily snapshot committed to ``branch`` without checking it out.

    ``fetch=True`` runs ``git fetch <remote> <branch>`` first (read-only
    network); the ref used is ``refs/remotes/<remote>/<branch>`` if present,
    else the local ``refs/heads/<branch>``. ``since`` keeps files with
    asof_date >= since."""
    if fetch:
        _git(repo, "fetch", "--quiet", remote, f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}")
    ref = None
    for cand in (f"refs/remotes/{remote}/{branch}", f"refs/heads/{branch}"):
        if subprocess.run(["git", "-C", repo, "rev-parse", "--verify", "--quiet", cand],
                          capture_output=True).returncode == 0:
            ref = cand
            break
    if ref is None:
        raise FileNotFoundError(f"branch {branch!r} not found in {repo} (fetch it or run the workflow first)")
    names = [n for n in _git(repo, "ls-tree", "-r", "--name-only", ref, "--", "estimates/").splitlines()
             if _DAILY_RE.search(n)]
    if since:
        names = [n for n in names if _DAILY_RE.search(n).group(1) >= since]
    frames = [normalize(pd.read_parquet(io.BytesIO(_git(repo, "cat-file", "blob", f"{ref}:{n}", binary=True))))
              for n in sorted(names)]
    if not frames:
        return normalize(pd.DataFrame(columns=SNAPSHOT_COLUMNS))
    return normalize(pd.concat(frames, ignore_index=True))


# =========================================================================== #
# Features
# =========================================================================== #
def _scaled_change(new, old):
    new, old = np.asarray(new, float), np.asarray(old, float)
    return (new - old) / np.maximum(np.abs(old), 0.01)


def _prepare(snapshots: pd.DataFrame) -> pd.DataFrame:
    s = normalize(snapshots)
    s = s[s["ticker"].notna() & s["asof_date"].notna()].copy()
    s["asof_date"] = pd.to_datetime(s["asof_date"])
    s["period_end"] = pd.to_datetime(s["period_end"], errors="coerce")
    s["available_date"] = pd.to_datetime(s["available_date"]).fillna(s["asof_date"])
    s["period"] = s["period"].fillna("")
    s["horizon"] = np.where(s["period"].isin(["0y", "+1y"]), "y", "q")
    s["_ord"] = s["period"].map(_PERIOD_ORDER).fillna(2)
    return s


def _targets(s: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """One target row per (ticker, asof_date): Yahoo ``0q`` / ``0y``; for
    period-less (FMP) quarterly rows the earliest period_end after asof_date."""
    if horizon == "y":
        t = s[s["period"] == "0y"]
    else:
        q = s[s["horizon"] == "q"]
        t = q[(q["period"] == "0q") | ((q["period"] == "") & (q["period_end"] > q["asof_date"]))]
    t = t.sort_values(["ticker", "asof_date", "_ord", "period_end"], kind="mergesort")
    return t.drop_duplicates(["ticker", "asof_date"], keep="first")


def _history(s: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Consensus history per (ticker, fiscal period end) for one horizon; within
    a snapshot a duplicated period_end (Yahoo occasionally repeats 0y's endDate
    for +1y) keeps the nearer period."""
    h = s[(s["horizon"] == horizon) & s["period_end"].notna() & s["eps_avg"].notna()]
    h = h.sort_values(["ticker", "period_end", "asof_date", "_ord"], kind="mergesort")
    h = h.drop_duplicates(["ticker", "period_end", "asof_date"], keep="first")
    return h[["ticker", "period_end", "asof_date", "eps_avg"]].rename(
        columns={"asof_date": "old_asof", "eps_avg": "old_eps"})


def _own_revisions(m: pd.DataFrame, hist: pd.DataFrame, days: int, tol_days: int | None) -> np.ndarray:
    """For each row of ``m`` (ticker, period_end, asof_date, eps_avg): change vs
    the latest snapshot of the same fiscal period with asof <= asof_date - days."""
    out = np.full(len(m), np.nan)
    ok = m["period_end"].notna() & m["eps_avg"].notna()
    if not ok.any() or hist.empty:
        return out
    left = m.loc[ok, ["ticker", "period_end", "asof_date", "eps_avg"]].copy()
    left["_pos"] = np.flatnonzero(ok.to_numpy())
    left["lookup"] = left["asof_date"] - pd.Timedelta(days=days)
    left = left.sort_values("lookup", kind="mergesort")
    r = pd.merge_asof(left, hist.sort_values("old_asof", kind="mergesort"), left_on="lookup", right_on="old_asof",
                      by=["ticker", "period_end"], direction="backward",
                      tolerance=pd.Timedelta(days=tol_days) if tol_days is not None else None)
    out[r["_pos"].to_numpy()] = _scaled_change(r["eps_avg"], r["old_eps"])
    return out


def estimate_features(snapshots: pd.DataFrame, keys: pd.DataFrame, max_age_days: int | None = 14,
                      rev_tolerance_days: int | None = None) -> pd.DataFrame:
    """PIT features for ``keys`` (ticker, Date), aligned to ``keys.index``.

    Only snapshots with ``available_date <= Date`` are used (legacy rows
    without available_date: ``asof_date <= Date``); the latest such snapshot
    must be at most ``max_age_days`` old (None = no limit), so a ticker that
    drops out of coverage does not carry stale values forward. Own revisions
    look back to the latest snapshot at least 30 / 91 days older
    (``rev_tolerance_days`` optionally caps how much older)."""
    out = pd.DataFrame(np.nan, index=keys.index, columns=ALL_FEATURE_COLUMNS)
    if snapshots is None or snapshots.empty or keys.empty:
        return out
    s = _prepare(snapshots)
    if s.empty:
        return out
    k = pd.DataFrame({"ticker": keys["ticker"].to_numpy(), "Date": pd.to_datetime(keys["Date"]).to_numpy(),
                      "_row": np.arange(len(keys))}).sort_values("Date", kind="mergesort")
    # latest usable snapshot per key (one row per (ticker, asof) snapshot day)
    snaps = (s.groupby(["ticker", "asof_date"], as_index=False)["available_date"].max()
               .sort_values(["available_date", "asof_date"], kind="mergesort"))
    m = pd.merge_asof(k, snaps, left_on="Date", right_on="available_date", by="ticker", direction="backward")
    if max_age_days is not None:
        m.loc[(m["Date"] - m["asof_date"]).dt.days > max_age_days, "asof_date"] = pd.NaT
    m = m.sort_values("_row").reset_index(drop=True)
    res = pd.DataFrame(np.nan, index=range(len(m)), columns=ALL_FEATURE_COLUMNS)
    has = m["asof_date"].notna()
    res.loc[has, "est_age_days"] = (m.loc[has, "Date"] - m.loc[has, "asof_date"]).dt.days.astype(float)

    vcols = ["eps_avg", "eps_high", "eps_low", "n_analysts", "period_end", "eps_trend_current", "eps_trend_7d",
             "eps_trend_30d", "eps_trend_90d", "eps_up_7d", "eps_up_30d", "eps_down_7d", "eps_down_30d"]
    for horizon in ("q", "y"):
        t = _targets(s, horizon)[["ticker", "asof_date"] + vcols]
        mm = m[["ticker", "asof_date"]].merge(t, on=["ticker", "asof_date"], how="left")
        hist = _history(s, horizon)
        n = mm["n_analysts"].to_numpy(float)
        trend_now = mm["eps_trend_current"].to_numpy(float)
        if horizon == "q":
            res["est_eps"] = mm["eps_avg"].to_numpy(float)
            res["est_n_analysts"] = n
            res["est_dispersion"] = ((mm["eps_high"] - mm["eps_low"]).to_numpy(float)
                                     / np.maximum(np.abs(mm["eps_avg"].to_numpy(float)), 0.01))
            for col, days in REV_DAYS.items():
                res[col] = _own_revisions(mm, hist, days, rev_tolerance_days)
            for d in ("7d", "30d", "90d"):
                res[f"estv_rev_{d}"] = _scaled_change(trend_now, mm[f"eps_trend_{d}"])
            for d in ("7d", "30d"):
                res[f"estv_breadth_{d}"] = ((mm[f"eps_up_{d}"] - mm[f"eps_down_{d}"]).to_numpy(float)
                                            / np.maximum(np.nan_to_num(n, nan=0.0), 1.0))
        else:
            for col, days in FY_REV_DAYS.items():
                res[col] = _own_revisions(mm, hist, days, rev_tolerance_days)
            for d in ("30d", "90d"):
                res[f"estv_rev_fy_{d}"] = _scaled_change(trend_now, mm[f"eps_trend_{d}"])
            res["estv_breadth_fy_30d"] = ((mm["eps_up_30d"] - mm["eps_down_30d"]).to_numpy(float)
                                          / np.maximum(np.nan_to_num(n, nan=0.0), 1.0))
    res.loc[~has.to_numpy()] = np.nan
    out.loc[:, :] = res.to_numpy()
    return out


# =========================================================================== #
# Universe + snapshot run
# =========================================================================== #
def build_universe_file(path: str = UNIVERSE_FILE, lookback_sessions: int = 21) -> list[str]:
    """Write the committed snapshot universe (needs the local WS1 store and
    price store under $BDA_DATA_V3): the bot's v3 live universe (active common
    stocks, ``bot.v3_live_universe``) restricted to names passing the v3
    ``eligible`` rule (price >= $3, 20d ADV >= $2M, >= 120 bars) on any of the
    last ``lookback_sessions`` stored sessions. Refresh monthly with the other
    snapshots; the workflow adds the tickers of the latest ops/predictions file."""
    from ..trading_agent.bot import v3_live_universe
    from .prices import load_prices
    from .universe import eligibility_details
    live = set(v3_live_universe(verbose=False))
    start = (dt.date.today() - dt.timedelta(days=400)).isoformat()
    px = load_prices(start=start, columns=["company_close", "company_volume"], with_raw=True)
    det = eligibility_details(px)
    dates = sorted(det["Date"].unique())[-lookback_sessions:]
    elig = set(det.loc[det["Date"].isin(dates) & det["eligible"], "ticker"])
    out = sorted(t for t in live & elig if "~" not in t)
    with open(path, "w") as f:
        f.write(f"# Analyst-estimate snapshot universe: v3 live common stocks eligible in the last "
                f"{lookback_sessions} sessions up to {dates[-1] if dates else '?'} (n={len(out)}).\n"
                f"# Regenerate: python -m DataAnalysisPipeline2.v3.estimates universe\n")
        f.write("\n".join(out) + "\n")
    return out


def snapshot_universe(universe_file: str = UNIVERSE_FILE, predictions_glob: str = PREDICTIONS_GLOB) -> list[str]:
    """Committed universe file + tickers of the newest ops/predictions file."""
    names: set[str] = set()
    if os.path.exists(universe_file):
        with open(universe_file) as f:
            names |= {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
    preds = sorted(glob.glob(predictions_glob), key=os.path.basename)
    if preds:
        try:
            names |= set(pd.read_csv(preds[-1], usecols=["ticker"])["ticker"].dropna().astype(str))
        except Exception:
            pass
    return sorted(n for n in names if re.fullmatch(r"[A-Z]{1,5}([./][A-Z]{1,2})?", n))


def coverage(df: pd.DataFrame, requested: int) -> dict:
    q = df[(df["period"] == "0q") & df["eps_avg"].notna()] if len(df) else df
    return {"rows": int(len(df)), "tickers_with_rows": int(df["ticker"].nunique()) if len(df) else 0,
            "tickers_with_0q_eps": int(q["ticker"].nunique()) if len(q) else 0,
            "coverage_0q": round(q["ticker"].nunique() / requested, 4) if requested and len(q) else 0.0,
            "median_n_analysts_0q": float(q["n_analysts"].median()) if len(q) else None,
            "non_usd_eps": int((df["eps_currency"].notna() & (df["eps_currency"] != "USD")).sum()) if len(df) else 0}


def run_snapshot(out_root: str, tickers: list[str], asof: str | None = None, max_minutes: float = 45.0,
                 rate_per_s: float = 4.0, workers: int = 4, log=print) -> dict:
    asof = asof or dt.datetime.now(dt.timezone.utc).date().isoformat()
    prov = YahooEstimates(rate_per_s=rate_per_s, workers=workers, log=log,
                          deadline=time.monotonic() + max_minutes * 60.0)
    cur = prov.current_consensus(tickers)
    snap = finalize_snapshot(cur, asof) if len(cur) else normalize(cur)
    path = write_daily(snap, out_root, asof) if len(snap) else None
    rep = {"asof_date": asof, "available_date": next_weekday(asof), "path": path,
           **prov.stats, **coverage(snap, len(tickers))}
    return rep


def _summary_md(rep: dict) -> str:
    lines = ["### Analyst-estimate snapshot (Yahoo)", "", "| field | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rep.items() if k != "errors_sample"]
    if rep.get("errors_sample"):
        lines += ["", "Sample errors:", ""] + [f"- `{e}`" for e in rep["errors_sample"]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m DataAnalysisPipeline2.v3.estimates")
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("universe", help="regenerate the committed snapshot universe (local data stores)")
    u.add_argument("--path", default=UNIVERSE_FILE)
    sp = sub.add_parser("snapshot", help="fetch today's Yahoo consensus snapshot")
    sp.add_argument("--out", required=True, help="root dir; writes <out>/estimates/YYYY/YYYY-MM-DD.parquet")
    sp.add_argument("--asof", default=None)
    sp.add_argument("--limit", type=int, default=None, help="first N tickers only (testing)")
    sp.add_argument("--tickers", default=None, help="comma-separated override")
    sp.add_argument("--max-minutes", type=float, default=45.0)
    sp.add_argument("--rate", type=float, default=4.0, help="max requests per second")
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--min-coverage", type=float, default=0.0,
                    help="exit 2 if the share of tickers with a 0q EPS consensus is below this")
    ld = sub.add_parser("load", help="summarise snapshots on the data branch")
    ld.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "universe":
        out = build_universe_file(a.path)
        print(f"[estimates] wrote {len(out)} tickers to {a.path}")
        return 0
    if a.cmd == "snapshot":
        tickers = list(dict.fromkeys(t.strip() for t in a.tickers.split(",") if t.strip())) \
            if a.tickers else snapshot_universe()
        if a.limit:
            tickers = tickers[:a.limit]
        print(f"[estimates] snapshot of {len(tickers)} tickers")
        rep = run_snapshot(a.out, tickers, a.asof, a.max_minutes, a.rate, a.workers)
        print("ESTIMATES_JSON: " + json.dumps(rep, default=str))
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                f.write(_summary_md(rep))
        if rep["path"] is None:
            return 1
        return 2 if rep["coverage_0q"] < a.min_coverage else 0
    if a.cmd == "load":
        df = load_branch_snapshots(fetch=not a.no_fetch)
        days = sorted(df["asof_date"].dropna().unique())
        print(f"[estimates] {len(df)} rows, {df['ticker'].nunique()} tickers, {len(days)} snapshot days"
              + (f" ({days[0]} .. {days[-1]})" if days else ""))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

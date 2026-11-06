"""Analyst-estimate adapter (consensus EPS, revisions, dispersion, coverage).

There is no paid estimates data in this project. This module defines the
interface the model would use and a Financial Modeling Prep (FMP)
implementation that is **disabled unless ``FMP_API_KEY``** is set (process env
or ``trading_agent/.env``). Nothing here creates accounts or calls a paid API
by itself; with no key every call returns empty frames.

Point-in-time design
--------------------
Vendors such as FMP serve the *current* consensus per fiscal period, not its
history. Using today's consensus for past dates would be look-ahead, so the
adapter is snapshot-based: ``SnapshotStore.record`` appends what the provider
returns with ``asof_date`` = the fetch date, and ``estimate_features`` only
uses snapshots with ``asof_date <= Date``. A usable history therefore starts
on the day snapshotting begins (run it daily from the scheduler); revisions
over 1/3 months become available 1/3 months later.

Features per (ticker, Date), for the next fiscal quarter not yet reported as of
the snapshot (``period_end > asof_date``):
  est_eps          consensus mean EPS
  est_rev_1m/3m    (consensus now - consensus ~21/63 sessions ago) / |price-free scale|,
                   scaled by max(|consensus 1-3m ago|, 0.01) -- same fiscal period
  est_dispersion   (high - low) / max(|mean|, 0.01)
  est_n_analysts   number of analysts
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Protocol

import numpy as np
import pandas as pd

SNAPSHOT_COLUMNS = ["ticker", "asof_date", "period_end", "eps_avg", "eps_high", "eps_low", "n_analysts"]
FEATURE_COLUMNS = ["est_eps", "est_rev_1m", "est_rev_3m", "est_dispersion", "est_n_analysts"]
FMP_URL = "https://financialmodelingprep.com/api/v3/analyst-estimates/{symbol}"
REV_DAYS = {"est_rev_1m": 30, "est_rev_3m": 91}


class EstimatesProvider(Protocol):
    enabled: bool

    def current_consensus(self, symbols: list[str]) -> pd.DataFrame:
        """Current consensus per (ticker, period_end): SNAPSHOT_COLUMNS minus asof_date."""


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
        if not self.enabled:
            return pd.DataFrame(columns=[c for c in SNAPSHOT_COLUMNS if c != "asof_date"])
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
                             "n_analysts": d.get("numberAnalystsEstimatedEps")})
        out = pd.DataFrame(rows, columns=[c for c in SNAPSHOT_COLUMNS if c != "asof_date"])
        for c in ("eps_avg", "eps_high", "eps_low", "n_analysts"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out


class SnapshotStore:
    """Append-only Parquet store of consensus snapshots under ``root``."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def record(self, provider: EstimatesProvider, symbols: list[str], asof: str | None = None) -> int:
        if not getattr(provider, "enabled", False):
            return 0
        asof = asof or dt.date.today().isoformat()
        cur = provider.current_consensus(symbols)
        if cur.empty:
            return 0
        cur = cur.assign(asof_date=asof)[SNAPSHOT_COLUMNS]
        cur.to_parquet(os.path.join(self.root, f"snapshot_{asof}.parquet"), index=False)
        return len(cur)

    def load(self) -> pd.DataFrame:
        files = sorted(f for f in os.listdir(self.root) if f.startswith("snapshot_"))
        if not files:
            return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
        return pd.concat([pd.read_parquet(os.path.join(self.root, f)) for f in files], ignore_index=True)


def estimate_features(snapshots: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """PIT features for ``keys`` (ticker, Date). Uses only snapshots with
    asof_date <= Date; the target period is the earliest period_end after the
    latest usable snapshot date. Returns FEATURE_COLUMNS aligned to ``keys``."""
    out = pd.DataFrame(np.nan, index=keys.index, columns=FEATURE_COLUMNS)
    if snapshots is None or snapshots.empty or keys.empty:
        return out
    s = snapshots.copy()
    s["asof_date"] = pd.to_datetime(s["asof_date"])
    s["period_end"] = pd.to_datetime(s["period_end"])
    by_t = {t: g.sort_values("asof_date") for t, g in s.groupby("ticker")}
    for i, (t, d) in enumerate(zip(keys["ticker"], pd.to_datetime(keys["Date"]))):
        g = by_t.get(t)
        if g is None:
            continue
        g = g[g["asof_date"] <= d]
        if g.empty:
            continue
        last = g["asof_date"].max()
        now = g[(g["asof_date"] == last) & (g["period_end"] > last)].sort_values("period_end")
        if now.empty:
            continue
        r = now.iloc[0]
        row = {"est_eps": r["eps_avg"], "est_n_analysts": r["n_analysts"],
               "est_dispersion": (r["eps_high"] - r["eps_low"]) / max(abs(r["eps_avg"]), 0.01)}
        same = g[g["period_end"] == r["period_end"]]
        for col, days in REV_DAYS.items():
            old = same[same["asof_date"] <= last - pd.Timedelta(days=days)]
            if len(old):
                o = old.sort_values("asof_date").iloc[-1]["eps_avg"]
                row[col] = (r["eps_avg"] - o) / max(abs(o), 0.01)
        for k, v in row.items():
            out.iat[i, FEATURE_COLUMNS.index(k)] = v
    return out

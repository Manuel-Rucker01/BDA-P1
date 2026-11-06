"""Thin, read-only Alpaca REST client shared by the v3 data modules.

Only GET endpoints are used (assets, corporate actions, historical bars):
nothing here can place an order or mutate the broker account.

Why not ``alpaca-py``'s ``StockHistoricalDataClient``?  We need control over
(1) a process-wide rate limiter (free plan: ~200 requests/min, shared by all
threads), (2) ``400 invalid symbol: X`` -- one bad symbol fails the whole
multi-symbol request, so we drop it and retry -- and (3) a fake ``session`` in
the unit tests (no network).
"""
from __future__ import annotations

import datetime as dt
import os
import re
import threading
import time
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = "/Users/manuelruckerabella/Workspace/UNI/Q6/BDA/bda_data_v3"
DATA_URL = "https://data.alpaca.markets"
NY_TZ = "America/New_York"
BAR_FIELDS = {"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close",
              "v": "volume", "n": "trade_count", "vw": "vwap"}


def data_dir(*parts: str) -> str:
    """``$BDA_DATA_V3`` (outside git) joined with ``parts``; directories are created."""
    root = os.environ.get("BDA_DATA_V3", DEFAULT_DATA_DIR)
    path = os.path.join(root, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def last_completed_close_utc(now: dt.datetime | None = None) -> dt.datetime:
    """Previous UTC midnight. Daily bars are stamped at session start (04:00/05:00Z),
    so ``end`` = previous midnight includes yesterday's (completed) session and
    excludes today's partial bar; the free SIP plan also forbids recent data."""
    now = now or dt.datetime.now(dt.timezone.utc)
    end = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if now - end < dt.timedelta(minutes=20):  # free plan: last 15 min are "recent"
        end -= dt.timedelta(minutes=20)
    return end


def md_table(df, index: bool = True, floatfmt: str = "{:,.3f}") -> str:
    """Markdown table without the optional ``tabulate`` dependency."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    d = df.reset_index() if index else df
    fmt = lambda v: ("" if v is None or (isinstance(v, float) and np.isnan(v)) else
                     floatfmt.format(v) if isinstance(v, (float, np.floating)) else
                     f"{v:,}" if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else str(v))
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def to_ny_date(ts: pd.Series) -> pd.Series:
    """UTC bar timestamps -> 'YYYY-MM-DD' session dates in New York time."""
    return pd.to_datetime(ts, utc=True).dt.tz_convert(NY_TZ).dt.strftime("%Y-%m-%d")


class RateLimiter:
    """Thread-safe sliding-window limiter: at most ``per_minute`` calls / 60 s."""

    def __init__(self, per_minute: int = 180, clock=time.monotonic, sleep=time.sleep):
        self.per_minute, self.clock, self.sleep = per_minute, clock, sleep
        self._calls: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = self.clock()
                self._calls = [t for t in self._calls if now - t < 60.0]
                if len(self._calls) < self.per_minute:
                    self._calls.append(now)
                    return
                wait = 60.0 - (now - self._calls[0]) + 0.05
            self.sleep(max(wait, 0.05))


class AlpacaHTTPError(RuntimeError):
    def __init__(self, status: int, text: str):
        super().__init__(f"HTTP {status}: {text[:300]}")
        self.status, self.text = status, text


_INVALID_SYMBOL = re.compile(r"invalid symbol[s]?:\s*([^\s\"',}]+)", re.I)


def credentials(account: str | None = None) -> tuple[str, str, str]:
    """(api_key, secret_key, trading_url) for ``account`` (default env
    ``BDA_V3_ALPACA_ACCOUNT`` or 'weekly'). The weekly account's data key is used
    so the bulk download does not eat the other processes' rate limit."""
    account = account or os.environ.get("BDA_V3_ALPACA_ACCOUNT", "weekly")
    os.environ["ACTIVE_ACCOUNT"] = account  # must precede the config import
    from DataAnalysisPipeline2.trading_agent import config
    acct = config.resolve_account(account)
    if not acct["configured"]:
        raise RuntimeError("Alpaca credentials not configured (trading_agent/.env)")
    return acct["api_key"], acct["secret_key"], config.ALPACA_URL


class AlpacaClient:
    """Read-only GET client with retries, backoff and a shared rate limiter."""

    def __init__(self, key: str | None = None, secret: str | None = None,
                 trading_url: str | None = None, session=None,
                 limiter: RateLimiter | None = None, retries: int = 6,
                 sleep=time.sleep, timeout: float = 120.0, log=print):
        if key is None:
            key, secret, trading_url = credentials()
        if session is None:
            import requests
            session = requests.Session()
        self.session, self.retries, self.sleep, self.timeout, self.log = session, retries, sleep, timeout, log
        self.limiter = limiter or RateLimiter(sleep=sleep)
        self.trading_url = trading_url or "https://paper-api.alpaca.markets"
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self.n_requests = 0

    # ------------------------------------------------------------------ core
    def get(self, url: str, params: dict) -> dict | list:
        last = None
        for attempt in range(1, self.retries + 1):
            self.limiter.acquire()
            self.n_requests += 1
            try:
                r = self.session.get(url, params=params, headers=self.headers, timeout=self.timeout)
            except Exception as e:  # network error: back off and retry
                last = e
                self.sleep(min(2 ** attempt, 60))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                reset = r.headers.get("X-RateLimit-Reset") if hasattr(r, "headers") else None
                wait = 5.0
                try:
                    wait = max(1.0, float(reset) - time.time() + 1.0) if reset else 2 ** attempt
                except ValueError:
                    pass
                self.log(f"[alpaca] 429 rate limited; sleeping {wait:.0f}s")
                self.sleep(min(wait, 90))
                last = AlpacaHTTPError(429, r.text)
                continue
            if r.status_code >= 500:
                last = AlpacaHTTPError(r.status_code, r.text)
                self.sleep(min(2 ** attempt, 60))
                continue
            raise AlpacaHTTPError(r.status_code, r.text)  # 4xx: caller's problem
        raise last if isinstance(last, Exception) else RuntimeError("request failed")

    def _paginate(self, url: str, params: dict, key: str):
        params = dict(params)
        while True:
            j = self.get(url, params)
            yield j.get(key) or {}
            tok = j.get("next_page_token")
            if not tok:
                return
            params["page_token"] = tok

    # ------------------------------------------------------------- endpoints
    def get_assets(self, status: str) -> list[dict]:
        return self.get(f"{self.trading_url}/v2/assets",
                        {"status": status, "asset_class": "us_equity"})

    def get_corporate_actions(self, types: Iterable[str], start: str, end: str) -> dict[str, list]:
        out: dict[str, list] = {}
        params = {"types": ",".join(types), "start": start, "end": end, "limit": 1000}
        for page in self._paginate(f"{DATA_URL}/v1/corporate-actions", params, "corporate_actions"):
            for k, v in page.items():
                out.setdefault(k, []).extend(v)
        return out

    def get_bars(self, symbols: list[str], start, end, timeframe: str = "1Day",
                 adjustment: str = "all", feed: str = "sip") -> tuple[pd.DataFrame, list[str]]:
        """All bars for ``symbols`` in [start, end] (pagination handled).

        Returns ``(df, invalid)``: ``df`` has columns symbol + BAR_FIELDS values
        (raw UTC timestamps); ``invalid`` lists symbols the API rejected as
        malformed (they are dropped and the request retried). Unknown but
        well-formed symbols are silently absent from ``df``.
        """
        syms, invalid = list(dict.fromkeys(symbols)), []
        fmt = lambda x: x.strftime("%Y-%m-%dT%H:%M:%SZ") if isinstance(x, dt.datetime) else str(x)
        while syms:
            params = {"symbols": ",".join(syms), "timeframe": timeframe, "start": fmt(start),
                      "end": fmt(end), "adjustment": adjustment, "feed": feed, "limit": 10000}
            rows = []
            try:
                for page in self._paginate(f"{DATA_URL}/v2/stocks/bars", params, "bars"):
                    for sym, bars in page.items():
                        for b in bars:
                            rec = {BAR_FIELDS[k]: v for k, v in b.items() if k in BAR_FIELDS}
                            rec["symbol"] = sym
                            rows.append(rec)
            except AlpacaHTTPError as e:
                m = _INVALID_SYMBOL.search(e.text) if e.status in (400, 422) else None
                if m and m.group(1) in syms:
                    invalid.append(m.group(1))
                    syms.remove(m.group(1))
                    continue
                raise
            cols = ["symbol"] + list(BAR_FIELDS.values())
            return pd.DataFrame(rows, columns=cols), invalid
        return pd.DataFrame(columns=["symbol"] + list(BAR_FIELDS.values())), invalid

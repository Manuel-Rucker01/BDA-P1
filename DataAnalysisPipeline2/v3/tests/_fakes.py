"""Network-free fakes of the Alpaca bars endpoint for the v3 tests."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from DataAnalysisPipeline2.v3._alpaca import AlpacaClient, RateLimiter


class FakeResponse:
    def __init__(self, status: int, payload, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self):
        return self._payload


class FakeBarsSession:
    """Serves ``/v2/stocks/bars`` from in-memory data.

    ``data[adjustment][symbol]`` is a DataFrame with columns
    t (ISO UTC), o, h, l, c, v, n, vw. Pages hold ``page_size`` bars.
    ``invalid`` symbols trigger the API's 400; ``fail_after`` raises
    (simulated crash) once that many requests have been served;
    ``rate_limit_first`` returns one 429 first.
    """

    def __init__(self, data, page_size=7, invalid=(), fail_after=None, rate_limit_first=False):
        self.data, self.page_size, self.invalid = data, page_size, set(invalid)
        self.fail_after, self.rate_limit_first = fail_after, rate_limit_first
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(dict(params or {}))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise KeyboardInterrupt("simulated crash")
        if self.rate_limit_first:
            self.rate_limit_first = False
            return FakeResponse(429, "too many requests", {"X-RateLimit-Reset": "0"})
        syms = params["symbols"].split(",")
        bad = [s for s in syms if s in self.invalid]
        if bad:
            return FakeResponse(400, {"message": f"invalid symbol: {bad[0]}"})
        adj = params.get("adjustment", "raw")
        start = pd.Timestamp(params["start"]).tz_localize("UTC") if pd.Timestamp(params["start"]).tzinfo is None \
            else pd.Timestamp(params["start"])
        end = pd.Timestamp(params["end"]).tz_localize("UTC") if pd.Timestamp(params["end"]).tzinfo is None \
            else pd.Timestamp(params["end"])
        rows = []
        for s in sorted(syms):
            df = self.data.get(adj, {}).get(s)
            if df is None:
                continue
            t = pd.to_datetime(df["t"], utc=True)
            for rec in df[(t >= start) & (t <= end)].to_dict("records"):
                rows.append((s, rec))
        off = int(params.get("page_token") or 0)
        page = rows[off:off + self.page_size]
        bars = {}
        for s, rec in page:
            bars.setdefault(s, []).append(rec)
        nxt = str(off + self.page_size) if off + self.page_size < len(rows) else None
        return FakeResponse(200, {"bars": bars, "next_page_token": nxt})


def fake_client(session) -> AlpacaClient:
    return AlpacaClient("k", "s", "https://paper", session=session,
                        limiter=RateLimiter(10_000, sleep=lambda s: None), sleep=lambda s: None,
                        log=lambda *a, **k: None)


def make_bars(dates, closes, volume=1e6, split_factor=None) -> pd.DataFrame:
    """Daily bars stamped like Alpaca: session start in UTC (04:00/05:00Z)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize("America/New_York").tz_convert("UTC")
    c = np.asarray(closes, dtype=float)
    v = np.full(len(c), volume, dtype=float) if np.isscalar(volume) else np.asarray(volume, float)
    return pd.DataFrame({"t": d.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": c, "h": c * 1.01, "l": c * 0.99,
                         "c": c, "v": v, "n": np.where(v > 0, 100, 0), "vw": c})

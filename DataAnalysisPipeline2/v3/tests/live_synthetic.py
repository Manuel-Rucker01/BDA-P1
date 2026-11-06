"""Live-shaped synthetic bars + EDGAR files for ``risk.live_panel`` and the
optimized_v3 book (no network, no data store)."""
import os

import numpy as np
import pandas as pd

LIVE_SECTORS = ["Business Equipment", "Finance", "Healthcare", "Manufacturing", "Other"]


def make_live_bars_and_fund(fund_dir, n=160, n_sessions=330, seed=0, start="2025-01-01"):
    """Bars as ``market_data.fetch_daily_bars`` returns them (ticker, Date, Open,
    High, Low, company_close, company_volume) for ``n`` factor-structured names
    ``L000``.., and EDGAR-style ``cik_map`` / ``fundamentals_asof`` / ``sectors``
    parquet files written to ``fund_dir``. Special names: ``PENNY`` (~$1, never
    eligible), ``NOFUND`` (no CIK: mcap from the ADV fit, sector Other),
    ``SHORTH`` (only 60 bars). ``L001`` files 20-F (foreign private issuer).
    Returns (bars, meta)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_sessions).strftime("%Y-%m-%d").tolist()
    T = n_sessions
    tick = [f"L{i:03d}" for i in range(n)]
    sec = rng.integers(0, len(LIVE_SECTORS) - 1, n)
    beta = rng.uniform(0.6, 1.4, n)
    spec = rng.uniform(0.01, 0.03, n)
    shares = np.exp(rng.normal(18.5, 1.2, n))
    mkt = rng.normal(0.0004, 0.01, T)
    fs = rng.normal(0, 0.006, (T, len(LIVE_SECTORS)))
    r = mkt[:, None] * beta + fs[:, sec] + rng.normal(0, 1, (T, n)) * spec
    px = rng.uniform(10, 120, n) * np.exp(np.cumsum(r, axis=0))
    vol = shares * rng.uniform(0.002, 0.01, n)
    rows = []
    for j, t in enumerate(tick):
        c = px[:, j]
        hl = np.abs(rng.normal(0, 0.01, T)) + 0.002
        rows.append(pd.DataFrame({"ticker": t, "Date": dates, "Open": c * (1 + rng.normal(0, 0.003, T)),
                                  "High": c * (1 + hl), "Low": c * (1 - hl), "company_close": c,
                                  "company_volume": vol[j] * rng.uniform(0.5, 1.5, T)}))
    extra = {"PENNY": (1.0, 5e7, 0), "NOFUND": (50.0, 2e5, 0), "SHORTH": (30.0, 3e5, T - 60)}
    for t, (p0, v0, first) in extra.items():
        c = p0 * np.exp(np.cumsum(rng.normal(0, 0.02, T)))
        rows.append(pd.DataFrame({"ticker": t, "Date": dates, "Open": c, "High": c * 1.01, "Low": c * 0.99,
                                  "company_close": c, "company_volume": v0}).iloc[first:])
    bars = pd.concat(rows, ignore_index=True)

    names = tick + ["PENNY", "SHORTH"]
    ciks = np.arange(1000, 1000 + len(names))
    cm = pd.DataFrame({"ticker": names, "cik": ciks,
                       "valid_from": pd.to_datetime(pd.Series([pd.NaT] * len(names))),
                       "valid_to": pd.to_datetime(pd.Series([pd.NaT] * len(names)))})
    q = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=200), dates[-1], freq="91D")
    sh_all = np.r_[shares, 1e9, 2e7]
    fa = []
    for i, c in enumerate(ciks):
        a = float(np.exp(rng.normal(21, 1)))
        for d in q:
            fa.append({"cik": int(c), "asof_date": d, "last_filed": d - pd.Timedelta(days=1),
                       "last_form": "20-F" if names[i] == "L001" else "10-Q",
                       "net_income_ttm": a * rng.normal(0.05, 0.03), "gross_profit_ttm": a * rng.uniform(0.1, 0.5),
                       "assets": a, "equity": a * rng.uniform(0.2, 0.6), "shares": float(sh_all[i])})
    fa = pd.DataFrame(fa)
    sec_all = np.r_[sec, 4, 4]
    secs = pd.DataFrame({"cik": ciks, "sic": 1000, "ff12": sec_all + 1,
                         "sector_name": np.array(LIVE_SECTORS)[sec_all]})
    os.makedirs(fund_dir, exist_ok=True)
    cm.to_parquet(os.path.join(fund_dir, "cik_map.parquet"), index=False)
    fa.to_parquet(os.path.join(fund_dir, "fundamentals_asof.parquet"), index=False)
    secs.to_parquet(os.path.join(fund_dir, "sectors.parquet"), index=False)
    return bars, {"dates": dates, "tickers": tick, "sector": dict(zip(tick, np.array(LIVE_SECTORS)[sec])),
                  "shares": dict(zip(tick, shares)), "last_filed": fa["last_filed"].max()}

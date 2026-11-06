"""Synthetic price panels for the v3 model tests (no network, no data store)."""
import numpy as np
import pandas as pd

SECTORS = ["Technology", "Finance", "Energy"]


def make_panel(n_per_sector=8, n_sessions=330, seed=0, start="2024-01-01"):
    """Factor-structured panel.

    * ``S{j}_{i}``: known-sector names (sector j), raw close = adjusted close;
    * ``UNK``: no metadata, driven by the Energy factor (proxy-sector test);
    * ``PNY``: adjusted close ~$10 but raw close ~$1 (never eligible: the $3 rule uses raw);
    * ``DEL``: stops trading 40 sessions before the end (delisting);
    * ``NEW``: listed 100 sessions before the end (< 120 bars);
    * ``ILQ``: tiny volume (ADV < $2M).
    Returns (panel, meta)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_sessions).strftime("%Y-%m-%d").tolist()
    mkt = rng.normal(0.0003, 0.01, n_sessions)
    fac = {s: rng.normal(0, 0.012, n_sessions) for s in SECTORS}
    rows, meta = [], []

    def add(t, rets, start_px=40.0, vol=2e5, first=0, last=None, raw_scale=1.0):
        last = n_sessions if last is None else last
        px = start_px * np.exp(np.cumsum(rets))
        v = vol * (1 + rng.random(n_sessions))
        for i in range(first, last):
            rows.append({"ticker": t, "Date": dates[i], "company_close": px[i],
                         "company_volume": v[i], "close_raw": px[i] * raw_scale})

    for j, s in enumerate(SECTORS):
        for i in range(n_per_sector):
            t = f"S{j}_{i}"
            add(t, mkt + fac[s] + rng.normal(0, 0.015, n_sessions), start_px=20 + 10 * i)
            meta.append({"ticker": t, "Sector": s})
    add("UNK", mkt + fac["Energy"] + rng.normal(0, 0.006, n_sessions))
    add("PNY", mkt + rng.normal(0, 0.01, n_sessions), start_px=10.0, vol=5e6, raw_scale=0.1)
    add("DEL", mkt + rng.normal(0, 0.02, n_sessions), last=n_sessions - 40)
    add("NEW", mkt + rng.normal(0, 0.02, n_sessions), first=n_sessions - 100)
    add("ILQ", mkt + rng.normal(0, 0.02, n_sessions), vol=1e3)
    meta += [{"ticker": "PNY", "Sector": "Finance"}, {"ticker": "DEL", "Sector": "Technology"},
             {"ticker": "NEW", "Sector": "None"}, {"ticker": "ILQ", "Sector": None}]
    return pd.DataFrame(rows), pd.DataFrame(meta)

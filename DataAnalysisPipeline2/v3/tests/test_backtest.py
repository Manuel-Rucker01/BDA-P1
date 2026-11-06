"""Backtest accounting on toy examples + evaluation helpers."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import backtest as bt
from DataAnalysisPipeline2.v3 import evaluation as ev

D = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"]


def _closes():
    return pd.DataFrame({"A": [10.0, 11.0, 11.0, 12.1, 12.1, 12.1],
                         "B": [20.0, 10.0, 10.0, np.nan, np.nan, np.nan],   # delisted after day 2
                         "C": [np.nan, np.nan, 5.0, 5.0, 10.0, 10.0]},       # listed on day 2
                        index=D)


def test_returns_matrix_delisting_and_listing():
    r = bt.daily_returns_matrix(_closes())
    assert np.isclose(r.loc[D[1], "A"], 0.1) and np.isclose(r.loc[D[1], "B"], -0.5)
    assert (r.loc[D[3]:, "B"] == 0).all()          # flat after the last bar
    assert np.isnan(r.loc[D[1], "C"])              # no return before the first bar


def test_simulate_long_only_costs_and_drift():
    r = bt.daily_returns_matrix(_closes())
    ret, log = bt.simulate({D[0]: {"A": 0.5, "B": 0.5}, D[3]: {"A": 1.0}}, r, D[0],
                           cost_bps=10.0, borrow_bps_yr=0.0)
    nav = 1.0 - 0.001 * 1.0                         # buy 100% at 10 bps
    assert np.isclose(ret[D[0]], -0.001)
    eA, eB = 0.5 * nav, 0.5 * nav
    pnl = 0.1 * eA - 0.5 * eB
    assert np.isclose(ret[D[1]], pnl / nav)
    nav, eA, eB = nav + pnl, eA * 1.1, eB * 0.5
    assert np.isclose(ret[D[2]], 0.0)              # nobody moves
    # day 3: A +10%, B delisted (flat); rebalance into 100% A at the close
    pnl = 0.1 * eA
    nav1, eA = nav + pnl, eA * 1.1
    traded = abs(1.0 - eA / nav1) + abs(0.0 - eB / nav1)
    nav2 = nav1 * (1 - 0.001 * traded)
    assert np.isclose(ret[D[3]], nav2 / nav - 1)
    assert np.isclose(log.loc[1, "turnover_oneway"], traded / 2)
    assert np.isclose(np.prod(1 + ret) - 1, nav2 * 1.0 - 1)   # A flat afterwards


def test_simulate_short_borrow_and_delisted_short():
    r = bt.daily_returns_matrix(_closes())
    ret, _ = bt.simulate({D[0]: {"A": 0.5, "B": -0.5}}, r, D[0], cost_bps=0.0, borrow_bps_yr=252.0)
    b = 0.0252 / 252
    # day 1: long A +10% (+0.05), short B -50% (+0.25), borrow on 0.5 short
    assert np.isclose(ret[D[1]], 0.05 + 0.25 - b * 0.5)
    # after B's last bar the short is frozen at its last price: only borrow accrues on it
    nav = 1.3 - b * 0.5
    eB = -0.25
    assert np.isclose(ret[D[2]], -b * 0.25 / nav)
    assert np.isclose(ret[D[3]], (0.1 * 0.55 - b * 0.25) / (nav - b * 0.25))
    del eB


def test_rebalance_plan_uses_strictly_earlier_weekly_date():
    cal = pd.bdate_range("2025-01-01", "2025-03-31").strftime("%Y-%m-%d").tolist()
    weekly = ["2025-01-03", "2025-01-31", "2025-02-28", "2025-03-07"]
    plan = bt.rebalance_plan(weekly, cal)
    assert plan == {"2025-02-03": "2025-01-31", "2025-03-03": "2025-02-28"}
    plan2 = bt.rebalance_plan(["2025-01-31", "2025-02-03"], cal)
    assert plan2["2025-02-03"] == "2025-01-31"      # same-day prediction never used


def test_books_respect_specs():
    rng = np.random.default_rng(0)
    n = 300
    preds = pd.DataFrame({"ticker": [f"T{i}" for i in range(n)], "pred_rank": rng.random(n),
                          "Sector": rng.choice(["A", "B", "C", "D", "E", "F", "G", "H"], n),
                          "return_volatility_20d": rng.uniform(0.01, 0.05, n), "company_close": 10.0})
    for name, spec in bt.CONSTRUCTIONS.items():
        w = bt.build_book(preds, spec)
        v = np.array(list(w.values()))
        if spec["kind"] == "long_only":
            assert len(w) == spec["k"] and v.max() <= spec["max_weight"] + 1e-9
            assert np.isclose(v.sum(), spec["target_exposure"])
        elif spec["kind"] == "long_short":
            assert np.isclose(v[v > 0].sum(), 0.8) and np.isclose(v[v < 0].sum(), -0.8, atol=1e-6)
        else:
            assert len(w) == 10 and np.isclose(v.sum(), 1.0)
            top = preds.nlargest(10, "pred_rank")["ticker"]
            assert set(w) == set(top)


def test_newey_west_and_dsr():
    rng = np.random.default_rng(1)
    x = rng.normal(0.05, 1, 2000)
    t0 = ev.newey_west_t(x, 0)
    assert np.isclose(t0, x.mean() / (x.std(ddof=0) / np.sqrt(len(x))))
    ar = np.convolve(rng.normal(0, 1, 2100), np.ones(6), "valid")[:2000] + 0.2
    assert ev.newey_west_t(ar, 6) < ev.newey_west_t(ar, 0)       # HAC widens overlapping data
    r = rng.normal(0.001, 0.01, 1500)
    d1 = ev.deflated_sharpe(r, [0.1, 0.0, 0.05])
    d2 = ev.deflated_sharpe(r, list(rng.normal(0, 0.05, 40)))
    assert d1["psr0"] > 0.99 and d2["dsr"] < d1["psr0"]


def test_stationary_bootstrap_indices():
    rng = np.random.default_rng(0)
    i = ev.stationary_bootstrap_indices(500, 20, rng)
    assert len(i) == 500 and i.min() >= 0 and i.max() < 500
    assert (np.diff(i) % 500 == 1).mean() > 0.9                  # mostly contiguous blocks


def test_signal_metrics():
    rng = np.random.default_rng(2)
    rows = []
    for k in range(20):
        y = rng.normal(0, 1, 200)
        rows.append(pd.DataFrame({"Date": f"2025-01-{k + 1:02d}", "pred": y + rng.normal(0, 1, 200),
                                  "fwd_ret_30": y, "Sector": rng.choice(["A", "B"], 200)}))
    pdx = ev.per_date_signal(pd.concat(rows))
    assert (pdx["ic"] > 0.5).all() and (pdx["decile_spread"] > 0).all()
    assert (pdx["sector_neutral_ic"] > 0.5).all()
    s = ev.signal_summary(pdx)
    assert s.loc["all", "pct_ic_pos"] == 1.0

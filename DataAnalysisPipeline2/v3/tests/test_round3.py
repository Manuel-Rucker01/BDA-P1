"""Round-3 construction mechanics on toy data."""
import numpy as np
import pandas as pd

from DataAnalysisPipeline2.v3 import backtest as bt
from DataAnalysisPipeline2.v3 import gate
from DataAnalysisPipeline2.v3 import round3 as r3

D = pd.bdate_range("2025-01-06", periods=8).strftime("%Y-%m-%d").tolist()


def _rets(cols):
    return bt.daily_returns_matrix(pd.DataFrame(cols, index=D))


def test_short_stop_covers_at_plus_50pct():
    rets = _rets({"L": [10.0] * 8, "S": [10, 12, 14, 16, 20, 40, 40, 40]})
    ret, lg = r3.simulate_ext({D[0]: {"L": 0.5, "S": -0.1}}, rets, D[0], stop_up=0.5, cost_bps=0.0,
                              default_borrow=0.0)
    # S hits +60% on D[3] (16/10) -> covered; the later doubling no longer hurts
    assert lg.attrs["n_stops"] == 1
    assert np.isclose(ret[D[5]], 0.0)
    assert np.isclose(np.prod(1 + ret) - 1, -0.1 * 0.6)


def test_weekly_reset_rescales_short_leg_only():
    rets = _rets({"L": [10, 11, 12, 13, 14, 15, 16, 17.0], "S": [10, 11, 12, 12, 12, 12, 12, 12.0]})
    ret, _ = r3.simulate_ext({D[0]: {"L": 0.5, "S": -0.2}}, rets, D[0], reset_dates=[D[2]], cost_bps=0.0,
                             default_borrow=0.0)
    nav = 1.0 + 0.5 * 0.1 - 0.2 * 0.1
    nav2 = nav + 0.5 * 1.1 * (12 / 11 - 1) - 0.2 * 1.1 * (12 / 11 - 1)
    assert np.isclose(np.prod(1 + ret[:3]), nav2)
    # after the reset the short is back at exactly -20% of NAV; the long kept drifting
    e_long = 0.5 * 1.2
    assert np.isclose(ret[D[3]], e_long * (13 / 12 - 1) / nav2)


def test_tiered_borrow():
    rets = _rets({"S": [10.0] * 8})
    ret, _ = r3.simulate_ext({D[0]: {"S": -0.5}}, rets, D[0], borrow_by_date={D[0]: {"S": 0.05}}, cost_bps=0.0)
    assert np.isclose(ret[D[1]], -0.05 / 252 * 0.5)


def test_vol_target_scales_and_caps_and_trend():
    idx = pd.bdate_range("2024-01-01", periods=80).strftime("%Y-%m-%d")
    rng = np.random.default_rng(0)
    book = pd.Series(rng.normal(0, 0.03, 80), index=idx)          # ~48% annual vol
    out, s = r3.vol_target(book, [idx[30]], target=0.15, window=20, cost_bps=0.0)
    assert (s[: 31] == 1.0).all()                                   # applied from the next session
    assert s.iloc[31] < 0.5 and np.isclose(out.iloc[40], s.iloc[40] * book.iloc[40])
    calm = pd.Series(rng.normal(0, 0.001, 80), index=idx)
    _, s2 = r3.vol_target(calm, [idx[30]], window=20)
    assert s2.iloc[-1] == 1.0                                       # never levered
    trend = pd.Series(False, index=idx)
    _, s3 = r3.vol_target(calm, [idx[30]], window=20, trend=trend)
    assert s3.iloc[-1] == 0.5


def test_floor_and_safe_pool():
    g = pd.DataFrame({"Date": "d", "ticker": list("ABCDEFGHIJ"), "mcap": np.arange(1, 11) * 1e9,
                      "adv_usd": [5e6, 30e6, 30e6, 30e6, 30e6, 30e6, 30e6, 30e6, 30e6, 30e6],
                      "close_raw": [20, 20, 5, 20, 20, 20, 20, 20, 20, 20.0],
                      "vol60": [0.01] * 9 + [0.2], "return_5d": [0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0.0]})
    f = r3.floor_mask(g, 50, 10e6)
    assert set(g.loc[f, "ticker"]) == set("FGHIJ")
    s = r3.safe_short_mask(g)
    assert set(g.loc[s, "ticker"]) == set("BEFGHI")    # A: ADV, C: price, D: run-up, J: top-decile vol


def test_preregistered_trials():
    assert gate.N_TRIALS_R3 == 43 and len(gate.R3_CONFIGS) == 9
    assert gate.choose_config_r3({"flo30_k50": {"sharpe": 1.0, "max_dd": -0.3},
                                  "ls_safe": {"sharpe": 2.0, "max_dd": -0.5}}) == "flo30_k50"

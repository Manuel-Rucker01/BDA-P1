"""Round-5 runner mechanics on a toy prep (no data store)."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import gate
from DataAnalysisPipeline2.v3 import risk
from DataAnalysisPipeline2.v3 import round5 as r5

CAL = pd.bdate_range("2024-01-01", periods=140).strftime("%Y-%m-%d").to_numpy()


def test_trailing_ic_uses_only_matured_labels_and_prior():
    ic = pd.Series({CAL[i]: 0.10 for i in range(0, 100, 5)})
    # at CAL[20] only labels of dates <= CAL[-10]... none matured (30-session label) -> prior
    est, n = r5.trailing_ic(ic, CAL, CAL[20])
    assert n == 0 and est == pytest.approx(gate.R5_COMMON["ic_prior"])
    est, n = r5.trailing_ic(ic, CAL, CAL[60])
    assert n == 7                     # dates 0,5,...,30 have pos + 30 <= 60
    assert est == pytest.approx((7 * 0.10 + 26 * 0.03) / 33)
    est, _ = r5.trailing_ic(-ic, CAL, CAL[130])
    assert est == 0.0                 # floored


def test_weekly_plan_trades_next_session():
    pdates = [CAL[4], CAL[9], CAL[24], CAL[29]]
    monthly = r5.plan_for("monthly", pdates, CAL)
    weekly = r5.plan_for("weekly", pdates, CAL)
    assert all(r > d for r, d in weekly.items())
    assert min(weekly) >= min(monthly)
    assert weekly[CAL[25]] == CAL[24]


def _toy_prep(n=60, seed=0):
    rng = np.random.default_rng(seed)
    tick = np.array([f"T{i:02d}" for i in range(n)])
    T = len(CAL)
    ret = rng.normal(0.0003, 0.015, (T, n)).astype(np.float32)
    ret[100:, 0] = 0.0                     # "delisted": frozen
    by_date = {}
    for d in CAL[5::5]:
        sec = np.arange(n) % 3
        X = np.hstack([rng.normal(size=(n, 2)), np.eye(3)[sec]])
        m = risk.RiskModel(d, tick, X, np.diag([1e-5, 1e-5, 1e-4, 1e-4, 1e-4]), np.full(n, 2e-4),
                           ["s1", "s2", "ind:a", "ind:b", "ind:c"], rng.lognormal(23, 1, n), sec, ["a", "b", "c"])
        by_date[d] = {"model": m, "pred": rng.normal(size=n), "hs": np.full(n, 5e-4), "sigma": np.full(n, 0.015),
                      "adv": np.full(n, 5e7), "price": np.full(n, 50.0), "safe": np.ones(n, dtype=bool),
                      "cs_ok": np.ones(n, dtype=bool)}
    ic = pd.Series({d: 0.05 for d in CAL[5::5]})
    return {"dates": CAL, "tickers": tick, "ret": ret, "by_date": by_date, "ic": ic, "calendar": CAL}


def test_universe_mask_top_n():
    prep = _toy_prep()
    x = next(iter(prep["by_date"].values()))
    m = r5.universe_mask(x, "top10")
    assert m.sum() == 10 and np.all(x["model"].mcap[m].min() >= x["model"].mcap[~m].max())


def test_run_config_costs_and_constraints(monkeypatch):
    prep = _toy_prep()
    monkeypatch.setitem(gate.R5_CONFIGS, "toy_te", {"kind": "active", "universe": "all", "te": 0.04, "lam": 1.0,
                                                    "active_cap": 0.03, "sector_band": 0.05, "beta_band": 0.05,
                                                    "rebalance": "monthly"})
    res = r5.run_config("toy_te", 1e6, prep=prep)
    lg = res["log"]
    assert len(lg) >= 4 and (lg["status"] == "solved").all()
    assert np.allclose(lg["net"], 1.0, atol=2e-3)
    assert (lg["ex_ante_te"] <= 0.04 * 1.02).all()
    assert (lg["cost"] > 0).all()
    # same book at 100x the AUM: impact and liquidity make it trade less
    big = r5.run_config("toy_te", 1e8, prep=prep)["log"]
    assert big["turnover_oneway"].iloc[1:].mean() < lg["turnover_oneway"].iloc[1:].mean()
    cw = r5.run_config("cw_all", 1e8, prep=prep)
    assert (cw["log"]["cost"] == 0).all()
    top = r5.run_config("cw_top500", 1e8, prep=prep)        # universe subset path
    assert (top["log"]["n_names"] <= 500).all()
    monkeypatch.setitem(r5.CW_SPECS, "cw_top20", {"kind": "cw", "universe": "top20", "rebalance": "monthly"})
    assert (r5.run_config("cw_top20", 1e8, prep=prep)["log"]["n_names"] == 20).all()

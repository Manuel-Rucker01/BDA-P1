"""Stress-test mechanics on small synthetic inputs (no data store)."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import stress as st

DATES = pd.bdate_range("2020-01-01", periods=60).strftime("%Y-%m-%d")


def test_window_stats_and_clip():
    r = pd.Series(0.01, index=DATES[:10])
    spy = pd.Series(-0.01, index=DATES[:10])
    s = st.window_stats(r, spy, "2019-12-01", DATES[4])
    assert s["n_days"] == 5 and s["clipped"]
    assert s["book"] == pytest.approx(1.01 ** 5 - 1)
    assert s["active"] == pytest.approx(1.01 ** 5 - 0.99 ** 5)
    assert s["max_dd"] == 0.0 and s["spy_max_dd"] == pytest.approx(0.99 ** 5 - 1)
    assert st.window_stats(r, spy, "2030-01-01", "2030-02-01")["n_days"] == 0


def test_buy_and_hold_drift_and_coverage():
    R = pd.DataFrame({"A": [0.10, 0.10], "B": [0.0, 0.0], "C": [np.nan, np.nan]}, index=DATES[:2])
    w = pd.Series({"A": 0.5, "B": 0.4, "C": 0.1})
    r, cov = st.buy_and_hold(w, R, DATES[0], DATES[1])
    assert cov == pytest.approx(0.9)
    wa, wb = 0.5 / 0.9, 0.4 / 0.9                        # C dropped, renormalised to net 1
    v1 = wa * 1.1 + wb
    v2 = wa * 1.21 + wb
    assert r.iloc[0] == pytest.approx(v1 - 1)
    assert r.iloc[1] == pytest.approx(v2 / v1 - 1)        # drift: A's weight grew


def test_conditional_factor_move_matches_regression():
    F = np.array([[4.0, 2.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    f, C = st.conditional_factor_move(F, np.array([1.0, 0, 0]), -2.0)
    assert f == pytest.approx([-2.0, -1.0, 0.0])        # beta of f2 on f1 = 0.5
    assert C[0, 0] == pytest.approx(0.0) and C[1, 1] == pytest.approx(2.0 - 1.0)


def test_shock_pnl_isolated_vs_correlated():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    F = np.array([[1e-4, 5e-5], [5e-5, 1e-4]])
    D = np.full(2, 1e-4)
    w = np.array([0.0, 1.0])                           # only exposed to factor 2
    r = st.shock_pnl(w, X, F, D, np.array([1.0, 0.0]), -0.1, horizon=1)
    assert r["isolated"] == pytest.approx(0.0)
    assert r["correlated"] == pytest.approx(-0.05)
    assert r["specific_sd"] == pytest.approx(0.01)


def test_market_shock_isolated_is_uniform():
    rng = np.random.default_rng(0)
    n = 30
    sec = rng.integers(0, 3, n)
    X = np.column_stack([rng.normal(size=n), np.eye(3)[sec]])
    factors = ["beta", "ind:A", "ind:B", "ind:C"]
    A = rng.normal(size=(4, 4)) * 0.01
    F = A @ A.T + np.eye(4) * 1e-5
    D = np.full(n, 1e-4)
    w = np.full(n, 1.0 / n)
    b = st.cap_proxy(rng.uniform(1, 10, n), top=20)
    assert b.sum() == pytest.approx(1.0) and (b > 0).sum() == 20
    sc = st.scenarios(w, X, F, D, factors, b)
    mk = sc[sc["group"] == "market"].iloc[0]
    assert mk["isolated"] == pytest.approx(st.MARKET_SHOCK)
    # the shocked portfolio itself loses exactly the shock under the conditional move
    r = st.shock_pnl(b, X, F, D, X.T @ b, st.MARKET_SHOCK)
    assert r["correlated"] == pytest.approx(st.MARKET_SHOCK)
    assert set(sc.loc[sc["group"] == "sector", "scenario"]) == {f"sector {s} -15%" for s in "ABC"}
    assert sc.loc[sc["group"] == "sector", "isolated"].sum() == pytest.approx(st.SECTOR_SHOCK)


def test_calibration_unbiased_and_underestimated():
    rng = np.random.default_rng(1)
    d = pd.bdate_range("2020-01-01", periods=1500).strftime("%Y-%m-%d")
    log = pd.DataFrame({"Date": d[::21], "ex_ante_vol": 0.16})
    spy = pd.Series(rng.normal(0, 0.01, len(d)), index=d)
    r = pd.Series(rng.normal(0, 0.16 / np.sqrt(252), len(d)), index=d)
    c = st.calibration(r, log, spy)
    assert c["bias_all"] == pytest.approx(1.0, abs=0.06)
    c2 = st.calibration(r * 1.5, log, spy)
    assert c2["bias_all"] == pytest.approx(1.5, abs=0.09)
    z = st.standardized_daily(r, log)
    assert np.isnan(z.iloc[0])                          # no rebalance before the first session
    assert z.iloc[1] == pytest.approx(r.iloc[1] / (0.16 / np.sqrt(252)))


def test_exposures_var_shares_sum_to_one():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 3))
    A = rng.normal(size=(3, 3)) * 0.01
    F = A @ A.T
    D = np.full(20, 2e-4)
    w = np.full(20, 0.05)
    e = st.exposures(w, X, F, D, ["a", "b", "c"], np.full(20, 0.05))
    assert e["var_share"].sum() == pytest.approx(1.0)
    assert np.allclose(e["active"].iloc[:3], 0.0)

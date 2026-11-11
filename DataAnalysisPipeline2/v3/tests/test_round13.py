"""Round 13 (lev125, pa_iwm) mechanics and point-in-time behaviour on toy inputs (no data store)."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import gate
from DataAnalysisPipeline2.v3 import round11 as r11
from DataAnalysisPipeline2.v3 import round13 as r13
from DataAnalysisPipeline2.v3.tests.test_round5 import CAL, _toy_prep


def _bl(prep, blackout_all=False):
    tick = prep["tickers"]
    if blackout_all:
        kept = pd.DataFrame({"cik": np.arange(len(tick)), "adate": sorted(prep["by_date"])[0]})
    else:
        kept = pd.DataFrame({"cik": pd.Series([], dtype="int64"), "adate": pd.Series([], dtype=object)})
    return {"kept": kept, "cik": {d: {t: i for i, t in enumerate(tick)} for d in prep["by_date"]}}


def _etf(prep, seed=3):
    rng = np.random.default_rng(seed)
    T = len(prep["dates"])
    spy = rng.normal(0.0004, 0.01, T)
    iwm = 1.1 * spy + rng.normal(0.0, 0.006, T)
    # small-cap-like toy market: stock returns load on IWM
    prep = dict(prep)
    prep["ret"] = (prep["ret"] + 0.9 * iwm[:, None]).astype(np.float32)
    return prep, {"spy": spy, "iwm": iwm}


# --------------------------------------------------------------------------- #
# Legs and betas
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("b", [(0.9, 0.95, 1.1), (1.2, 1.0, 1.3), (0.0, 0.5, 1.0), (0.3, 0.4, 0.9)])
def test_pa_legs_hit_beta_and_gross(b):
    b_pI, b_pS, b_IS = b
    legs = r13.pa_legs(b_pI, b_pS, b_IS)
    assert legs["gross"] <= gate.R13_PA_GROSS_CAP + 1e-12
    assert 0 < legs["s"] <= 1.0 and legs["h"] >= 0 and legs["w_spy"] >= 0
    assert legs["beta_spy"] == pytest.approx(gate.R13_PA_BETA_TARGET, abs=1e-12)
    assert legs["h"] == pytest.approx(legs["s"] * b_pI)
    # the cap binds unless s hits 1
    assert legs["s"] == pytest.approx(1.0) or legs["gross"] == pytest.approx(gate.R13_PA_GROSS_CAP)


def test_pa_legs_negative_spy_branch():
    legs = r13.pa_legs(0.1, 3.0, 1.0)       # residual SPY beta > 1 -> SPY leg would be negative
    assert legs["w_spy"] == 0.0
    assert legs["s"] == pytest.approx(min(1.0, gate.R13_PA_GROSS_CAP / 1.1))
    assert legs["gross"] <= gate.R13_PA_GROSS_CAP + 1e-12


def test_pa_betas_ols_and_fallback():
    rng = np.random.default_rng(0)
    T, n = 300, 5
    rs = rng.normal(0, 0.01, T)
    ri = 1.2 * rs + rng.normal(0, 0.005, T)
    R = 0.8 * ri[:, None] + rng.normal(0, 0.01, (T, n))
    R[3, 2] = np.nan
    w = np.full(n, 1.0 / n)
    b = r13.pa_betas(R, w, ri, rs)
    rp = np.nan_to_num(R) @ w
    assert b["b_pI"] == pytest.approx(np.polyfit(ri, rp, 1)[0])
    assert b["b_IS"] == pytest.approx(np.polyfit(rs, ri, 1)[0])
    assert b["b_pS"] == pytest.approx(np.polyfit(rs, rp, 1)[0])
    ri2 = ri.copy()
    ri2[:200] = np.nan
    assert r13.pa_betas(R, w, ri2, rs)["fallback"]
    assert r13.pa_betas(-R, w, ri, rs)["b_pI"] == 0.0     # floored


def test_exclude_top_drops_largest():
    mcap = np.array([5.0, 1.0, 9.0, 3.0, np.nan, 7.0])
    ok = np.array([True, True, True, True, False, False])
    keep = r13.exclude_top(mcap, ok, 2)
    assert keep.tolist() == [False, True, False, True, False, False]


# --------------------------------------------------------------------------- #
# Books
# --------------------------------------------------------------------------- #
def test_earn_blackout_equals_round11():
    prep = _toy_prep()
    for all_ in (False, True):
        bl = _bl(prep, all_)
        orig = r11.expected_dates
        try:
            if all_:
                r11.expected_dates = lambda k, d, **kw: pd.Series({i: d for i in range(len(prep["tickers"]))})
            a = r11.run_book("earn_blackout", prep, aum=1e8, bl=bl)
            b = r13.run_book("earn_blackout", prep, aum=1e8, bl=bl)
        finally:
            r11.expected_dates = orig
        assert np.array_equal(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_lev125_constant_leverage_and_financing():
    prep = _toy_prep()
    bl = _bl(prep)
    lo = r13.run_book("lev125", prep, aum=1e8, rates=np.zeros(len(CAL)), bl=bl)
    hi = r13.run_book("lev125", prep, aum=1e8, rates=np.full(len(CAL), 0.05), bl=bl)
    lg = lo["log"]
    assert (lg["lev"] == gate.R13_LEV).all()
    assert np.allclose(lg["gross"], gate.R13_LEV, atol=3e-3)
    assert np.prod(1 + hi["ret"]) < np.prod(1 + lo["ret"])
    # financing ~ 0.25 x (rate + spread) per year
    f = -hi["daily"]["financing"].iloc[1:]
    assert f.mean() * 252 == pytest.approx(0.25 * (0.05 + gate.R11_FIN_SPREAD), rel=0.1)


def test_pa_iwm_book(monkeypatch):
    monkeypatch.setattr(gate, "R13_PA_EXCLUDE_TOP", 10)
    prep, etf = _etf(_toy_prep())
    bl = _bl(prep)
    res = r13.run_book("pa_iwm", prep, aum=1e8, rates=np.full(len(CAL), 0.02), bl=bl, etf=etf)
    lg = res["log"]
    assert (lg["status"] == "solved").all()
    assert (lg["gross"] <= gate.R13_PA_GROSS_CAP + 1e-9).all()
    assert np.allclose(lg["ex_ante_beta_spy"], 1.0)
    assert (lg["iwm_w"] < 0).all() and (lg["spy_w"] > 0).all()
    assert np.allclose(lg["stock_w"], lg["leg_s"], atol=3e-3)
    assert (lg["n_universe"] == 50).all()
    assert (lg["n_names"] <= 50).all()
    d = res["daily"]
    assert (d["iwm_borrow"] <= 0).all() and (d["financing"] <= 0).all()
    # P&L decomposition adds up (between rebalances)
    no_reb = ~d.index.isin(lg["Date"])
    tot = d[["stock", "spy", "iwm", "financing", "iwm_borrow"]].sum(axis=1)
    assert np.allclose(tot[no_reb], res["ret"][no_reb], atol=1e-12)


def test_pa_iwm_point_in_time(monkeypatch):
    """Changing ETF / stock returns after a decision date leaves that
    rebalance's betas and legs unchanged."""
    monkeypatch.setattr(gate, "R13_PA_EXCLUDE_TOP", 10)
    prep, etf = _etf(_toy_prep())
    bl = _bl(prep)
    a = r13.run_book("pa_iwm", prep, aum=1e8, rates=np.zeros(len(CAL)), bl=bl, etf=etf)["log"]
    d2 = a["decision"].iloc[1]
    k = int(np.searchsorted(np.asarray(prep["dates"]).astype(str), d2, side="right"))
    etf2 = {key: v.copy() for key, v in etf.items()}
    k += 1                       # keep the trade session's own return (it drifts w_prev)
    etf2["iwm"][k:] *= 5.0
    prep2 = dict(prep)
    prep2["ret"] = prep["ret"].copy()
    prep2["ret"][k:] *= -3.0
    b = r13.run_book("pa_iwm", prep2, aum=1e8, rates=np.zeros(len(CAL)), bl=bl, etf=etf2)["log"]
    for c in ("b_pI", "b_pS", "b_IS", "leg_s", "leg_h", "leg_w_spy"):
        assert a[c].iloc[1] == pytest.approx(b[c].iloc[1], rel=1e-12, abs=1e-15)
        assert a[c].iloc[0] == pytest.approx(b[c].iloc[0], rel=1e-12, abs=1e-15)


# --------------------------------------------------------------------------- #
# Adoption rule
# --------------------------------------------------------------------------- #
def test_adopt_r13():
    inc = {"ir_spy": -0.1, "max_dd": -0.28, "sharpe": 1.04, "cagr": 0.156}
    t = {"lev125": {"ir_spy": 0.1, "max_dd": -0.33, "sharpe": 0.95, "cagr": 0.17},
         "pa_iwm": {"ir_spy": 0.2, "max_dd": -0.30, "sharpe": 0.93, "cagr": 0.165}}
    out = gate.adopt_r13(t, inc, 0.161, {"lev125": False, "pa_iwm": False})
    assert out["adoptable"] == ["lev125"]                      # pa_iwm Sharpe 0.93 < 1.04 - 0.10
    assert out["winner"] == "lev125"
    t["pa_iwm"]["sharpe"] = 0.95
    assert gate.adopt_r13(t, inc, 0.161, {})["winner"] == "pa_iwm"   # both -> higher IR
    t["lev125"]["max_dd"] = -0.36
    t["pa_iwm"]["cagr"] = 0.16
    out = gate.adopt_r13(t, inc, 0.161, {"lev125": True})
    assert out["adoptable"] == ["lev125"] and not out["per_trial"]["pa_iwm"]["checks"]["cagr_gt_spy"]

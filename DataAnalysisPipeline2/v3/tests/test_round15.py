"""Round 15 (intra-month crisis triggers on lev125) mechanics on toy inputs (no data store)."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import costs as cs
from DataAnalysisPipeline2.v3 import gate
from DataAnalysisPipeline2.v3 import round5 as r5
from DataAnalysisPipeline2.v3 import round13 as r13
from DataAnalysisPipeline2.v3 import round15 as r15
from DataAnalysisPipeline2.v3.tests.test_round5 import CAL, _toy_prep
from DataAnalysisPipeline2.v3.tests.test_round13 import _bl

RATES = np.full(len(CAL), 0.03)


def _plan(prep):
    return r5.plan_for("monthly", list(prep["by_date"]), prep["calendar"])


def _i(d):
    return int(np.searchsorted(CAL, d))


def _state(on_from=None, on_to=None):
    """Detection state per CAL session: ON on [on_from, on_to) (indices)."""
    s = np.zeros(len(CAL), dtype=bool)
    if on_from is not None:
        s[on_from:on_to] = True
    return s


def _free_detection(prep, after=0):
    """A detection index d such that neither d + 1 nor d + 2 is a plan session
    (and d + 1 is past the first plan session)."""
    plan = set(_plan(prep))
    first = _i(min(plan))
    for k in range(max(first + 2, after), len(CAL) - 3):
        if CAL[k + 1] not in plan and CAL[k + 2] not in plan and CAL[k] not in plan:
            return k
    raise AssertionError


# --------------------------------------------------------------------------- #
# Triggers
# --------------------------------------------------------------------------- #
def test_rv21_and_drawdown_definitions():
    rng = np.random.default_rng(1)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 400))
    rv = r15.rv21(close)
    r = np.diff(close) / close[:-1]
    assert np.isnan(rv[:21]).all() and np.isfinite(rv[21])
    assert rv[21] == pytest.approx(np.std(r[0:21], ddof=1) * np.sqrt(252))
    assert rv[300] == pytest.approx(np.std(r[279:300], ddof=1) * np.sqrt(252))
    dd = r15.drawdown_252(close)
    assert np.isnan(dd[:251]).all()
    assert dd[251] == pytest.approx(close[251] / close[:252].max() - 1)
    assert dd[399] == pytest.approx(close[399] / close[148:400].max() - 1)


def test_hysteresis_state_machine():
    on, off = (lambda v: v > 0.25), (lambda v: v < 0.20)
    x = np.array([np.nan, 0.30, 0.10, 0.26, 0.22, 0.25, 0.20, 0.199, 0.24, 0.251, np.nan, 0.22])
    s = r15.hysteresis(x, on, off)
    # undefined -> OFF; 0.30 ON; 0.10 OFF; 0.26 ON; stays ON in the band (0.22, 0.25, 0.20 not < 0.20);
    # 0.199 OFF; 0.24 stays OFF (not > 0.25); 0.251 ON; NaN forces OFF; 0.22 stays OFF
    assert s.tolist() == [False, True, False, True, True, True, True, False, False, True, False, False]
    assert not r15.hysteresis(np.array([0.25, 0.25]), on, off).any()      # strict '>'


def test_vol_state_initial_and_undefined_window():
    rng = np.random.default_rng(2)
    r = np.r_[rng.normal(0, 0.05, 20), rng.normal(0, 0.003, 60), rng.normal(0, 0.04, 30), rng.normal(0, 0.003, 80)]
    close = 100 * np.cumprod(1 + np.r_[0.0, r])
    s = r15.vol_state(close)
    rv = r15.rv21(close)
    assert not s[:21].any()                       # huge vol, but fewer than 21 returns -> OFF
    k_on = int(np.argmax(s))
    assert rv[k_on] > gate.R15_VOL_ON and (rv[21:k_on] <= gate.R15_VOL_ON).all()
    k_off = k_on + int(np.argmax(~s[k_on:]))
    assert rv[k_off] < gate.R15_VOL_OFF and (rv[k_on:k_off] >= gate.R15_VOL_OFF).all()


def test_dd_state_thresholds():
    close = np.r_[np.full(260, 100.0), 95.0, 90.1, 89.999, 92.0, 94.9, 95.001, 96.0, 89.0]
    s = r15.dd_state(close)
    assert not s[:262].any()                      # -5%, -9.9%: still OFF
    assert s[262] and s[263] and s[264]           # -10.001% ON; -8%, -5.1% stay ON
    assert not s[265]                             # -4.999% OFF
    assert not s[266] and s[267]


def test_trigger_state_has_no_look_ahead():
    rng = np.random.default_rng(3)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.02, 600))
    a_v, a_d = r15.vol_state(close), r15.dd_state(close)
    for d in (100, 300, 450):
        c2 = close.copy()
        c2[d + 1:] *= np.cumprod(1 + rng.normal(-0.01, 0.05, len(c2) - d - 1))
        assert np.array_equal(a_v[:d + 1], r15.vol_state(c2)[:d + 1])
        assert np.array_equal(a_d[:d + 1], r15.dd_state(c2)[:d + 1])


def test_episodes():
    s = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=bool)
    eps = r15.episodes(s, CAL[:9])
    assert [(e["on"], e["off"], e["n_on"]) for e in eps] == [(CAL[0], CAL[2], 2), (CAL[4], CAL[7], 3),
                                                             (CAL[8], None, 1)]


# --------------------------------------------------------------------------- #
# Books
# --------------------------------------------------------------------------- #
def test_trigger_off_equals_round13_lev125():
    prep = _toy_prep()
    bl = _bl(prep)
    a = r13.run_book("lev125", prep, aum=1e8, rates=RATES, bl=bl)
    for trial in (gate.R15_BASE_TRIAL,) + gate.R15_TRIALS:
        b = r15.run_book(trial, prep, aum=1e8, rates=RATES, bl=bl, state=None)
        assert np.array_equal(a["ret"].to_numpy(), b["ret"].to_numpy()), trial
        assert np.array_equal(a["daily"]["financing"].to_numpy(), b["daily"]["financing"].to_numpy())
        assert np.array_equal(a["log"]["cost"].to_numpy(), b["log"]["cost"].to_numpy())
        assert len(b["events"]) == 0
    # the base book ignores any state
    c = r15.run_book(gate.R15_BASE_TRIAL, prep, aum=1e8, rates=RATES, bl=bl, state=_state(30, 60))
    assert np.array_equal(a["ret"].to_numpy(), c["ret"].to_numpy())


def test_timing_d_to_d_plus_1_and_pro_rata_scaling():
    prep = _toy_prep()
    bl = _bl(prep)
    k = _free_detection(prep)
    base = r15.run_book("delev_vol", prep, aum=1e8, rates=RATES, bl=bl, state=None)
    trig = r15.run_book("delev_vol", prep, aum=1e8, rates=RATES, bl=bl, state=_state(k, k + 15))
    rb, rt = base["ret"], trig["ret"]
    sess = list(rb.index)
    j = sess.index(CAL[k])                        # detection session in book coordinates
    # identical through the detection session d; the trade at the close of d + 1 only costs that day
    assert np.array_equal(rb.to_numpy()[:j + 1], rt.to_numpy()[:j + 1])
    dt = trig["daily"]
    assert dt["lev"].iloc[j] == gate.R13_LEV and dt["lev"].iloc[j + 1] == 1.0
    assert dt["trade_state"].iloc[j] == 0 and dt["trade_state"].iloc[j + 1] == 1
    ev_ = trig["events"]
    e0 = ev_.iloc[0]
    assert e0["Date"] == CAL[k + 1] and e0["detect"] == CAL[k] and e0["action"] == "scale" and e0["to_on"]
    assert e0["decision"] == max(d for d in prep["by_date"] if d <= CAL[k])
    assert dt["scale_cost"].iloc[j + 1] == pytest.approx(e0["cost"]) and e0["cost"] > 0
    # pro rata: one-way turnover = (gross before - 1.0) / 2, cost >= half spread x |dw|
    assert e0["turnover_oneway"] == pytest.approx(0.125, abs=0.01)
    assert e0["cost"] >= 5e-4 * 2 * e0["turnover_oneway"] - 1e-15
    # the day-(d+1) return differs only by that cost (same pre-trade P&L)
    nav_ratio = (1 + rb.iloc[j + 1]) - (1 + rt.iloc[j + 1])
    assert nav_ratio == pytest.approx(e0["cost"] * (1 + rb.iloc[j + 1]), rel=1e-9)
    # from d + 2 the stock P&L is the untriggered book's x 1.0 / (drifted gross ~ 1.25)
    ratio = dt["stock"].iloc[j + 2] / base["daily"]["stock"].iloc[j + 2]
    assert ratio == pytest.approx(1.0 / 1.25, rel=0.02)
    # OFF again: scaled back up to 1.25
    e1 = ev_.iloc[1]
    assert e1["Date"] == CAL[k + 16] and not e1["to_on"] and e1["L_new"] == gate.R13_LEV
    assert e1["action"] in ("scale", "rebalance")


def test_no_look_ahead_in_book():
    """State changes after d do not change the book through d + 1."""
    prep = _toy_prep()
    bl = _bl(prep)
    k = _free_detection(prep)
    a = r15.run_book("rebal_delev_vol", prep, aum=1e8, rates=RATES, bl=bl, state=_state(k, k + 10))
    s2 = _state(k, k + 10)
    s2[k + 1:] = ~s2[k + 1:]
    b = r15.run_book("rebal_delev_vol", prep, aum=1e8, rates=RATES, bl=bl, state=s2)
    j = list(a["ret"].index).index(CAL[k])
    assert np.array_equal(a["ret"].to_numpy()[:j + 2], b["ret"].to_numpy()[:j + 2])


def test_pro_rata_and_trade_cost_arithmetic():
    w = np.array([0.5, 0.25, 0.5, 0.0])
    t = r15.pro_rata(w, 1.0)
    assert t.sum() == pytest.approx(1.0) and np.allclose(t / t.sum(), w / w.sum())
    assert np.array_equal(r15.pro_rata(np.zeros(3), 1.0), np.zeros(3))
    x = {"hs": np.array([1e-3, 2e-3, 3e-3]), "sigma": np.full(3, 0.02), "adv": np.full(3, 1e7)}
    cols = {"A": 0, "B": 1, "C": 2, "D": 3}
    dw = t - w
    cost, mc = r15.trade_cost(x, np.array(["A", "B", "C"]), cols, 4, 1e8, dw)
    imp = cs.impact_coef(x["sigma"], x["adv"], 1e8)
    hs = np.r_[x["hs"], cs.DEFAULT_HS]
    want = (hs * np.abs(dw)).sum() + (np.r_[imp, 0.0] * np.abs(dw) ** 1.5).sum()
    assert cost == pytest.approx(want, rel=1e-12) and mc.tolist() == [0, 1, 2]


def test_derisk_holds_cash_without_financing():
    prep = _toy_prep()
    bl = _bl(prep)
    k = _free_detection(prep)
    res = r15.run_book("derisk_vol", prep, aum=1e8, rates=RATES, bl=bl, state=_state(k, k + 12))
    d = res["daily"]
    j = list(d.index).index(CAL[k])
    assert d["lev"].iloc[j + 1] == 0.5
    assert (d["financing"].iloc[j + 2:j + 12] == 0).all()
    assert (d["financing"].iloc[1:j + 1] < 0).all()
    assert res["events"].iloc[0]["turnover_oneway"] == pytest.approx(0.375, abs=0.02)


def test_state_change_on_rebalance_session_only_rebalances():
    prep = _toy_prep()
    bl = _bl(prep)
    plan = sorted(_plan(prep))
    k = _i(plan[2]) - 1                           # detection the session before a scheduled rebalance
    for trial in ("delev_vol", "rebal_delev_vol", "rebal_vol"):
        res = r15.run_book(trial, prep, aum=1e8, rates=RATES, bl=bl, state=_state(k, k + 30))
        e0 = res["events"].iloc[0]
        assert e0["Date"] == plan[2] and e0["action"] == "rebalance"
        lg = res["log"]
        assert (lg["kind"] == "monthly").all() and len(lg) == len(plan)      # no off-cycle in this episode
        assert lg.set_index("Date").loc[plan[2], "lev"] == gate.R15_SPECS[trial][1]
        assert res["daily"].loc[plan[2], "scale_cost"] == 0.0


def test_one_offcycle_per_episode():
    prep = _toy_prep()
    bl = _bl(prep)
    plan = sorted(_plan(prep))
    k = _free_detection(prep)
    # one long episode that spans a scheduled rebalance -> exactly one off-cycle
    res = r15.run_book("rebal_vol", prep, aum=1e8, rates=RATES, bl=bl, state=_state(k, k + 40))
    lg = res["log"]
    off = lg[lg["kind"] == "offcycle"]
    assert len(off) == 1 and off["Date"].iloc[0] == CAL[k + 1] and off["trigger_date"].iloc[0] == CAL[k]
    assert off["decision"].iloc[0] == max(d for d in prep["by_date"] if d <= CAL[k])
    assert off["lev"].iloc[0] == 1.25 and (lg["kind"] == "monthly").sum() == len(plan)
    assert res["events"].iloc[0]["action"] == "offcycle"
    # rebal_vol: the OFF event does not trade (L unchanged)
    assert (res["events"]["action"].iloc[1:] != "scale").all()
    # two episodes -> two off-cycle rebalances; L = 1.00 in the same trade
    k2 = _free_detection(prep, after=k + 25)
    s = _state(k, k + 5)
    s[k2:k2 + 5] = True
    res2 = r15.run_book("rebal_delev_vol", prep, aum=1e8, rates=RATES, bl=bl, state=s)
    off2 = res2["log"][res2["log"]["kind"] == "offcycle"]
    assert off2["Date"].tolist() == [CAL[k + 1], CAL[k2 + 1]] and (off2["lev"] == 1.0).all()
    assert np.allclose(off2["gross"], 1.0, atol=3e-3)


# --------------------------------------------------------------------------- #
# Adoption rule
# --------------------------------------------------------------------------- #
def test_adopt_r15():
    inc = {"sharpe": 0.98, "max_dd": -0.34, "cagr": 0.179, "ir_spy": 0.12}
    t = {"delev_vol": {"sharpe": 0.96, "max_dd": -0.30, "cagr": 0.170, "ir_spy": 0.08},     # all hold
         "derisk_vol": {"sharpe": 1.00, "max_dd": -0.25, "cagr": 0.150, "ir_spy": 0.10},    # CAGR fails
         "rebal_vol": {"sharpe": 0.99, "max_dd": -0.32, "cagr": 0.18, "ir_spy": 0.12},      # DD gain < 3 pt
         "rebal_delev_vol": {"sharpe": 0.94, "max_dd": -0.28, "cagr": 0.175, "ir_spy": 0.12},  # Sharpe fails
         "delev_dd": {"sharpe": 0.97, "max_dd": -0.31, "cagr": 0.172, "ir_spy": 0.06}}      # IR fails
    out = gate.adopt_r15(t, inc, {})
    assert out["adoptable"] == ["delev_vol"] and out["selected"] == "delev_vol" and out["adopt"]
    pt = out["per_trial"]
    assert not pt["derisk_vol"]["checks"]["cagr"] and not pt["rebal_vol"]["checks"]["max_dd"]
    assert not pt["rebal_delev_vol"]["checks"]["sharpe"] and not pt["delev_dd"]["checks"]["ir_spy"]
    # boundaries are inclusive: exactly 3.0 pt DD gain, exactly -0.03 Sharpe, -1.0 pt CAGR, -0.05 IR
    edge = {"x": {"sharpe": 0.95, "max_dd": -0.31, "cagr": 0.169, "ir_spy": 0.07}}
    assert gate.adopt_r15(edge, inc, {})["adoptable"] == ["x"]
    # gate pass alone makes a trial adoptable
    assert "rebal_vol" in gate.adopt_r15(t, inc, {"rebal_vol": True})["adoptable"]
    # none adoptable -> highest Sharpe is selected, not adopted
    t.pop("delev_vol")
    out = gate.adopt_r15(t, inc, {})
    assert out["adoptable"] == [] and out["selected"] == "derisk_vol" and not out["adopt"]

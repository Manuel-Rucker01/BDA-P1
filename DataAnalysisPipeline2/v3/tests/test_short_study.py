"""Short-overlay event study helpers on toy inputs (no data store)."""
import numpy as np
import pandas as pd

from DataAnalysisPipeline2.v3 import short_study as ss


def _ret(T=320, N=6, seed=0):
    return np.random.default_rng(seed).normal(0.0005, 0.02, (T, N)).astype(np.float32)


def test_trailing_stats_use_nothing_after_t():
    r = _ret()
    t = 300
    a = ss.trailing_stats(r, t)
    r2 = r.copy()
    r2[t + 1:] = 9.0                       # change the future only
    b = ss.trailing_stats(r2, t)
    for k in a:
        assert np.allclose(a[k], b[k], equal_nan=True)
    # momentum skips the last 21 sessions
    r3 = r.copy()
    r3[t - ss.MOM_SKIP + 1:t + 1] = 0.5
    assert np.allclose(ss.trailing_stats(r3, t)["mom"], a["mom"])
    assert np.allclose(a["max21"], r[t - 20:t + 1].max(0))
    assert np.allclose(a["vol60"], r[t - 59:t + 1].astype(float).std(0, ddof=1))


def test_trailing_stats_need_history():
    r = _ret()
    r[:200, 0] = np.nan                    # too few observations for momentum
    assert np.isnan(ss.trailing_stats(r, 300)["mom"][0])


def test_window_outcomes_and_mae():
    r = np.zeros((10, 2), dtype=np.float32)
    r[3] = [0.10, -0.10]                   # first day after entry at row 2
    r[4] = [-0.20, 0.05]
    r[8] = [5.0, 5.0]                      # outside a 3-session window
    o = ss.window_outcomes(r, 2, np.array([0, 1]), 3)
    assert np.allclose(o["ret"], [1.1 * 0.8 - 1, 0.9 * 1.05 - 1])
    assert np.allclose(o["mae"], [0.10, 0.0])   # never above entry -> 0
    # the entry day's own return is not part of the window
    r[2] = [0.9, 0.9]
    assert np.allclose(ss.window_outcomes(r, 2, np.array([0, 1]), 3)["ret"], o["ret"])


def _frame(n=40, seed=1):
    g = np.random.default_rng(seed)
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)], "Date": "2020-01-31", "entry": "2020-02-03",
        "pred": g.uniform(0, 1, n), "sec": np.repeat([0, 1], n // 2), "price": 50.0, "mcap": 5e9, "adv": 1e8,
        "si_ratio": g.uniform(0.0, 0.09, n), "si_dtc": 3.0, "si_chg": g.normal(0, 0.1, n), "si_age": 5.0,
        "return_5d": 0.0, "blackout": False, "ear_src10q": 0.0, "ear_days": 20.0, "ear": g.normal(0, 0.08, n),
        "sue_eps": g.normal(0, 1, n), "sue_days": 20.0, "fund_imputed": False, "fund_staleness_days": 30.0,
        "net_issuance_12m": g.normal(0.02, 0.05, n), "accruals": g.normal(0, 0.1, n), "roa": g.normal(0.03, 0.1, n),
        "mom": g.normal(0.1, 0.3, n), "vol60": g.uniform(0.01, 0.05, n), "max21": g.uniform(0.02, 0.2, n)})


def test_filters():
    f = _frame()
    assert ss.shortable_mask(f).all()
    for col, bad in (("price", 9.99), ("mcap", 1.9e9), ("adv", 4.9e7), ("si_ratio", 0.11), ("si_dtc", 8.5),
                     ("si_age", 60.0), ("return_5d", 0.2), ("blackout", True), ("si_ratio", np.nan),
                     ("pred", np.nan), ("sec", -1)):
        g = f.copy()
        g.loc[0, col] = bad
        m = ss.shortable_mask(g)
        assert not m.iloc[0] and m.iloc[1:].all(), col


def test_conditions():
    f = _frame()
    L = ss.add_conditions(f)
    # C1: bottom decile per sector = the 2 lowest of 20
    for s in (0, 1):
        g = L[L["sec"] == s]
        assert set(g.index[g["C1_model"]]) == set(g["pred"].nsmallest(2).index)
    assert (L.loc[L["C2_si"], "si_chg"] > 0).all()
    assert (L.loc[L["C2_si"], "si_ratio"] >= L["si_ratio"].quantile(0.79)).all()
    assert ((L["ear"] <= ss.C3_EAR_MAX) & (L["sue_eps"] < 0)).equals(L["C3_earn"])
    assert (L["n_agree"] == L[ss.BASE].sum(axis=1)).all()
    assert (L["C6_consensus"] == (L["n_agree"] >= ss.C6_K)).all()
    # imputed fundamentals never fire C4; small sectors never fire C1
    g = f.copy()
    g["fund_imputed"] = True
    assert not ss.add_conditions(g)["C4_fund"].any()
    g = f.copy()
    g["sec"] = np.arange(len(g)) % 8      # 5 names per sector < MIN_SECTOR_N
    assert not ss.add_conditions(g)["C1_model"].any()
    # an absurd issuance value is treated as unknown
    g = f.copy()
    g["accruals"], g["roa"], g["net_issuance_12m"] = -1.0, -0.1, 50.0
    g.loc[0, "accruals"] = 1.0
    c4 = ss.add_conditions(g)["C4_fund"]
    assert c4.iloc[0] and not c4.iloc[1:].any()


def test_cluster_t_and_verdict():
    x = np.array([0.01, 0.02, 0.03, 0.02])
    # every observation its own cluster -> the usual (CR0) t
    e = x - x.mean()
    assert np.isclose(ss.cluster_t(x, np.arange(4)), x.mean() / (np.sqrt((e ** 2).sum()) / 4))
    # perfectly duplicated clusters do not add information
    assert np.isclose(ss.cluster_t(np.repeat(x, 2), np.repeat(np.arange(4), 2)), ss.cluster_t(x, np.arange(4)))
    s = {"events": 100, "dates": 40, "net_spy_21": {"p_one_sided": 0.001, "mean_fm": 0.01},
         "net_sec_21": {"mean_fm": 0.01}, "net_spy_30": {"mean_fm": 0.01}}
    assert ss.verdict_discovery(s)["survives"]
    assert not ss.verdict_discovery({**s, "net_spy_21": {"p_one_sided": 0.009, "mean_fm": 0.01}})["survives"]
    assert not ss.verdict_discovery({**s, "net_sec_21": {"mean_fm": -0.01}})["survives"]
    assert not ss.verdict_discovery({**s, "events": 10})["survives"]
    assert ss.verdict_confirmation(s, 1)["confirmed"]
    assert not ss.verdict_confirmation({**s, "net_spy_21": {"p_one_sided": 0.03, "mean_fm": 0.01}}, 2)["confirmed"]


def test_summarize_signs():
    ev = pd.DataFrame({"entry": ["a", "a", "b", "b", "c", "c"], "ticker": list("uvwxyz"), "C": [True] * 6,
                       "cost_rt": 0.002})
    for h in ss.HORIZONS:
        ev[f"ret_{h}"] = [-0.05, -0.02, -0.03, 0.01, -0.04, -0.01]
        ev[f"mae_{h}"] = [0.0, 0.01, 0.0, 0.03, 0.0, 0.0]
        for k in ("gross_spy", "gross_sec"):
            ev[f"{k}_{h}"] = -ev[f"ret_{h}"]
        for k in ("net_spy", "net_spy_hi", "net_sec", "abs_short"):
            ev[f"{k}_{h}"] = -ev[f"ret_{h}"] - 0.002
    s = ss.summarize(ev, "C")
    assert s["events"] == 6 and s["dates"] == 3 and s["events_per_month"] == 2.0
    assert s["net_spy_21"]["mean_fm"] > 0 and s["net_spy_21"]["t_nw"] > 0
    assert np.isclose(s["net_spy_21"]["hit"], 5 / 6)
    assert np.isclose(s["adverse_21"]["mae_max"], 0.03) and np.isclose(s["adverse_21"]["stock_ret_max"], 0.01)

"""Round 11 mechanics and point-in-time behaviour on toy inputs (no data store)."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import gate
from DataAnalysisPipeline2.v3 import optimizer as op
from DataAnalysisPipeline2.v3 import round5 as r5
from DataAnalysisPipeline2.v3 import round11 as r11
from DataAnalysisPipeline2.v3.tests.test_round5 import CAL, _toy_prep

SPEC = {**gate.R5_COMMON, **gate.R5_CONFIGS["mv_abs_all"]}


def _args(prep, d):
    x = prep["by_date"][d]
    m = x["model"]
    return dict(tickers=m.tickers, scores=x["pred"], X=m.X, F_daily=m.F, D_daily=m.D, mcap=m.mcap,
                sector=m.sector, factors=m.factors, half_spread=x["hs"], sigma_daily=x["sigma"], adv=x["adv"],
                ic=0.05)


def _spy(sessions, seed=1):
    rng = np.random.default_rng(seed)
    n = len(sessions)
    return {"ret": rng.normal(0.0004, 0.01, n), "adv": np.full(n, 3e10), "vol": np.full(n, 0.01)}


# --------------------------------------------------------------------------- #
# Optimizer extension
# --------------------------------------------------------------------------- #
def test_optimize_ext_equals_optimize_portfolio():
    prep = _toy_prep(n=40)
    d = sorted(prep["by_date"])[2]
    a = _args(prep, d)
    wp = {t: 1.0 / 40 for t in a["tickers"][:20]}
    w1, _ = op.optimize_portfolio(SPEC, aum=1e8, w_prev=wp, **a)
    w2, _ = r11.optimize_ext(SPEC, aum=1e8, w_prev=wp, **a)
    assert set(w1) == set(w2)
    assert max(abs(w1[t] - w2[t]) for t in w1) < 1e-12


def test_ub_cap_blocks_new_and_increased_positions():
    prep = _toy_prep(n=40)
    d = sorted(prep["by_date"])[2]
    a = _args(prep, d)
    free, _ = r11.optimize_ext(SPEC, aum=1e8, **a)
    top = sorted(free, key=free.get, reverse=True)[:5]       # names the optimizer wants most
    held = {top[0]: 0.01}                                     # one of them held at 1%
    cap = np.where(np.isin(a["tickers"], top), 0.0, np.inf)
    cap[list(a["tickers"]).index(top[0])] = 0.01
    w, _ = r11.optimize_ext(SPEC, aum=1e8, w_prev=held, ub_cap=cap, **a)
    for t in top[1:]:
        assert w.get(t, 0.0) == 0.0
    assert w.get(top[0], 0.0) <= 0.01 + 1e-9
    assert abs(sum(w.values()) - 1.0) < 2e-3


def test_etf_asset_hits_beta_target_and_budget():
    prep = _toy_prep(n=40)
    d = sorted(prep["by_date"])[2]
    a = _args(prep, d)
    m = prep["by_date"][d]["model"]
    b = r11.spy_proxy(m.mcap, top=20)
    assert b.sum() == pytest.approx(1.0) and (b > 0).sum() == 20
    etf = {"b": b, "w_prev": 0.0, "half_spread": 0.5e-4, "imp_coef": 1e-4, "beta_target": 1.0}
    w, diag = r11.optimize_ext(SPEC, aum=1e8, etf=etf, **a)
    assert "SPY" in w and 0.0 <= w["SPY"] <= 1.0
    assert sum(w.values()) == pytest.approx(1.0, abs=2e-3)
    assert diag["ex_ante_beta_spy"] == pytest.approx(1.0, abs=5e-3)
    # the reported beta is the exact one: recompute from the full covariance
    beta = r11.betas_to(m.X, m.F, m.D, b)
    ws = np.array([w.get(t, 0.0) for t in m.tickers])
    assert float(beta @ ws) + w["SPY"] == pytest.approx(diag["ex_ante_beta_spy"], abs=1e-9)


def test_betas_to_proxy_is_one_for_the_proxy():
    prep = _toy_prep(n=30)
    m = next(iter(prep["by_date"].values()))["model"]
    b = r11.spy_proxy(m.mcap, top=10)
    beta = r11.betas_to(m.X, m.F, m.D, b)
    assert float(beta @ b) == pytest.approx(1.0)
    S = m.cov()
    assert np.allclose(beta, S @ b / (b @ S @ b))


# --------------------------------------------------------------------------- #
# Rates
# --------------------------------------------------------------------------- #
def test_rate_series_uses_previous_observation_only():
    raw = pd.DataFrame({"observation_date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
                        "DTB3": ["5.00", ".", "4.00", "3.00"]})
    s = r11.rate_series(raw, ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    assert np.isnan(s[0])                       # nothing strictly before
    assert s[1] == pytest.approx(0.05)
    assert s[2] == pytest.approx(0.05)          # 01-03 missing -> forward-filled 01-02
    assert s[3] == pytest.approx(0.04)          # never the same-day 3.00
    assert s[4] == pytest.approx(0.03)


# --------------------------------------------------------------------------- #
# Earnings blackout: point in time
# --------------------------------------------------------------------------- #
def _events(rows):
    return pd.DataFrame([{"cik": c, "day0": 0, "src10q": s, "acceptance": pd.Timestamp(a, tz="UTC"),
                          "avail_pos": 0, "prev_day0": np.nan} for c, a, s in rows])


def test_retained_events_drop_intra_quarter_and_10q():
    ev_ = _events([(1, "2024-01-25 21:00", 0), (1, "2024-02-05 21:00", 0),   # extra 2.02 11 days later
                   (1, "2024-04-25 21:00", 0), (1, "2024-05-01 21:00", 1),   # 10-Q fallback ignored
                   (2, "2024-03-01 02:00", 0)])                              # 21:00 NY on 02-29
    k = r11.retained_events(ev_)
    assert list(k[k["cik"] == 1]["adate"]) == ["2024-01-25", "2024-04-25"]
    assert list(k[k["cik"] == 2]["adate"]) == ["2024-02-29"]


def test_expected_dates_ignore_future_events():
    ev_ = _events([(1, "2024-01-25 21:00", 0), (1, "2024-04-25 21:00", 0), (2, "2024-02-10 21:00", 0)])
    k = r11.retained_events(ev_)
    e = r11.expected_dates(k, "2024-03-29")
    assert e[1] == (pd.Timestamp("2024-01-25") + pd.Timedelta(days=91)).strftime("%Y-%m-%d")
    # adding / changing events after the decision date changes nothing
    ev2 = pd.concat([ev_, _events([(1, "2024-03-30 12:00", 0), (2, "2024-04-01 12:00", 0)])], ignore_index=True)
    e2 = r11.expected_dates(r11.retained_events(ev2), "2024-03-29")
    assert e.equals(e2)


def test_blackout_window_and_mask():
    lo, hi = r11.blackout_window(CAL[20], CAL)
    assert hi == CAL[25]
    assert lo == (pd.Timestamp(CAL[20]) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    exp = pd.Series({1: CAL[22], 2: CAL[30], 3: lo})
    mask = r11.blackout_tickers(np.array(["A", "B", "C", "D"]), {"A": 1, "B": 2, "C": 3}, exp, (lo, hi))
    assert mask.tolist() == [True, False, True, False]


# --------------------------------------------------------------------------- #
# Plans and the book runner
# --------------------------------------------------------------------------- #
def test_biweekly_plan_every_second_week_next_session():
    pdates = list(CAL[4::5])
    plan = r11.biweekly_plan(pdates, CAL)
    dec = sorted(plan.values())
    monthly = r5.plan_for("monthly", pdates, CAL)
    assert dec[0] == monthly[min(monthly)]
    assert dec == [d for d in pdates if d >= dec[0]][::2][:len(dec)]
    assert all(r > d for r, d in plan.items())


def test_run_book_base_equals_round5_run_config():
    prep = _toy_prep()
    a = r5.run_config("mv_abs_all", 1e8, prep=prep)
    b = r11.run_book("base", prep, aum=1e8)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy(), atol=1e-14)


def test_run_book_lev_caps_leverage_and_pays_financing():
    prep = _toy_prep()
    rates0 = np.zeros(len(CAL))
    lo = r11.run_book("beta1_lev", prep, aum=1e8, rates=rates0)
    hi = r11.run_book("beta1_lev", prep, aum=1e8, rates=np.full(len(CAL), 0.05))
    lg = lo["log"]
    assert (lg["lev"] <= gate.R11_LEV_CAP + 1e-12).all()
    assert np.allclose(lg["gross"], lg["lev"], atol=3e-3)
    assert np.allclose(lg["ex_ante_beta_spy"].where(lg["lev"] < gate.R11_LEV_CAP).dropna(), 1.0)
    # financing only costs when levered
    if (lg["lev"] > 1).any():
        assert np.prod(1 + hi["ret"]) < np.prod(1 + lo["ret"])


def test_run_book_spy_and_blackout():
    prep = _toy_prep()
    s = r11.run_book("beta1_spy", prep, aum=1e8, spy=_spy(CAL))
    lg = s["log"]
    assert (lg["spy_w"] >= 0).all() and (lg["spy_w"] <= 1).all()
    assert np.allclose(lg["net"], 1.0, atol=3e-3)
    assert np.allclose(lg["ex_ante_beta_spy"], 1.0, atol=1e-2)
    # every name blacked out at every rebalance: nothing may be bought or increased
    tick = prep["tickers"]
    d1 = sorted(prep["by_date"])[1]
    kept = pd.DataFrame({"cik": np.arange(len(tick)), "adate": d1})
    cik = {d: {t: i for i, t in enumerate(tick)} for d in prep["by_date"]}

    def far(*_a, **_k):
        return pd.Series({i: "2099-01-01" for i in range(len(tick))})
    bl = {"kept": kept, "cik": cik}
    base = r11.run_book("base", prep, aum=1e8)
    import DataAnalysisPipeline2.v3.round11 as mod
    orig = mod.expected_dates
    try:
        mod.expected_dates = lambda k, d, **kw: pd.Series({i: d for i in range(len(tick))})
        b = r11.run_book("earn_blackout", prep, aum=1e8, bl=bl)
        mod.expected_dates = far
        nb = r11.run_book("earn_blackout", prep, aum=1e8, bl=bl)
    finally:
        mod.expected_dates = orig
    assert np.allclose(nb["ret"].to_numpy(), base["ret"].to_numpy(), atol=1e-12)   # no blackout == base
    lgb = b["log"]
    assert (lgb["n_blackout"] == len(tick)).all()
    assert lgb["n_names"].iloc[0] == 0          # first rebalance from cash: nothing may be bought
    assert (lgb["cost"].iloc[1:] < 1e-9).all() if len(lgb) > 1 else True


def test_residual_target_is_orthogonal_to_exposures():
    rng = np.random.default_rng(3)
    n = 400
    X = np.hstack([rng.normal(size=(n, 2)), np.eye(3)[rng.integers(0, 3, n)], np.zeros((n, 1))])
    f = np.array([0.02, -0.01, 0.01, 0.0, -0.02, 0.0])
    eps = rng.normal(0, 0.05, n)
    y = X @ f + eps
    w = rng.uniform(1, 4, n)
    y[0] = np.nan
    res, fh = r11.residual_target(y, X, w)
    assert np.isnan(res[0]) and fh[-1] == 0.0
    ok = np.isfinite(res)
    assert np.allclose((X[ok, :5] * w[ok, None]).T @ res[ok], 0.0, atol=1e-8)
    assert np.allclose(fh[:5], f[:5], atol=0.02)

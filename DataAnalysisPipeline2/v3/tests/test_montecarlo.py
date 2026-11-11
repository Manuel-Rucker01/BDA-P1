"""Monte-Carlo bands and live monitoring rules (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.v3 import montecarlo as mc


def _df(n=600, seed=0, mu=0.0008):
    rng = np.random.default_rng(seed)
    spy = rng.normal(0.0004, 0.01, n)
    strat = 0.8 * spy + rng.normal(mu - 0.8 * 0.0004, 0.006, n)
    idx = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({"strat": strat, "spy": spy}, index=idx)


def test_stationary_indices_block_structure():
    rng = np.random.default_rng(1)
    idx = mc.stationary_indices(500, 2000, 100, 21, rng)
    assert idx.shape == (2000, 100) and idx.min() >= 0 and idx.max() < 500
    cont = (idx[:, 1:] == (idx[:, :-1] + 1) % 500).mean()
    assert abs(cont - (1 - 1 / 21)) < 0.01  # P(continue block) ~ 1 - 1/mean_block


def test_simulate_deterministic_and_haircut():
    df = _df()
    a = mc.simulate(df, n_paths=500, n_days=63, seed=7)
    b = mc.simulate(df, n_paths=500, n_days=63, seed=7)
    np.testing.assert_array_equal(a["raw"]["cum"], b["raw"]["cum"])
    assert np.median(a["haircut"]["cum"][:, -1]) < np.median(a["raw"]["cum"][:, -1])
    assert (a["raw"]["mdd"] <= 0).all()
    assert (np.diff(a["raw"]["mdd"], axis=1) <= 1e-12).all()  # running max DD is monotone


def test_joint_resampling_keeps_correlation():
    df = _df(mu=0.0)
    s = mc.simulate(df, n_paths=2000, n_days=21, seed=3)["raw"]
    cum_spy = s["cum"][:, -1] - s["active"][:, -1]
    assert np.corrcoef(s["cum"][:, -1], cum_spy)[0, 1] > 0.6


def test_band_table_and_frozen(monkeypatch):
    df = _df()
    sims = mc.simulate(df, n_paths=1000, seed=1)
    bands = mc.band_table(sims)
    assert len(bands) == 8 and set(bands["version"]) == {"raw", "haircut"}
    assert (bands["cum_p5"] <= bands["cum_p50"]).all() and (bands["cum_p50"] <= bands["cum_p95"]).all()
    assert bands["p_beat_spy"].between(0, 1).all()
    fr = mc.build_frozen(df, sims)
    t = fr["thresholds"]
    assert fr["created"] == "2026-11-11" and fr["refrozen"]["date"] == "2026-11-11" and fr["seed"] == mc.SEED and len(fr["data_hash_sha256"]) == 64
    assert all(r <= y for r, y in zip(t["active_q01"], t["active_q05"]))
    assert all(r <= y for r, y in zip(t["mdd_q01"], t["mdd_q05"]))
    # interpolation lies between neighbouring knots
    v = mc.threshold_at(fr, "mdd_q05", 30)
    i = t["knots_days"].index(21)
    assert min(t["mdd_q05"][i], t["mdd_q05"][i + 1]) <= v <= max(t["mdd_q05"][i], t["mdd_q05"][i + 1])


def _frozen():
    k = list(mc.KNOTS)
    lin = lambda a: [a * d / 252 for d in k]  # noqa: E731
    return {"thresholds": {"knots_days": k, "active_q05": lin(-0.20), "active_q01": lin(-0.30),
                           "mdd_q05": lin(-0.25), "mdd_q01": lin(-0.35)}}


def _series(rets, start="2027-01-04"):
    idx = pd.bdate_range(start, periods=len(rets) + 1)
    return pd.Series(np.cumprod(np.r_[1.0, 1.0 + np.asarray(rets)]), index=idx)


def test_check_live_statuses():
    fr = _frozen()
    n = 126  # thresholds at d=126: active q05 -10%, q01 -15%; mdd q05 -12.5%, q01 -17.5%
    spy = _series(np.zeros(n))
    start = spy.index[0]
    g = mc.check_live(_series(np.full(n, 0.0002)), spy, start, fr)
    assert g["status"] == "GREEN" and g["elapsed_days"] == n
    y = mc.check_live(_series(np.full(n, -0.12 / n)), spy, start, fr)
    assert y["status"] == "YELLOW" and any("active" in r for r in y["reasons"])
    r = mc.check_live(_series(np.r_[np.full(n - 1, 0.0), -0.2]), spy, start, fr)
    assert r["status"] == "RED" and any("best_model.pkl" in x for x in r["reasons"])


def test_check_live_sparse_monthly_points_and_empty():
    fr = _frozen()
    eq = pd.Series([1.0, 1.01, 1.02], index=["2027-01-04", "2027-02-01", "2027-03-01"])
    out = mc.check_live(eq, eq, "2027-01-04", fr)
    assert out["status"] == "GREEN" and 35 <= out["elapsed_days"] <= 42
    assert mc.check_live(eq.iloc[:1], eq.iloc[:1], "2027-01-04", fr)["elapsed_days"] == 0


def test_live_from_monitoring(tmp_path):
    p = tmp_path / "monitoring.csv"
    pd.DataFrame(columns=mc.pd.Index(["account", "decision_date", "end_date", "status",
                                      "book_return", "spy_return"])).to_csv(p, index=False)
    assert mc.live_from_monitoring(p) is None
    pd.DataFrame({"account": ["monthly"] * 3, "decision_date": ["2026-10-01", "2026-12-01", "2027-01-04"],
                  "end_date": ["2026-11-03", "2027-01-04", "2027-02-01"],
                  "status": ["closed", "closed", "open"], "book_return": [0.5, 0.02, 0.01],
                  "spy_return": [0.0, 0.01, 0.0]}).to_csv(p, index=False)
    eq, sp, start = mc.live_from_monitoring(p)
    assert start == "2026-12-01" and len(eq) == 2
    assert eq.iloc[-1] == pytest.approx(1.02) and sp.iloc[-1] == pytest.approx(1.01)

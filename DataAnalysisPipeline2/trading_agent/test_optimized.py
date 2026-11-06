"""optimized_v3: the live mean-variance book, its fundamentals source and the
fallback to long_only_v3. Synthetic data, no network."""
import os

import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.trading_agent import bot, config, optimized
from DataAnalysisPipeline2.v3 import optimizer as op
from DataAnalysisPipeline2.v3 import risk
from DataAnalysisPipeline2.v3.tests.live_synthetic import make_live_bars_and_fund

TARGET = 0.98


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("fund"))
    bars, meta = make_live_bars_and_fund(d, n=200, n_sessions=330, seed=3)
    rng = np.random.default_rng(7)
    tick = sorted(bars["ticker"].unique())
    preds = pd.DataFrame({"ticker": tick, "pred_rank": rng.uniform(size=len(tick)),
                          "Sector": [meta["sector"].get(t, "Other") for t in tick],
                          "return_volatility_20d": rng.uniform(0.01, 0.04, len(tick)),
                          "kalman_beta": 1.0, "company_close": 50.0, "pred_proba": 0.5})
    today = (meta["last_filed"] + pd.Timedelta(days=5)).date()
    return bars, meta, d, preds, today


def _book(live, **kw):
    bars, _, d, preds, today = live
    args = dict(aum=1e6, ic=0.07, target_exposure=TARGET, candidates=[("test", d)], today=today,
                min_universe=50, min_names=10, log=lambda *a: None)
    args.update(kw)
    return optimized.optimized_book(preds, bars, **args)


def _inputs(live):
    bars, _, d, preds, _ = live
    return optimized.build_inputs(bars, preds.set_index("ticker")["pred_rank"], d)


# --------------------------------------------------------------------------- #
# The book
# --------------------------------------------------------------------------- #
def test_optimized_book_respects_constraints(live):
    w, diag = _book(live)
    inp, meta = _inputs(live)
    sp = optimized.live_spec()
    v = np.array([w.get(t, 0.0) for t in inp["tickers"]])
    assert set(w) <= set(inp["tickers"]) and not {"PENNY", "SHORTH"} & set(w)
    assert v.min() >= 0.0
    assert v.sum() <= TARGET + 1e-9 and v.sum() == pytest.approx(TARGET, abs=3e-3)
    cap = np.minimum(sp["name_cap"], sp["liq_frac"] * inp["adv"] / 1e6)
    assert np.all(v <= cap + 1e-12)
    b = inp["mcap"] / inp["mcap"].sum()
    for s in np.unique(inp["sector"]):
        m = inp["sector"] == s
        assert abs(v[m].sum() / TARGET - b[m].sum()) <= sp["sector_band"] + 2e-3
    assert np.all((v == 0) | (v / TARGET >= sp["min_weight"] - 1e-6))
    assert diag["construction"] == "optimized_v3" and diag["status"] == "solved"
    assert diag["n_names"] == int((v > 0).sum()) >= 10
    assert 0 < diag["ex_ante_vol_ann"] < 0.6 and diag["solve_seconds"] >= 0
    assert diag["fundamentals"]["source"] == "test"
    assert sum(diag["sector_weights"].values()) == pytest.approx(TARGET, abs=1e-3)
    assert meta["n_universe"] == diag["n_universe"]


def test_optimized_book_liquidity_cap_binds_at_large_aum(live):
    aum = 1e8
    w, _ = _book(live, aum=aum)
    inp, _ = _inputs(live)
    liq = 0.10 * inp["adv"] / aum
    v = np.array([w.get(t, 0.0) for t in inp["tickers"]])
    assert np.all(v <= np.minimum(0.05, liq) + 1e-6)
    assert np.any((liq < 0.05) & (v >= liq * TARGET - 1e-4))      # some names held at their ADV cap


def test_optimized_book_is_deterministic(live):
    w1, d1 = _book(live)
    w2, d2 = _book(live)
    assert w1 == w2 and d1["iters"] == d2["iters"]


def test_optimized_book_min_weight_and_w_prev(live):
    w, _ = _book(live, spec={"min_weight": 0.01})
    assert min(w.values()) >= 0.01 * TARGET - 1e-6
    # holding the previous book: (almost) nothing to trade
    w2, d2 = _book(live, w_prev={t: x / TARGET for t, x in w.items()}, spec={"min_weight": 0.01})
    assert d2["turnover_oneway"] < 0.05 and d2["n_prev_held"] == len(w)


def test_optimized_book_rejects_small_universe_and_unsolved(live, monkeypatch):
    with pytest.raises(optimized.OptimizerUnavailable, match="universe"):
        _book(live, min_universe=10_000)
    real = op.optimize_portfolio

    def unsolved(*a, **k):
        w, d = real(*a, **k)
        return w, {**d, "status": "max_iter"}

    monkeypatch.setattr(op, "optimize_portfolio", unsolved)
    with pytest.raises(optimized.OptimizerUnavailable, match="solver status"):
        _book(live)


def test_check_book_catches_violations(live):
    inp, _ = _inputs(live)
    sp = optimized.live_spec()
    n = len(inp["tickers"])
    good = {"status": "solved", "iters": 1}
    w = np.zeros(n)
    w[:3] = [0.5, 0.3, 0.2]                         # name cap
    with pytest.raises(optimized.OptimizerUnavailable, match="name cap"):
        optimized.check_book(w, inp, sp, 1e6, good)
    with pytest.raises(optimized.OptimizerUnavailable, match="budget"):
        optimized.check_book(np.full(n, 0.5 / n), inp, sp, 1e6, good)
    w = np.full(n, 1.0 / n)
    w[0] = -0.01
    with pytest.raises(optimized.OptimizerUnavailable, match="negative"):
        optimized.check_book(w, inp, sp, 1e6, good)


def test_short_history_is_refused(live):
    bars, meta, d, preds, today = live
    short = bars[bars["Date"] >= meta["dates"][-200]]
    with pytest.raises(optimized.OptimizerUnavailable, match="history"):
        optimized.optimized_book(preds, short, aum=1e6, ic=0.07, target_exposure=TARGET,
                                 candidates=[("test", d)], today=today, min_universe=50, log=lambda *a: None)


# --------------------------------------------------------------------------- #
# Fundamentals source + snapshot
# --------------------------------------------------------------------------- #
def test_resolve_fund_dir_order_and_staleness(live, tmp_path):
    _, meta, d, _, today = live
    missing = str(tmp_path / "nope")
    fd = optimized.resolve_fund_dir(candidates=[("env", missing), ("store", d)], today=today, max_age_days=30)
    assert fd["source"] == "store" and fd["path"] == d and fd["age_days"] == 5
    late = (pd.Timestamp(today) + pd.Timedelta(days=400)).date()
    with pytest.raises(optimized.OptimizerUnavailable, match="stale") as e:
        optimized.resolve_fund_dir(candidates=[("env", missing), ("store", d)], today=late, max_age_days=120)
    assert "missing" in str(e.value)


def test_snapshot_roundtrip_gives_the_same_panel(live, tmp_path):
    bars, meta, d, _, today = live
    out = str(tmp_path / "snap")
    man = optimized.build_snapshot(d, out, years=3, today=today)
    assert man["as_of"] == str(meta["last_filed"].date())
    assert optimized.fund_as_of(out) == meta["last_filed"]
    fa = pd.read_parquet(os.path.join(out, "fundamentals_asof.parquet"))
    assert list(fa.columns) == risk.LIVE_FUND_COLS
    p0 = risk.live_panel(bars, fund_dir=d)
    p1 = risk.live_panel(bars, fund_dir=out)
    assert np.array_equal(p0.elig, p1.elig) and np.array_equal(p0.sector, p1.sector)
    np.testing.assert_allclose(p0.mcap, p1.mcap, rtol=1e-5)
    fd = optimized.resolve_fund_dir(candidates=[("snapshot", out)], today=today)
    assert fd["source"] == "snapshot" and fd["as_of"] == man["as_of"]


def test_committed_snapshot_is_small_and_has_a_manifest():
    files = [os.path.join(optimized.SNAPSHOT_DIR, f) for f in optimized.FUND_FILES]
    assert all(os.path.exists(f) for f in files)
    assert sum(os.path.getsize(f) for f in files) < 5e6
    assert optimized.fund_as_of(optimized.SNAPSHOT_DIR) >= pd.Timestamp("2026-01-01")


# --------------------------------------------------------------------------- #
# Bot wiring and the fallback
# --------------------------------------------------------------------------- #
@pytest.fixture
def agent_cfg(live, monkeypatch):
    bars, _, d, preds, _ = live
    monkeypatch.setattr(config, "TICKERS", preds["ticker"].tolist())
    for k, v in dict(V3_K=30, V3_MAX_PER_SECTOR=5, V3_MAX_WEIGHT=0.06, V3_TARGET_EXPOSURE=TARGET,
                     OPT_AUM=1e6, OPT_IC=0.07, OPT_TARGET_EXPOSURE=TARGET, OPT_MIN_WEIGHT=0.0025,
                     OPT_FUND_DIR=d, OPT_FUND_MAX_AGE_DAYS=100_000, OPT_MIN_UNIVERSE=50, OPT_MIN_NAMES=10,
                     ALPACA_API_KEY="", ALPACA_SECRET_KEY="").items():
        monkeypatch.setattr(config, k, v, raising=False)
    agent = object.__new__(bot.BDATradingAgent)
    agent.last_bars = bars
    return agent, preds


def _lo30(agent, preds):
    a2 = object.__new__(bot.BDATradingAgent)
    return {t: x for t, x in a2.calculate_target_weights(preds, None, strategy="long_only_v3").items() if x}


def test_bot_uses_the_optimized_book(agent_cfg):
    agent, preds = agent_cfg
    w = agent.calculate_target_weights(preds, is_bull=None, strategy="optimized_v3")
    held = {t: x for t, x in w.items() if x}
    assert agent.last_construction["construction"] == "optimized_v3"
    assert agent.last_construction["n_names"] == len(held) >= 10
    assert sum(held.values()) <= TARGET + 1e-9 and min(held.values()) >= 0.0025 * TARGET - 1e-9
    assert held != _lo30(agent, preds)


@pytest.mark.parametrize("break_it", ["no_fundamentals", "no_bars", "solver_raises", "unsolved"])
def test_bot_falls_back_to_long_only_v3(agent_cfg, monkeypatch, capsys, tmp_path, break_it):
    agent, preds = agent_cfg
    if break_it == "no_fundamentals":
        monkeypatch.setattr(config, "OPT_FUND_DIR", str(tmp_path / "none"))
        monkeypatch.setattr(optimized, "_store_fund_dir", lambda: str(tmp_path / "none2"))
        monkeypatch.setattr(optimized, "SNAPSHOT_DIR", str(tmp_path / "none3"))
    elif break_it == "no_bars":
        agent.last_bars = None
    elif break_it == "solver_raises":
        def boom(*a, **k):
            raise np.linalg.LinAlgError("singular")
        monkeypatch.setattr(op, "optimize_portfolio", boom)
    else:
        real = op.optimize_portfolio
        monkeypatch.setattr(op, "optimize_portfolio",
                            lambda *a, **k: (lambda r: (r[0], {**r[1], "status": "max_iter"}))(real(*a, **k)))
    w = agent.calculate_target_weights(preds, is_bull=None, strategy="optimized_v3")
    held = {t: x for t, x in w.items() if x}
    assert held == _lo30(agent, preds)
    c = agent.last_construction
    assert c["requested"] == "optimized_v3" and c["construction"] == "long_only_v3" and c["fallback_reason"]
    out = capsys.readouterr().out
    assert "FALLING BACK to long_only_v3" in out


def test_stale_fundamentals_fall_back(agent_cfg, monkeypatch):
    agent, preds = agent_cfg
    monkeypatch.setattr(config, "OPT_FUND_MAX_AGE_DAYS", 1)
    monkeypatch.setattr(optimized, "_store_fund_dir", lambda: "/nonexistent/store")
    monkeypatch.setattr(optimized, "SNAPSHOT_DIR", "/nonexistent/snapshot")
    agent.calculate_target_weights(preds, is_bull=None, strategy="optimized_v3")
    assert agent.last_construction["construction"] == "long_only_v3"
    assert "stale" in agent.last_construction["fallback_reason"]


def test_optimized_v3_is_selectable_and_skips_the_regime(capsys):
    assert "optimized_v3" in bot.STRATEGIES
    agent = object.__new__(bot.BDATradingAgent)
    assert agent.check_market_regime(strategy="optimized_v3") is None


def test_fetch_live_data_keeps_long_history_but_feeds_inference_its_window(live, monkeypatch):
    from DataAnalysisPipeline2.trading_agent import market_data
    bars, _, _, preds, _ = live
    calls = []

    def fake(tickers, lookback_days=250, **k):
        calls.append(lookback_days)
        return bars.copy(), []

    monkeypatch.setattr(market_data, "fetch_daily_bars", fake)
    monkeypatch.setattr(config, "TICKERS", sorted(bars["ticker"].unique()))
    agent = bot.BDATradingAgent()
    agent.feature_version = 2
    agent.risk_history_days = 800
    df = agent.fetch_live_data()
    assert calls == [800]
    assert len(agent.last_bars) == len(bars)
    span = (pd.to_datetime(df["Date"]).max() - pd.to_datetime(df["Date"]).min()).days
    assert span <= config.LIVE_HISTORY_DAYS and len(df) < len(bars)
    # default strategies: one fetch of the usual window, nothing trimmed
    calls.clear()
    agent2 = bot.BDATradingAgent()
    agent2.feature_version = 2
    assert len(agent2.fetch_live_data()) == len(bars) and calls == [config.LIVE_HISTORY_DAYS]


def test_duplicate_share_classes_keeps_most_liquid(tmp_path):
    import numpy as np
    import pandas as pd
    from DataAnalysisPipeline2.trading_agent.optimized import duplicate_share_classes
    pd.DataFrame({"cik": [1, 1, 2, 3, 3], "ticker": ["GOOG", "GOOGL", "AAPL", "FOX", "FOXA"],
                  "valid_to": [pd.NaT] * 5}).to_parquet(tmp_path / "cik_map.parquet")
    t = np.array(["GOOG", "GOOGL", "AAPL", "FOX", "FOXA", "XYZ"])
    adv = np.array([2e9, 3e9, 9e9, 1e8, 2e8, 1e7])
    keep = np.array([True, True, True, False, True, True])      # FOX not a candidate
    assert duplicate_share_classes(t, adv, keep, str(tmp_path)) == {"GOOG"}
    assert duplicate_share_classes(t, adv, keep, str(tmp_path / "missing")) == set()

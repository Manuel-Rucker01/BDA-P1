"""Round-13 lev125 on the optimized_v3 + altall path: config clamp, the account
check, the levered book (mirrors v3/round13.run_book), every fallback at 1.0x
and order sizing at gross > 1. Synthetic data, fake broker, no network."""
import importlib

import numpy as np
import pandas as pd
import pytest

from DataAnalysisPipeline2.trading_agent import bot, config, market_data, operational, optimized
from DataAnalysisPipeline2.v3.tests.live_synthetic import make_live_bars_and_fund

TARGET = 0.98
L = 1.25


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("v,exp", [(1.25, 1.25), (1.0, 1.0), (0.5, 1.0), (3, 1.5), (1.5, 1.5),
                                   ("1.3", 1.3), ("abc", 1.0), (None, 1.0), (float("nan"), 1.0),
                                   (float("inf"), 1.0)])
def test_clamp_leverage(v, exp):
    assert config.clamp_leverage(v) == exp


@pytest.mark.parametrize("env,exp", [(None, 1.25), ("", 1.25), ("1", 1.0), ("2", 1.5), ("0.2", 1.0),
                                     ("junk", 1.0), ("1.1", 1.1)])
def test_leverage_env(monkeypatch, env, exp):
    if env is None:
        monkeypatch.delenv("BDA_LEVERAGE", raising=False)
    else:
        monkeypatch.setenv("BDA_LEVERAGE", env)
    try:
        assert importlib.reload(config).LEVERAGE == exp
    finally:
        monkeypatch.delenv("BDA_LEVERAGE", raising=False)
        importlib.reload(config)


# --------------------------------------------------------------------------- #
# account check
# --------------------------------------------------------------------------- #
def _acct(**kw):
    a = dict(multiplier=4, equity=100_000.0, buying_power=300_000.0, regt_buying_power=100_000.0,
             long_market_value=98_000.0, target_exposure=TARGET)
    a.update(kw)
    return a


def test_resolve_leverage_ok():
    lev, info = bot.resolve_leverage(L, **_acct())
    assert lev == L and info["applied"] == L and info["reason"] == "ok"
    assert info["required_notional"] == pytest.approx(L * TARGET * 100_000 - 98_000)
    assert info["buying_power_used"] == 100_000.0           # overnight (Reg T) buying power, not 4x


@pytest.mark.parametrize("kw,why", [
    (dict(multiplier=1), "not margin-enabled"),
    (dict(multiplier="1"), "not margin-enabled"),
    (dict(regt_buying_power=20_000.0, long_market_value=0.0), "insufficient buying power"),
    (dict(buying_power=float("nan")), "account check failed"),
    (dict(equity=0.0), "account check failed"),
    (dict(multiplier=None), "account check failed"),
])
def test_resolve_leverage_falls_back(kw, why):
    lev, info = bot.resolve_leverage(L, **_acct(**kw))
    assert lev == 1.0 and info["applied"] == 1.0 and why in info["reason"]


def test_resolve_leverage_cushion_edge():
    need = L * TARGET * 100_000 - 50_000
    ok, _ = bot.resolve_leverage(L, **_acct(long_market_value=50_000.0, regt_buying_power=need * 1.05 + 1))
    no, info = bot.resolve_leverage(L, **_acct(long_market_value=50_000.0, regt_buying_power=need * 1.04))
    assert ok == L and no == 1.0 and "insufficient" in info["reason"]


def test_resolve_leverage_clamps_and_unlevered():
    assert bot.resolve_leverage(9.0, **_acct(regt_buying_power=1e6))[0] == 1.5
    lev, info = bot.resolve_leverage(1.0, **_acct())
    assert lev == 1.0 and "1.0 requested" in info["reason"]


# --------------------------------------------------------------------------- #
# the levered book
# --------------------------------------------------------------------------- #
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


def test_levered_book_is_the_sleeve_at_aum_x_L_times_L(live):
    """round13.run_book('lev125'): optimizer at AUM x L (caps on the sleeve),
    then x L. Identical to solving at AUM x L unlevered and scaling."""
    wl, dl = _book(live, leverage=L, aum=2e8)       # large AUM: the ADV cap binds on some names
    wu, _ = _book(live, leverage=1.0, aum=2e8 * L)
    assert set(wl) == set(wu)
    for t in wu:
        assert wl[t] == pytest.approx(wu[t] * L, rel=1e-12, abs=1e-15)
    assert dl["leverage"] == L and dl["aum_opt"] == 2e8 * L and dl["aum"] == 2e8
    assert sum(wl.values()) == pytest.approx(L * TARGET, abs=4e-3)
    assert dl["long"] == pytest.approx(sum(wl.values())) and dl["cash"] < 0
    inp, _ = optimized.build_inputs(live[0], live[3].set_index("ticker")["pred_rank"], live[2])
    v = np.array([wl.get(t, 0.0) for t in inp["tickers"]])
    assert v.max() <= 0.05 * L + 1e-12                               # 5% name cap on the sleeve
    assert np.all(v <= 0.10 * inp["adv"] / 2e8 + 1e-12)               # <= 10% of ADV after scaling


def test_leverage_one_is_the_unchanged_book(live):
    a, _ = _book(live)
    b, d = _book(live, leverage=1.0)
    assert a == b and d["leverage"] == 1.0
    assert sum(a.values()) <= TARGET + 1e-9


@pytest.mark.parametrize("bad", [0.9, 1.6])
def test_levered_book_rejects_out_of_range(live, bad):
    with pytest.raises(optimized.OptimizerUnavailable, match="leverage"):
        _book(live, leverage=bad)


# --------------------------------------------------------------------------- #
# bot wiring
# --------------------------------------------------------------------------- #
class _Account:
    def __init__(self, **kw):
        self.multiplier = kw.get("multiplier", "4")
        self.equity = str(kw.get("equity", 100_000.0))
        self.portfolio_value = self.equity
        self.buying_power = str(kw.get("buying_power", 300_000.0))
        self.regt_buying_power = str(kw.get("regt_buying_power", 200_000.0))
        self.long_market_value = str(kw.get("long_market_value", 0.0))


class _Pos:
    def __init__(self, symbol, mv):
        self.symbol, self.market_value = symbol, str(mv)


def _fake_tc(positions=(), **acct):
    calls = {"orders": [], "closed": []}

    class Fake:
        def __init__(self, *a, **k):
            pass

        def get_account(self):
            return _Account(**acct)

        def get_all_positions(self):
            return [_Pos(s, v) for s, v in positions]

        def submit_order(self, order):
            calls["orders"].append((order.symbol, float(order.qty), str(getattr(order.side, "value", order.side))))
            return type("O", (), {"id": f"o{len(calls['orders'])}", "status": "filled"})()

        def close_position(self, sym):
            calls["closed"].append(sym)
            return type("O", (), {"id": f"c{sym}", "status": "filled"})()

        def get_order_by_id(self, oid):
            return type("O", (), {"id": oid, "status": "filled"})()

    return Fake, calls


@pytest.fixture
def agent_cfg(live, monkeypatch):
    bars, _, d, preds, _ = live
    monkeypatch.setattr(config, "TICKERS", preds["ticker"].tolist())
    for k, v in dict(V3_K=30, V3_MAX_PER_SECTOR=5, V3_MAX_WEIGHT=0.06, V3_TARGET_EXPOSURE=TARGET,
                     OPT_AUM=1e6, OPT_IC=0.07, OPT_IC_ALTALL=0.07, OPT_TARGET_EXPOSURE=TARGET,
                     OPT_MIN_WEIGHT=0.0025, OPT_FUND_DIR=d, OPT_FUND_MAX_AGE_DAYS=100_000,
                     OPT_MIN_UNIVERSE=50, OPT_MIN_NAMES=10, EARN_BLACKOUT=False, LEVERAGE=L,
                     ALPACA_API_KEY="PKTEST", ALPACA_SECRET_KEY="secret").items():
        monkeypatch.setattr(config, k, v, raising=False)
    agent = object.__new__(bot.BDATradingAgent)
    agent.last_bars = bars
    agent.signal = "altall"
    return agent, preds


def _run(agent, preds):
    w = agent.calculate_target_weights(preds, is_bull=None, strategy="optimized_v3")
    return {t: x for t, x in w.items() if x}


def test_bot_levers_the_altall_optimized_book(agent_cfg, monkeypatch, capsys):
    agent, preds = agent_cfg
    fake, _ = _fake_tc()
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    held = _run(agent, preds)
    c = agent.last_construction
    assert c["construction"] == "optimized_v3" and c["leverage"] == L
    assert c["leverage_check"]["applied"] == L and c["leverage_check"]["multiplier"] == 4
    assert sum(held.values()) == pytest.approx(L * TARGET, abs=4e-3)
    assert "leverage 1.25x (lev125)" in capsys.readouterr().out


def _assert_unlevered(agent, held, why):
    c = agent.last_construction
    assert c["construction"] == "optimized_v3" and c["leverage"] == 1.0
    assert c["leverage_check"]["applied"] == 1.0 and why in c["leverage_check"]["reason"]
    assert sum(held.values()) <= TARGET + 1e-9


@pytest.mark.parametrize("case,why", [
    ("cash_account", "not margin-enabled"),
    ("no_buying_power", "insufficient buying power"),
    ("account_error", "account check failed"),
    ("no_credentials", "no Alpaca credentials"),
    ("production_signal", "not altall"),
    ("disabled", "BDA_LEVERAGE=1"),
])
def test_bot_trades_unlevered_when_unsafe(agent_cfg, monkeypatch, capsys, case, why):
    agent, preds = agent_cfg
    acct = {"cash_account": dict(multiplier="1"),
            "no_buying_power": dict(regt_buying_power=10_000.0)}.get(case, {})
    fake, _ = _fake_tc(**acct)
    if case == "account_error":
        fake.get_account = lambda self: 1 / 0
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    if case == "no_credentials":
        monkeypatch.setattr(config, "ALPACA_API_KEY", "")
    if case == "production_signal":
        agent.signal = "production"
    if case == "disabled":
        monkeypatch.setattr(config, "LEVERAGE", 1.0)
    held = _run(agent, preds)
    _assert_unlevered(agent, held, why)
    if case not in ("production_signal", "disabled"):
        assert "trading UNLEVERED at 1.0x" in capsys.readouterr().out


def test_levered_build_failure_rebuilds_at_one(agent_cfg, monkeypatch, capsys):
    agent, preds = agent_cfg
    fake, _ = _fake_tc()
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    real = optimized.optimized_book

    def picky(*a, leverage=1.0, **k):
        if leverage > 1:
            raise optimized.OptimizerUnavailable("levered book infeasible")
        return real(*a, leverage=leverage, **k)
    monkeypatch.setattr(optimized, "optimized_book", picky)
    held = _run(agent, preds)
    _assert_unlevered(agent, held, "rebuilt at 1.0x")
    assert "Rebuilding UNLEVERED" in capsys.readouterr().out


def test_long_only_v3_fallback_is_unlevered(agent_cfg, monkeypatch):
    agent, preds = agent_cfg
    fake, _ = _fake_tc()
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)

    def boom(*a, **k):
        raise optimized.OptimizerUnavailable("no book")
    monkeypatch.setattr(optimized, "optimized_book", boom)
    held = _run(agent, preds)
    assert agent.last_construction["construction"] == "long_only_v3"
    assert sum(held.values()) <= TARGET + 1e-9


def test_levered_sleeve_w_prev_and_blackout_caps(agent_cfg, monkeypatch):
    """With L > 1 the optimizer's w_prev and the blackout caps get the SLEEVE
    weights (current / their sum), as round13.run_book."""
    agent, preds = agent_cfg
    fake, _ = _fake_tc(positions=[("T001", 30_000.0), ("T002", 30_000.0)], long_market_value=60_000.0)
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    monkeypatch.setattr(config, "EARN_BLACKOUT", True)
    seen = {}

    def spy_caps(tickers, decision_date, w_cur, te, *a, **k):
        seen["caps_w"] = dict(w_cur)
        return {}, {"applied": True, "n_blackout": 0, "n_blackout_held": 0, "blackout_held_w": 0.0,
                    "window": ["x", "y"], "names": []}
    from DataAnalysisPipeline2.trading_agent import earnings
    monkeypatch.setattr(earnings, "live_caps", spy_caps)
    real = optimized.optimized_book

    def spy_book(*a, **k):
        seen["w_prev"], seen["leverage"] = dict(k["w_prev"]), k["leverage"]
        return real(*a, **k)
    monkeypatch.setattr(optimized, "optimized_book", spy_book)
    _run(agent, preds)
    assert seen["leverage"] == L
    assert seen["w_prev"] == pytest.approx({"T001": 0.5, "T002": 0.5})
    assert seen["caps_w"] == pytest.approx({"T001": 0.5, "T002": 0.5})


# --------------------------------------------------------------------------- #
# execution at gross 1.25
# --------------------------------------------------------------------------- #
@pytest.fixture
def exec_env(tmp_path, monkeypatch):
    old = operational.AGENT_LOG_DIR
    operational.set_log_dir(str(tmp_path / "logs"))
    monkeypatch.setattr(config, "ALPACA_API_KEY", "PKTEST", raising=False)
    monkeypatch.setattr(config, "ALPACA_SECRET_KEY", "secret", raising=False)
    monkeypatch.setattr(config, "DRAWDOWN_LIMIT_PCT", 15.0, raising=False)
    monkeypatch.setattr(config, "MIN_ORDER_VALUE", 5.0, raising=False)
    monkeypatch.setattr(config, "MIN_TRADE_PCT", 0.0025, raising=False)
    monkeypatch.setattr(bot.time, "sleep", lambda s: None)
    monkeypatch.setattr(market_data, "fetch_daily_bars",
                        lambda *a, **k: (pd.DataFrame(columns=market_data.OUTPUT_COLUMNS), []))
    yield
    operational.set_log_dir(old)


def _levered_targets(n=25):
    return {f"S{i:02d}": L * TARGET / n for i in range(n)}       # 4.9% each, gross 1.225


def _exec(monkeypatch, targets, positions=(), **acct):
    fake, calls = _fake_tc(positions=positions, **acct)
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    monkeypatch.setattr(config, "TICKERS", sorted(targets), raising=False)
    px = pd.DataFrame([{"ticker": t, "Date": "2026-11-04", "company_close": 10.0} for t in targets])
    res = bot.BDATradingAgent().execute_alpaca_rebalance(targets, prices_df=px, dry_run=False)
    return res, calls


def test_orders_are_sized_for_gross_125_on_a_margin_account(exec_env, monkeypatch):
    targets = _levered_targets()
    held = [(t, 40_000.0 / 25) for t in targets]                    # 40% invested now
    res, calls = _exec(monkeypatch, targets, positions=held)
    assert res["status"] == "executed" and res["failed"] == 0
    buys = {s: q for s, q, side in calls["orders"] if side == "buy"}
    assert len(buys) == 25
    # every name to 4.9% of $100k equity: ($4,900 - $1,600) / $10 = 330 shares
    assert all(q == 330.0 for q in buys.values())
    assert sum(q * 10 for q in buys.values()) + 40_000 == pytest.approx(L * TARGET * 100_000)


@pytest.mark.parametrize("mult", ["1", None])
def test_orders_are_scaled_to_one_x_without_margin(exec_env, monkeypatch, capsys, mult):
    res, calls = _exec(monkeypatch, _levered_targets(), multiplier=mult)
    total = sum(q * 10 for s, q, side in calls["orders"] if side == "buy")
    assert total <= 100_000 and total == pytest.approx(100_000, abs=25 * 10)
    assert "scaling the long book to 1.0x" in capsys.readouterr().out


def test_guard_long_exposure_keeps_unlevered_books():
    w = {"A": 0.5, "B": 0.48, "C": -0.2}
    assert bot.guard_long_exposure(w, None) is w
    lev = {"A": 0.7, "B": 0.55, "C": -0.1}
    assert bot.guard_long_exposure(lev, 4) is lev
    s = bot.guard_long_exposure(lev, 1)
    assert sum(v for v in s.values() if v > 0) == pytest.approx(1.0) and s["C"] == -0.1

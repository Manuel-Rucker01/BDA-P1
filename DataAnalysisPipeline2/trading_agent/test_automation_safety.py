"""Safety guarantees the unattended (scheduled / cloud) rebalance relies on.

No network: the Alpaca trading and data clients are replaced by fakes.
"""
import datetime as dt
import os

import pandas as pd
import pytest

from DataAnalysisPipeline2.trading_agent import bot, config, market_data, operational


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _Acct:
    def __init__(self, value):
        self.portfolio_value = str(value)


class _Pos:
    def __init__(self, symbol, mv):
        self.symbol, self.market_value = symbol, str(mv)


def _make_fake_client(account_value=100_000.0, positions=(), fail_close=(), fail_submit=()):
    log = {"closed": [], "orders": []}

    class FakeTradingClient:
        def __init__(self, *a, **k):
            pass

        def get_account(self):
            return _Acct(account_value)

        def get_all_positions(self):
            return [_Pos(s, v) for s, v in positions]

        def close_position(self, symbol):
            if symbol in fail_close:
                raise RuntimeError("asset not tradable")
            log["closed"].append(symbol)

        def submit_order(self, order):
            if order.symbol in fail_submit:
                raise RuntimeError("order rejected")
            log["orders"].append((order.symbol, float(order.qty), str(order.side)))

    return FakeTradingClient, log


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Per-test agent state dir + fake credentials; restore globals after."""
    old_dir = operational.AGENT_LOG_DIR
    operational.set_log_dir(str(tmp_path / "logs"))
    monkeypatch.setattr(config, "ALPACA_API_KEY", "PKTEST", raising=False)
    monkeypatch.setattr(config, "ALPACA_SECRET_KEY", "secret", raising=False)
    monkeypatch.setattr(config, "DRAWDOWN_LIMIT_PCT", 15.0, raising=False)
    yield tmp_path
    operational.set_log_dir(old_dir)


def _prices(rows):
    return pd.DataFrame([{"ticker": t, "Date": "2026-11-04", "company_close": p} for t, p in rows],
                        columns=["ticker", "Date", "company_close"])


# --------------------------------------------------------------------------- #
# execute_alpaca_rebalance
# --------------------------------------------------------------------------- #
def test_single_failed_order_does_not_abort_rebalance(isolated, monkeypatch):
    fake, log = _make_fake_client(
        positions=[("OLD", 5_000.0), ("ECHO", 3_000.0)],  # ECHO: renamed, off-universe
        fail_close={"ECHO"}, fail_submit={"BAD"},
    )
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    # No price anywhere for NOPRICE (also not from the data API).
    monkeypatch.setattr(market_data, "fetch_daily_bars",
                        lambda *a, **k: (pd.DataFrame(columns=market_data.OUTPUT_COLUMNS), []))
    monkeypatch.setattr(config, "TICKERS", ["AAA", "BAD", "NOPRICE"], raising=False)

    agent = bot.BDATradingAgent()
    res = agent.execute_alpaca_rebalance(
        {"AAA": 0.30, "BAD": 0.20, "NOPRICE": 0.10},
        prices_df=_prices([("AAA", 100.0), ("BAD", 50.0)]),
        dry_run=False,
    )

    assert res["status"] == "executed"
    assert log["closed"] == ["OLD"]                       # liquidated despite ECHO failing
    assert ("AAA", 300.0, "OrderSide.BUY") in log["orders"] or \
           ("AAA", 300.0, "buy") in log["orders"]          # buys still ran after failures
    failed = {f["ticker"] for f in res["failures"]}
    assert failed == {"ECHO", "BAD", "NOPRICE"}
    assert res["submitted"] == 2 and res["failed"] == 3


def test_never_sizes_an_order_with_a_fake_price(isolated, monkeypatch):
    fake, log = _make_fake_client()
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    monkeypatch.setattr(market_data, "fetch_daily_bars",
                        lambda *a, **k: (pd.DataFrame(columns=market_data.OUTPUT_COLUMNS), []))
    monkeypatch.setattr(config, "TICKERS", ["XYZ"], raising=False)

    res = bot.BDATradingAgent().execute_alpaca_rebalance({"XYZ": 0.5}, prices_df=_prices([]),
                                                         dry_run=False)
    assert log["orders"] == []          # old code would have bought 50,000 "shares"
    assert res["failures"][0]["error"] == "no price"


def test_halt_file_blocks_all_orders(isolated, monkeypatch):
    fake, log = _make_fake_client()
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    open(operational.HALT_FILE, "w").close()
    res = bot.BDATradingAgent().execute_alpaca_rebalance({"AAA": 1.0},
                                                         prices_df=_prices([("AAA", 10.0)]),
                                                         dry_run=False)
    assert res == {"status": "halted", "reason": "halt_file"}
    assert log["orders"] == [] and log["closed"] == []


def test_drawdown_beyond_limit_halts(isolated, monkeypatch):
    agent = bot.BDATradingAgent()
    agent.circuit_breaker.update_and_check(100_000.0)       # peak
    fake, log = _make_fake_client(account_value=80_000.0)   # -20% > 15% limit
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    res = agent.execute_alpaca_rebalance({"AAA": 1.0}, prices_df=_prices([("AAA", 10.0)]),
                                         dry_run=False)
    assert res["status"] == "halted" and res["reason"] == "drawdown"
    assert log["orders"] == []


def test_dry_run_places_nothing(isolated, monkeypatch):
    fake, log = _make_fake_client()
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    res = bot.BDATradingAgent().execute_alpaca_rebalance({"AAPL": 1.0},
                                                         prices_df=_prices([("AAPL", 10.0)]),
                                                         dry_run=True)
    assert res == {"status": "dry_run"} and log["orders"] == []


# --------------------------------------------------------------------------- #
# Per-account state
# --------------------------------------------------------------------------- #
def test_account_log_dirs_are_separate():
    a, b = config.account_log_dir("weekly"), config.account_log_dir("monthly")
    assert a != b
    assert a.endswith(os.path.join("accounts", "weekly"))
    assert b.endswith(os.path.join("accounts", "monthly"))


def test_set_log_dir_moves_all_state_paths(tmp_path):
    old = operational.AGENT_LOG_DIR
    try:
        operational.set_log_dir(str(tmp_path / "acct"))
        assert operational.HALT_FILE.startswith(str(tmp_path))
        assert operational.EQUITY_HISTORY_PATH.startswith(str(tmp_path))
        assert operational.TradeLogger().path.startswith(str(tmp_path))
        assert operational.DrawdownCircuitBreaker().history_path.startswith(str(tmp_path))
    finally:
        operational.set_log_dir(old)


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
def test_alpaca_bars_schema_and_ny_date(monkeypatch):
    monkeypatch.setattr(config, "ALPACA_API_KEY", "PKTEST", raising=False)
    monkeypatch.setattr(config, "ALPACA_SECRET_KEY", "secret", raising=False)
    idx = pd.MultiIndex.from_tuples(
        [("AAA", pd.Timestamp("2026-11-04 05:00", tz="UTC")),
         ("AAA", pd.Timestamp("2026-11-03 05:00", tz="UTC"))], names=["symbol", "timestamp"])
    raw = pd.DataFrame({"open": [1.0, 1.1], "high": [2.0, 2.1], "low": [0.5, 0.6],
                        "close": [1.5, 1.6], "volume": [100.0, 110.0]}, index=idx)

    class FakeBars:
        df = raw

    class FakeDataClient:
        def __init__(self, *a, **k):
            pass

        def get_stock_bars(self, req):
            return FakeBars()

    monkeypatch.setattr("alpaca.data.historical.StockHistoricalDataClient", FakeDataClient)
    df, failures = market_data.fetch_daily_bars(["AAA", "GONE"], lookback_days=30,
                                                provider="alpaca", verbose=False)
    assert list(df.columns) == market_data.OUTPUT_COLUMNS
    assert sorted(df["Date"]) == ["2026-11-03", "2026-11-04"]   # NY calendar date, not UTC day
    assert failures == ["GONE"]


def test_partial_universe_is_refused(monkeypatch):
    tickers = [f"T{i}" for i in range(40)]
    have = pd.DataFrame([{"ticker": t, "Date": "2026-11-04", "Open": 1, "High": 1, "Low": 1,
                          "company_close": 1.0, "company_volume": 1.0} for t in tickers[:20]])
    monkeypatch.setattr(config, "TICKERS", tickers, raising=False)
    monkeypatch.setattr(market_data, "fetch_daily_bars", lambda *a, **k: (have, tickers[20:]))
    with pytest.raises(RuntimeError, match="coverage"):
        bot.BDATradingAgent().fetch_live_data()


# --------------------------------------------------------------------------- #
# Execution v3: sells -> wait for fills -> buys, shorts, min trade size
# --------------------------------------------------------------------------- #
class _Order:
    def __init__(self, oid, status="new"):
        self.id, self.status = oid, status


class _Asset:
    def __init__(self, shortable=True, etb=True):
        self.shortable, self.easy_to_borrow = shortable, etb


def _make_exec_client(account_value=100_000.0, positions=(), fill_after=2, never_fill=(),
                      assets=None):
    """Fake broker that records an ordered event log. Every order reports 'new'
    for ``fill_after`` polls and then 'filled' (or stays 'new' forever for
    symbols in ``never_fill``)."""
    ev = {"events": [], "asset_calls": []}
    state = {"n": 0, "polls": {}, "sym": {}}
    assets = assets or {}

    class FakeTradingClient:
        def __init__(self, *a, **k):
            pass

        def get_account(self):
            return _Acct(account_value)

        def get_all_positions(self):
            return [_Pos(s, v) for s, v in positions]

        def _new(self, sym):
            state["n"] += 1
            oid = f"oid{state['n']}"
            state["polls"][oid] = 0
            state["sym"][oid] = sym
            return _Order(oid)

        def close_position(self, symbol):
            ev["events"].append(("close", symbol))
            return self._new(symbol)

        def submit_order(self, order):
            side = getattr(order.side, "value", order.side)
            ev["events"].append(("submit", order.symbol, str(side), float(order.qty)))
            return self._new(order.symbol)

        def get_order_by_id(self, oid):
            state["polls"][oid] += 1
            sym = state["sym"][oid]
            ev["events"].append(("poll", sym))
            done = sym not in never_fill and state["polls"][oid] > fill_after
            return _Order(oid, "filled" if done else "new")

        def get_asset(self, sym):
            ev["asset_calls"].append(sym)
            return assets.get(sym, _Asset())

    return FakeTradingClient, ev


@pytest.fixture
def exec_env(isolated, monkeypatch):
    sleeps = []
    monkeypatch.setattr(bot.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(market_data, "fetch_daily_bars",
                        lambda *a, **k: (pd.DataFrame(columns=market_data.OUTPUT_COLUMNS), []))
    monkeypatch.setattr(config, "MIN_ORDER_VALUE", 5.0, raising=False)
    monkeypatch.setattr(config, "MIN_TRADE_PCT", 0.0025, raising=False)
    monkeypatch.setattr(config, "FILL_WAIT_TIMEOUT_S", 60.0, raising=False)
    monkeypatch.setattr(config, "FILL_POLL_INTERVAL_S", 2.0, raising=False)
    return sleeps


def _submits(ev):
    return [e for e in ev["events"] if e[0] in ("submit", "close")]


def test_buys_wait_for_sell_fills(exec_env, monkeypatch):
    fake, ev = _make_exec_client(positions=[("OLD", 20_000.0), ("GONE", 5_000.0),
                                            ("SHRT", -10_000.0)], fill_after=3)
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    monkeypatch.setattr(config, "TICKERS", ["OLD", "NEW", "SHRT"], raising=False)
    res = bot.BDATradingAgent().execute_alpaca_rebalance(
        {"OLD": 0.10, "NEW": 0.30, "SHRT": -0.05},
        prices_df=_prices([("OLD", 100.0), ("NEW", 50.0), ("SHRT", 10.0)]), dry_run=False)

    events = ev["events"]
    buy_idx = events.index(("submit", "NEW", "buy", 600.0))
    # every exposure-reducing order was submitted before the buy ...
    reducing = [i for i, e in enumerate(events)
                if e in (("submit", "OLD", "sell", 100.0), ("close", "GONE"),
                         ("submit", "SHRT", "buy", 500.0))]
    assert len(reducing) == 3 and max(reducing) < buy_idx
    # ... and the buy only went out after each of them had been polled to 'filled'
    for sym in ("OLD", "GONE", "SHRT"):
        polls = [i for i, e in enumerate(events) if e == ("poll", sym)]
        assert len(polls) == 4 and max(polls) < buy_idx
    assert exec_env and all(s == 2.0 for s in exec_env)
    assert res["status"] == "executed" and res["unfilled"] == [] and res["failed"] == 0


def test_unfilled_sells_are_reported_after_timeout(exec_env, monkeypatch):
    monkeypatch.setattr(config, "FILL_WAIT_TIMEOUT_S", 10.0, raising=False)
    fake, ev = _make_exec_client(positions=[("OLD", 20_000.0)], never_fill={"OLD"})
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    monkeypatch.setattr(config, "TICKERS", ["OLD", "NEW"], raising=False)
    res = bot.BDATradingAgent().execute_alpaca_rebalance(
        {"NEW": 0.10}, prices_df=_prices([("NEW", 50.0)]), dry_run=False)
    assert ev["events"].count(("poll", "OLD")) == 6        # 10s / 2s + 1, then give up
    assert len(exec_env) == 5
    assert [u["ticker"] for u in res["unfilled"]] == ["OLD"]
    assert "timeout" in res["unfilled"][0]["status"]
    assert ("submit", "NEW", "buy", 200.0) in ev["events"]   # still proceeds, reported


def test_never_shorts_non_shortable_or_hard_to_borrow(exec_env, monkeypatch):
    fake, ev = _make_exec_client(assets={"HTB": _Asset(etb=False),
                                         "NOSHORT": _Asset(shortable=False)})
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    monkeypatch.setattr(config, "TICKERS", ["OK", "HTB", "NOSHORT"], raising=False)
    res = bot.BDATradingAgent().execute_alpaca_rebalance(
        {"OK": -0.05, "HTB": -0.05, "NOSHORT": -0.05},
        prices_df=_prices([("OK", 10.0), ("HTB", 10.0), ("NOSHORT", 10.0)]), dry_run=False)
    assert _submits(ev) == [("submit", "OK", "sell", 500.0)]
    reasons = {f["ticker"]: f["error"] for f in res["failures"]}
    assert reasons == {"HTB": "short refused: not easy-to-borrow",
                       "NOSHORT": "short refused: not shortable"}
    assert sorted(ev["asset_calls"]) == ["HTB", "NOSHORT", "OK"]


def test_borrow_lookup_failure_refuses_short(exec_env, monkeypatch):
    fake, ev = _make_exec_client()

    def broken(self, sym):
        raise RuntimeError("asset not found")

    monkeypatch.setattr(fake, "get_asset", broken)
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    monkeypatch.setattr(config, "TICKERS", ["X"], raising=False)
    res = bot.BDATradingAgent().execute_alpaca_rebalance(
        {"X": -0.05}, prices_df=_prices([("X", 10.0)]), dry_run=False)
    assert _submits(ev) == []
    assert res["failures"][0]["error"].startswith("short refused: asset lookup failed")


def test_cover_short_to_zero_and_partial_cover(exec_env, monkeypatch):
    fake, ev = _make_exec_client(positions=[("S1", -5_000.0), ("S2", -5_000.0)], fill_after=0)
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    monkeypatch.setattr(config, "TICKERS", ["S1", "S2"], raising=False)
    res = bot.BDATradingAgent().execute_alpaca_rebalance(
        {"S1": 0.0, "S2": -0.02}, prices_df=_prices([("S1", 10.0), ("S2", 10.0)]),
        dry_run=False)
    subs = _submits(ev)
    assert ("close", "S1") in subs                       # cover to zero: close_position
    assert ("submit", "S2", "buy", 300.0) in subs        # -5000 -> -2000: buy 300 @ $10
    assert not any(e[0] == "submit" and e[1] == "S1" for e in subs)
    assert ev["asset_calls"] == []                       # covering needs no borrow check
    assert res["failed"] == 0 and res["submitted"] == 2


def test_long_to_short_flip_closes_then_shorts(exec_env, monkeypatch):
    fake, ev = _make_exec_client(positions=[("F", 3_000.0)], fill_after=1)
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    monkeypatch.setattr(config, "TICKERS", ["F"], raising=False)
    bot.BDATradingAgent().execute_alpaca_rebalance(
        {"F": -0.02}, prices_df=_prices([("F", 10.0)]), dry_run=False)
    subs = _submits(ev)
    assert subs == [("close", "F"), ("submit", "F", "sell", 200.0)]
    assert ev["events"].index(("poll", "F")) < ev["events"].index(subs[1])


def test_min_trade_size_skips_small_trades(exec_env, monkeypatch):
    # equity 100k, MIN_TRADE_PCT 0.25% -> trades under $250 are skipped
    fake, ev = _make_exec_client(positions=[("A", 10_000.0), ("C", 100.0)], fill_after=0)
    monkeypatch.setattr("alpaca.trading.client.TradingClient", fake)
    monkeypatch.setattr(config, "TICKERS", ["A", "B", "C", "D"], raising=False)
    res = bot.BDATradingAgent().execute_alpaca_rebalance(
        {"A": 0.102, "B": 0.002, "D": 0.01},
        prices_df=_prices([("A", 10.0), ("B", 10.0), ("C", 10.0), ("D", 10.0)]), dry_run=False)
    subs = _submits(ev)
    assert not any(e[1] in ("A", "B") for e in subs)     # $200 drift / $200 new: skipped
    assert ("close", "C") in subs                        # tiny leftover still liquidated
    assert ("submit", "D", "buy", 100.0) in subs         # $1,000 >= $250
    assert sorted(res["skipped_small"]) == ["A", "B"]


def test_v3_knobs_are_env_overridable(monkeypatch):
    import importlib
    monkeypatch.setenv("BDA_MIN_TRADE_PCT", "0.01")
    monkeypatch.setenv("BDA_V3_K", "45")
    monkeypatch.setenv("BDA_LS_GROSS", "1.2")
    try:
        cfg = importlib.reload(config)
        assert cfg.MIN_TRADE_PCT == 0.01 and cfg.V3_K == 45 and cfg.LS_GROSS == 1.2
    finally:
        monkeypatch.delenv("BDA_MIN_TRADE_PCT")
        monkeypatch.delenv("BDA_V3_K")
        monkeypatch.delenv("BDA_LS_GROSS")
        importlib.reload(config)
    assert config.MIN_TRADE_PCT == 0.0025 and config.V3_K == 30 and config.LS_GROSS == 1.6

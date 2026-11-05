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

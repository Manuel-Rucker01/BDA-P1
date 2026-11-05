"""Decision logic of the stateless scheduled rebalance (no network)."""
import datetime as dt

import pytest

from DataAnalysisPipeline2.trading_agent import scheduled_rebalance as sr
from DataAnalysisPipeline2.trading_agent import config

NY = sr.NY
DAY = dt.date(2026, 12, 1)  # Tuesday, first trading day of December 2026
OPEN = dt.datetime(2026, 12, 1, 9, 30, tzinfo=NY)
CLOSE = dt.datetime(2026, 12, 1, 16, 0, tzinfo=NY)
ACTIVE = {"status": "ACTIVE", "trading_blocked": False, "account_blocked": False}


def _decide(**over):
    kw = dict(account=ACTIVE, clock={"is_open": True}, trading_days=[DAY],
              bot_orders_this_period=0, drawdown_pct=2.0, limit_pct=15.0, today=DAY,
              cadence="monthly", retry_days=3, now=OPEN + dt.timedelta(hours=1),
              session_open=OPEN, session_close=CLOSE, min_after_open=30, min_before_close=30)
    kw.update(over)
    return sr.decide(**kw)


def test_trades_on_first_trading_day_of_month():
    assert _decide()["action"] == "trade"


def test_skips_when_market_closed():
    assert _decide(clock={"is_open": False})["action"] == "skip"


def test_skips_right_after_open_and_right_before_close():
    assert _decide(now=OPEN + dt.timedelta(minutes=10))["action"] == "skip"
    assert _decide(now=CLOSE - dt.timedelta(minutes=10))["action"] == "skip"


def test_retries_on_following_days_if_not_done_yet():
    days = [dt.date(2026, 12, 1), dt.date(2026, 12, 2), dt.date(2026, 12, 3)]
    assert _decide(trading_days=days, today=days[2])["action"] == "trade"


def test_not_due_after_retry_window():
    days = [dt.date(2026, 12, d) for d in (1, 2, 3, 4)]
    r = _decide(trading_days=days, today=days[3])
    assert r["action"] == "skip" and "not due" in r["reason"]


def test_idempotent_once_bot_orders_exist_this_period():
    r = _decide(bot_orders_this_period=7)
    assert r["action"] == "skip" and "already rebalanced" in r["reason"]


def test_empty_new_account_is_built_at_next_session_even_off_schedule():
    days = [dt.date(2026, 12, d) for d in (1, 2, 3, 4, 7)]
    r = _decide(trading_days=days, today=days[4], n_positions=0)
    assert r["action"] == "trade" and "initial build" in r["reason"]
    # ...but never off-hours, and not again once the bot has placed orders
    assert _decide(trading_days=days, today=days[4], n_positions=0,
                   clock={"is_open": False})["action"] == "skip"
    assert _decide(trading_days=days, today=days[4], n_positions=0,
                   bot_orders_this_period=5)["action"] == "skip"
    # a funded account still waits for its window
    assert _decide(trading_days=days, today=days[4], n_positions=10)["action"] == "skip"


def test_drawdown_breaker_halts():
    r = _decide(drawdown_pct=16.0)
    assert r["action"] == "halt"


def test_account_problems_are_errors():
    assert _decide(account={**ACTIVE, "status": "ACCOUNT_CLOSED_PENDING"})["action"] == "error"
    assert _decide(account={**ACTIVE, "trading_blocked": True})["action"] == "error"


def test_force_due_bypasses_schedule_but_not_safety():
    assert _decide(bot_orders_this_period=3, force_due=True)["action"] == "trade"
    assert _decide(clock={"is_open": False}, force_due=True)["action"] == "skip"
    assert _decide(drawdown_pct=40.0, force_due=True)["action"] == "halt"


def test_period_start():
    assert sr.period_start(dt.date(2026, 12, 17), "monthly") == dt.date(2026, 12, 1)
    assert sr.period_start(dt.date(2026, 12, 17), "weekly") == dt.date(2026, 12, 14)  # Monday


def test_drawdown_respects_operator_ack_date():
    eq = [100_000, 120_000, 90_000, 95_000]
    dates = ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"]
    assert round(sr.max_drawdown_pct(eq, 90_000), 2) == 25.0                # from 120k peak
    assert round(sr.max_drawdown_pct(eq, 90_000, "2026-11-04", dates), 2) == 5.26  # peak 95k after reset


def test_refuses_without_dedicated_keys(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_SANDBOX", raising=False)
    assert sr.main(["--account", "sandbox"]) == sr.EXIT_ERROR


def test_refuses_non_paper_endpoint(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_LIVEACCT", "AKLIVE")
    monkeypatch.setattr(config, "resolve_account", lambda name=None: {
        "api_key": "AKLIVE", "secret_key": "x", "url": "https://api.alpaca.markets"})
    assert sr.main(["--account", "liveacct"]) == sr.EXIT_ERROR

#!/usr/bin/env python3
"""Stateless scheduled rebalance for an Alpaca **paper** account.

Safe to invoke as often as you like (hourly from launchd, or a few times on the
first days of each month from a cloud cron). It only trades when ALL hold:

* the target account has its own keys and points at the PAPER endpoint
  (hard rule -- this runner refuses live trading),
* the account is ACTIVE and neither trading- nor account-blocked,
* the market is open and we are inside a safe intraday window,
* a rebalance is due for the cadence (first ``retry_days`` trading days of the
  month/week) and the bot has not already submitted orders this period,
* the account's drawdown from its peak (from Alpaca portfolio history) is within
  ``DRAWDOWN_LIMIT_PCT``.

Every input comes from Alpaca's REST API, so the runner needs no local state and
behaves the same on a laptop and in the cloud. Orders are identified via the
``client_order_id`` prefix the bot stamps on every order.

Usage::

    python -m DataAnalysisPipeline2.trading_agent.scheduled_rebalance --account monthly          # check only
    python -m DataAnalysisPipeline2.trading_agent.scheduled_rebalance --account monthly --execute

Monitoring (WS5): ``--monitor`` -- implied after an actual ``--execute`` rebalance --
refreshes ``ops/monitoring.{csv,md}`` from read-only market data and prints
``MONITOR_JSON``. Monitoring never changes the exit code unless
``--fail-on-alert`` is given and an alert fires (then exit 4 if it would
otherwise be 0).

Exit codes: 0 nothing to do or done, 1 error / not tradable, 2 halted by the
drawdown breaker, 3 executed with some failed orders, 4 monitoring alert (only
with ``--fail-on-alert``).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
REPO_ROOT = Path(__file__).resolve().parents[2]
EXIT_OK, EXIT_ERROR, EXIT_HALTED, EXIT_PARTIAL, EXIT_ALERT = 0, 1, 2, 3, 4


def _log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%SZ}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Pure decision logic (unit-tested; no network)
# --------------------------------------------------------------------------- #
def period_start(today: dt.date, cadence: str) -> dt.date:
    if cadence == "weekly":
        return today - dt.timedelta(days=today.weekday())
    return today.replace(day=1)


def max_drawdown_pct(equity_series, current_equity, ack_date=None, dates=None) -> float:
    """Drawdown of ``current_equity`` from the peak of the series (optionally only
    counting points on/after ``ack_date`` -- the operator's breaker reset)."""
    vals = []
    for i, v in enumerate(equity_series or []):
        if v is None or v <= 0:
            continue
        if ack_date and dates is not None and dates[i] < ack_date:
            continue
        vals.append(float(v))
    peak = max(vals + [float(current_equity)])
    return max(0.0, (1.0 - float(current_equity) / peak) * 100.0) if peak > 0 else 0.0


def decide(*, account: dict, clock: dict, trading_days: list, bot_orders_this_period: int,
           drawdown_pct: float, limit_pct: float, today: dt.date, cadence: str,
           retry_days: int, now: dt.datetime, session_open: dt.datetime | None,
           session_close: dt.datetime | None, min_after_open: int, min_before_close: int,
           force_due: bool = False, n_positions: int = 1) -> dict:
    """Return {"action": "trade"|"skip"|"halt"|"error", "reason": str}.

    A brand-new, empty account (no positions and no bot orders this period) is
    built at the next valid session instead of waiting for the next period."""
    if account.get("status") != "ACTIVE":
        return {"action": "error", "reason": f"account status {account.get('status')}"}
    if account.get("trading_blocked") or account.get("account_blocked"):
        return {"action": "error", "reason": "account is blocked"}
    if not clock.get("is_open"):
        return {"action": "skip", "reason": "market closed"}
    if session_open and now < session_open + dt.timedelta(minutes=min_after_open):
        return {"action": "skip", "reason": f"within {min_after_open} min of the open"}
    if session_close and now > session_close - dt.timedelta(minutes=min_before_close):
        return {"action": "skip", "reason": f"within {min_before_close} min of the close"}
    initial_build = n_positions == 0 and bot_orders_this_period == 0
    if not force_due and not initial_build:
        if today not in trading_days:
            return {"action": "skip", "reason": "today is not a trading day"}
        nth = trading_days.index(today) + 1
        if nth > retry_days:
            return {"action": "skip",
                    "reason": f"not due: trading day #{nth} of the {cadence} period "
                              f"(rebalance window is the first {retry_days})"}
        if bot_orders_this_period > 0:
            return {"action": "skip", "reason": f"already rebalanced this {cadence} period"}
    if drawdown_pct > limit_pct:
        return {"action": "halt",
                "reason": f"drawdown {drawdown_pct:.2f}% exceeds limit {limit_pct:.2f}%"}
    why = "initial build of an empty account" if initial_build and not force_due else f"due ({cadence})"
    return {"action": "trade", "reason": f"{why}, market open, drawdown {drawdown_pct:.2f}%"}


# --------------------------------------------------------------------------- #
# Alpaca I/O
# --------------------------------------------------------------------------- #
class Alpaca:
    def __init__(self, key, secret, base_url):
        import requests
        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        self.base = base_url.rstrip("/")

    def get(self, path, **params):
        r = self.s.get(f"{self.base}{path}", params=params or None, timeout=30)
        r.raise_for_status()
        return r.json()


def _bot_orders_since(api: Alpaca, since: dt.date, prefix: str) -> int:
    after = dt.datetime.combine(since, dt.time(0, 0), tzinfo=NY).isoformat()
    orders = api.get("/v2/orders", status="all", after=after, limit=500, direction="desc")
    return sum(1 for o in orders if str(o.get("client_order_id", "")).startswith(prefix))


def _notify(title: str, msg: str) -> None:
    """macOS notification (local runs) + GitHub job summary (cloud runs)."""
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"### {title}\n\n{msg}\n\n")
    if sys.platform == "darwin":
        safe = msg.replace('"', "'")[:200]
        subprocess.run(["osascript", "-e", f'display notification "{safe}" with title "{title}"'],
                       check=False, capture_output=True)


# --------------------------------------------------------------------------- #
# Monitoring post-step
# --------------------------------------------------------------------------- #
def should_monitor(args, outcome: dict) -> bool:
    """--monitor always; --execute only when a rebalance was actually run."""
    return bool(args.monitor or (args.execute and outcome.get("decision") == "trade"
                                 and "result" in outcome))


def monitor_exit_code(code: int, alerts, fail_on_alert: bool) -> int:
    return EXIT_ALERT if (fail_on_alert and alerts and code == EXIT_OK) else code


def run_monitoring(account: str, updater=None) -> dict:
    """Refresh the monitoring report (read-only market data); never raises."""
    try:
        if updater is None:
            from .monitoring import update_monitoring_report as updater
        return updater(account)
    except Exception as e:
        return {"account": account, "error": f"{type(e).__name__}: {e}", "alerts": []}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--account", default=os.getenv("BDA_ACCOUNT", "monthly"))
    p.add_argument("--cadence", choices=["monthly", "weekly"], default=os.getenv("BDA_CADENCE", "monthly"))
    p.add_argument("--execute", action="store_true", help="actually rebalance when due (else check only)")
    p.add_argument("--force-due", action="store_true", help="ignore the cadence/idempotency check")
    p.add_argument("--retry-days", type=int, default=int(os.getenv("BDA_RETRY_DAYS", "3")))
    p.add_argument("--min-after-open", type=int, default=30)
    p.add_argument("--min-before-close", type=int, default=30)
    p.add_argument("--universe", default=os.getenv("BDA_UNIVERSE", "full"))
    p.add_argument("--strategy", default=os.getenv("BDA_STRATEGY", "top_k"))
    p.add_argument("--notify", action="store_true", help="macOS notification / GitHub summary")
    p.add_argument("--monitor", action="store_true",
                   help="refresh ops/monitoring.{csv,md} (read-only; implied after an --execute rebalance)")
    p.add_argument("--fail-on-alert", action="store_true",
                   help="exit 4 when monitoring raises an alert (otherwise the exit code is unaffected)")
    args = p.parse_args(argv)

    outcome: dict = {}
    code = _rebalance(args, outcome)
    if should_monitor(args, outcome):
        summary = run_monitoring(args.account.lower())
        print("MONITOR_JSON: " + json.dumps(summary, default=str))
        for a in summary.get("alerts", []):
            _log(f"MONITOR ALERT [{a['rule']}] {a['message']}")
        if summary.get("error"):
            _log(f"monitoring failed (non-fatal): {summary['error']}")
        code = monitor_exit_code(code, summary.get("alerts"), args.fail_on_alert)
    return code


def _rebalance(args, outcome: dict) -> int:
    """The original preflight/decision/execute flow; fills ``outcome`` with the
    extra fields of the final SCHEDULER_JSON (decision, result, ...)."""
    account_name = args.account.lower()
    os.environ["ACTIVE_ACCOUNT"] = account_name  # before importing config
    from . import config  # noqa: E402
    from .bot import BOT_ORDER_PREFIX  # noqa: E402

    def finish(code, title, msg, **extra):
        outcome.update(extra)
        _log(msg)
        print("SCHEDULER_JSON: " + json.dumps({"exit": code, "account": account_name,
                                                "cadence": args.cadence, "message": msg, **extra}))
        if args.notify and code != EXIT_OK or (args.notify and extra.get("traded")):
            _notify(title, msg)
        return code

    # 1. Account must have its own keys and be PAPER (hard rule).
    key_var = f"ALPACA_API_KEY_{account_name.upper()}"
    if not os.getenv(key_var):
        return finish(EXIT_ERROR, "BDA rebalance: config error",
                      f"{key_var} is not set -- refusing to fall back to other keys.")
    acct = config.resolve_account(account_name)
    if "paper-api" not in acct["url"] or not acct["api_key"].startswith("PK"):
        return finish(EXIT_ERROR, "BDA rebalance: refused",
                      "Target is not a paper account -- this runner only trades paper.")
    api = Alpaca(acct["api_key"], acct["secret_key"], acct["url"])

    try:
        account = api.get("/v2/account")
        clock = api.get("/v2/clock")
        import pandas as pd  # Alpaca timestamps carry nanoseconds; pandas parses them
        now = pd.Timestamp(clock["timestamp"]).tz_convert(NY).to_pydatetime(warn=False)
        today = now.date()
        pstart = period_start(today, args.cadence)
        cal = api.get("/v2/calendar", start=pstart.isoformat(), end=today.isoformat())
        trading_days = [dt.date.fromisoformat(d["date"]) for d in cal]
        sess = next((d for d in cal if d["date"] == today.isoformat()), None)
        s_open = s_close = None
        if sess:
            s_open = dt.datetime.combine(today, dt.time.fromisoformat(sess["open"]), tzinfo=NY)
            s_close = dt.datetime.combine(today, dt.time.fromisoformat(sess["close"]), tzinfo=NY)
        n_bot = _bot_orders_since(api, pstart, BOT_ORDER_PREFIX)
        n_pos = len(api.get("/v2/positions"))
        hist = api.get("/v2/account/portfolio/history", period="all", timeframe="1D")
        ack = os.getenv("BDA_BREAKER_ACK_DATE")  # operator's manual breaker reset, YYYY-MM-DD
        hdates = [dt.datetime.fromtimestamp(t, NY).date().isoformat() for t in (hist.get("timestamp") or [])]
        dd = max_drawdown_pct(hist.get("equity"), float(account["equity"]), ack_date=ack, dates=hdates)
    except Exception as e:
        return finish(EXIT_ERROR, "BDA rebalance: API error", f"Alpaca API error during preflight: {e}")

    d = decide(account=account, clock=clock, trading_days=trading_days,
               bot_orders_this_period=n_bot, n_positions=n_pos, drawdown_pct=dd,
               limit_pct=float(config.DRAWDOWN_LIMIT_PCT), today=today, cadence=args.cadence,
               retry_days=args.retry_days, now=now, session_open=s_open, session_close=s_close,
               min_after_open=args.min_after_open, min_before_close=args.min_before_close,
               force_due=args.force_due)
    _log(f"account #{account.get('account_number')} equity=${float(account['equity']):,.2f} "
         f"drawdown={dd:.2f}% bot_orders_this_period={n_bot} -> {d['action']}: {d['reason']}")

    if d["action"] == "skip":
        return finish(EXIT_OK, "BDA rebalance", f"Nothing to do: {d['reason']}.", decision="skip")
    if d["action"] == "error":
        return finish(EXIT_ERROR, "BDA rebalance: account problem", d["reason"], decision="error")
    if d["action"] == "halt":
        return finish(EXIT_HALTED, "BDA rebalance HALTED",
                      f"{d['reason']}. No orders placed. Review, then set BDA_BREAKER_ACK_DATE to resume.",
                      decision="halt")
    if not args.execute:
        return finish(EXIT_OK, "BDA rebalance", f"Check only: WOULD rebalance now ({d['reason']}).",
                      decision="trade")

    # 2. Run the exact CLI pipeline live for this account.
    cmd = [sys.executable, "-m", "DataAnalysisPipeline2.trading_agent.run",
           "--universe", args.universe, "--strategy", args.strategy, "--live"]
    _log("Executing: " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env={**os.environ, "ACTIVE_ACCOUNT": account_name},
                          capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-20000:])
    sys.stderr.write(proc.stderr[-5000:])
    result = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON: "):
            result = json.loads(line[len("RESULT_JSON: "):])
    code = {0: EXIT_OK, 2: EXIT_HALTED, 3: EXIT_PARTIAL}.get(proc.returncode, EXIT_ERROR)
    msg = (f"Rebalance {result.get('status', 'failed')}: {result.get('submitted', 0)} orders, "
           f"{result.get('failed', 0)} failed (exit {proc.returncode}).")
    return finish(code, "BDA rebalance", msg, decision="trade",
                  traded=result.get("submitted", 0) > 0, result=result)


if __name__ == "__main__":
    sys.exit(main())

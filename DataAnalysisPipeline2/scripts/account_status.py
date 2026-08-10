#!/usr/bin/env python3
"""Read-only snapshot of the active Alpaca paper account.

Prints equity, today's P/L, cash, open positions and the drawdown-breaker halt
status. Places NO orders. Used by `make account`.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)

from trading_agent import config  # noqa: E402

try:
    import requests  # noqa: E402
except Exception:
    print("requests not installed; run `make install`.")
    sys.exit(1)


def main() -> None:
    acct = config.resolve_account() if hasattr(config, "resolve_account") else None
    label = acct["label"] if acct else "default"
    key, secret = config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY
    print(f"Active account : {label}  |  endpoint {config.ALPACA_URL}")
    if not (key and secret):
        print("!! No API credentials for this account — paste keys in trading_agent/.env")
        return

    h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    base = config.ALPACA_URL
    a = requests.get(f"{base}/v2/account", headers=h, timeout=20).json()
    if a.get("code") or a.get("message"):
        print("Alpaca error:", a)
        return
    eq, leq = float(a.get("equity", 0)), float(a.get("last_equity", 0))
    day_pl = eq - leq
    print(f"Equity         : ${eq:,.2f}")
    print(f"Today P/L      : ${day_pl:,.2f} ({(day_pl / leq * 100) if leq else 0:+.2f}%)")
    print(f"Cash           : ${float(a.get('cash', 0)):,.2f}")

    pos = requests.get(f"{base}/v2/positions", headers=h, timeout=20).json()
    if isinstance(pos, list) and pos:
        print(f"\nPositions ({len(pos)}):")
        print(f"  {'SYM':<6}{'MKT VALUE':>12}{'UNREAL P/L':>13}{'%':>8}")
        for p in sorted(pos, key=lambda q: -abs(float(q["market_value"]))):
            print(f"  {p['symbol']:<6}{float(p['market_value']):>12,.0f}"
                  f"{float(p['unrealized_pl']):>13,.2f}{float(p['unrealized_plpc']) * 100:>7.1f}%")
    else:
        print("\nPositions: (none)")

    halt = os.path.join(config.AGENT_DIR, "agent_logs", ".agent_halted")
    print(f"\nCircuit breaker: {'HALTED (remove .agent_halted to resume)' if os.path.exists(halt) else 'OK'}")


if __name__ == "__main__":
    main()

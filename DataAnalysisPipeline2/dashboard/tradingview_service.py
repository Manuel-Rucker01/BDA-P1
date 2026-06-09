"""
TradingView export helpers for dashboard proposals.

TradingView retail accounts do not expose a general order-placement API that a
third-party dashboard can call directly. The practical integration points are:
watchlist import, manual order tickets in a connected broker, Pine Script helper
notes, and webhook alert payloads. This module converts a model proposal into
those TradingView-compatible artifacts.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone


DISCLAIMER = (
    "TradingView does not provide a direct retail account order API for this "
    "dashboard. Use these artifacts to import a watchlist, review suggested "
    "orders manually, or configure TradingView alert webhooks with your own "
    "broker/automation endpoint."
)


def _tv_symbol(ticker: str, exchange_prefix: str = "NASDAQ") -> str:
    ticker = str(ticker).strip().upper()
    if ":" in ticker:
        return ticker
    return f"{exchange_prefix}:{ticker}"


def _action_for_side(side: str, weight: float) -> str:
    side = (side or "").upper()
    if weight < 0 or side == "SHORT":
        return "SELL_SHORT"
    return "BUY"


def build_tradingview_package(job_id: str, result: dict, exchange_prefix: str = "NASDAQ") -> dict:
    """Build watchlist/order/webhook/Pine artifacts from a dashboard proposal."""
    proposals = list((result or {}).get("proposals") or [])
    generated_at = datetime.now(timezone.utc).isoformat()
    symbols = [_tv_symbol(p["ticker"], exchange_prefix) for p in proposals]

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "symbol", "ticker", "action", "side", "target_weight",
            "target_weight_pct", "pred_rank_pct", "reference_price",
            "sector", "notes",
        ],
    )
    writer.writeheader()
    for p in proposals:
        weight = float(p.get("weight") or 0.0)
        writer.writerow({
            "symbol": _tv_symbol(p.get("ticker", ""), exchange_prefix),
            "ticker": p.get("ticker", ""),
            "action": _action_for_side(p.get("side"), weight),
            "side": p.get("side", ""),
            "target_weight": f"{weight:.8f}",
            "target_weight_pct": f"{float(p.get('weight_pct') or weight * 100):.4f}",
            "pred_rank_pct": "" if p.get("pred_rank") is None else f"{float(p['pred_rank']):.4f}",
            "reference_price": "" if p.get("price") is None else f"{float(p['price']):.4f}",
            "sector": p.get("sector", ""),
            "notes": "Review manually in TradingView / connected broker before placing orders.",
        })
    orders_csv = output.getvalue()

    webhook_alerts = []
    for p in proposals:
        weight = float(p.get("weight") or 0.0)
        payload = {
            "source": "bda_quant_dashboard",
            "job_id": job_id,
            "generated_at": generated_at,
            "symbol": _tv_symbol(p.get("ticker", ""), exchange_prefix),
            "ticker": p.get("ticker", ""),
            "action": _action_for_side(p.get("side"), weight),
            "target_weight": weight,
            "target_weight_pct": float(p.get("weight_pct") or weight * 100),
            "strategy": result.get("strategy"),
            "universe": result.get("universe"),
            "regime": result.get("regime"),
        }
        webhook_alerts.append({
            "symbol": payload["symbol"],
            "name": f"BDA {payload['action']} {payload['symbol']}",
            "message": json.dumps(payload, separators=(",", ":")),
        })
    webhook_json = json.dumps({
        "disclaimer": DISCLAIMER,
        "job_id": job_id,
        "generated_at": generated_at,
        "alerts": webhook_alerts,
    }, indent=2)

    pine_script = _build_pine_script(symbols, proposals, result, generated_at)
    watchlist = "\n".join(symbols) + ("\n" if symbols else "")

    return {
        "ok": True,
        "job_id": job_id,
        "generated_at": generated_at,
        "disclaimer": DISCLAIMER,
        "metadata": {
            "strategy": result.get("strategy"),
            "universe": result.get("universe"),
            "regime": result.get("regime"),
            "n_holdings": len(proposals),
            "gross_exposure_pct": result.get("gross_exposure_pct"),
            "exchange_prefix": exchange_prefix,
        },
        "artifacts": {
            "watchlist": watchlist,
            "orders_csv": orders_csv,
            "webhook_json": webhook_json,
            "pine_script": pine_script,
        },
        "instructions": [
            "Import the watchlist text into TradingView, or paste symbols manually.",
            "Use the order-ticket CSV as the source of target weights for manual broker orders.",
            "If you use TradingView alerts, paste the JSON messages into alert webhook bodies.",
            "Do not place live trades until you have reviewed every ticker, size, and side.",
        ],
    }


def artifact_payload(package: dict, artifact: str) -> tuple[str, str, str]:
    """Return (content, mime_type, filename) for a package artifact."""
    artifacts = package.get("artifacts", {})
    if artifact not in artifacts:
        raise KeyError(f"Unknown TradingView artifact: {artifact}")
    mime = {
        "watchlist": "text/plain",
        "orders_csv": "text/csv",
        "webhook_json": "application/json",
        "pine_script": "text/plain",
    }.get(artifact, "text/plain")
    ext = {
        "watchlist": "txt",
        "orders_csv": "csv",
        "webhook_json": "json",
        "pine_script": "pine",
    }.get(artifact, "txt")
    return artifacts[artifact], mime, f"tradingview_{artifact}.{ext}"


def _build_pine_script(symbols: list[str], proposals: list[dict], result: dict, generated_at: str) -> str:
    rows = []
    for p in proposals[:20]:
        rows.append(
            f"{p.get('ticker')} {p.get('side')} "
            f"{float(p.get('weight_pct') or 0):.2f}% "
            f"rank={'' if p.get('pred_rank') is None else round(float(p['pred_rank']), 1)}"
        )
    escaped_rows = ", ".join(json.dumps(r) for r in rows)
    title = f"BDA proposal {result.get('strategy')} {generated_at[:10]}"
    return f"""//@version=5
indicator({json.dumps(title)}, overlay=true)
// Generated by BDA Quant Dashboard at {generated_at}
// {DISCLAIMER}

var table proposal = table.new(position.top_right, 1, {max(len(rows), 1) + 1})
if barstate.islast
    table.cell(proposal, 0, 0, "BDA TradingView Proposal", text_color=color.white, bgcolor=color.new(color.blue, 20))
    rows = array.from({escaped_rows})
    for i = 0 to array.size(rows) - 1
        table.cell(proposal, 0, i + 1, array.get(rows, i), text_color=color.white)
"""

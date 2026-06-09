# BDA Quant Dashboard

A lightweight web UI on top of the production trading agent. It lets a
non-technical user run the full pipeline, review the *proposed* portfolio,
push it to Alpaca with one click, export the proposal for TradingView, and
track holdings and performance over time.

It does **not** re-implement any trading logic. `pipeline_service.py` drives the
exact same `BDATradingAgent` methods the CLI runner (`trading_agent/run.py`)
uses, so the dashboard and the CLI always agree.

## What you can do

| Tab | What it shows |
|-----|---------------|
| **① Run & Propose** | Pick universe / strategy / top-K, click **Run Pipeline**. The server loads the model, detects the S&P regime, fetches live prices, runs the ensemble, and computes the target book. You see every proposed holding (weight, predicted rank, price, Kalman β, sector). Click **Send Orders to Alpaca** to submit *that exact book*, or **Export for TradingView** to generate watchlist/order/webhook/Pine artifacts. |
| **② Portfolio** | Live Alpaca account (equity, cash, buying power, day P/L) plus a holdings table: quantity, average cost, current price, cost basis, market value, and unrealized P/L ($ and %). |
| **③ History** | Equity curve + peak, drawdown, per-rebalance realized-vs-benchmark bars, and the recent trade blotter — all from `trading_agent/agent_logs/`. |

## Run it

```bash
# 1. From the repo root, install the dashboard extra (Flask) on top of the
#    agent's existing environment:
pip install -r DataAnalysisPipeline2/dashboard/requirements.txt

# 2. Make sure Alpaca creds are set (same .env the agent already uses):
#    DataAnalysisPipeline2/trading_agent/.env
#      ALPACA_API_KEY=...
#      ALPACA_SECRET_KEY=...
#      ALPACA_PAPER_TRADING=True      # keep True unless you really mean it

# 3. Launch (either invocation works):
python -m DataAnalysisPipeline2.dashboard.app
# or
cd DataAnalysisPipeline2/dashboard && python app.py

# 4. Open http://127.0.0.1:8000
```

The header pill shows **PAPER trading** (green) or **● LIVE trading** (red) so
you always know which account a click will hit. `Send Orders` is disabled until
Alpaca credentials are detected, and always asks for confirmation first.

## TradingView mode

TradingView retail accounts do **not** expose a general direct order-placement
API for third-party dashboards. The dashboard therefore does not pretend to send
orders to TradingView. Instead, after a proposal is ready, **Export for
TradingView** creates:

- `watchlist.txt` style symbols, defaulting to `NASDAQ:TICKER`;
- `orders.csv` with action, side, target weight, predicted rank, price, sector;
- `webhook.json` alert-message templates for users with TradingView alert
  webhooks and their own broker/automation endpoint;
- a small Pine Script helper that displays the proposal on a chart.

For a family/friend demo, the safest workflow is: run the proposal, import or
paste the watchlist into TradingView, inspect the charts, and place any trades
manually in the broker connected to TradingView. Keep this paper/small-size
until the strategy has stronger validation than the current B&H comparison.

## Notes & safety

- **Paper by default.** `ALPACA_PAPER_TRADING` defaults to `True` in the agent
  config. The dashboard surfaces the mode prominently and confirms before any
  order submission.
- **The proposal is what trades.** A run caches both the target weights and the
  price snapshot used to compute them; `Send Orders` submits exactly that, so
  there's no drift between what you reviewed and what executes.
- **TradingView export is not broker execution.** It is a review/manual/webhook
  package. You remain responsible for any manual orders placed in TradingView or
  a connected broker.
- **Full universe is slow.** Scoring ~1,890 tickers takes a few minutes (live
  price pulls + ensemble inference). The progress line shows the current stage.
  Use the *High-Alpha basket* (20 names) for a fast demo.
- The model (~46 MB) is loaded once and reused across runs.
- `DASHBOARD_PORT` env var overrides the default port (8000).

## Architecture

```
 Browser (static/ : index.html · app.js · Chart.js)
    │  fetch JSON
    ▼
 Flask (app.py)  ── thin HTTP layer, no logic
    │
    ▼
 pipeline_service.py  ── background job runner + caching + Alpaca read APIs
    │  reuses, unmodified:
    ▼
 trading_agent/bot.py  (BDATradingAgent)  +  config.py  +  agent_logs/*.csv
```

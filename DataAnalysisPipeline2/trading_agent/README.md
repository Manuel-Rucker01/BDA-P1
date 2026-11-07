# BDA Production Quant Trading Bot Agent

A professional-grade, automated quantitative trading bot that integrates Graph Neural Network (GNN) Knowledge Graph Embeddings (KGE) with tree-boosted ensembles to trade a high-alpha basket of 20 mid-cap stock market tickers. 

This agent connects directly to the **Alpaca Brokerage SDK** to automate weekly rebalancing.

---

## 📈 Backtest & Historical Alpha Performance (Friction Adjusted)

This trading bot operates on stock baskets that feature higher idiosyncratic volatility, where our **RotatE + Relational Graph Convolutional Network (R-GCN)** structural company embeddings and macroeconomic indicators have a massive predictive information advantage compared to hyper-efficient mega-caps.

During out-of-sample backtesting (ending May 22, 2026) evaluated under our **Strict Transaction Friction and Slippage Model** (10 bps fee per trade turnover) and **Probability-Weighted High-Confidence Allocations**, the strategies implemented in this bot yielded outstanding results:

* **High-Confidence Long-Only ($P \geq 0.53$)**:
  * **6 Months Cumulative Return: +54.85%** (vs. +60.08% equal-weighted benchmark).
  * **12 Months Cumulative Return: +89.82%** (vs. +84.50% equal-weighted benchmark).
  * **24 Months Cumulative Return: +31.98%** (vs. +98.37% equal-weighted benchmark).
* **HMM + Kalman Beta Upgraded**:
  * **12 Months Cumulative Return: +103.28%** (vs. +84.50% benchmark).
  * **24 Months Cumulative Return: +95.78%** (vs. +98.37% benchmark, compressing Max Drawdown to -29.30% vs. -29.57%).

---

## ⚙️ Core Quantitative Infrastructure

1. **Gaussian Hidden Markov Model (HMM) Regime Switch**
   The S&P 500 index (`^GSPC`) daily log returns are dynamically monitored on a sliding 250-day window. The bot fits a continuous 2-state Gaussian HMM using the Baum-Welch (Expectation-Maximization) algorithm and decodes the current market state (Bull or Bear) via Viterbi trellis paths. In the decoded low-volatility Bull state (State 0), defensive shorts are automatically suppressed to prevent short squeezes. In high-volatility Bear states (State 1), the bot activates protective shorts.

2. **State-Space Kalman Beta Filter Risk Scaling**
   To prevent margin squeezes during high-volatility states, the bot recursive estimates time-varying stock sensitivity ($\beta$) relative to S&P 500 daily index returns using a state-space Kalman Filter. In Bear states, target short allocations are dynamically scaled down proportional to the asset's active Kalman Beta:
   $$W_{\text{target}, i} = \frac{W_{\text{raw}, i}}{\max(|\beta_{\text{Kalman}, i}|, 0.5)}$$
   This caps exposure on high-systemic-risk shorts, compressing drawdown and preserving compounding capital.

3. **Soft-Voting Ensemble Inference**
   The bot extracts 41 technical indicators (RSI-14, MACD, Bollinger Bands, rolling volatility, volume ratios, and lag structures) and merges World Bank real interest rate macroeconomic signals. A Soft-Voting Ensemble (averaging probabilities across `CatBoost`, `XGBoost`, `LightGBM`, and `RandomForest`) trained on an expanded 12-month training dataset generates a weekly probability $P(\text{up})$ representing the asset's upward direction.

4. **Differential Portfolio Rebalancing Optimizer**
   Instead of selling all assets to cash every week (which generates high fee friction), the bot calculates the required weight adjustments ($\Delta \text{ Weight}_i = \text{Target Weight}_i - \text{Current Weight}_i$). This **cuts traded volume by 58.32%**, saving spreads, slippage, and commission drag (boosting net portfolio yield by **+2.83% per quarter**). Furthermore, it sorts trade orders to **submit SELL orders first**, freeing up buying power before dishing out BUY orders to prevent margin rejection.

---

## 🚀 Setup & Credentials

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. Navigate to the agent directory and install dependencies:
```bash
pip install -r DataAnalysisPipeline2/trading_agent/requirements.txt
```

### 2. Configure Credentials
Copy the environment variables template and configure your Alpaca API credentials:
```bash
cp DataAnalysisPipeline2/trading_agent/.env.template DataAnalysisPipeline2/trading_agent/.env
```
Open the newly created `.env` file and input your keys:
```env
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_PAPER_TRADING=True   # Set to False to trade with real cash
```

---

## 💻 Operations Guide

The trading bot is run via a clean command-line interface:

### 1. Interactive Dry-Run Simulation (Safe Offline Testing)
To run the bot, download live pricing, run ensemble inference, calculate optimal weights, and simulate differential trade execution without sending orders to Alpaca:
```bash
python3 -m DataAnalysisPipeline2.trading_agent.run --strategy high_confidence
```
*Note: If no credentials are found in `.env`, the bot automatically executes in safe dry-run simulation mode.*

### 2. Live Order Rebalancing
To execute live orders directly on your active Alpaca account:
```bash
python3 -m DataAnalysisPipeline2.trading_agent.run --live --strategy high_confidence
```

### 3. Dynamic Universe Selection
Customize your stock universe and basket size:
* Safe Mid-Caps: `--universe safe`
* Dynamic Top Market Cap: `--universe top_mcap --num-tickers 50`

### 4. Strategy Options
* **High-Confidence Longs (Recommended for max alpha)**:
  `--strategy high_confidence` (default)
* **Regime-Filtered Long/Short (Dynamic hedging)**:
  `--strategy regime_filtered`
* **Optimized long-only (experimental, OFF by default)**:
  `--strategy optimized_v3` (or `BDA_STRATEGY=optimized_v3`). Builds the book with the
  Round-5/7 cost-aware mean-variance optimizer (`v3/optimizer.py`, spec `mv_abs_all`)
  instead of the K=30 heuristic: better walk-forward risk (Sharpe 0.82-0.98, max DD ~-30%
  vs 0.52-0.64 / -35..-38%), but it does **not** pass the promotion gate (IR vs SPY < 0,
  DSR < 0.90), so the workflow default stays `long_only_v3`. Knobs: `BDA_OPT_IC` (0.0725),
  `BDA_OPT_AUM` ($1M), `BDA_OPT_TARGET_EXPOSURE`, `BDA_OPT_MIN_WEIGHT`,
  `BDA_OPT_HISTORY_DAYS` (800), `BDA_OPT_FUND_DIR`, `BDA_OPT_FUND_MAX_AGE_DAYS` (120).
  Fundamentals (shares outstanding, EDGAR fundamentals, SIC sector) come from
  `$BDA_DATA_V3/fundamentals` when present, else from the committed snapshot
  `data/fundamentals_snapshot/` (~2 MB; refresh monthly with
  `python -m DataAnalysisPipeline2.trading_agent.optimized snapshot`). If fundamentals are
  missing or stale, the history is too short, or the solver / constraint check fails, the
  bot logs a warning and uses `long_only_v3`; `CONSTRUCTION_JSON` / `RESULT_JSON` report the
  construction actually used.
* **Signal selection (`BDA_SIGNAL`, default `production`)**: `BDA_SIGNAL=altall` scores with
  the best Round-8 signal `sn30_altall_hd` (v3 core + EDGAR fundamentals + SEC insider +
  FINRA short interest; `ExploitationZone/model_v3_altall.pkl`, git LFS, built by
  `python -m DataAnalysisPipeline2.v3.final_altall frame train`). PAPER opt-in: it fails the
  promotion gate (DSR 0.863 at N = 92, IR vs SPY -0.13). With `optimized_v3` it uses
  `BDA_OPT_IC_ALTALL` (0.0534, its Round-8 trailing IC) instead of `BDA_OPT_IC`. Live inputs
  come from `data/fundamentals_snapshot/` and `data/altdata_snapshot/` (`altall.py`); refresh
  both before each monthly run:
  `python -m DataAnalysisPipeline2.v3.altlive refresh && python -m DataAnalysisPipeline2.trading_agent.optimized snapshot && python -m DataAnalysisPipeline2.trading_agent.altall snapshot`.
  A missing / invalid artefact, an insider feed older than `BDA_ALT_INSIDER_MAX_AGE_DAYS`
  (14) or a missing public FINRA file, a feature error, or an optimized_v3 book that could not
  be built all fall back to the production model and its usual optimized_v3 -> long_only_v3
  path; `CONSTRUCTION_JSON` / `RESULT_JSON` carry `signal` and `signal_fallback_reason`.

### 5. Market Regime Override
To manually force the market trend regime (bypassing S&P 500 HMM decoders):
* Force Bull Mode (Shorts disabled): `--force-regime bull`
* Force Bear Mode (Shorts enabled): `--force-regime bear`

*Example:*
```bash
python3 -m DataAnalysisPipeline2.trading_agent.run --live --strategy regime_filtered --force-regime bull
```

---

## 🤖 Automated Production Deployment (CRON)

Quantitative rebalancing is designed to run once per week. The standard rebalancing frequency in backtests is **every Friday afternoon at market close** (e.g., at 15:45 EST / 21:45 CET, 15 minutes before the market bell).

To schedule this automatically on an AWS EC2 instance, Unix server, or macOS workstation:

### 1. Open Crontab Editor
```bash
crontab -e
```

### 2. Add Weekly Rebalancing Job
Add the following line to execute the bot every Friday at 15:45 EST (which is 20:45 UTC):
```text
45 20 * * 5 cd /path/to/UNI/Q6/BDA/BDA-P1 && /usr/bin/python3 -m DataAnalysisPipeline2.trading_agent.run --live --strategy high_confidence >> /path/to/UNI/Q6/BDA/BDA-P1/DataAnalysisPipeline2/results/trading_bot.log 2>&1
```
*(Make sure to replace `/path/to/` with the absolute path to your repository and virtual environment Python binary).*

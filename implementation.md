# Operations & Implementation Manual

This guide provides clean, step-by-step instructions to clone this repository, set up the system environment, execute the end-to-end semantic data engineering pipeline, retrain the models, run out-of-sample backtests, and launch the real-time quantitative trading bot rebalancer.

---

## 🛠️ System Prerequisites

Ensure your system meets the following software requirements before starting:
1. **Python 3.10 or 3.11** (recommended version)
2. **Java 17 or higher** (Required by PySpark for data quality processing)
   * *On macOS (via Homebrew):*
     ```bash
     brew install --cask temurin@17
     export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home
     ```
3. **C/C++ Compiler** (gcc/clang, required to compile model dependencies)
4. **Alpaca Brokerage Account** (Optional; required only for active paper/live trade execution)

---

## ⚙️ Environment Setup

1. **Clone the Repository** and navigate to the project directory:
   ```bash
   git clone <repository_url> BDA-P1
   cd BDA-P1
   ```

2. **Establish a Virtual Environment** and activate it:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Core Dependencies** (includes PyTorch, PySpark, DuckDB, CatBoost, LightGBM, XGBoost, RDFLib, and YFinance):
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔄 Running the End-to-End Pipeline

Run all execution commands from the repository root.

### Step 1: Raw Data Collection (Landing Zone)
Fetch raw stock prices and currency exchange rates from external APIs and write them to CSV logs in `/datasets`:
```bash
python3 LandingZone/nasdaq.py
python3 LandingZone/company_history.py
python3 LandingZone/exchange.py
python3 LandingZone/additional_information.py
```

### Step 2: Database Ingestion (Formatted Zone)
Homogenize and import raw CSV files into DuckDB schemas using PySpark:
```bash
python3 FormattedZone/formatted_zone_pipeline.py
```

### Step 3: Data Quality & Cleaning (Trusted Zone)
Apply strict quality filters (Denial Constraints) to the relational tables, build an enriched companies directory, and export clean datasets to `TrustedZone.duckdb`:
```bash
python3 TrustedZone/dataQuality.py
```
*To verify the data quality suite via unit tests:*
```bash
python3 -m pytest TrustedZone/test_dataQuality.py -v
```

### Step 4: Semantic Graph Construction (Exploitation Zone)
Convert relational database records into Turtle RDF graphs, resolve sovereign parameters via APIs, and build the linked data layer:
```bash
# 4a. Integrate tabular master datasets
python3 ExploitationZone/data_integration.py

# 4b. Generate the corporate Financial Knowledge Graph (~2.3M triples)
python3 ExploitationZone/graph_generation.py

# 4c. Generate the Macroeconomic & Geopolitical Graph (~2K triples)
python3 ExploitationZone/geopolitical_macroeconomic.py
```
*To verify macroeconomic graph structure:*
```bash
python3 -m pytest ExploitationZone/test_macroeconomic_graph.py -v
```

---

## 🧠 Model Training & Embedding Baking

To retrain the complex **RotatE** relational graph embeddings on the structural triples of the KG, project them through PCA (dimensionality reduction to 16 principal orthogonal components), and fit the soft-voting ensemble classifiers:
```bash
python3 DataAnalysisPipeline2/scripts/kg_embeddings_classifier.py
```
*This command runs a multi-model bake-off across tabular and embedding features and bakes the final fitted base classifiers (`CatBoost`, `XGBoost`, `LightGBM`, `RandomForest`), standard scaler, and PCA projection matrices directly to `/ExploitationZone/best_model.pkl` for real-time inference.*

---

## 📈 Running Out-of-Sample Backtests (with 10 bps friction)

We have implemented a strict **10 basis points (10 bps)** transaction fee and slippage model on all rebalancing turnover, complete with weekly asset weight drift propagation.

### 1. Master Investment Horizons (6, 12, and 24 Months)
Evaluate the performance, Sharpe ratios, and max drawdowns of Buy & Hold, SMA50, and HMM + Kalman models:
```bash
PYTHONPATH=DataAnalysisPipeline2/scripts python3 DataAnalysisPipeline2/scripts/backtests/verify_hmm_kalman_horizons.py
```
*This generates a premium markdown performance report in `horizon_comparison.md`.*

### 2. Multi-Subset Generalization (5 Sector Subsets, 50 Tickers Each)
Evaluate how the strategies generalize on a 100% unseen future out-of-sample window across different liquidity profiles:
```bash
PYTHONPATH=DataAnalysisPipeline2/scripts python3 DataAnalysisPipeline2/scripts/backtests/verify_subsets_comparison.py
```
*This generates a premium markdown report in `subsets_comparison_report.md`.*

---

## 🤖 Launching the Production Trading Bot CLI

The modular production bot lives inside the `/DataAnalysisPipeline2/trading_agent` directory.

### 1. Local Credentials Configuration
Create a `.env` file in the workspace directory:
```bash
cp DataAnalysisPipeline2/trading_agent/.env.template DataAnalysisPipeline2/trading_agent/.env
```
Open `.env` in a text editor and input your credentials:
```env
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_PAPER_TRADING=True   # False to route trades to live capital
```

### 2. Run Interactive Offline Simulations (Dry Run)
Download live pricing, fetch broad-market indices, fit the Gaussian HMM to decode regimes, calculate Kalman Betas, run soft-voting inference, and execute simulated differential rebalancing without placing real brokerage trades:
```bash
python3 -m DataAnalysisPipeline2.trading_agent.run --universe high_alpha --strategy high_confidence
```
*Note: If no credentials are found in `.env`, the script automatically defaults to this simulated offline mode for safety.*

### 3. Route Real-Time Orders (Live Execution)
Place actual rebalancing orders directly on your Alpaca account:
```bash
python3 -m DataAnalysisPipeline2.trading_agent.run --universe high_alpha --strategy high_confidence --live
```
*The bot leverages the Differential Portfolio Rebalancing Optimizer, which only trades the weight adjustments, cutting transaction fee volume by over 58%. It automatically submits SELL orders first to free up buying power, preventing margin rejections.*

### 4. Dynamic Universe and Command Parameters
* **Strategy Selection**:
  * High-Confidence Long-Only: `--strategy high_confidence`
  * Regime-Filtered Hedging: `--strategy regime_filtered`
* **Target Stock Basket**:
  * Safe Mid-Caps: `--universe safe`
  * Dynamic Top Market Cap: `--universe top_mcap --num-tickers 50`
* **Force Market Regimes (Override HMM Index Decoders)**:
  * Force Bull (Shorts disabled): `--force-regime bull`
  * Force Bear (Shorts enabled): `--force-regime bear`

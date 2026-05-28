# Large-Scale Data Engineering and Algorithmic Trading with Knowledge Graphs (P1 + P2)

**Authors:** Yufeng Chen, Marc Delgado, Manuel Rucker  
**Institution:** Universitat Politècnica de Catalunya (UPC)  
**Date:** 23rd May 2026  

---

## 📈 Executive Project Summary

This project implements a fully integrated, semantic quantitative trading and investment infrastructure across two distinct phases. 

* **Phase 1 (P1)**: Constructed a classical, high-capacity tabular data lake. Raw stock market prices and currency indicators are ingested, standardized into relational structures via PySpark/DuckDB, cleaned using rigorous data quality rules, and fed into ARIMA time series and baseline machine learning classifiers (RandomForest and MLP) to predict 7-day stock price direction.
* **Phase 2 (P2)**: Integrates an advanced **Knowledge Graph (KG)** semantic layer. Cleaned relational data is mapped into linked RDF graphs, enriched with live sovereign macroeconomic and geopolitical parameters queried from World Bank and geopolitical APIs, and analyzed through two distinct pipelines:
  1. **SPARQL Analytical Engine**: Evaluates cross-graph queries that link corporate taxonomy straight to geopolitical borders and regional risk indicators.
  2. **GNN Structural Embeddings & Ensemble ML**: Trains a **RotatE** relational graph model in PyTorch, projects structural company embeddings through PCA, and integrates them with tabular indicators inside a soft-voting classifier ensemble (`CatBoost`, `XGBoost`, `LightGBM`, `RandomForest`).
* **Operational Trading Infrastructure (Advanced Addition)**: Goes far beyond standard academic models by building a production-ready **Algorithmic Trading Bot CLI runner** that connects to the Alpaca Brokerage SDK. The bot dynamically switched regimes using a dynamic 2-state **Gaussian Hidden Markov Model (HMM)**, scales systemic exposure using state-space **Kalman Beta Filters**, and executes orders under a strict **10 basis points (10 bps)** transaction friction and slippage model using a **Differential Portfolio Rebalancing Optimizer**.

---

## 📁 Repository Architecture

```
.
├── LandingZone/                  # Ingestion scripts (APIs → Raw CSVs)
│   ├── nasdaq.py
│   ├── company_history.py
│   ├── exchange.py
│   └── additional_information.py
├── datasets/                     # Raw stock and exchange rate CSV storage
├── FormattedZone/                # Spark standardization & DuckDB ingestion
│   └── formatted_zone_pipeline.py
├── TrustedZone/                  # Data quality & Denial Constraints pipeline
│   ├── dataQuality.py                # PySpark cleaning & enriched companies table
│   └── test_dataQuality.py           # Unit tests for data quality rules
├── ExploitationZone/             # Tabular integration & RDF Graph generation
│   ├── data_integration.py           # SQL window features integration
│   ├── graph_generation.py           # Generates corporate Financial Knowledge Graph (RDF)
│   ├── geopolitical_macroeconomic.py # Generates sovereign Macroeconomic Graph (RDF)
│   ├── financial_knowledge_graph.ttl # Output Turtle Graph (~2.3M triples)
│   ├── macroeconomic_graph.ttl       # Output Turtle Graph (~2K triples)
│   ├── test_macroeconomic_graph.py   # Tests for macro graph relations
│   └── best_model.pkl                # Baked ensemble classifiers, scaler, and PCA state
├── DataAnalysisPipeline1/        # Classical ARIMA time series modeling
│   ├── scripts/arima_models.py       # Auto-ARIMA prices vs. returns forecasting
│   └── scripts/arima_results_validation.py # ARIMA validation & performance heatmap plot
├── DataAnalysisPipeline2/        # Advanced ML, SPARQL, and live trading agent
│   ├── scripts/
│   │   ├── stock_prediction_mlp.py          # Classical MLP classifier
│   │   ├── stock_prediction_random_forest.py # Classical RandomForest classifier
│   │   ├── sparql_analysis.py               # Pattern-matching SPARQL engine
│   │   └── kg_embeddings_classifier.py      # Retrains PyTorch RotatE KGE & ensemble
│   └── trading_agent/            # Modular automated quantitative trading bot
│       ├── config.py                 # Path resolutions and trading constants
│       ├── bot.py                    # Agent engine (HMM, Kalman, Alpaca rebalancing)
│       └── run.py                    # Production CLI runner
├── implementation.md             # Operations, setup, and deployment manual
├── report.tex                    # Professional 5-page LaTeX academic project report
└── requirements.txt              # Unified dependencies configuration
```

---

## 📥 Cloning the Repository (Git LFS Required)

Two large artefacts in this repository are stored via **[Git LFS](https://git-lfs.com)** rather than as ordinary Git blobs:

| File | Size | Why it's in LFS |
|---|---|---|
| `ExploitationZone/best_model.pkl` | 157 MB | Trained ensemble (CatBoost + XGB + LightGBM + RF) + fitted scaler + PCA state |
| `ExploitationZone/financial_knowledge_graph.ttl` | 98 MB | Generated RDF graph (~590k triples) |

A plain `git clone` without LFS will leave you with **pointer stubs** instead of these files, and `graph_generation.py` / the trading bot will fail at load time. Do the following once:

```bash
# 1. Install git-lfs (one-time)
brew install git-lfs                   # macOS
# sudo apt install git-lfs             # Ubuntu / Debian
# choco install git-lfs                # Windows (Chocolatey)

# 2. Register the LFS smudge/clean filters in your user config (one-time)
git lfs install

# 3. Clone the repo — LFS objects download automatically as part of the checkout
git clone git@github.com:Manuel-Rucker01/BDA-P1.git
cd BDA-P1
```

If you already cloned the repo *before* installing LFS, the two files above will be tiny pointer text files. Fix it with:

```bash
git lfs install
git lfs pull                           # download the real binary blobs
```

To verify everything is in order:

```bash
git lfs ls-files
# expected output:
#   b6bdb5b557 - ExploitationZone/best_model.pkl
#   ade219497a - ExploitationZone/financial_knowledge_graph.ttl

ls -lh ExploitationZone/best_model.pkl ExploitationZone/financial_knowledge_graph.ttl
# expected:  ~157M and ~98M (not a few KB)
```

> **Note for existing contributors:** the LFS migration on 26 May 2026 force-pushed a rewritten history to `main`. If you have a clone from before that date, your local `main` no longer matches the remote. The safe fix is `git fetch origin && git reset --hard origin/main` after backing up any local branches, or simply re-clone.

---

## 🚀 Step-by-Step Operations Pipeline

Run all execution commands from the repository root directory.

### 1. Landing Zone Ingestion
Fetch raw daily price bars and currency conversions from external APIs and write to raw CSV logs:
```bash
python3 LandingZone/nasdaq.py
python3 LandingZone/company_history.py
python3 LandingZone/exchange.py
python3 LandingZone/additional_information.py
```

### 2. Formatted Zone Standardization
Ingest CSV datasets and standardize relational schemata inside DuckDB using Spark SQL:
```bash
python3 FormattedZone/formatted_zone_pipeline.py
```

### 3. Trusted Zone Data Quality
Clean data records using PySpark Denial Constraints, build country-resolving directories, and write output datasets to `TrustedZone.duckdb`:
```bash
python3 TrustedZone/dataQuality.py
```
*To verify the data quality rules:*
```bash
python3 -m pytest TrustedZone/test_dataQuality.py -v
```

### 4. Exploitation Zone Graph Generation
Link company observations straight to sovereign indicators:
```bash
# 4a. Integrate tabular master datasets
python3 ExploitationZone/data_integration.py

# 4b. Generate the corporate Financial Knowledge Graph (~2.3M triples)
python3 ExploitationZone/graph_generation.py

# 4c. Generate the Macroeconomic & Geopolitical Graph (~2K triples)
python3 ExploitationZone/geopolitical_macroeconomic.py
```

### 5. Analytical Inferences (ARIMA & SPARQL)
Execute pattern-matching time-series and semantic queries:
```bash
# Run Pipeline 1: Auto-ARIMA models comparison
python3 DataAnalysisPipeline1/scripts/arima_models.py
python3 DataAnalysisPipeline1/scripts/arima_results_validation.py

# Run Pipeline 2a: SPARQL analytical queries (including cross-graph borders queries)
python3 DataAnalysisPipeline2/scripts/sparql_analysis.py
```

### 6. Relational Graph Embeddings and Retraining
Retrain the structural **RotatE** embeddings in PyTorch, project them through PCA (16 components), and train the tree-boosting Soft-Voting ensembles on the complete 12-month expanded training dataset:
```bash
python3 DataAnalysisPipeline2/scripts/kg_embeddings_classifier.py
```
*This command retrains on 481,000 observations and bakes the final fitted classifiers, StandardScaler, and PCA parameters straight to `/ExploitationZone/best_model.pkl`.*

### 7. Run Historical Out-of-Sample Backtests (Strict 10 bps Friction)
Evaluate our strategies over horizons and corporate liquidity profiles under dynamic weight drift and institutional transaction cost drag:
```bash
# Horizons comparison backtest (horizon_comparison.md)
PYTHONPATH=DataAnalysisPipeline2/scripts python3 DataAnalysisPipeline2/scripts/backtests/verify_hmm_kalman_horizons.py

# Thematic subsets OOS future backtest (subsets_comparison_report.md)
PYTHONPATH=DataAnalysisPipeline2/scripts python3 DataAnalysisPipeline2/scripts/backtests/verify_subsets_comparison.py
```

### 8. Deploy the Live Quantitative Rebalancer Bot
Launch the production bot CLI:
```bash
# Execute local simulated dry run
python3 -m DataAnalysisPipeline2.trading_agent.run --universe high_alpha --strategy high_confidence

# Execute live order rebalancing directly on Alpaca (requires credentials in .env)
python3 -m DataAnalysisPipeline2.trading_agent.run --universe high_alpha --strategy high_confidence --live
```
*For comprehensive instructions, setup specifications, and cron scheduling guidelines, see the [Operations & Implementation Manual](file:///Users/manuelruckerabella/Workspace/UNI/Q6/BDA/BDA-P1/implementation.md).*

---

## 📊 Summary of Friction-Adjusted Horizons Backtests

The empirical out-of-sample backtests evaluate capital performance under a strict 10 basis points transaction cost model, with the passive Buy & Hold benchmark charged entry/exit fees on entry and exit. The model is run in its intended deployment mode: every Friday it scores all ~1,890 modelled tickers, the top 5% by predicted rank are kept, capped at the top **K=10** names, equal-weighted. This matches the cross-section size used during training, which is required for the per-date cross-sectional Z preprocessing to behave consistently.

> ⚠️ **Survivorship-bias warning.** The 12- and 24-month rows below use the current NASDAQ universe, so tickers that delisted in those windows are silently absent. The 17-week Clean OOS and the 2-month Unseen Future windows are the cleanest results — both are strictly post the model's training range (which ended 2026-03-19).

### Full-universe top-K=10 backtests (the production deployment mode)

| Horizon | Strategy | Cumulative Return | Sharpe | Max DD |
| :--- | :--- | :---: | :---: | :---: |
| **Clean OOS Weekly** (17 wk, post 2026‑01‑14) | Buy & Hold | +4.88% | 0.828 | -7.17% |
| | High-Confidence Long-Only | **+94.94%** | **4.504** | -11.07% |
| **Clean OOS Monthly** (4 reb, post 2026‑01‑14) | Buy & Hold | +6.15% | 3.193 | **-2.77%** |
| | High-Confidence Long-Only | **+66.78%** | **6.736** | -4.88% |
| **Unseen Future** (2 mo, 2026‑03‑20 → 2026‑05‑15) | Buy & Hold | +22.09% | 5.198 | **-2.41%** |
| | High-Confidence Long-Only | **+47.00%** | **6.233** | -2.98% |
| 📅 6 Months | Buy & Hold | +34.54% | 2.162 | -6.18% |
| | High-Confidence Long-Only | **+188.06%** | **5.547** | -11.07% |
| 📅 12 Months ⚠️ | Buy & Hold | +41.82% | 2.308 | -6.14% |
| | High-Confidence Long-Only | **+748.14%** | **6.683** | -11.08% |
| 📅 24 Months ⚠️ | Buy & Hold | +1,431.19% | 0.815 | -19.45% |
| | High-Confidence Long-Only | +711.80% | 2.765 | -40.16% |

### Sector-conditional behaviour (2-month unseen future, 50-ticker subsets)

| Subset | Buy & Hold | High-Confidence Top-K=10 | Δ |
| :--- | :---: | :---: | :---: |
| Mega-Cap Titans | +14.67% | **+26.82%** | +12.2 pp |
| Technology Sector | +31.36% | **+77.52%** | +46.2 pp |
| Consumer Services | +1.43% | **+8.58%** | +7.2 pp |
| Healthcare Pioneers | +5.55% | -3.54% | -9.1 pp |
| Financial Giants | +9.51% | -8.63% | -18.1 pp |

*The headline result is the strictly-post-training **2-month unseen future window**: High-Confidence Longs `+47.00%` vs Buy & Hold `+22.09%`, Sharpe `6.23`, Max DD `-2.98%`. The model carries genuine cross-sectional rank signal (walk-forward CV IC `+0.1053`, 5/5 folds positive), and that signal translates into portfolio P&L when the deployment cross-section matches the training cross-section (full universe → top-K=10). Sector-conditional results are mixed: the model wins decisively on Tech and Mega-Cap names (which dominate the training distribution) but loses to Buy & Hold on Healthcare and Financials, where structural KG embeddings underspecify the dominant sector-specific dynamics. The concentrated K=10 portfolio also runs materially higher drawdown than a diversified index (`-11%` to `-40%` depending on horizon).*


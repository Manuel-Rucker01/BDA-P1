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

# Production deployment: full universe, concentrated top-K, inverse-vol sizing, news tilt
python3 -m DataAnalysisPipeline2.trading_agent.run --universe full --strategy top_k --top-pct 5 --top-k 10 --live
```

> **News-Sentiment — v1 live tilt → v2 backtested PIT feature.** *(v1)* The bot first used news as a live-only *tilt*: pull real-time company news (Alpaca News API), score each headline (finance lexicon / optional FinBERT), build an ephemeral RDF `NewsEvent` graph, aggregate per-ticker sentiment via **SPARQL**, and nudge the ranking by `score = pred_rank + λ·sentiment` (λ=0.10). *(v2 — adopted)* We then turned it into a **backtested point-in-time feature**: Alpaca returns each article's true `created_at`, so we build 5 features per (ticker, date) — 7d mean sentiment, news volume, 30d mean, 7-vs-30d momentum, 7d dispersion — using **only articles published strictly before** that date (no leak), scored with a frozen lexicon. One canonical implementation feeds training, live inference, and backtests.
>
> **Validation.** Same walk-forward CV (only +5 columns): Combined IC **+0.1053 → +0.1135** (positive on all 8 models, 4/5 folds; marginal, p≈0.19). Clean post-training OOS portfolio: news-trained top-K=10 **+57.49% vs +47.28%** for the news-free model (B&H +22.18%), **IR +4.52 vs +4.37**, at slightly higher volatility (Sharpe 6.55 vs 6.66, DD −3.39% vs −2.40%). The PIT features are **adopted into the deployed model + computed live**; the v1 tilt is disabled (`USE_NEWS_SENTIMENT=0`) to avoid double-counting. Caveats: PIT integrity assumed from Alpaca timestamps, large-cap-skewed coverage (1,518/1,890 tickers), single 2-month validation window. *(This v2 model lives on branch `feature/news-in-training-v2`; the other backtests in this README use the news-free baseline model.)*

*For comprehensive instructions, setup specifications, and cron scheduling guidelines, see the [Operations & Implementation Manual](file:///Users/manuelruckerabella/Workspace/UNI/Q6/BDA/BDA-P1/implementation.md).*

---

## 📊 Summary of Friction-Adjusted Horizons Backtests

The empirical out-of-sample backtests evaluate capital performance under a strict 10 basis points transaction cost model, with the passive Buy & Hold benchmark charged entry/exit fees on entry and exit. The model is run in its intended deployment mode: every Friday it scores all ~1,890 modelled tickers, the top 5% by predicted rank are kept, capped at the top **K=10** names, then sized by **inverse-volatility weighting** (`w_i ∝ 1/σ_i`, capped at 25% per name). This matches the cross-section size used during training, which is required for the per-date cross-sectional Z preprocessing to behave consistently.

> **Risk control — inverse-volatility sizing.** The HMM regime gate and Kalman β filter manage *systematic* risk, which dominates a diversified book; but a concentrated 10-name book is dominated by *idiosyncratic* single-name risk. The model selects *which* names to hold; the `1/σ` rule (per-name cap `0.25`) decides *how much*, so high-volatility names receive less capital. We justify it on **risk-management** grounds (the cap provably stops one name dominating the book) rather than a backtested return edge — an earlier inverse-vol-vs-equal-weight comparison rested on an 18-month window we later found overlaps training, so those deltas were withdrawn.

> ⚠️ **What counts as out-of-sample.** The deployed model was fit on feature-dates `2025-03-31 → 2026-01-14` (data ends `2026-02-13`; last 30 days trimmed for the 30-day forward target). Only windows *outside* that interval are genuinely OOS. Earlier drafts headlined 6/12/24-month horizons whose windows **overlap training** (the model scoring data it learned from); those have been **removed**. We report the two genuinely-OOS windows below. The **pre-training** window additionally carries current-membership survivorship bias + mild static-embedding look-ahead, so it is indicative, not deployable.

> **Model/backtest hardening notes.**
> - **No multi-horizon adoption:** the shipped model remains the canonical 30-day-forward ranker; multi-horizon experiments are diagnostics only until retrained, versioned, and validated without train-window overlap.
> - **Acquisition-edge leak caveat:** rolling-vintage acquisition adjustments reduce future structural leakage, but any "acquisition edge" measurement remains approximate while some corporate KG/static embedding inputs are not fully point-in-time.
> - **Macro/static KG caveats:** macro features use vintage-aware/emulated lags where possible, while generated macro and corporate KG snapshots still contain static membership/taxonomy assumptions; pre-training results should therefore be read as indicative, not deployable evidence.
> - **No `regime_filtered` deployment yet:** the long/short regime strategy stays disabled for production promotion until short exposure caps, borrow constraints, and gross/net exposure limits are explicitly enforced and tested.
> - **Canonical OOS guidance:** use the full-universe, post-training Top-K=10 inverse-vol backtest below as the cleanest deployment proxy; do not headline overlapping 6/12/24-month horizon runs as OOS.
> - **Manifest/schema validation:** retraining should export both `ExploitationZone/best_model.pkl` and `ExploitationZone/best_model_manifest.json`; consumers should verify the manifest's artifact schema version, feature column order, training date bounds, and selected recipe before treating a model as deployable.

### Genuinely out-of-sample full-universe top-K=10 backtests (production deployment mode)

| OOS Window | Strategy | Return | Sharpe | Max DD | IR vs B&H |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Pre-Training** ‡ (20 mo, 2023‑07 → 2025‑03) | Buy & Hold | +33.59% | 1.92 | -7.17% | — |
| | Top-K=10 | +15.62% | 0.70 | -13.47% | -0.60 |
| **Post-Training** (2 mo, 2026‑03‑20 → 2026‑05‑15) | Buy & Hold | +22.18% | 5.22 | -2.41% | — |
| | **Top-K=10 (cleanest)** | **+47.28%** | **6.66** | **-2.40%** | **+4.37** |

‡ pre-training window: survivorship + static-embedding caveats apply — read as indicative.

### Sector-conditional behaviour (2-month unseen future, 50-ticker subsets)

| Subset | Buy & Hold | High-Confidence Top-K=10 | Δ |
| :--- | :---: | :---: | :---: |
| Mega-Cap Titans | +14.67% | **+19.98%** | +5.3 pp |
| Technology Sector | +31.36% | **+54.37%** | +23.0 pp |
| Consumer Services | +1.43% | **+7.88%** | +6.5 pp |
| Healthcare Pioneers | +5.55% | -2.37% | -7.9 pp |
| Financial Giants | +9.51% | -6.28% | -15.8 pp |

*The headline result is the strictly-post-training **2-month window**: Top-K=10 `+47.28%` vs Buy & Hold `+22.18%`, Sharpe `6.66`, Max DD `-2.40%`, **Information Ratio `+4.37`** (excess return over benchmark is large relative to tracking error → consistent, not a lucky single bet). The model carries genuine cross-sectional rank signal (walk-forward CV IC `+0.1053`, 5/5 folds positive). Sector-conditional results are mixed: it wins on Tech/Mega-Cap (which dominate the training distribution) but loses to Buy & Hold on Healthcare/Financials. The concentration is deliberate: a **breadth experiment** widening to K=20 with a per-sector cap was **worse** (post-training IR `+3.32` vs `+4.37`) because the model's skill is sector-concentrated — forcing diversification spends capital on its negative-IC sectors. We keep K=10.*


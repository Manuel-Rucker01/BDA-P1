# Project — Large-Scale Data Engineering for AI (P1 + P2)

**Authors:** Yufeng Chen, Marc Delgado, Manuel Rucker  
**Date:** 12th December 2026

---

## Overview

This project implements an end-to-end data science pipeline for financial market analysis in two phases. **P1** built a classical tabular pipeline: raw market data is collected, homogenised into DuckDB, cleaned through data quality rules, and fed into ARIMA and ML models (MLP, Random Forest) that predict whether a stock will rise over the next 7 trading days.

**P2** extends the Exploitation Zone with a Knowledge Graph (KG) layer. Cleaned relational data from the Trusted Zone is transformed into RDF graphs using RDFLib, enriched with live macroeconomic and geopolitical data from public APIs, and consumed by two new analytical pipelines: a SPARQL pattern-matching pipeline and a KG-embedding ML pipeline based on TransE.

---

## Repository Structure

```
.
├── LandingZone/                  # Data collectors (APIs → CSV)
│   ├── nasdaq.py
│   ├── company_history.py
│   ├── exchange.py
│   └── additional_information.py
├── datasets/                     # Raw CSV storage
├── FormattedZone/                # PySpark/DuckDB ingestion
│   └── formatted_zone_pipeline.py
├── TrustedZone/                  # Data quality pipeline
│   ├── dataQuality.py                # Spark DQ + enriched `companies` table
│   └── test_dataQuality.py
├── ExploitationZone/             # KG generation & integration
│   ├── data_integration.py           # P1: tabular master dataset
│   ├── graph_generation.py           # P2: financial_knowledge_graph.ttl
│   ├── geopolitical_macroeconomic.py # P2: macroeconomic_graph.ttl
│   ├── financial_knowledge_graph.ttl # Output: company/sector/country graph
│   ├── macroeconomic_graph.ttl       # Output: country GDP/borders/tensions
│   └── test_macroeconomic_graph.py   # Tests for the macro graph
├── DataAnalysisPipeline1/        # Classical time series (ARIMA)
│   └── scripts/arima_models.py
└── DataAnalysisPipeline2/        # Advanced ML + KG pipelines
    └── scripts/
        ├── stock_prediction_mlp.py          # P1: MLP baseline
        ├── stock_prediction_random_forest.py # P1: RF baseline
        ├── sparql_analysis.py               # P2: 8 SPARQL queries
        └── kg_embeddings_classifier.py      # P2: TransE + 5-model bake-off
```

---

## Execution Pipeline

Run all scripts from the repository root.

### 1. Landing Zone

Fetch raw market data from external APIs and append to CSV storage.

```bash
python3 LandingZone/nasdaq.py
python3 LandingZone/company_history.py
python3 LandingZone/exchange.py
python3 LandingZone/additional_information.py
```

### 2. Formatted Zone

Ingest CSVs into DuckDB using PySpark.

```bash
python3 FormattedZone/formatted_zone_pipeline.py
```

### 3. Trusted Zone

Apply data quality rules and write cleaned tables to `TrustedZone.duckdb`. This stage also builds an enriched `companies` table that resolves each NASDAQ ticker's country via `sp500_companies.Country` → Forbes name-match → fallback `United States`, so the Exploitation Zone receives clean, non-null country data.

```bash
python3 TrustedZone/dataQuality.py
```

To run the unit-test suite:

```bash
python3 -m pytest TrustedZone/test_dataQuality.py -v
```

### 4. Exploitation Zone

#### 4a — Tabular master dataset (P1)

Joins all cleaned tables with SQL window functions to produce the feature matrix used by the P1 models.

```bash
python3 ExploitationZone/data_integration.py
```

#### 4b — Financial Knowledge Graph (P2)

Reads `TrustedZone.duckdb`, fetches geopolitical data from the RESTCountries API, and serialises the graph to `ExploitationZone/financial_knowledge_graph.ttl`.

Node types: `Company`, `Sector`, `Industry`, `SizeClass`, `VolatilityClass`, `Country`, `Region`, `SubRegion`, `Observation`  
Key edges: `operatesInSector`, `belongsToIndustry`, `hasSize`, `hasVolatilityProfile`, `headquarteredIn`, `locatedInRegion`, `sharesBorderWith`, `hasObservation`  
Each `Observation` carries `closePrice`, `tradeVolume`, `eurExchangeRate`, and the binary ML target `target7dUp`.

```bash
python3 ExploitationZone/graph_generation.py
```

#### 4c — Macroeconomic & Geopolitical Graph (P2)

Fetches all ~250 countries from RESTCountries (population, area, region, borders) and World Bank (GDP, GDP growth for 2022), and encodes static geopolitical tensions. Outputs `ExploitationZone/macroeconomic_graph.ttl` (~2091 triples).

Node types: `Country`, `Region`  
Key edges: `sharesBorderWith`, `hasTensionWith`, `locatedInRegion`  
Key literals: `gdpUSD`, `gdpGrowthPercent`, `population`, `areaSqKm`

```bash
python3 ExploitationZone/geopolitical_macroeconomic.py
```

#### 4d — Verify the macro graph

```bash
python3 -m pytest ExploitationZone/test_macroeconomic_graph.py -v
```

### 5. Data Analysis

#### Pipeline 1 — ARIMA (P1)

Classical time series analysis per ticker.

```bash
python3 DataAnalysisPipeline1/scripts/arima_models.py
```

#### Pipeline 2a — SPARQL Pattern Matching (P2)

Loads `financial_knowledge_graph.ttl` and `macroeconomic_graph.ttl` and runs ten analytical queries:

| Query | Description |
|---|---|
| **Q1**  | 7-day upward rate grouped by sector × market-cap class (≥100 obs filter) |
| **Q2**  | High-volatility companies in countries with geopolitical tensions, enriched with both home and rival GDP (cross-graph) |
| **Q3**  | Peer pairs sharing sector + industry + size class (deduplicated) |
| **Q4**  | Mega-cap US companies per sector with US GDP as macroeconomic context |
| **Q5**  | Acquisition fingerprint per acquirer (count, total + average spend) |
| **Q6**  | 7-day upward rate by volatility class — does the risk premium pay off? |
| **Q7**  | Border-proximity risk: companies whose HQ shares a border with a tense country (2-hop cross-graph) |
| **Q8**  | Sector concentration by macro region |
| **Q9**  | Large/mega-cap companies in high-volatility sectors that have made NO acquisitions (anti-join) |
| **Q10** | Top-3 most-acquisitive companies per sector (rank via correlated subquery) |

Q2, Q4 and Q7 demonstrate cross-graph querying: financial and macroeconomic graphs are linked at runtime by rewriting country URIs between the two namespaces.

```bash
python3 DataAnalysisPipeline2/scripts/sparql_analysis.py
```

#### Pipeline 2b — KG Embeddings + Multi-Model Bake-Off (P2)

Trains a **RotatE** model (Sun et al. 2019, PyTorch, 128-dim complex = 256 real, self-adversarial negative sampling, sigmoid-log loss, early stopping) on the structural triples of the financial KG. RotatE handles symmetric, antisymmetric and compositional relations — a strict superset of what TransE can model, which matters for the `sharesBorderWith` (symmetric) and `madeAcquisition` ↔ `acquisitionCountry` (compositional) edges in our graph.

The 256-dim complex embeddings are then **compressed via PCA to 16 axes** to balance the signal-to-noise against the small tabular feature set. Per-observation features are computed by DuckDB window functions: `log_market_cap`, `daily_return`, `price_vs_ma5`, `volume_ratio`, `rolling_volatility_10d`, `vol_adjusted_return`, `volume_zscore_20d`, plus a cross-sectional `sector_daily_return`.

The script runs a **multi-model bake-off** across three feature configurations (tabular-only, embedding-only, tabular+embedding) and **six classifiers**: **RandomForest**, **MLP**, **CatBoost**, **XGBoost**, **LightGBM**, plus a **StackingClassifier** that combines the three boosted models with a logistic meta-learner. Reports accuracy, F1 and ROC-AUC for every combination. Company embeddings are exported to `company_embeddings.parquet` for downstream reuse.

Latest bake-off (114,780 obs, 80/20 temporal split, positive rate = 0.39):

| Feature set | Best model | Accuracy | F1 | ROC-AUC |
|---|---|---:|---:|---:|
| tabular_only | **RandomForest** | 0.6607 | 0.5840 | **0.7122** |
| embedding_only (16-dim PCA of RotatE) | RandomForest | 0.5499 | 0.3961 | 0.5394 |
| tabular+embedding | Stack | 0.6669 | 0.5085 | 0.7063 |

The 16-dim KG axes only carry ~0.54 AUC on their own — sector/industry/size membership are nearly-static categorical labels and don't encode short-horizon price dynamics. The combined model lands slightly below tabular-only on ROC-AUC but recovers a couple of points of accuracy via the Stack ensemble. The +1.6 pp AUC over the P1 RF baseline (0.6964 → 0.7122) is mostly driven by the new tabular features (`vol_adjusted_return`, `sector_daily_return`, `volume_zscore_20d`).

```bash
python3 DataAnalysisPipeline2/scripts/kg_embeddings_classifier.py
```

#### Pipeline 2c — P1 Baselines (for comparison)

```bash
python3 DataAnalysisPipeline2/scripts/stock_prediction_random_forest.py
python3 DataAnalysisPipeline2/scripts/stock_prediction_mlp.py
```

---

## Pending

A fourth team member is adding ownership and inter-company relationship nodes (e.g. `ownsStakeIn`, `subsidiaryOf`) to `financial_knowledge_graph.ttl`. The TransE model and all SPARQL queries will incorporate these edges automatically once the graph is regenerated — no code changes required.

---

## Java requirement

The `TrustedZone/dataQuality.py` pipeline uses PySpark 4.x, which requires **Java 17+**. On macOS:

```bash
brew install --cask temurin@17
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home
```

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
│   ├── dataQuality.py
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
        ├── sparql_analysis.py               # P2: SPARQL queries
        └── kg_embeddings_classifier.py      # P2: TransE + RF
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

Apply data quality rules and write cleaned tables to `TrustedZone.duckdb`.

```bash
python3 TrustedZone/dataQuality.py
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

Loads `financial_knowledge_graph.ttl` and `macroeconomic_graph.ttl` and runs four analytical queries:

| Query | Description |
|---|---|
| **Q1** | 7-day upward rate grouped by sector × market-cap class |
| **Q2** | High-volatility companies headquartered in countries with active geopolitical tensions, enriched with World Bank GDP (cross-graph) |
| **Q3** | Intra-sector peer pairs sharing the same sector and size class — structural graph similarity |
| **Q4** | Mega-cap US companies per sector with US GDP as macroeconomic context |

Q2 and Q4 demonstrate cross-graph querying: financial and macroeconomic graphs are linked at runtime by rewriting country URIs between the two namespaces.

```bash
python3 DataAnalysisPipeline2/scripts/sparql_analysis.py
```

#### Pipeline 2b — KG Embeddings + Classifier (P2)

Trains a **TransE** model (PyTorch) on the structural triples of the financial KG to learn 64-dimensional embeddings for every entity. Observation nodes and literal-valued triples are excluded from embedding training — they are joined back from DuckDB after the fact. The resulting company embeddings are concatenated with the tabular features used in P1 (close price, volume, EUR/JPY rate, market cap) and fed into the same **Random Forest** classifier, enabling a direct comparison between the P1 tabular baseline and the KG-augmented model.

Steps: parse KG → build entity/relation index → train TransE (margin-ranking loss, 100 epochs) → extract company embeddings → join with DuckDB observations → temporal train/test split (80/20) → train RF → evaluate.

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

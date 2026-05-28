# P1 Compliance Audit — `Manuel-Rucker01/BDA-P1`

**Author:** Anthropic Claude on behalf of Y. Chen, M. Delgado, M. Rucker
**Date:** 2026-05-26
**Scope:** Retrospective audit of the submitted P1 codebase. **Read-only.** No P1 zone code was modified.

This document has two parts.

* **Part 1** audits the four zone implementations against the P1 spec items the team listed (Landing / Formatted / Trusted / Exploitation + Data Analysis pipelines). Each item is labelled `PASS` / `PARTIAL` / `FAIL` with a single-line evidence pointer and a single-line note on what to say about it in the P2 paper or in evaluation.
* **Part 2** documents production-quality concerns that are **outside the P1 graded scope** but affect whether the P2 paper's empirical claims (e.g. backtest returns) are defensible. Severity is `high` / `medium` / `low` for P2 defensibility, not for P1.

Findings are blunt on purpose. Soft findings help nobody — if a reviewer would notice it, this report should notice it first.

---

## Part 1 — P1 spec compliance (read-only)

### Landing Zone

| Item | Status | Evidence | Paper/eval note |
|---|---|---|---|
| ≥ 3 data sources ingested | **PASS** | `LandingZone/nasdaq.py` (Kaggle NASDAQ list), `LandingZone/company_history.py` (Yahoo Finance daily bars), `LandingZone/exchange.py` (forex-python USD rates), `LandingZone/additional_information.py` (Kaggle Forbes + S&P 500), `LandingZone/acquisitions.py` (Kaggle Top-7 acquisitions). **5 distinct sources**, comfortably above the threshold. | Say "five data sources" — well above the spec floor. |
| Data Collector exists for each source | **PASS** | One script per source as listed above. | — |
| Collector parameterised / re-runnable on schedule | **PARTIAL** | `nasdaq.py`, `additional_information.py`, `acquisitions.py` are re-runnable but **non-parametric** — they always download the same Kaggle dataset and overwrite. `company_history.py` is genuinely incremental (reads existing `company_history.csv`, downloads only data since `last_date+1`, dedupes on `(Date, Company)` and writes back — lines 56–119). `exchange.py` is **fixed at "the last 100 business days from now"** (lines 22–32) — no `--start` / `--end` parameters. | "Periodic re-running works because each Kaggle dataset is a snapshot; only `company_history.py` does true incremental updates. The exchange collector is hardcoded to a rolling 100-day window — acceptable for fresh runs but not for back-dated re-collection." Pre-empt this one if asked. |
| Raw storage in a suitable format | **PASS** | All collectors write CSV under `datasets/` (e.g. `nasdaq_companies.csv`, `company_history.csv`, `US_exchange.csv`, `forbes_employers.csv`, `sp500_companies.csv`, `company_acquisitions.csv`). | — |
| **"No data transformations applied here"** | **PARTIAL** | `nasdaq.py`, `exchange.py`, `additional_information.py`, `acquisitions.py` write the source data verbatim. `company_history.py` does `pd.concat([df_master, df_combined_new])` + `drop_duplicates(subset=['Date','Company'])` + `set_index('Date')` (lines 105–119). This is *append + dedupe*, defensible as "required for incremental writes" but a strict spec reading flags it. | Frame the concat/dedupe as "incremental merge needed to preserve append-only semantics, not a semantic transformation." Pre-empt if asked. |

**Landing-Zone overall:** 3 PASS + 2 PARTIAL. No FAIL. Defensible.

---

### Formatted Zone

| Item | Status | Evidence | Paper/eval note |
|---|---|---|---|
| Data homogenised into the relational data model | **PASS** | Each CSV is loaded as a Spark `DataFrame` and written as a single DuckDB table (`FormattedZone/formatted_zone_pipeline.py:65–95` for the reads, `:117–156` for the writes). One table per source — `nasdaq`, `company_history`, `us_exchange`, `sp500_companies`, `forbes_employers`, `company_acquisitions`. | — |
| Uses Spark / SparkSQL (DataFrame API or SQL) — **HARD requirement** | **PASS (literal) / PARTIAL (spirit)** | Spark IS invoked: `SparkSession.builder.appName("FormattedZonePipeline")` (line 46), `spark.read.csv(...)` returns a Spark `DataFrame` (lines 65, 70, 75, 80, 85, 91), `.count()` is called (lines 66, 71, 76, 81, 86, 92). But the only DataFrame operations are `read.csv` and `count`; **all subsequent processing happens after `.toPandas()`** (lines 118, 125, 132, 139, 146, 153). Spark is effectively used as a CSV reader. | The spec says "uses Spark or SparkSQL (DataFrame API or SQL)" — that bar is technically met. If pushed in evaluation, frame as "Spark for ingestion-stage homogenisation; downstream zones use the Spark-loaded DataFrames for SQL DCs." Don't oversell it as a Spark-heavy pipeline. |
| One table per dataset | **PASS** | 6 source CSVs → 6 DuckDB tables, exact 1:1. | — |

**Formatted-Zone overall:** All PASS, but the Spark usage is thin. If a reviewer asks "*what processing does Spark actually do here?*" the honest answer is *parallel CSV reading and row counts*.

---

### Trusted Zone

| Item | Status | Evidence | Paper/eval note |
|---|---|---|---|
| DQ rules identified & documented | **PASS** | Documented in two places: as Spark SQL `CREATE VIEW` constraints (`TrustedZone/dataQuality.py:140–187`) and as a structured `denial_constraints` dict written into the `data_quality_metrics` table (lines 293–306). Both are human-readable. | — |
| Quality assessment performed (not just cleaning) | **PASS** | `raw_counts`, `clean_counts`, `rows_removed`, `removal_rate` computed per table and persisted to the `data_quality_metrics` table (lines 288–292 + the `write_to_trusted_zone` call). Pre/post comparison is explicit. | "We measured DQ — not only filtered. The `data_quality_metrics` table preserves before/after row counts and per-table removal rates." |
| Data cleaning applied per-dataset | **PASS** | Each of the six tables has its own filter block (`dataQuality.py:191–252`). | — |
| **Rules expressed as Denial Constraints** | **PARTIAL** | The DCs are defined in Spark SQL views (`dataQuality.py:140–187`) — that part is genuinely DC-style. However, **the views are not what cleans the data**: the actual filtering is `pandas`-boolean-mask logic that *re-encodes the same predicates* (lines 192–251). The Spark-SQL views are effectively documentation; the executed logic is pandas. | This is the single biggest "if a reviewer reads carefully" risk in Trusted Zone. Acknowledge in the paper: "*Denial constraints are documented in Spark SQL CREATE VIEW statements and applied in equivalent pandas predicates against the same source tables; the two encode the same logical constraint.*" Do not claim "DCs are applied in Spark" — that is not quite true. |
| Pipeline uses Spark / SparkSQL — **HARD requirement** | **PASS (literal) / PARTIAL (spirit)** | `SparkSession` is initialised (`dataQuality.py:67–74`), source DataFrames are registered as temp views via `createOrReplaceTempView` (lines 132–137), and `spark.sql(...)` builds the constraint views (lines 140–187). Same caveat as Formatted Zone: executed logic is pandas. | Same defence as Formatted Zone. |
| Schema parity with Formatted Zone | **FAIL** | Formatted Zone produces 6 tables: `nasdaq`, `company_history`, `us_exchange`, `sp500_companies`, `forbes_employers`, `company_acquisitions`. Trusted Zone produces those 6 *plus* an enriched `companies` table (`dataQuality.py:371–410`) that joins `nasdaq` + `sp500_companies` + `forbes_employers` to resolve a `country` column. The new table breaks "the same tables as the Formatted Zone where their quality has been improved" — it is a new integration product, not a cleaned version of an existing table. | **Real issue.** Acknowledge in the paper: "*The `companies` table is an integration product — a country reconciliation across NASDAQ, S&P 500, and Forbes — and would more naturally live in the Exploitation Zone. It was placed in Trusted because downstream consumers (KG generation) needed a single non-null country attribute per ticker, but a strict reading of P1 would place this in Exploitation.*" Pre-empt this — a reviewer who reads `prepare_trusted_database` will see the 7th table. |

**Trusted-Zone overall:** 4 PASS + 1 PARTIAL + 1 FAIL. The schema-parity FAIL and the "DCs documented in Spark but executed in pandas" wrinkle are the two things to address in the paper.

---

### Exploitation Zone

| Item | Status | Evidence | Paper/eval note |
|---|---|---|---|
| Trusted Zone datasets merged / integrated for downstream consumption | **PASS** | `ExploitationZone/data_integration.py` produces `master_dataset` joining `company_history`, `us_exchange` (forward-filled), and `nasdaq` metadata (`data_integration.py:30–98`). `ExploitationZone/graph_generation.py` produces the financial RDF graph from `nasdaq`, `sp500_companies`, `forbes_employers`, `company_acquisitions`, `companies`. `ExploitationZone/geopolitical_macroeconomic.py` produces the macro RDF graph from RESTCountries + World Bank. | — |
| Data reconciliation across sources | **PARTIAL — wrong zone** | The country reconciliation (SP500 → Forbes → default-US) **lives in `TrustedZone/dataQuality.py:371–410`**, not in `ExploitationZone`. The spec calls reconciliation an Exploitation activity. The team built the right reconciliation in the wrong zone. | Same paper note as the schema-parity FAIL above. Two findings, one root cause. |

**Exploitation-Zone overall:** PASS on integration; the reconciliation finding is shared with Trusted Zone.

---

### Data Analysis Pipelines

| Item | Status | Evidence | Paper/eval note |
|---|---|---|---|
| ≥ 2 analytical pipelines exist and are functional | **PASS** | Pipeline 1: `DataAnalysisPipeline1/scripts/arima_models.py` (auto-ARIMA price / return forecasting), plus `arima_results_validation.py` for performance plots. Pipeline 2: `DataAnalysisPipeline2/scripts/kg_embeddings_classifier.py` (RotatE + walk-forward CV + multi-model bake-off) and `sparql_analysis.py` (cross-graph SPARQL). Both are referenced in the paper and have been executed successfully in the prior session (commits `1887f9b`, `c99cd77`). | — |

---

### Part 1 summary

| Zone | PASS | PARTIAL | FAIL |
|---|---:|---:|---:|
| Landing | 3 | 2 | 0 |
| Formatted | 3 (1 literal-pass) | 0 | 0 |
| Trusted | 4 | 1 | 1 |
| Exploitation | 1 | 1 | 0 |
| Pipelines | 1 | 0 | 0 |
| **Total** | **12** | **4** | **1** |

The single FAIL is the schema-parity issue in Trusted Zone (the `companies` enriched table belongs in Exploitation). The four PARTIALs are defensible if pre-empted in the paper. If the reviewer reads `prepare_trusted_database` end-to-end, they will see the extra table.

**Bottom line for the team:** The pipeline is solid. There is one structural mistake (country reconciliation in the wrong zone) that should be acknowledged in the P2 paper rather than discovered in evaluation. Everything else is fine.

---

## Part 2 — Beyond P1 scope: production concerns

These are **not P1 graded requirements.** They affect whether the empirical claims in the P2 paper (especially backtest returns) hold up under a careful read. Severity here is for **P2 paper defensibility**, not for P1.

### 1. Point-in-time universe / survivorship bias

* **Severity:** **high**
* **Evidence:** `LandingZone/nasdaq.py` downloads the *current* NASDAQ company list from Kaggle (`dhimananubhav/nasdaq-company-list`). `LandingZone/company_history.py` then collects daily prices only for tickers present in that list. The historical universe used in the backtest is therefore **today's surviving NASDAQ membership**, projected backwards.
* **Effect on the paper:** Companies that were listed in 2024 and delisted in 2025 (bankruptcies, mergers, regulatory delistings) are absent from the universe. The backtest's "Buy & Hold the universe" benchmark and any long-only / long-short strategy systematically overstates returns because the worst outcomes have been pruned. The 24-month backtest is most affected.
* **Fix in P2 / acknowledge as future work:** Acknowledge as future work. A point-in-time universe would require historical NASDAQ membership snapshots (e.g. CRSP), which is not freely available. The paper should add a sentence: "*The backtest universe is forward-only NASDAQ membership; survivorship bias may inflate strategy returns and the Buy & Hold benchmark alike.*"

### 2. Corporate action handling (splits & dividends)

* **Severity:** **medium**
* **Evidence:** `company_history.csv` columns are `Date, Open, High, Low, Close, Volume, Dividends, Stock Splits, Company, Capital Gains`. The `Close` column saved is the yfinance default `auto_adjust=True` close — i.e. **split-adjusted but not dividend-adjusted**. `Dividends` is a separate column carrying the raw dividend amount, but it is never added back into a "total-return" close anywhere in the pipeline.
* **Effect on the paper:** Splits are handled correctly. Dividends are not. For a 30-day return target this is small in expectation (a handful of bps per month at NASDAQ-typical yields) but it is a real downward bias against high-dividend names. The backtest "Buy & Hold" benchmark is also underreported by the same amount.
* **Fix in P2 / acknowledge as future work:** Acknowledge. The fix is trivial — `Close + cumulative_dividend_reinvestment` — but propagates through every backtest. Note in the paper: "*Returns are computed on split-adjusted close; dividend reinvestment is not modelled. The effect is symmetric across strategies and benchmark.*"

### 3. Schema contracts on ingest

* **Severity:** **medium**
* **Evidence:** None of the collectors assert anything about column names, row counts, or date ranges after download. `LandingZone/additional_information.py:35–55` looks for files by name pattern (`if "company" in file_name.lower() or "info" in file_name.lower() or "constituents" in file_name.lower()`) — a Kaggle rename would silently pick the wrong CSV. `company_history.py` assumes columns `Open, High, Low, Close, Volume, Dividends, Stock Splits` — a Yahoo rename (which has happened historically) would propagate `NaN` into the pipeline without erroring.
* **Effect on the paper:** None today. But if you regenerate the data and a column has been renamed upstream, the model would silently retrain on a degraded feature set with no failure signal.
* **Fix in P2 / acknowledge as future work:** Acknowledge as future work. Cheap to add (`assert set(REQUIRED_COLS).issubset(df.columns)`) but out of scope for the current submission.

### 4. Freshness checks

* **Severity:** **low**
* **Evidence:** No script asserts that the data it reads is current. Trusted-Zone and Exploitation-Zone scripts will happily run on a `company_history.csv` that hasn't been updated in months, and the only signal of staleness is the absence of recent dates in the downstream output.
* **Effect on the paper:** None for the submitted version (data was collected fresh). Real concern for a re-run.
* **Fix in P2 / acknowledge as future work:** Acknowledge — fix is "`max(Date) >= today - timedelta(days=3)`" check at the top of each consumer. Out of scope today.

### 5. Macroeconomic graph robustness on outage / silent NaN propagation

* **Severity:** **medium**
* **Evidence:** `ExploitationZone/geopolitical_macroeconomic.py` queries RESTCountries (`:23–46`) and World Bank (`:49–62`) live. Both have `try/except Exception` blocks that **return an empty dict on failure** and continue. No caching layer. No alerting. The downstream KG construction proceeds with whatever it got — silently producing a graph with missing GDP / region / borders triples if either API was rate-limited or down.
* **Effect on the paper:** If the team regenerates the macro graph during a RESTCountries outage, the resulting `macroeconomic_graph.ttl` will be smaller than the committed version (which has ~2,091 triples). The downstream model will still load, but with degraded macro features. Currently the committed `.ttl` is good, so no immediate impact.
* **Fix in P2 / acknowledge as future work:** Acknowledge as future work. A defensible mitigation already exists: the committed `macroeconomic_graph.ttl` in LFS *is* the cache; regeneration is opt-in. State this in the paper: "*The macroeconomic graph is generated once and committed; downstream pipelines load from the cached `.ttl` rather than re-querying.*"

---

### Part 2 summary

| Concern | Severity for P2 paper | Suggested handling |
|---|---|---|
| Survivorship bias in NASDAQ universe | **high** | Acknowledge in paper; cannot fix without paid CRSP-equivalent |
| No dividend reinvestment | medium | Acknowledge; symmetric across strategies |
| No ingest schema contracts | medium | Acknowledge as future work |
| No data freshness checks | low | Acknowledge as future work |
| Macro graph silently NaN-propagates on outage | medium | Acknowledge; `.ttl` cache mitigates today |

The high-severity finding (survivorship) is the only one a referee is likely to push on. Pre-empt it in the paper. Everything else is reasonable to call out as "future work" without weakening the contribution.

---

## What this audit does NOT do

* It does **not** modify any P1 zone code.
* It does **not** rerun the pipelines (the committed artefacts — `TrustedZone.duckdb`, `financial_knowledge_graph.ttl`, `best_model.pkl` — are taken as the as-submitted state).
* It does **not** audit the P2 analytical layer beyond a sanity check that both pipelines exist and are functional (that is Part 3 of the broader workstream).

The team should use this audit as a checklist before P2 evaluation: every PARTIAL or FAIL above corresponds to a question a reviewer might ask. Knowing the honest answer in advance is the point.

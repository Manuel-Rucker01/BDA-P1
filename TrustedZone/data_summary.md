# Trusted Zone Data Storage Architecture

The Trusted Zone serves as the single source of truth for all analytical processes and downstream Exploitation Zone tasks. Data in this phase has been thoroughly cleaned, strictly typed, and validated against predefined Denial Constraints.

## Storage Engine

The data is persisted in a single file-based database named `TrustedZone.duckdb`. DuckDB is an in-process SQL OLAP (Online Analytical Processing) database management system. 

Unlike traditional row-based databases (like PostgreSQL or MySQL), DuckDB stores data in a columnar format. This architectural choice significantly accelerates read-heavy analytical queries, aggregations, and joins. When the Exploitation Zone needs to query the historical stock prices alongside the company metadata, DuckDB can scan the specific columns requested without loading entire rows into memory, minimizing the I/O bottleneck.

## Enforced Schema and Data Typing

A critical aspect of the Trusted Zone storage is the strict enforcement of data types. During the pipeline execution, Apache Spark schema inference is bypassed in favor of explicit schema definitions. All text fields are strictly cast as strings and missing values are translated to native nulls rather than floating-point NaNs. This guarantees that the downstream machine learning models and knowledge graph generators will not encounter mixed-type errors when ingesting the tables.

## Stored Entities

The DuckDB database contains seven specific tables. Six of these represent the core business data, and one serves as an operational metadata registry.

### 1. `nasdaq`
Contains the curated catalog of NASDAQ-listed companies.
* **Primary Identifier:** `Symbol` (Strictly non-null)
* **Key Attributes:** `Name`, `Sector`, `Industry`, `LastSale`, `MarketCap`, `IPOyear`
* **Integrity Rules Applied:** Market capitalization and last sale prices are guaranteed to be non-negative. IPO years are bounded to realistic historical limits.

### 2. `company_history`
The largest table in the database, storing the daily trading metrics for the tracked companies.
* **Primary Identifier:** `Date` and `Company` (Ticker)
* **Key Attributes:** `Open`, `High`, `Low`, `Close`, `Volume`
* **Integrity Rules Applied:** High prices are mathematically verified to be greater than or equal to Low prices. Trading volumes and closing prices are non-negative.

### 3. `us_exchange`
Stores the historical exchange rates of the US Dollar against global currencies.
* **Primary Identifier:** `Date`
* **Key Attributes:** `EUR`, `JPY`, `GBP`, `CHF`, etc.
* **Integrity Rules Applied:** Critical currency rates (like EUR and JPY) are verified to be strictly greater than zero to prevent division-by-zero errors in downstream financial conversions.

### 4. `sp500_companies`
Holds the dimensional data for the companies included in the S&P 500 index.
* **Primary Identifier:** `Ticker` (Strictly non-null)
* **Key Attributes:** `Name`, `Sector`, `Industry`, `MarketCap`, `Country`, `Employees`
* **Integrity Rules Applied:** Ensures that no company is recorded with a negative employee count or market capitalization.

### 5. `forbes_employers`
Contains the ranking data from the Forbes World's Best Employers list.
* **Primary Identifier:** `company` and `rank`
* **Key Attributes:** `industries`, `country_territory`, `employees`, `publish_year`
* **Integrity Rules Applied:** All records have a valid rank greater than zero, and the publication year cannot exceed the current temporal bounds.

### 6. `company_acquisitions`
Records the historical mergers and acquisitions carried out by seven of the largest technology corporations (Microsoft, Google, IBM, HP, Apple, Amazon, and Facebook/Twitter).
* **Primary Identifier:** `ParentCompany` and `AcquiredCompany`
* **Key Attributes:** `AcquisitionYear`, `AcquisitionMonth`, `Business`, `Country`, `AcquisitionPrice`, `Derived Products`, `Category`
* **Integrity Rules Applied:** Acquisition years are bounded between the founding year of the parent company and the current year. The acquisition price, when present, is verified to be non-negative, and the `ParentCompany` field is restricted to the canonical set of seven acquirers to prevent foreign-key drift when joining against the `nasdaq` and `sp500_companies` tables.

### 7. `data_quality_metrics`
This operational table does not store financial data. Instead, it acts as an immutable audit log for the data quality pipeline. 
* **Structure:** `metric_name` (VARCHAR), `metric_value` (JSON String), `timestamp` (TIMESTAMP).
* **Purpose:** It stores the exact row counts, the number of records removed by the Denial Constraints, and the removal rates for each pipeline execution. This allows data engineers to trace back exactly how much data was filtered out before it reached the Exploitation Zone.
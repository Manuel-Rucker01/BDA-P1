# PySpark Ingestion Layer (Formatted Zone)

This folder contains standardization scripts, representing the **Formatted Zone** of our financial DataOps lake.

---

## 📁 File Directory

### 1. **formatted_zone_pipeline.py**
* **Purpose**: Ingests raw stock prices, company parameters, exchange rates, and sovereign records from `/datasets` and standardizes their schemas using PySpark SQL.
* **Outputs**: Standardized relational tables mapped into DuckDB tables (`master_dataset`).

---

## 💻 Execution

To run standardizations and establish relational schemata inside DuckDB, run from the repository root:
```bash
python3 FormattedZone/formatted_zone_pipeline.py
```
*The pipeline enforces homogeneous relational data models, ensuring all datasets conform to unified structures before entering data quality sanitization.*

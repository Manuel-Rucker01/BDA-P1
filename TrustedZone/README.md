# Sanitization & Data Quality Layer (Trusted Zone)

This folder contains data quality pipelines, representing the **Trusted Zone** of our financial DataOps lake.

---

## 📁 File Directory

### 1. **dataQuality.py**
* **Purpose**: Implements rigorous data quality audits using PySpark Denial Constraints:
  * Verifies closing prices are positive and non-zero.
  * Verifies trade volumes are non-negative.
  * Checks for duplicate records and null values.
  * Resolves each stock's country via SP500 name matching and Forbes directories, producing a clean, enriched companies directory.
* **Outputs**: Cleans and writes synchronized tables straight to `TrustedZone.duckdb`.

### 2. **test_dataQuality.py**
* **Purpose**: Contains the unit-test suite checking the mathematical validity of data quality filters.

---

## 💻 Execution

To execute data quality cleaning, run from the repository root:
```bash
python3 TrustedZone/dataQuality.py
```

To run the unit-test suite:
```bash
python3 -m pytest TrustedZone/test_dataQuality.py -v
```

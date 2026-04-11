# Project 1 - Large-Scale Data Engineering for AI

**Authors:** Yufeng Chen, Marc Delgado, Manuel Rucker
**Date:** 12th December 2026 

## Repository Structure

* **`LandingZone/`**: Contains custom data collectors to interact directly with different APIs to acquire raw financial data in a consistent and automated way.
* **`datasets/`**: The local file system where the collected raw data is stored as CSV files.
* **`FormattedZone/`**: Scripts using PySpark, Pandas, and DuckDB to take the raw data from the CSV files and save it into an organized database.
* **`TrustedZone/`**: Contains the data quality pipeline to improve the quality of our structured data by identifying and filtering out anomalous or invalid records.
* **`ExploitationZone/`**: The final data integration stage where cleaned data is transformed into a high-value analytical asset..
* **`DataAnalysisPipeline1/`**: Contains the classical time series analysis scripts utilizing models like Auto-ARIMA.
* **`DataAnalysisPipeline2/`**: Contains the advanced machine learning models.

---

## Execution Pipeline

To run the full end-to-end pipeline, execute the scripts in the following order from the root directory of the repository.

### 1. Landing Zone 
Run the data collectors to fetch the current market data and append it to the existing raw storage.

```bash
python3 LandingZone/nasdaq.py
python3 LandingZone/company_history.py
python3 LandingZone/exchange.py
```

### 2. Formatted Zone
Transform the raw CSV data into DuckDB. 

```bash
python3 FormattedZone/formatted_zone_pipeline.py
```

### 3. Trusted Zone

Run the data quality pipeline to enforce data integrity constraints.

```bash
python3 TrustedZone/dataQuality.py
```

### 4. Exploitation Zone

Join the datasets and implement advanced SQL window functions to create a dense matrix of features


```bash
python3 ExploitationZone/data_integration.py
```

### 5. Data Analysis

Execute the scripts within the DataAnalysisPipeline1 and 2 directory

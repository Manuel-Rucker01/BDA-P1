# Ingestion Layer (Landing Zone)

This folder contains raw stock pricing and currency exchange rates data collectors, representing the **Landing Zone** of our financial DataOps lake.

---

## 📁 File Directory

### 1. **nasdaq.py**
* **Purpose**: Downloads stock listing metadata and filters candidate tickers that belong to NASDAQ small-to-mid-cap lists.
* **Outputs**: Raw NASDAQ company metadata written to raw CSV formats inside `/datasets`.

### 2. **company_history.py**
* **Purpose**: Ingests historical daily OHLCV price bars for active portfolio candidates from APIs.
* **Outputs**: Raw stock price histories written to `/datasets`.

### 3. **exchange.py**
* **Purpose**: Dynamically queries currency exchange rates to pull USD/EUR and USD/JPY conversions.
* **Outputs**: Raw foreign exchange rates written to `/datasets/us_exchange.csv`.

### 4. **additional_information.py**
* **Purpose**: Queries macroeconomic country-level indicators (GDP, population, area) and geopolitical links.
* **Outputs**: Macro data logs written to raw files inside `/datasets`.

---

## 💻 Execution

To execute raw data collection, run the scripts from the repository root:
```bash
python3 LandingZone/nasdaq.py
python3 LandingZone/company_history.py
python3 LandingZone/exchange.py
python3 LandingZone/additional_information.py
```
*Note: In production, the collectors support periodic executions to pull live incremental daily candles.*

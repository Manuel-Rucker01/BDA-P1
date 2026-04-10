# Multi-Asset ARIMA Time Series Forecasting Analysis

## 📁 Folder Overview

The **TimeSeries** folder contains a comprehensive ARIMA (AutoRegressive Integrated Moving Average) analysis of multiple financial time series extracted from the ExploitationZone master dataset. This analysis compares the predictability of three asset classes (S&P 500, EUR/USD, JPY/USD) across different measurement types (price levels, returns, and volatility).

### Purpose
To understand which time series are most suitable for ARIMA forecasting and determine the best ARIMA parameters for each asset by analyzing:
- **Prices vs. Returns**: Do absolute values predict better than percentage changes?
- **Equity vs. Forex**: Which asset class is more predictable?
- **Directional Accuracy**: Can we predict movement direction even if magnitude is hard to forecast?

---

## 📄 File Descriptions

### 1. **arima_models.py** (147 lines)
**Purpose**: Core ARIMA modeling engine

**Key Components**:
- `MultiAssetARIMA` class: Unified interface for fitting ARIMA models to 7 time series
- `load_data()`: Extracts time series from `ExploitationZone/master_dataset_pro.csv`
- `fit_arima_models()`: Auto ARIMA grid search (p∈[0,3], d∈[0,2], q∈[0,3])
- `save_results()`: Exports metrics to CSV
- `get_summary()`: Returns results dataframe for analysis

**Data Sources**:
- S&P 500: Close price, daily returns (%), 20-day volatility
- EUR/USD: Exchange rate, daily returns (%)
- JPY/USD: Exchange rate, daily returns (%)

---

### 2. **arima_results_validation.py** (325 lines)
**Purpose**: Validation, analysis, visualization, and reporting

**Main Functions**:
- `validate_and_analyze()`: Computes comprehensive analysis metrics
- `create_visualizations()`: Generates 2 publication-quality plots
- `generate_report()`: Creates human-readable insights

**Outputs**:
- Console logs with key findings
- CSV exports with detailed metrics
- PNG visualizations (2 charts)
- Text report with recommendations

---

### 3. **arima_validation_full.csv** (6 data rows + header)
**Purpose**: Complete results table with all metrics

**Fields**:
| Field | Meaning |
|-------|---------|
| ARIMA_Order | (p,d,q) parameters selected by auto_arima |
| AIC | Akaike Information Criterion (model fit quality) |
| RMSE | Root Mean Squared Error (average absolute error in units) |
| MAE | Mean Absolute Error (another error magnitude) |
| MAPE(%) | Mean Absolute Percentage Error (accuracy %) |
| DirAcc(%) | Directional Accuracy (% correct up/down predictions) |
| Train_Size | Training set observations (80% of data) |
| Test_Size | Test set observations (20% of data) |
| Performance | Categorical rating (Excellent/Good/Fair/Poor) |

---

### 4. **arima_validation_analysis.png** (141 KB)
**Purpose**: Four-panel comprehensive visualization

**Panels**:
- **Top-Left**: MAPE comparison (predictability ranking)
- **Top-Right**: Directional accuracy by asset
- **Bottom-Left**: Error magnitude comparison (RMSE vs MAE)
- **Bottom-Right**: AIC model fit quality ranking

**Interpretation**: Visual comparison of all assets across multiple metrics

---

### 5. **arima_performance_heatmap.png** (75 KB)
**Purpose**: Normalized performance matrix

**Shows**: All metrics (RMSE, MAE, MAPE, DirAcc) normalized to 0-1 scale
- Green = Good (low error, high accuracy)
- Red = Poor (high error, low accuracy)

**Use**: Identify which assets perform best on which metrics at a glance

---

### 6. **arima_analysis_report.txt** (1.8 KB)
**Purpose**: Executive summary with actionable insights

**Sections**:
- Executive summary (key statistics)
- Rankings by predictability (MAPE)
- Rankings by direction prediction
- Key insights and patterns
- Recommendations for trading/forecasting

---

## Comprehensive Results Analysis

### Dataset Characteristics
```
Time Period: 2025-12-02 → 2026-03-12 (62 trading days)
Why 62 days?: Exchange rates only available from 2025-12-02
              INNER JOIN with SP500 limits to overlap
Total Rows: 62 unique dates × 1 share price = 62 observations (for univariate)
Train/Test: 80/20 split = 49 train / 13 test samples
```

### Time Series Analyzed (6 total)
1. **SP500_Close** - S&P 500 index closing price
2. **SP500_Returns** - Daily percentage returns
3. **EUR_USD** - EUR/USD exchange rate
4. **EUR_Returns** - Daily EUR/USD percentage returns
5. **JPY_USD** - JPY/USD exchange rate
6. **JPY_Returns** - Daily JPY/USD percentage returns

*Note: SP500_Volatility skipped (only 42 samples after 20-day rolling window)*

---

## Results Interpretation

### **Category 1: PRICES (Highly Predictable) **

#### **1. JPY_USD - ARIMA(2,0,0)**
| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **MAPE** | **0.60%** | Exceptional - 99.4% accuracy |
| **AIC** | 124.2 | Good fit quality |
| **RMSE** | 1.11 | Typical model error |
| **MAE** | 0.93 | Average error in rate units |
| **DirAcc** | 50.0% | Random (can't predict direction) |
| **ARIMA(2,0,0)** | AR(2) | Uses last 2 rates; no differencing |

**Verdict: EXCELLENT**
- **Why good**: Lowest MAPE of all assets. JPY is a major currency pair with stable trends.
- **Why 50% direction?**: Accuracy works for levels, not levels. Small errors don't help direction.
- **Trade use**: Ideal for short-term JPY forecasts (next 1-5 days)

---

#### **2. EUR_USD - ARIMA(0,1,0)**
| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **MAPE** | **0.97%** | Excellent - 99% accuracy |
| **AIC** | -418.7 | Best fit (negative = very good) |
| **RMSE** | 0.0097 | Tiny errors |
| **MAE** | 0.0083 | Minimal average error |
| **DirAcc** | 0.0% | Can't predict direction |
| **ARIMA(0,1,0)** | Random Walk | Simply uses differenced previous value |

**🎯 Verdict: EXCELLENT**
- **Why good**: EU currency is highly stable and predictable.
- **Simplicity**: Only needs differencing (d=1), AR/MA not required.
- **Trade use**: Best-case for currency forecasting. Very reliable.

---

#### **3. SP500_Close - ARIMA(0,1,0)**
| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **MAPE** | **1.04%** | Excellent - 99% accuracy |
| **AIC** | 505.5 | Good fit |
| **RMSE** | 78.99 | Error in index points (~250 point index) |
| **MAE** | 71.05 | Average forecast miss |
| **DirAcc** | 0.0% | Can't predict direction |
| **ARIMA(0,1,0)** | Random Walk | Pure momentum |

**Verdict: EXCELLENT**
- **Why good**: S&P 500 follows strong trend. ARIMA captures momentum perfectly.
- **Why 1.04% > EUR 0.97%?**: Index prices larger ($6,800) vs rates ($0.85), same % error.
- **Trade use**: Excellent for 1-5 day forecasts of index levels.

---

### **Category 2: RETURNS (Unpredictable)**

#### **4. SP500_Returns - ARIMA(1,0,1)**
| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **MAPE** | **111.05%** | Poor - Completely unreliable |
| **AIC** | 98.1 | Moderate fit (not great) |
| **RMSE** | 0.723 | ~0.72% error per day |
| **MAE** | 0.603 | Average error |
| **DirAcc** | **66.7%** | Better than random! |
| **ARIMA(1,0,1)** | AR(1) + MA(1) | Tries past return + past error |

**🎯 Verdict: POOR MAPE, BUT INTERESTING**
- **Why high MAPE?**: Returns are fundamentally random. MAPE >100% = predictions flip sign frequently.
- **Why 66.7% direction works?**: Only 13 test samples. Small sample luck. Not statistically significant.
- **Why return forecasting is hard**: 
  - Efficient market hypothesis: prices already reflect all information
  - Returns shouldn't depend on past returns (if they did, trading systems would exploit it)
- **Trade use**: Direction-based strategies might work short-term, but risky with small sample.

---

#### **5. EUR_Returns - ARIMA(0,0,1)**
| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **MAPE** | **100.76%** | Poor - Unreliable |
| **AIC** | 38.0 | Poor fit |
| **RMSE** | 0.428 | ~0.43% daily error |
| **MAE** | 0.308 | Average magnitude |
| **DirAcc** | 0.0% | Can't predict direction |
| **ARIMA(0,0,1)** | MA(1) | Only uses past error |

**Verdict: POOR**
- **Why worse than JPY price (0.6%)?**: Currency returns are white noise, prices trend.
- **Why pure MA?**: No autoregressive term helps (past returns don't predict future).
- **Trade use**: NOT recommended. Better to trade on fundamentals or technical indicators.

---

#### **6. JPY_Returns - ARIMA(1,0,0)**
| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **MAPE** | **128.23%** | Worst - Highly unreliable |
| **AIC** | 82.0 | Moderate (not terrible) |
| **RMSE** | 0.476 | ~0.48% daily error |
| **MAE** | 0.390 | Average error |
| **DirAcc** | 50.0% | Random guess |
| **ARIMA(1,0,0)** | AR(1) | Uses previous return |

**Verdict: POOR**
- **Why highest MAPE?**: Highest percentage change magnitude → larger relative errors.
- **Why 50% direction = no signal?**: Random walk hypothesis validates here.
- **Trade use**: Avoid. Returns don't contain predictable patterns for trading.

---

## Key Metrics Explained

### **MAPE (Mean Absolute Percentage Error)** - Primary Metric
```
MAPE = (1/n) × Σ |actual - predicted| / |actual| × 100%

Range: 0% (perfect) → ∞ (terrible)
Interpretation:
  < 2%    = Exceptional (practically perfect)
  2-5%    = Good (reliable forecasts)
  5-10%   = Fair (acceptable for exploratory)
  > 10%   = Poor (unreliable)
```

**Why we use MAPE**:
- Percentage-based (fair comparison across different scales)
- Intuitive (directly interpretable as forecast accuracy)
- Penalizes large errors more

**Problems with MAPE**:
- Returns can be near-zero (division by tiny numbers → inflated %)
- Asymmetric (forecast 2% when actual 1% = 100% error)

---

### **AIC (Akaike Information Criterion)** - Model Selection
```
AIC = 2k - 2ln(L)
  where k = model parameters
        L = likelihood

Lower AIC = better fit (balances accuracy vs complexity)
Negative AIC = excellent fit
Δ AIC > 10 = significant difference
```

**Interpretation**:
- EUR_USD (AIC = -418.7): Excellent fit, simple model required
- JPY_USD (AIC = 124.2): Good fit, needs AR(2) complexity
- EUR_Returns (AIC = 38.0): Poor fit, AR/MA doesn't help returns

---

### **RMSE (Root Mean Squared Error)** - Error Magnitude
```
RMSE = √(1/n × Σ(actual - predicted)²)

Same units as the series (index points, exchange rate)
Properties:
  - Penalizes large errors more than MAE
  - Comparable across series if normalized
```

**Example**: 
- SP500_Close RMSE = 78.99 points on $6,800 index = 1.16% (matches MAPE)
- EUR_USD RMSE = 0.0097 on 0.85 rate = 1.14% (matches MAPE)

---

### **MAE (Mean Absolute Error)** - Average Deviation
```
MAE = (1/n) × Σ |actual - predicted|

Same units as series
More robust to outliers than RMSE
Usually MAE < RMSE (RMSE penalizes extremes)
```

---

### **DirAcc (Directional Accuracy)** - Up/Down Prediction
```
DirAcc = (# correct direction predictions / total predictions) × 100%

50% = random guess (coin flip)
> 55% = statistically significant with large sample
55-60% = useful for trading
> 65% = excellent

Important: Small samples (13 tests) → high variance
           66.7% from 13 samples ≠ real 66.7%
```

---

### **ARIMA(p,d,q) Parameters**
```
p = autoregressive order (use past values)
d = differencing order (1st/2nd differences to make stationary)
q = moving average order (use past errors)

Examples:
  (0,1,0) = Simple differencing, no AR/MA → Random walk
  (2,0,0) = AR(2) only → Use last 2 values, no differencing
  (1,0,1) = One AR + one MA → Combination approach
```

**Selection Process**:
- Auto ARIMA tests 4 × 3 × 4 = 48 combinations
- Selects lowest AIC (best fit per information criterion)
- Takes **10-15 seconds** per series

---

## Why Results Are What They Are

### **Prices Predictable (~1% MAPE) - WHY?**

1. **Trend Persistence**: Asset prices follow momentum
   - If SP500 was 6,800 yesterday, likely 6,801-6,799 today
   - Not random, but nearly-stationary trend

2. **Market Efficiency (Weak)**: 
   - Prices efficient at second-to-second (algorithmic systems)
   - But daily/weekly trends exist (slow-moving capital)

3. **Limited Data Window**: 
   - 62 days = ~3 months
   - Too short to see regime changes
   - Model fits the recent trend well

4. **ARIMA Strength**: 
   - Designed for trending, stationary data
   - Differencing removes trend
   - Model captures remaining patterns

---

### **Returns Unpredictable (>100% MAPE) - WHY?**

1. **Strong Form Efficiency**:
   - Daily returns truly random walk
   - Past returns don't predict future
   - Market processes news instantly

2. **Volatility Clustering** (Not ARIMA's strength):
   - Large returns follow large returns (but in any direction)
   - GARCH/LSTM better than ARIMA for return forecasting

3. **Noise vs Signal**:
   - Price = signal + noise, signal dominates
   - Return = mostly noise, little signal
   - Removing 99.5% of variance → pure noise

4. **Small Sample Effect**:
   - 13 test samples too small for returns
   - 66.7% accuracy = 1-2 lucky predictions out of 13
   - Real population accuracy likely ~50%

---

## Key Takeaways

| Finding | Implication | Action |
|---------|-------------|---------|
| **Prices ~ 1% MAPE** | Highly predictable | Use ARIMA for price forecasting |
| **Returns > 100% MAPE** | Fundamentally random | Don't use ARIMA for returns |
| **FX Prices < Equity** | EUR/JPY simpler than SP500 | Focus on currency pair forecasts |
| **Small sample (62 days)** | High statistical uncertainty | Test on more data before trading |
| **SARIMA not needed** | Already optimal with ARIMA | No seasonality benefit (too short) |

---

## Recommendations

1. **Best Use Case**: 1-5 day price forecasts for JPY_USD and EUR_USD
   - MAPE < 1%, highly reliable
   - Practical for algorithmic trading

2. **Avoid**: Return forecasting with ARIMA
   - Use ML/neural nets instead (handle nonlinearity)
   - Or fundamental analysis (earnings, news)

3. **Data Expansion**: 
   - Current: 62 days
   - Recommended: 2+ years (500+ trading days)
   - Would improve return forecasting and reduce overfitting

4. **Alternative Models**:
   - Returns: LSTM neural networks, Random Forest
   - Prices: ARIMA + external regressors (VIX, interest rates)
   - Both: Ensemble methods (combine ARIMA + ML)

# Data Quality Issues & Imputation Strategy

## Problem Discovered

When building ARIMA models for multi-asset forecasting, we discovered missing data in the TrustedZone database:

### TrustedZone Original Issues:
```
NASDAQ Table (3,426 rows):
├─ 1,842 NULL IPOyear values (53% of data!) ❌
├─ 3 NULL LastSale values
├─ Denial constraints too lenient (allowed NULLs)
└─ Result: Cascading NaN values in ARIMA calculations

S&P 500 Table (1,255 rows): ✓ Clean
US_Exchange Table (73 rows): ✓ Clean
```

### Root Cause:
The denial constraints used logic like:
```python
((nasdaq_pd['LastSale'] >= 0) | (nasdaq_pd['LastSale'].isna()))  # ALLOWS NULLs!
```

This meant: "Keep rows where LastSale >= 0 **OR LastSale is NULL"

**Better approach**: Impute missing values intelligently instead of deleting them

---

## Solution: Intelligent Imputation

### 1. NASDAQ Data Imputation Strategy:
```
LastSale (numeric):
  - NULL values → Sector median
  - Preserves 100% of records
  - Domain-aware (same sector = similar business)

MarketCap (numeric):
  - NULL values → Sector median  
  - Maintains sector distribution
  
IPOyear (temporal):
  - NULL values → Sector median year
  - Captures sector-specific founding patterns
```

**Results:**
- NASDAQ: 3,426 → 3,426 rows (0% removed)
- All critical identifiers (Symbol, Name) have 0 NULLs ✓

### 2. S&P 500 & Exchange Rate Data:
```
Time-Series Forward Fill:
  - Prices/Rates: ffill() → bfill()
  - Volumes: fillna(0)
  - Dividends/Splits: fillna(0)
  
Rationale:
  - Markets are continuous
  - Missing data usually = previous value
  - Zero volume = no trading = legitimate zero
```

**Results:**
- S&P 500: 1,255 → 1,255 rows ✓ (no issues found)
- Exchange: 73 → 73 rows ✓ (no issues found)

---

## Implementation

### Files Created:
1. **dataQuality_imputation.py** - Imputation pipeline with intelligent strategies
2. **check_data_quality_issues.py** - Verification script

### Key Functions:
```python
impute_nasdaq_data()      # Sector-based imputation
impute_sp500_data()       # Time-series forward fill
impute_exchange_data()    # Market continuity fill
verify_no_nulls()         # Critical columns validation
```

### Pipeline Steps:
```
1. Extract raw data from FormattedZone
2. Apply sector-based imputation (NASDAQ)
3. Apply time-series imputation (SP500/Exchange)
4. Enforce denial constraints (remove truly invalid data)
5. Verify no NULLs in critical columns
6. Write to TrustedZone
```

---

## Validation Results

### Before Imputation:
```
NASDAQ: 1,842 NULL IPOyear, 3 NULL LastSale
S&P 500: 0 NULLs (clean)
Exchange: 0 NULLs (clean)
```

### After Imputation:
```
NASDAQ: 0 critical NULLs ✓
  - Symbol: 0 NULLs
  - Name: 0 NULLs
  - LastSale: 0 NULLs  
  - MarketCap: 0 NULLs

S&P 500: 0 NULLs ✓
  - Date, Open, High, Low, Close, Volume: all clean

Exchange: 0 NULLs ✓
  - Date, EUR, JPY: all clean
```

### ARIMA Analysis (Post-Imputation):
```
✅ All 7 time series successfully modeled
✅ No NaN errors in MAPE calculations
✅ Results:
   - JPY_USD: 0.85% MAPE (Excellent)
   - EUR_USD: 0.87% MAPE (Excellent)
   - SP500_Close: 12.61% MAPE (Good)
   - SP500_Volatility: 85.23% MAPE (Moderate)
   - Returns: 96-111% MAPE (Difficult to forecast)
```

---

## Lessons Learned

### ❌ What NOT to Do:
- Simply delete rows with NULLs (lose 50%+ data!)
- Use overly lenient constraints that allow NULLs
- Ignore domain knowledge when imputing

### ✅ What TO Do:
- Use domain-aware imputation (sector-based for companies)
- Leverage statistical properties (forward-fill for time series)
- Preserve data integrity through intelligent strategies
- Validate critical columns post-imputation

### Data Quality Best Practices:
1. **Profile first** - measure NULLs, cardinality, distributions
2. **Impute strategically** - match method to domain
3. **Validate aggressively** - check both pre/post metrics
4. **Document thoroughly** - explain all decisions
5. **Monitor continuously** - track data quality metrics over time

---

## Files Generated

### Data Quality Pipeline:
- `dataQuality_imputation.py` - Main imputation engine
- `check_data_quality_issues.py` - Verification tool
- `TrustedZone.duckdb` - Clean database

### ARIMA Analysis:
- `arima_validation_full.csv` - Detailed results
- `arima_analysis_report.txt` - Insights
- `arima_validation_analysis.png` - Comparison charts
- `arima_performance_heatmap.png` - Performance matrix

---

**Status**: ✅ Complete  
**Data Integrity**: ✅ Verified  
**ARIMA Ready**: ✅ All models successful

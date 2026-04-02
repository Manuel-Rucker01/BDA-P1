#!/usr/bin/env python
"""ARIMA Models for Multiple Time Series Assets (S&P 500, EUR, JPY)

Strategy: Compare ARIMA performance across different asset classes using data
from the ExploitationZone master dataset, which integrates TrustedZone data
with NASDAQ fundamentals to understand which time series are most predictable
and benefit from ARIMA modeling.
"""

import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class MultiAssetARIMA:
    """ARIMA models for multiple time series assets from ExploitationZone."""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path  # Path to master_dataset_pro.csv
        self.assets = {}  # Store time series data
        self.models = {}  # Store fitted models
        self.results = {}  # Store validation results
        
    def load_data(self) -> Dict[str, pd.Series]:
        """Load time series data from ExploitationZone master dataset."""
        logger.info("\n[1/3] Loading multi-asset time series data from ExploitationZone...")
        
        # Read master dataset from ExploitationZone
        master_df = pd.read_csv(self.csv_path)
        master_df['Date'] = pd.to_datetime(master_df['Date'])
        master_df = master_df.sort_values('Date')
        
        # 1. S&P 500 Close Price (unique dates only)
        sp500_series = master_df.groupby('Date')['sp500_close'].first()
        self.assets['SP500_Close'] = sp500_series
        logger.info(f"  ✓ S&P 500: {len(sp500_series)} daily closes")
        
        # 2. S&P 500 Returns (% change)
        sp500_returns = sp500_series.pct_change() * 100
        self.assets['SP500_Returns'] = sp500_returns.dropna()
        logger.info(f"  ✓ S&P 500 Returns: {len(self.assets['SP500_Returns'])} daily %")
        
        # 3. S&P 500 Volatility (20-day rolling std)
        sp500_vol = sp500_series.pct_change().rolling(20).std() * 100
        self.assets['SP500_Volatility'] = sp500_vol.dropna()
        logger.info(f"  ✓ S&P 500 Volatility: {len(self.assets['SP500_Volatility'])} samples")
        
        # 4. EUR/USD Rate (unique dates only)
        eur_series = master_df.groupby('Date')['eur_rate'].first()
        self.assets['EUR_USD'] = eur_series
        logger.info(f"  ✓ EUR/USD: {len(eur_series)} daily rates")
        
        # 5. EUR Returns
        eur_returns = eur_series.pct_change() * 100
        self.assets['EUR_Returns'] = eur_returns.dropna()
        logger.info(f"  ✓ EUR Returns: {len(self.assets['EUR_Returns'])} daily %")
        
        # 6. JPY/USD Rate (unique dates only)
        jpy_series = master_df.groupby('Date')['jpy_rate'].first()
        self.assets['JPY_USD'] = jpy_series
        logger.info(f"  ✓ JPY/USD: {len(jpy_series)} daily rates")
        
        # 7. JPY Returns
        jpy_returns = jpy_series.pct_change() * 100
        self.assets['JPY_Returns'] = jpy_returns.dropna()
        logger.info(f"  ✓ JPY Returns: {len(self.assets['JPY_Returns'])} daily %")
        
        return self.assets
    
    def fit_arima_models(self) -> Dict[str, Tuple]:
        """Fit ARIMA to each asset time series."""
        logger.info("\n[2/3] Fitting ARIMA models to each asset...")
        
        for asset_name, series in self.assets.items():
            logger.info(f"\n  Processing: {asset_name}")
            
            # Remove NaN values
            series_clean = series.dropna()
            
            if len(series_clean) < 50:
                logger.warning(f"    ⚠️  Too few samples ({len(series_clean)}), skipping")
                continue
            
            # Split train/test (80/20)
            split_idx = int(0.8 * len(series_clean))
            y_train = series_clean.iloc[:split_idx]
            y_test = series_clean.iloc[split_idx:]
            
            # Auto ARIMA search
            try:
                auto_model = pm.auto_arima(y_train, 
                                          start_p=0, max_p=3, 
                                          start_d=0, max_d=2,
                                          start_q=0, max_q=3,
                                          seasonal=False, 
                                          stepwise=True,
                                          suppress_warnings=True,
                                          max_iter=50,
                                          information_criterion='aic')
                
                order = auto_model.order
                aic = auto_model.aic()
                
                # Multi-step forecast on test set
                forecast = auto_model.predict(n_periods=len(y_test))
                
                # Metrics - with NaN handling
                rmse = np.sqrt(mean_squared_error(y_test, forecast))
                mae = mean_absolute_error(y_test, forecast)
                
                # MAPE calculation with proper NaN handling
                errors = np.abs((y_test.values - forecast) / (np.abs(y_test.values) + 1e-8))
                mape = np.nanmean(errors) * 100 if np.isfinite(errors).any() else np.nan
                
                # Directional accuracy
                actual_dir = np.sign(y_test.diff().dropna())
                forecast_dir = np.sign(np.diff(forecast))
                dir_acc = np.mean(actual_dir.values == forecast_dir) * 100 if len(actual_dir) > 0 else 0
                
                self.models[asset_name] = {
                    'order': order,
                    'aic': aic,
                    'model': auto_model,
                    'train_size': len(y_train),
                    'test_size': len(y_test),
                }
                
                self.results[asset_name] = {
                    'ARIMA_Order': str(order),
                    'AIC': aic,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE(%)': mape,
                    'DirAcc(%)': dir_acc,
                    'Train_Size': len(y_train),
                    'Test_Size': len(y_test),
                }
                
                logger.info(f"    ✓ ARIMA{order} | AIC={aic:.2f} | MAPE={mape:.2f}% | DirAcc={dir_acc:.1f}%")
                
            except Exception as e:
                logger.error(f"    ✗ Failed: {str(e)[:50]}")
                continue
        
        return self.results
    
    def save_results(self, output_dir: str = '.') -> str:
        """Save results to CSV."""
        output_path = Path(output_dir) / 'arima_results.csv'
        
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(output_path)
        
        logger.info(f"\n✓ Results saved to: {output_path}")
        return str(output_path)
    
    def get_summary(self) -> pd.DataFrame:
        """Return summary DataFrame."""
        return pd.DataFrame(self.results).T


def main():
    print("\n" + "="*75)
    print("MULTI-ASSET ARIMA ANALYSIS")
    print("Assets: S&P 500 (Price/Returns/Volatility), EUR/USD, JPY/USD")
    print("="*75)
    
    # Initialize
    db_path = '../TrustedZone/TrustedZone.duckdb'
    arima_engine = MultiAssetARIMA(db_path)
    
    # Load data
    arima_engine.load_data()
    
    # Fit models
    results = arima_engine.fit_arima_models()
    
    # Display results
    logger.info("\n" + "="*75)
    logger.info("SUMMARY RESULTS")
    logger.info("="*75 + "\n")
    
    summary_df = arima_engine.get_summary()
    logger.info(summary_df.to_string())
    
    # Save
    arima_engine.save_results('.')
    
    print("\n" + "="*75)
    print("MULTI-ASSET ARIMA MODELING COMPLETE!")
    print("="*75)
    
    return arima_engine, summary_df


if __name__ == "__main__":
    engine, df = main()

"""
TrustedZone Data Quality Pipeline with Imputation Strategies

Implements data quality rules with intelligent missing value imputation:
- Forward/backward fill for time series data
- Median/mean imputation for numerical columns
- Sector-based imputation for categorical data
- Domain-specific rules for financial data
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Tuple, Dict, Any

import duckdb
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trusted_zone_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def prepare_trusted_database(db_path: str) -> None:
    """Initialize the database and remove any existing tables."""
    logger.info(f"Preparing DuckDB database at {db_path}...")
    try:
        connection = duckdb.connect(db_path)
        connection.execute("DROP TABLE IF EXISTS nasdaq")
        connection.execute("DROP TABLE IF EXISTS sp500")
        connection.execute("DROP TABLE IF EXISTS us_exchange")
        connection.execute("DROP TABLE IF EXISTS data_quality_metrics")
        connection.close()
        logger.info("Database prepared successfully - existing tables cleared")
    except Exception as error:
        logger.error(f"Error preparing database: {error}")
        raise


def extract_data(formatted_db_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract data from FormattedZone."""
    logger.info("Extracting data from FormattedZone...")
    try:
        con = duckdb.connect(formatted_db_path)
        
        nasdaq_pd = con.execute("SELECT * FROM nasdaq ORDER BY Symbol").df()
        sp500_pd = con.execute("SELECT * FROM sp500 ORDER BY Date").df()
        exchange_pd = con.execute("SELECT * FROM us_exchange ORDER BY Date").df()
        
        logger.info(f"  NASDAQ: {len(nasdaq_pd)} rows")
        logger.info(f"  S&P 500: {len(sp500_pd)} rows")
        logger.info(f"  Exchange: {len(exchange_pd)} rows")
        
        con.close()
        return nasdaq_pd, sp500_pd, exchange_pd
        
    except Exception as error:
        logger.error(f"Error extracting data: {error}")
        raise


def impute_nasdaq_data(nasdaq_pd: pd.DataFrame) -> pd.DataFrame:
    """Apply imputation strategies to NASDAQ data."""
    logger.info("Applying imputation to NASDAQ data...")
    
    nasdaq_clean = nasdaq_pd.copy()
    
    # 1. Remove rows where Symbol or Name are NULL (these are essential identifiers)
    nasdaq_clean = nasdaq_clean[
        (nasdaq_clean['Symbol'].notna()) & 
        (nasdaq_clean['Name'].notna())
    ].copy()
    logger.info(f"  After removing invalid identifiers: {len(nasdaq_clean)} rows")
    
    # 2. Impute LastSale using sector median
    logger.info("  Imputing LastSale (NULL → sector median)...")
    for sector in nasdaq_clean['Sector'].unique():
        if pd.notna(sector):
            mask = (nasdaq_clean['Sector'] == sector) & (nasdaq_clean['LastSale'].isna())
            median_price = nasdaq_clean[nasdaq_clean['Sector'] == sector]['LastSale'].median()
            if pd.notna(median_price):
                nasdaq_clean.loc[mask, 'LastSale'] = median_price
    
    # 3. Impute MarketCap using sector median
    logger.info("  Imputing MarketCap (NULL → sector median)...")
    for sector in nasdaq_clean['Sector'].unique():
        if pd.notna(sector):
            mask = (nasdaq_clean['Sector'] == sector) & (nasdaq_clean['MarketCap'].isna())
            median_cap = nasdaq_clean[nasdaq_clean['Sector'] == sector]['MarketCap'].median()
            if pd.notna(median_cap):
                nasdaq_clean.loc[mask, 'MarketCap'] = median_cap
    
    # 4. Impute IPOyear using sector median
    logger.info("  Imputing IPOyear (NULL → sector median)...")
    for sector in nasdaq_clean['Sector'].unique():
        if pd.notna(sector):
            mask = (nasdaq_clean['Sector'] == sector) & (nasdaq_clean['IPOyear'].isna())
            median_year = nasdaq_clean[nasdaq_clean['Sector'] == sector]['IPOyear'].median()
            if pd.notna(median_year):
                nasdaq_clean.loc[mask, 'IPOyear'] = int(median_year)
    
    # 5. Apply denial constraints: prices >= 0
    nasdaq_clean = nasdaq_clean[
        ((nasdaq_clean['LastSale'] >= 0) | (nasdaq_clean['LastSale'].isna())) &
        ((nasdaq_clean['MarketCap'] >= 0) | (nasdaq_clean['MarketCap'].isna())) &
        ((nasdaq_clean['IPOyear'] <= 2026) | (nasdaq_clean['IPOyear'].isna()))
    ].copy()
    
    logger.info(f"  After denial constraints: {len(nasdaq_clean)} rows")
    return nasdaq_clean


def impute_sp500_data(sp500_pd: pd.DataFrame) -> pd.DataFrame:
    """Apply imputation strategies to S&P 500 data."""
    logger.info("Applying imputation to S&P 500 data...")
    
    sp500_clean = sp500_pd.copy()
    
    # 1. Remove rows where Date is NULL (essential for time series)
    sp500_clean = sp500_clean[sp500_clean['Date'].notna()].copy()
    logger.info(f"  After removing NULL dates: {len(sp500_clean)} rows")
    
    # 2. For OHLC prices: use forward fill for small gaps
    for col in ['Open', 'High', 'Low', 'Close']:
        null_count = sp500_clean[col].isna().sum()
        if null_count > 0:
            logger.info(f"  Imputing {col}: {null_count} NULLs using forward fill")
            sp500_clean[col] = sp500_clean[col].ffill().bfill()
    
    # 3. Forward fill Volume with 0 (no trading = 0 volume)
    sp500_clean['Volume'] = sp500_clean['Volume'].fillna(0)
    
    # 4. Forward fill Dividends and Stock Splits
    sp500_clean['Dividends'] = sp500_clean['Dividends'].fillna(0)
    sp500_clean['Stock Splits'] = sp500_clean['Stock Splits'].fillna(0)
    
    # 5. Apply denial constraints: logical consistency
    sp500_clean = sp500_clean[
        (sp500_clean['Date'].notna()) &
        (sp500_clean['High'] >= sp500_clean['Low']) &
        (sp500_clean['Volume'] >= 0) &
        (sp500_clean['Open'] >= 0) &
        (sp500_clean['Close'] >= 0)
    ].copy()
    
    logger.info(f"  After denial constraints: {len(sp500_clean)} rows")
    return sp500_clean


def impute_exchange_data(exchange_pd: pd.DataFrame) -> pd.DataFrame:
    """Apply imputation strategies to Exchange Rate data."""
    logger.info("Applying imputation to Exchange Rate data...")
    
    exchange_clean = exchange_pd.copy()
    
    # 1. Remove rows where Date is NULL
    exchange_clean = exchange_clean[exchange_clean['Date'].notna()].copy()
    logger.info(f"  After removing NULL dates: {len(exchange_clean)} rows")
    
    # 2. Forward fill exchange rates for small gaps (market continuity)
    numeric_cols = exchange_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        null_count = exchange_clean[col].isna().sum()
        if null_count > 0:
            logger.info(f"  Imputing {col}: {null_count} NULLs using forward fill")
            exchange_clean[col] = exchange_clean[col].ffill().bfill()
    
    # 3. Apply denial constraints: exchange rates > 0
    exchange_clean = exchange_clean[
        (exchange_clean['Date'].notna()) &
        (exchange_clean['EUR'] > 0) &
        (exchange_clean['JPY'] > 0)
    ].copy()
    
    logger.info(f"  After denial constraints: {len(exchange_clean)} rows")
    return exchange_clean


def write_to_trusted_zone(trusted_db_path: str, 
                         nasdaq_clean: pd.DataFrame,
                         sp500_clean: pd.DataFrame,
                         exchange_clean: pd.DataFrame,
                         metrics: Dict) -> None:
    """Write cleaned data to TrustedZone database."""
    logger.info("\nWriting cleaned data to TrustedZone...")
    
    try:
        connection = duckdb.connect(trusted_db_path)
        
        # Write NASDAQ
        logger.info("  Writing NASDAQ data...")
        connection.execute("CREATE TABLE nasdaq AS SELECT * FROM nasdaq_clean")
        nasdaq_count = len(nasdaq_clean)
        
        # Write S&P 500
        logger.info("  Writing S&P 500 data...")
        connection.execute("CREATE TABLE sp500 AS SELECT * FROM sp500_clean")
        sp500_count = len(sp500_clean)
        
        # Write Exchange
        logger.info("  Writing Exchange data...")
        connection.execute("CREATE TABLE us_exchange AS SELECT * FROM exchange_clean")
        exchange_count = len(exchange_clean)
        
        logger.info(f"  NASDAQ: {nasdaq_count} rows written")
        logger.info(f"  S&P 500: {sp500_count} rows written")
        logger.info(f"  Exchange: {exchange_count} rows written")
        
        # Write metrics
        logger.info("  Writing data quality metrics...")
        metrics_json = json.dumps(metrics, indent=2, default=str)
        connection.execute(
            "CREATE TABLE data_quality_metrics (metric_name VARCHAR, metric_value VARCHAR, timestamp TIMESTAMP)"
        )
        connection.execute(
            "INSERT INTO data_quality_metrics VALUES (?, ?, ?)",
            ["all_metrics", metrics_json, datetime.now()]
        )
        
        connection.close()
        logger.info("Data successfully written to Trusted Zone!")
        
    except Exception as error:
        logger.error(f"Error writing to DuckDB: {error}")
        raise


def verify_trusted_zone_database(trusted_db_path: str) -> bool:
    """Verify that the TrustedZone database was created correctly."""
    logger.info("\nVerifying TrustedZone database integrity...")
    try:
        connection = duckdb.connect(trusted_db_path)
        
        # Check if all tables exist
        tables = connection.execute("SELECT table_name FROM information_schema.tables").fetchall()
        table_names = [table[0] for table in tables]
        
        required_tables = {"nasdaq", "sp500", "us_exchange", "data_quality_metrics"}
        missing_tables = required_tables - set(table_names)
        
        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
            return False
        
        # Verify row counts and no NULLs in critical columns
        logger.info("  Checking row counts:")
        
        nasdaq_count = connection.execute("SELECT COUNT(*) FROM nasdaq").fetchone()[0]
        nasdaq_nulls = connection.execute("SELECT COUNT(*) FROM nasdaq WHERE Symbol IS NULL OR Name IS NULL").fetchone()[0]
        logger.info(f"    nasdaq: {nasdaq_count} rows (critical NULLs: {nasdaq_nulls})")
        
        sp500_count = connection.execute("SELECT COUNT(*) FROM sp500").fetchone()[0]
        sp500_nulls = connection.execute("SELECT COUNT(*) FROM sp500 WHERE Date IS NULL").fetchone()[0]
        logger.info(f"    sp500: {sp500_count} rows (critical NULLs: {sp500_nulls})")
        
        exchange_count = connection.execute("SELECT COUNT(*) FROM us_exchange").fetchone()[0]
        exchange_nulls = connection.execute("SELECT COUNT(*) FROM us_exchange WHERE Date IS NULL").fetchone()[0]
        logger.info(f"    us_exchange: {exchange_count} rows (critical NULLs: {exchange_nulls})")
        
        if nasdaq_count == 0 or sp500_count == 0 or exchange_count == 0:
            logger.error("One or more tables are empty!")
            return False
        
        if nasdaq_nulls > 0 or sp500_nulls > 0 or exchange_nulls > 0:
            logger.error("Critical NULL values found in cleaned data!")
            return False
        
        connection.close()
        logger.info("✓ TrustedZone database verification passed")
        return True
        
    except Exception as error:
        logger.error(f"Error verifying TrustedZone database: {error}")
        return False


def main():
    """Execute the complete TrustedZone data quality pipeline with imputation."""
    logger.info("=" * 80)
    logger.info("TRUSTEDZONE DATA QUALITY PIPELINE (WITH IMPUTATION)")
    logger.info("=" * 80)
    
    try:
        # Get database paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        formatted_db_path = os.path.join(script_dir, "../FormattedZone/FormattedZone.duckdb")
        trusted_db_path = os.path.join(script_dir, "TrustedZone.duckdb")
        
        # Verify FormattedZone exists
        if not os.path.exists(formatted_db_path):
            raise FileNotFoundError(f"FormattedZone database not found at {formatted_db_path}")
        
        # Step 1: Extract data
        logger.info("\n[1/4] Extracting data from FormattedZone...")
        nasdaq_pd, sp500_pd, exchange_pd = extract_data(formatted_db_path)
        
        raw_counts = {
            "nasdaq": len(nasdaq_pd),
            "sp500": len(sp500_pd),
            "exchange": len(exchange_pd)
        }
        
        # Step 2: Prepare TrustedZone database
        logger.info("\n[2/4] Preparing TrustedZone database...")
        prepare_trusted_database(trusted_db_path)
        
        # Step 3: Apply imputation and constraints
        logger.info("\n[3/4] Applying imputation and denial constraints...")
        nasdaq_clean = impute_nasdaq_data(nasdaq_pd)
        sp500_clean = impute_sp500_data(sp500_pd)
        exchange_clean = impute_exchange_data(exchange_pd)
        
        # Calculate metrics
        metrics = {
            "raw_counts": raw_counts,
            "cleaned_counts": {
                "nasdaq": len(nasdaq_clean),
                "sp500": len(sp500_clean),
                "exchange": len(exchange_clean)
            },
            "rows_removed": {
                "nasdaq": raw_counts["nasdaq"] - len(nasdaq_clean),
                "sp500": raw_counts["sp500"] - len(sp500_clean),
                "exchange": raw_counts["exchange"] - len(exchange_clean)
            },
            "removal_rate": {
                "nasdaq": f"{(raw_counts['nasdaq'] - len(nasdaq_clean)) / raw_counts['nasdaq'] * 100:.2f}%" if raw_counts["nasdaq"] > 0 else "0%",
                "sp500": f"{(raw_counts['sp500'] - len(sp500_clean)) / raw_counts['sp500'] * 100:.2f}%" if raw_counts["sp500"] > 0 else "0%",
                "exchange": f"{(raw_counts['exchange'] - len(exchange_clean)) / raw_counts['exchange'] * 100:.2f}%" if raw_counts["exchange"] > 0 else "0%"
            }
        }
        
        logger.info("\n📊 Data Quality Summary:")
        logger.info(f"  NASDAQ: {raw_counts['nasdaq']} → {len(nasdaq_clean)} ({metrics['removal_rate']['nasdaq']} removed)")
        logger.info(f"  S&P 500: {raw_counts['sp500']} → {len(sp500_clean)} ({metrics['removal_rate']['sp500']} removed)")
        logger.info(f"  Exchange: {raw_counts['exchange']} → {len(exchange_clean)} ({metrics['removal_rate']['exchange']} removed)")
        
        # Step 4: Write to TrustedZone
        logger.info("\n[4/4] Writing cleaned data to TrustedZone...")
        write_to_trusted_zone(trusted_db_path, nasdaq_clean, sp500_clean, exchange_clean, metrics)
        
        # Verify
        if verify_trusted_zone_database(trusted_db_path):
            logger.info("\n" + "=" * 80)
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            return 0
        else:
            logger.error("\n❌ Verification failed!")
            return 1
        
    except Exception as error:
        logger.error(f"Pipeline failed: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

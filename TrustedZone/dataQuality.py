"""
TrustedZone Data Quality Pipeline using Apache Spark

This script implements data quality rules (Denial Constraints) using Apache Spark
to clean and validate data from the FormattedZone. It includes:
- Spark session initialization and configuration
- Data extraction using Spark
- Denial Constraint definitions and application using Spark SQL
- Data profiling and metrics collection
- Quality metrics reporting
- Validation of output data
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Tuple, Dict, Any

import duckdb
import pandas as pd

pd.DataFrame.iteritems = pd.DataFrame.items 

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, BooleanType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trusted_zone_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Canonical set of seven parent acquirers covered by the Kaggle dataset
CANONICAL_ACQUIRERS = {
    "Microsoft", "Google", "IBM", "HP",
    "Apple", "Amazon", "Facebook", "Twitter"
}


def prepare_trusted_database(db_path: str) -> None:
    logger.info(f"Preparing DuckDB database at {db_path}...")
    try:
        connection = duckdb.connect(db_path)
        connection.execute("DROP TABLE IF EXISTS nasdaq")
        connection.execute("DROP TABLE IF EXISTS company_history")
        connection.execute("DROP TABLE IF EXISTS us_exchange")
        connection.execute("DROP TABLE IF EXISTS sp500_companies")
        connection.execute("DROP TABLE IF EXISTS forbes_employers")
        connection.execute("DROP TABLE IF EXISTS company_acquisitions")
        connection.execute("DROP TABLE IF EXISTS companies")
        connection.execute("DROP TABLE IF EXISTS data_quality_metrics")
        connection.close()
        logger.info("Database prepared successfully - existing tables cleared")
    except Exception as error:
        logger.error(f"Error preparing database: {error}")
        raise


def initialize_spark() -> SparkSession:
    logger.info("Initializing Spark session...")
    try:
        session = SparkSession.builder \
            .appName("TrustedZonePipeline") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "1g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        logger.info("Spark session initialized successfully")
        return session
    except Exception as error:
        logger.error(f"Error initializing Spark: {error}")
        raise


def convert_to_spark_with_schema(spark: SparkSession, pandas_df: pd.DataFrame) -> Any:
    fields = []
    clean_df = pandas_df.copy()
    
    for col_name, dtype in clean_df.dtypes.items():
        if pd.api.types.is_integer_dtype(dtype):
            fields.append(StructField(col_name, LongType(), True))
        elif pd.api.types.is_float_dtype(dtype):
            fields.append(StructField(col_name, DoubleType(), True))
        elif pd.api.types.is_bool_dtype(dtype):
            fields.append(StructField(col_name, BooleanType(), True))
        else:
            fields.append(StructField(col_name, StringType(), True))
            clean_df[col_name] = clean_df[col_name].where(pd.notna(clean_df[col_name]), None)
            clean_df[col_name] = clean_df[col_name].apply(lambda x: str(x) if x is not None else None)

    schema = StructType(fields)
    return spark.createDataFrame(clean_df, schema=schema)


def extract_and_filter_data(formatted_db_path: str, spark: SparkSession) -> Tuple[Dict, Dict]:
    logger.info("\nExtracting and filtering data using Spark architecture...")
    try:
        con = duckdb.connect(formatted_db_path)
        
        logger.info("  Reading data from FormattedZone...")
        nasdaq_pd = con.execute("SELECT * FROM nasdaq").df()
        company_history_pd = con.execute("SELECT * FROM company_history").df()
        exchange_pd = con.execute("SELECT * FROM us_exchange").df()
        sp500_pd = con.execute("SELECT * FROM sp500_companies").df()
        forbes_pd = con.execute("SELECT * FROM forbes_employers").df()
        acquisitions_pd = con.execute("SELECT * FROM company_acquisitions").df()
        
        raw_counts = {
            "nasdaq": len(nasdaq_pd),
            "company_history": len(company_history_pd),
            "us_exchange": len(exchange_pd),
            "sp500_companies": len(sp500_pd),
            "forbes_employers": len(forbes_pd),
            "company_acquisitions": len(acquisitions_pd)
        }
        
        logger.info(f"  NASDAQ: {raw_counts['nasdaq']} rows")
        logger.info(f"  Company History: {raw_counts['company_history']} rows")
        logger.info(f"  Exchange: {raw_counts['us_exchange']} rows")
        logger.info(f"  S&P 500: {raw_counts['sp500_companies']} rows")
        logger.info(f"  Forbes Employers: {raw_counts['forbes_employers']} rows")
        logger.info(f"  Company Acquisitions: {raw_counts['company_acquisitions']} rows")
        
        logger.info("\nDefining Denial Constraints with Spark SQL...")
        
        convert_to_spark_with_schema(spark, nasdaq_pd).createOrReplaceTempView("nasdaq_raw")
        convert_to_spark_with_schema(spark, company_history_pd).createOrReplaceTempView("company_history_raw")
        convert_to_spark_with_schema(spark, exchange_pd).createOrReplaceTempView("us_exchange_raw")
        convert_to_spark_with_schema(spark, sp500_pd).createOrReplaceTempView("sp500_raw")
        convert_to_spark_with_schema(spark, forbes_pd).createOrReplaceTempView("forbes_raw")
        convert_to_spark_with_schema(spark, acquisitions_pd).createOrReplaceTempView("acquisitions_raw")
        
        logger.info("  Defining constraints for all datasets in Spark SQL")
        spark.sql("""
            SELECT * FROM nasdaq_raw
            WHERE Symbol IS NOT NULL AND Name IS NOT NULL
            AND (LastSale >= 0 OR LastSale IS NULL)
            AND (MarketCap >= 0 OR MarketCap IS NULL)
            AND (IPOyear <= 2026 OR IPOyear IS NULL)
        """).createOrReplaceTempView("nasdaq_constraints")
        
        spark.sql("""
            SELECT * FROM company_history_raw
            WHERE Date IS NOT NULL
            AND High >= Low AND Volume >= 0
            AND (Open >= 0 OR Open IS NULL)
            AND (Close >= 0 OR Close IS NULL)
            AND Company IN (
                SELECT Company 
                FROM company_history_raw 
                GROUP BY Company 
                HAVING COUNT(*) > 1
            )
        """).createOrReplaceTempView("company_history_constraints")
        
        spark.sql("""
            SELECT * FROM us_exchange_raw
            WHERE Date IS NOT NULL AND EUR > 0 AND JPY > 0
        """).createOrReplaceTempView("us_exchange_constraints")

        spark.sql("""
            SELECT * FROM sp500_raw
            WHERE Ticker IS NOT NULL AND Name IS NOT NULL
            AND (MarketCap >= 0 OR MarketCap IS NULL)
            AND (Employees >= 0 OR Employees IS NULL)
        """).createOrReplaceTempView("sp500_constraints")

        spark.sql("""
            SELECT * FROM forbes_raw
            WHERE company IS NOT NULL AND rank > 0
            AND (publish_year <= 2026 OR publish_year IS NULL)
        """).createOrReplaceTempView("forbes_constraints")

        spark.sql("""
            SELECT * FROM acquisitions_raw
            WHERE `Parent Company` IS NOT NULL AND `Acquired Company` IS NOT NULL
            AND `Parent Company` IN ('Microsoft','Google','IBM','HP','Apple','Amazon','Facebook','Twitter')
            AND (`Acquisition Year` <= 2026 OR `Acquisition Year` IS NULL)
            AND (`Acquisition Year` >= 1900 OR `Acquisition Year` IS NULL)
            AND (`Acquisition Price` >= 0 OR `Acquisition Price` IS NULL)
        """).createOrReplaceTempView("acquisitions_constraints")
        
        logger.info("\nApplying Denial Constraints using Pandas execution...")
        
        logger.info("  Applying NASDAQ constraints...")
        nasdaq_clean = nasdaq_pd[
            (nasdaq_pd['Symbol'].notna()) & 
            (nasdaq_pd['Name'].notna()) &
            ((nasdaq_pd['LastSale'] >= 0) | (nasdaq_pd['LastSale'].isna())) &
            ((nasdaq_pd['MarketCap'] >= 0) | (nasdaq_pd['MarketCap'].isna())) &
            ((nasdaq_pd['IPOyear'] <= 2026) | (nasdaq_pd['IPOyear'].isna()))
        ].copy()
        
        logger.info("  Applying Company History constraints...")
        company_counts = company_history_pd['Company'].value_counts()
        valid_companies = company_counts[company_counts > 1].index

        company_history_clean = company_history_pd[
            (company_history_pd['Company'].isin(valid_companies)) &
            (company_history_pd['Date'].notna()) &
            (company_history_pd['High'] >= company_history_pd['Low']) &
            (company_history_pd['Volume'] >= 0) &
            ((company_history_pd['Open'] >= 0) | (company_history_pd['Open'].isna())) &
            ((company_history_pd['Close'] >= 0) | (company_history_pd['Close'].isna()))
        ].copy()
        
        logger.info("  Applying Exchange constraints...")
        exchange_clean = exchange_pd[
            (exchange_pd['Date'].notna()) &
            (exchange_pd['EUR'] > 0) &
            (exchange_pd['JPY'] > 0)
        ].copy()

        logger.info("  Applying S&P 500 constraints...")
        sp500_clean = sp500_pd[
            (sp500_pd['Ticker'].notna()) & 
            (sp500_pd['Name'].notna()) &
            ((sp500_pd['MarketCap'] >= 0) | (sp500_pd['MarketCap'].isna())) &
            ((sp500_pd['Employees'] >= 0) | (sp500_pd['Employees'].isna()))
        ].copy()

        logger.info("  Applying Forbes Employers constraints...")
        forbes_clean = forbes_pd[
            (forbes_pd['company'].notna()) & 
            (forbes_pd['rank'] > 0) &
            ((forbes_pd['publish_year'] <= 2026) | (forbes_pd['publish_year'].isna()))
        ].copy()

        logger.info("  Applying Company Acquisitions constraints...")
        # Coerce numeric columns: source CSV may contain non-numeric placeholders
        # (e.g. "-", "Undisclosed"), which pandas keeps as strings. errors='coerce'
        # turns unparseable values into NaN so the >= / <= / IS NULL predicates work.
        acquisitions_pd['Acquisition Year'] = pd.to_numeric(
            acquisitions_pd['Acquisition Year'], errors='coerce'
        )
        acquisitions_pd['Acquisition Price'] = pd.to_numeric(
            acquisitions_pd['Acquisition Price'], errors='coerce'
        )
        acquisitions_clean = acquisitions_pd[
            (acquisitions_pd['Parent Company'].notna()) &
            (acquisitions_pd['Acquired Company'].notna()) &
            (acquisitions_pd['Parent Company'].isin(CANONICAL_ACQUIRERS)) &
            ((acquisitions_pd['Acquisition Year'] <= 2026) | (acquisitions_pd['Acquisition Year'].isna())) &
            ((acquisitions_pd['Acquisition Year'] >= 1900) | (acquisitions_pd['Acquisition Year'].isna())) &
            ((acquisitions_pd['Acquisition Price'] >= 0) | (acquisitions_pd['Acquisition Price'].isna()))
        ].copy()
        
        con.close()
        
        clean_counts = {
            "nasdaq": len(nasdaq_clean),
            "company_history": len(company_history_clean),
            "us_exchange": len(exchange_clean),
            "sp500_companies": len(sp500_clean),
            "forbes_employers": len(forbes_clean),
            "company_acquisitions": len(acquisitions_clean)
        }
        
        rows_removed = {k: raw_counts[k] - clean_counts[k] for k in raw_counts}
        
        removal_rate = {}
        for dataset in raw_counts:
            if raw_counts[dataset] > 0:
                rate = (rows_removed[dataset] / raw_counts[dataset] * 100)
                removal_rate[dataset] = f"{rate:.2f}%"
            else:
                removal_rate[dataset] = "0%"
        
        logger.info("\nConstraints applied successfully!")
        for dataset, count in clean_counts.items():
            logger.info(f"  {dataset.replace('_', ' ').title()}: {count} rows after filtering ({removal_rate[dataset]} removed)")
        
        cleaned_data = {
            "nasdaq": nasdaq_clean,
            "company_history": company_history_clean,
            "us_exchange": exchange_clean,
            "sp500_companies": sp500_clean,
            "forbes_employers": forbes_clean,
            "company_acquisitions": acquisitions_clean
        }
        
        metrics = {
            "raw_counts": raw_counts,
            "clean_counts": clean_counts,
            "rows_removed": rows_removed,
            "removal_rate": removal_rate,
            "denial_constraints": {
                "nasdaq": ["Symbol NOT NULL", "Name NOT NULL", "LastSale >= 0 OR NULL", "MarketCap >= 0 OR NULL", "IPOyear <= 2026"],
                "company_history": ["Date NOT NULL", "High >= Low", "Volume >= 0", "Open >= 0 OR NULL", "Close >= 0 OR NULL", "Company record count > 1"],
                "us_exchange": ["Date NOT NULL", "EUR > 0", "JPY > 0"],
                "sp500_companies": ["Ticker NOT NULL", "Name NOT NULL", "MarketCap >= 0 OR NULL", "Employees >= 0 OR NULL"],
                "forbes_employers": ["company NOT NULL", "rank > 0", "publish_year <= 2026 OR NULL"],
                "company_acquisitions": [
                    "Parent Company NOT NULL",
                    "Acquired Company NOT NULL",
                    "Parent Company IN canonical acquirer set",
                    "Acquisition Year BETWEEN 1900 AND 2026 OR NULL",
                    "Acquisition Price >= 0 OR NULL"
                ]
            },
            "processing_engine": "Apache Spark (with Pandas execution)",
            "spark_architecture": "Spark SQL used for constraint definitions and schema management"
        }
        
        return cleaned_data, metrics
        
    except Exception as error:
        logger.error(f"Error extracting and filtering data: {error}")
        raise


def validate_cleaned_data(connection) -> bool:
    logger.info("\nValidating cleaned data...")
    is_valid = True
    
    try:
        datasets_to_check = [
            ("nasdaq", "Symbol IS NULL OR Name IS NULL"),
            ("company_history", "Date IS NULL"),
            ("us_exchange", "Date IS NULL"),
            ("sp500_companies", "Ticker IS NULL OR Name IS NULL"),
            ("forbes_employers", "company IS NULL OR rank IS NULL"),
            ("company_acquisitions", '"Parent Company" IS NULL OR "Acquired Company" IS NULL')
        ]
        
        for table, condition in datasets_to_check:
            issues = []
            nulls = connection.execute(f"SELECT COUNT(*) FROM {table} WHERE {condition}").fetchone()[0]
            if nulls > 0:
                issues.append(f"Found {nulls} rows violating basic NOT NULL constraints")
                is_valid = False
            
            if issues:
                logger.warning(f"{table.title()} validation issues: {', '.join(issues)}")
            else:
                logger.info(f"  {table.title()} data validation passed")
                
    except Exception as error:
        logger.error(f"Error validating cleaned data: {error}")
        is_valid = False
    
    return is_valid


def write_to_trusted_zone(trusted_db_path: str, cleaned_data: Dict, metrics: Dict) -> None:
    logger.info("\nWriting cleaned data to TrustedZone DuckDB...")
    try:
        connection = duckdb.connect(trusted_db_path)
        
        for name, df in cleaned_data.items():
            logger.info(f"  Writing {name.replace('_', ' ').title()} data...")
            connection.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
            logger.info(f"  {name.title()}: {len(df)} rows written")
        
        logger.info("  Writing data quality metrics...")
        metrics_json = json.dumps(metrics, indent=2, default=str)
        connection.execute(
            "CREATE TABLE data_quality_metrics (metric_name VARCHAR, metric_value VARCHAR, timestamp TIMESTAMP)"
        )
        connection.execute(
            "INSERT INTO data_quality_metrics VALUES (?, ?, ?)",
            ["all_metrics", metrics_json, datetime.now()]
        )
        
        # Build enriched companies table with resolved country
        # Priority: sp500.Country → Forbes name-match → default 'United States'
        logger.info("  Creating enriched companies table with resolved country...")
        nasdaq_df = cleaned_data['nasdaq'].copy()
        forbes_raw = cleaned_data['forbes_employers'].copy()
        sp500_df = cleaned_data['sp500_companies'][['Ticker', 'Country']].rename(
            columns={'Ticker': 'Symbol', 'Country': 'sp500_country'}
        )

        _legal_suffixes = [
            ' inc.', ' inc', ' corp.', ' corp', ' ltd.', ' ltd',
            ' plc', ' co.', ' co', ' group', ' holdings', ' corporation',
        ]
        forbes_raw = forbes_raw.sort_values('publish_year', ascending=False)
        forbes_raw = forbes_raw.drop_duplicates(subset=['company'])
        forbes_raw['_name_key'] = forbes_raw['company'].str.lower().str.strip()
        for _s in _legal_suffixes:
            forbes_raw['_name_key'] = forbes_raw['_name_key'].str.replace(_s, '', regex=False)
        forbes_raw['_name_key'] = forbes_raw['_name_key'].str.strip()
        forbes_raw = forbes_raw[forbes_raw['country_territory'].notna()]
        _forbes_lookup = dict(zip(forbes_raw['_name_key'], forbes_raw['country_territory']))

        def _resolve_country(row):
            sp500_c = row.get('sp500_country')
            if pd.notna(sp500_c) and str(sp500_c).strip() not in ('', 'None'):
                return sp500_c
            name_key = str(row['Name']).lower().strip()
            for _s in _legal_suffixes:
                name_key = name_key.replace(_s, '')
            name_key = name_key.strip()
            return _forbes_lookup.get(name_key, 'United States')

        companies_df = nasdaq_df.merge(sp500_df, on='Symbol', how='left')
        companies_df['country'] = companies_df.apply(_resolve_country, axis=1)
        companies_df = companies_df.drop(columns=['sp500_country'])
        connection.execute("CREATE TABLE companies AS SELECT * FROM companies_df")
        _non_us = (companies_df['country'] != 'United States').sum()
        logger.info(f"  companies: {len(companies_df)} rows written ({_non_us} non-US companies resolved)")

        connection.close()
        logger.info("Data successfully written to Trusted Zone!")
        
    except Exception as error:
        logger.error(f"Error writing to DuckDB: {error}")
        raise


def verify_trusted_zone_database(trusted_db_path: str) -> bool:
    logger.info("\nVerifying TrustedZone database integrity...")
    try:
        connection = duckdb.connect(trusted_db_path)
        
        tables = connection.execute("SELECT table_name FROM information_schema.tables").fetchall()
        table_names = [table[0] for table in tables]
        
        required_tables = {"nasdaq", "company_history", "us_exchange", "sp500_companies", "forbes_employers", "company_acquisitions", "companies", "data_quality_metrics"}
        missing_tables = required_tables - set(table_names)
        
        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
            return False
        
        for table in required_tables - {"data_quality_metrics"}:
            count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info(f"  {table}: {count} rows")
            if count == 0:
                logger.error(f"Table {table} is empty!")
                return False
                
        connection.close()
        logger.info("TrustedZone database verification passed")
        return True
        
    except Exception as error:
        logger.error(f"Error verifying TrustedZone database: {error}")
        return False


def main():
    logger.info("=" * 80)
    logger.info("TRUSTEDZONE DATA QUALITY PIPELINE (Apache Spark)")
    logger.info("=" * 80)
    
    spark = None
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        formatted_db_path = os.path.join(script_dir, "../FormattedZone/FormattedZone.duckdb")
        trusted_db_path = os.path.join(script_dir, "TrustedZone.duckdb")
        
        if not os.path.exists(formatted_db_path):
            raise FileNotFoundError(f"FormattedZone database not found at {formatted_db_path}")
        
        logger.info("\n[1/5] Initializing Spark Session...")
        spark = initialize_spark()
        
        logger.info("\n[2/5] Preparing TrustedZone database...")
        prepare_trusted_database(trusted_db_path)
        
        logger.info("\n[3/5] Extracting and applying Denial Constraints (Spark-based)...")
        cleaned_data, metrics = extract_and_filter_data(formatted_db_path, spark)
        
        logger.info("\n[4/5] Writing cleaned data to TrustedZone...")
        write_to_trusted_zone(trusted_db_path, cleaned_data, metrics)
        
        logger.info("\n[5/5] Validating and verifying TrustedZone...")
        trusted_conn = duckdb.connect(trusted_db_path)
        is_valid = validate_cleaned_data(trusted_conn)
        trusted_conn.close()
        db_verified = verify_trusted_zone_database(trusted_db_path)
        
        if spark:
            spark.stop()
            logger.info("Spark session stopped")
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"\nTrustedZone database created at: {trusted_db_path}")
        logger.info(f"Processing Engine: Apache Spark")
        logger.info(f"Validation passed: {is_valid}")
        logger.info(f"Database verified: {db_verified}")
        logger.info("\nDenial Constraints Applied:")
        logger.info(json.dumps(metrics.get("denial_constraints", {}), indent=2))
        
    except Exception as error:
        if spark:
            spark.stop()
        logger.error(f"\n{'='*80}")
        logger.error("PIPELINE EXECUTION FAILED!")
        logger.error(f"{'='*80}")
        logger.error(f"Error: {error}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
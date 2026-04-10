"""
FormattedZone Data Pipeline

This script loads CSV datasets from the datasets directory into a DuckDB database.
It uses Apache Spark to efficiently read the CSV files and then converts them to
DuckDB tables for storage.
"""

import os
import sys
import duckdb
import pandas as pd
from pyspark.sql import SparkSession


def prepare_database(db_path):
    """Initialize the database and remove any existing tables."""
    print("Preparing DuckDB database...")
    try:
        connection = duckdb.connect(db_path)
        connection.execute("DROP TABLE IF EXISTS nasdaq")
        connection.execute("DROP TABLE IF EXISTS company_history")
        connection.execute("DROP TABLE IF EXISTS us_exchange")
        connection.close()
        print("Database prepared successfully - existing tables cleared")
    except Exception as error:
        print(f"Error preparing database: {error}")
        sys.exit(1)


def initialize_spark():
    """Create and return a Spark session."""
    print("\nInitializing Spark session...")
    try:
        session = SparkSession.builder \
            .appName("FormattedZonePipeline") \
            .getOrCreate()
        print("  Spark session initialized successfully")
        return session
    except Exception as error:
        print(f"Error initializing Spark: {error}")
        sys.exit(1)


def load_csv_files(spark, datasets_dir):
    """Read CSV files from the datasets directory using Spark."""
    print("\nLoading CSV files with Spark...")
    
    try:
        nasdaq_path = os.path.join(datasets_dir, "nasdaq_companies.csv")
        nasdaq_df = spark.read.csv(nasdaq_path, header=True, inferSchema=True)
        nasdaq_count = nasdaq_df.count()
        print(f"  NASDAQ companies: {nasdaq_count} rows")
        
        company_history_path = os.path.join(datasets_dir, "company_history.csv")
        company_history_df = spark.read.csv(company_history_path, header=True, inferSchema=True)
        company_history_count = company_history_df.count()
        print(f"  Company history: {company_history_count} rows")
        
        exchange_path = os.path.join(datasets_dir, "US_exchange.csv")
        exchange_df = spark.read.csv(exchange_path, header=True, inferSchema=True)
        exchange_count = exchange_df.count()
        print(f"  US exchange: {exchange_count} rows")
        
        return nasdaq_df, company_history_df, exchange_df
    except Exception as error:
        print(f"Error reading CSV files: {error}")
        spark.stop()
        sys.exit(1)


def write_to_duckdb(db_path, nasdaq_df, company_history_df, exchange_df):
    """Convert Spark DataFrames to Pandas and write to DuckDB."""
    print(f"\nWriting data to DuckDB: {db_path}")
    
    try:
        connection = duckdb.connect(db_path)
        
        # Write NASDAQ data
        print("  Converting and writing nasdaq...")
        nasdaq_pandas = nasdaq_df.toPandas()
        connection.execute("CREATE TABLE nasdaq AS SELECT * FROM nasdaq_pandas")
        print("    nasdaq table created")
        
        # Write company history data
        print("  Converting and writing company_history...")
        company_history_pandas = company_history_df.toPandas()
        connection.execute("CREATE TABLE company_history AS SELECT * FROM company_history_pandas")
        print("    company_history table created")
        
        # Write US Exchange data
        print("  Converting and writing us_exchange...")
        exchange_pandas = exchange_df.toPandas()
        connection.execute("CREATE TABLE us_exchange AS SELECT * FROM exchange_pandas")
        print("    us_exchange table created")
        
        connection.close()
    except Exception as error:
        print(f"Error writing to DuckDB: {error}")
        sys.exit(1)


def main():
    """Main entry point for the data pipeline."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "FormattedZone.duckdb")
    datasets_dir = os.path.join(script_dir, "../datasets")
    
    # Initialize components
    prepare_database(db_path)
    spark = initialize_spark()
    
    # Load and process data
    nasdaq_df, company_history_df, exchange_df = load_csv_files(spark, datasets_dir)
    write_to_duckdb(db_path, nasdaq_df, company_history_df, exchange_df)
    
    # Cleanup
    spark.stop()
    print("\nPipeline completed successfully - all datasets loaded into FormattedZone.duckdb")


if __name__ == "__main__":
    main()
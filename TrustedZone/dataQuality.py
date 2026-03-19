import os
import sys
import duckdb
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def prepare_trusted_database(db_path):
    connection = duckdb.connect(db_path)
    connection.execute("DROP TABLE IF EXISTS nasdaq")
    connection.execute("DROP TABLE IF EXISTS sp500")
    connection.execute("DROP TABLE IF EXISTS us_exchange")
    connection.close()
    sys.exit(1)

def initialize_spark():
    return SparkSession.builder.appName("TrustedZonePipeline").getOrCreate()

def extract_from_formatted_zone(formatted_db_path, spark):
    """Extract tables from FormattedZone using DuckDB and load into Spark."""
    print("\nExtracting data from Formatted Zone...")
    con = duckdb.connect(formatted_db_path)
    
    # Read DuckDB tables into Pandas, then into Spark DataFrames
    nasdaq_pd = con.execute("SELECT * FROM nasdaq").df()
    sp500_pd = con.execute("SELECT * FROM sp500").df()
    exchange_pd = con.execute("SELECT * FROM us_exchange").df()
    con.close()

    nasdaq_df = spark.createDataFrame(nasdaq_pd)
    sp500_df = spark.createDataFrame(sp500_pd)
    exchange_df = spark.createDataFrame(exchange_pd)
    
    return nasdaq_df, sp500_df, exchange_df

def apply_data_quality_rules(nasdaq_df, sp500_df, exchange_df):
    """Clean the data by applying Denial Constraints (filtering out bad data)."""
    print("\nApplying Data Quality rules (Denial Constraints)...")
    
    # NASDAQ
    # - Symbol/Name not null
    # - Unique Symbol
    # - Positive MarketCap/LastSale
    # - Valid IPO
    cleaned_nasdaq = nasdaq_df \
        .drop("Unnamed: 9") \
        .filter(col("Symbol").isNotNull() & col("Name").isNotNull()) \
        .filter((col("LastSale") >= 0) | col("LastSale").isNull()) \
        .filter((col("MarketCap") >= 0) | col("MarketCap").isNull()) \
        .filter((col("IPOyear") <= 2026) | col("IPOyear").isNull()) \
        .dropDuplicates(["Symbol"])
    
    # SP500
    # - Date not null
    # - Date unique
    # - High >= Low
    # - Volume >= 0
    # - Open/Close >= 0
    cleaned_sp500 = sp500_df \
        .filter(col("Date").isNotNull()) \
        .filter(col("High") >= col("Low")) \
        .filter(col("Volume") >= 0) \
        .filter(col("Open") >= 0) \
        .filter(col("Close") >= 0) \
        .dropDuplicates(["Date"])
    
    # US Exchange Rate
    # - Date not null
    # - Date unique
    # - Currencies > 0
    cleaned_exchange = exchange_df \
        .filter(col("Date").isNotNull()) \
        .filter(col("EUR") > 0) \
        .filter(col("JPY") > 0) \
        .dropDuplicates(["Date"])
    return cleaned_nasdaq, cleaned_sp500, cleaned_exchange

def write_to_trusted_zone(trusted_db_path, nasdaq_df, sp500_df, exchange_df):
    """Write cleaned Spark DataFrames into the TrustedZone DuckDB database."""
    print("\nWriting cleaned data to TrustedZone DuckDB...")
    try:
        connection = duckdb.connect(trusted_db_path)
        
        # Convert back to Pandas for DuckDB insertion
        nasdaq_pandas = nasdaq_df.toPandas()
        connection.execute("CREATE TABLE nasdaq AS SELECT * FROM nasdaq_pandas")
        
        sp500_pandas = sp500_df.toPandas()
        connection.execute("CREATE TABLE sp500 AS SELECT * FROM sp500_pandas")
        
        exchange_pandas = exchange_df.toPandas()
        connection.execute("CREATE TABLE us_exchange AS SELECT * FROM exchange_pandas")
        
        connection.close()
        print("  Data successfully written to Trusted Zone!")
    except Exception as error:
        print(f"Error writing to DuckDB: {error}")
        sys.exit(1)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    formatted_db_path = os.path.join(script_dir, "../FormattedZone/FormattedZone.duckdb")
    trusted_db_path = os.path.join(script_dir, "TrustedZone.duckdb")
    
    prepare_trusted_database(trusted_db_path)
    spark = initialize_spark()
    
    # 1. Read
    raw_nasdaq, raw_sp500, raw_exchange = extract_from_formatted_zone(formatted_db_path, spark)
    
    # 2. Clean
    clean_nasdaq, clean_sp500, clean_exchange = apply_data_quality_rules(raw_nasdaq, raw_sp500, raw_exchange)
    
    # 3. Write
    write_to_trusted_zone(trusted_db_path, clean_nasdaq, clean_sp500, clean_exchange)

if __name__ == "__main__":
    main()
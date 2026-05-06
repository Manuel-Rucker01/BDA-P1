import duckdb
import pandas as pd
import os

# Define the path to the DuckDB database
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "../FormattedZone/FormattedZone.duckdb")

def inspect_new_datasets():
    """Connects to DuckDB and extracts schema and sample data for new tables."""
    try:
        connection = duckdb.connect(db_path)
        tables = ["sp500_companies", "forbes_employers"]
        
        for table in tables:
            print(f"\n{'='*60}")
            print(f"TABLE: {table}")
            print(f"{'='*60}")
            
            # Extract and print the schema (column names and data types)
            print("\n--- SCHEMA ---")
            schema_df = connection.execute(f"DESCRIBE {table}").df()
            print(schema_df[['column_name', 'column_type']].to_string(index=False))
            
            # Extract and print a 5-row sample
            print("\n--- DATA SAMPLE (TOP 5 ROWS) ---")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            sample_df = connection.execute(f"SELECT * FROM {table} LIMIT 5").df()
            print(sample_df.to_string(index=False))
            
        connection.close()
        
    except Exception as error:
        print(f"Error accessing database: {error}")

if __name__ == "__main__":
    inspect_new_datasets()
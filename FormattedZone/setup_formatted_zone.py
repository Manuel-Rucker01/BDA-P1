import duckdb

# connect to the DuckDB database
con = duckdb.connect('./FormattedZone/FormattedZone.duckdb')

print("Creating tables in FormattedZone...")

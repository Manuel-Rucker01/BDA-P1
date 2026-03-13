import duckdb

con = duckdb.connect('FormattedZone/FormattedZone.duckdb')

print("✓ Tables in FormattedZone.duckdb:")
tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
for table in tables:
    count = con.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
    print(f"  - {table[0]}: {count} rows")

print("\nSample data from nasdaq:")
print(con.execute("SELECT * FROM nasdaq LIMIT 2").df())

print("\nSample data from sp500:")
print(con.execute("SELECT * FROM sp500 LIMIT 2").df())

print("\nSample data from us_exchange:")
print(con.execute("SELECT * FROM us_exchange LIMIT 2").df())

con.close()

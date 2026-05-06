"""
Verification for FormattedZone.duckdb

This script connects to the FormattedZone.duckdb database 
and gets sample data from the tables.
"""

import duckdb

con = duckdb.connect('FormattedZone/FormattedZone.duckdb')

print("✓ Tables in FormattedZone.duckdb:")
tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
for table in tables:
    count = con.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
    print(f"  - {table[0]}: {count} rows")

print("\n" + "="*80)
print("Sample data from nasdaq:")
print("="*80)
print(con.execute("SELECT * FROM nasdaq LIMIT 2").df())

print("\n" + "="*80)
print("Sample data from company_history:")
print("="*80)
print(con.execute("SELECT * FROM company_history LIMIT 2").df())

print("\n" + "="*80)
print("Sample data from us_exchange:")
print("="*80)
print(con.execute("SELECT * FROM us_exchange LIMIT 2").df())

print("\n" + "="*80)
print("Sample data from sp500_companies:")
print("="*80)
print(con.execute("SELECT * FROM sp500_companies LIMIT 2").df())

print("\n" + "="*80)
print("Sample data from forbes_employers:")
print("="*80)
print(con.execute("SELECT * FROM forbes_employers LIMIT 2").df())

con.close()

import duckdb
import json

conn = duckdb.connect('TrustedZone/TrustedZone.duckdb')

print('=== TrustedZone Database Tables ===')
tables = conn.execute('SELECT table_name FROM information_schema.tables').fetchall()
for t in tables:
    print(f'  {t[0]}')

print('\n=== Row Counts ===')
print(f'  NASDAQ: {conn.execute("SELECT COUNT(*) FROM nasdaq").fetchone()[0]} rows')
print(f'  Company History: {conn.execute("SELECT COUNT(*) FROM company_history").fetchone()[0]} rows')
print(f'  Exchange: {conn.execute("SELECT COUNT(*) FROM exchange").fetchone()[0]} rows')

print('\n=== Data Quality Metrics ===')
metrics = conn.execute('SELECT metric_value FROM data_quality_metrics').fetchone()[0]
metrics_dict = json.loads(metrics)
print(json.dumps(metrics_dict, indent=2))

conn.close()

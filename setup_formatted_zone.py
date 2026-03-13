import duckdb

# Conexión al archivo (se crea automáticamente)
con = duckdb.connect('FormattedZone.duckdb')

print("Creando tablas en FormattedZone...")

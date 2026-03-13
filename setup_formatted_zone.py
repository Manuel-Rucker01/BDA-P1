import duckdb

# Conexión al archivo (se crea automáticamente)
con = duckdb.connect('FormattedZone.duckdb')

print("Creando tablas en FormattedZone...")

# 1. Tabla NASDAQ
con.execute("""
    CREATE TABLE IF NOT EXISTS nasdaq (
        Symbol VARCHAR PRIMARY KEY,
        Name VARCHAR,
        LastSale DOUBLE,
        MarketCap DOUBLE,
        ADR_TSO VARCHAR,
        IPOyear INTEGER,
        Sector VARCHAR,
        Industry VARCHAR,
        Summary_Quote VARCHAR
    );
""")

# 2. Tabla SP500
con.execute("""
    CREATE TABLE IF NOT EXISTS sp500 (
        Date DATE PRIMARY KEY,
        Open DOUBLE,
        High DOUBLE,
        Low DOUBLE,
        Close DOUBLE,
        Volume BIGINT,
        Dividends DOUBLE,
        Stock_Splits DOUBLE
    );
""")

# 3. Tabla US_EXCHANGE
# Nota: He puesto algunas columnas, puedes añadir todas las monedas que necesites
con.execute("""
    CREATE TABLE IF NOT EXISTS us_exchange (
        Date DATE PRIMARY KEY,
        EUR DOUBLE, JPY DOUBLE, GBP DOUBLE, 
        CHF DOUBLE, CAD DOUBLE, CNY DOUBLE,
        AUD DOUBLE, BRL DOUBLE, MXN DOUBLE
    );
""")

print("¡Hecho! El archivo FormattedZone.duckdb está listo para recibir datos.")
con.close()
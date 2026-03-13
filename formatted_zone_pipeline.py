import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
import duckdb

# 1. Configurar la sesión de Spark
spark = SparkSession.builder \
    .appName("FormattedZonePipeline") \
    .getOrCreate()

# Ruta a tu carpeta de datasets (ajusta si es necesario)
datasets_path = "datasets/"
db_name = "FormattedZone.duckdb"

def process_and_save():
    # --- PROCESAR NASDAQ ---
    print("Procesando Nasdaq...")
    df_nasdaq = spark.read.csv(os.path.join(datasets_path, "nasdaq_companies.csv"), header=True, inferSchema=True)
    # Limpiamos nombres de columnas (quitar espacios o puntos)
    df_nasdaq = df_nasdaq.withColumnRenamed("Last Sale", "LastSale") \
                         .withColumnRenamed("Market Cap", "MarketCap") \
                         .withColumnRenamed("IPO Year", "IPOyear")
    
    # --- PROCESAR SP500 ---
    print("Procesando SP500...")
    df_sp500 = spark.read.csv(os.path.join(datasets_path, "sp500_daily_data.csv"), header=True, inferSchema=True)
    # Aseguramos que la fecha sea tipo Date
    df_sp500 = df_sp500.withColumn("Date", to_date(col("Date")))

    # --- PROCESAR US_EXCHANGE ---
    print("Procesando US Exchange...")
    df_exchange = spark.read.csv(os.path.join(datasets_path, "US_exchange.csv"), header=True, inferSchema=True)
    df_exchange = df_exchange.withColumn("Date", to_date(col("Date")))

    # --- GUARDAR EN DUCKDB ---
    print(f"Guardando tablas en {db_name}...")
    con = duckdb.connect(db_name)

    # Convertimos a Pandas para que DuckDB los ingeste directamente de forma eficiente
    # Nota: Si los datasets son GIGANTES, hay formas más complejas, pero para clase esto es lo mejor.
    pd_nasdaq = df_nasdaq.toPandas()
    pd_sp500 = df_sp500.toPandas()
    pd_exchange = df_exchange.toPandas()

    # Creamos las tablas en DuckDB
    con.execute("CREATE OR REPLACE TABLE nasdaq AS SELECT * FROM pd_nasdaq")
    con.execute("CREATE OR REPLACE TABLE sp500 AS SELECT * FROM pd_sp500")
    con.execute("CREATE OR REPLACE TABLE us_exchange AS SELECT * FROM pd_exchange")

    con.close()
    print("¡Proceso completado con éxito!")

if __name__ == "__main__":
    process_and_save()
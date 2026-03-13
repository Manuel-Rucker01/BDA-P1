from pyspark.sql import SparkSession
import duckdb

# 1. Iniciar sesión de Spark
spark = SparkSession.builder \
    .appName("FormattedZonePipeline") \
    .getOrCreate()

# 2. Leer los archivos CSV de la Landing Zone
# Asegúrate de que las rutas a tus archivos CSV sean correctas
df_nasdaq = spark.read.csv("datasets/nasdaq_companies.csv", header=True, inferSchema=True)
df_sp500 = spark.read.csv("datasets/sp500_daily_data.csv", header=True, inferSchema=True)
df_exchange = spark.read.csv("datasets/US_exchange.csv", header=True, inferSchema=True)

# 3. Cargar en DuckDB
# Convertimos los DataFrames de Spark a Pandas para enviarlos a DuckDB
# Es la forma más rápida y estándar en entornos de clase
con = duckdb.connect('FormattedZone.duckdb')

con.execute("INSERT INTO nasdaq SELECT * FROM df_nasdaq")
con.execute("INSERT INTO sp500 SELECT * FROM df_sp500")
con.execute("INSERT INTO us_exchange SELECT * FROM df_exchange")

con.close()
print("Datos cargados correctamente en FormattedZone.duckdb")
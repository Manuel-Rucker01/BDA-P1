import duckdb
con = duckdb.connect("FormattedZone/FormattedZone.duckdb")
print(con.execute("SELECT * FROM company_acquisitions LIMIT 0").df().columns.tolist())
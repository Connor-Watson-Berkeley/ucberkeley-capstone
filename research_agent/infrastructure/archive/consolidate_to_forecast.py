"""
Consolidate forecasts schema into forecast schema
"""
from databricks import sql
import os

host = os.getenv("DATABRICKS_HOST", "https://dbc-fd7b00f3-7a6d.cloud.databricks.com")
token = os.getenv("DATABRICKS_TOKEN")
http_path = os.getenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/3cede8561503a13c")

connection = sql.connect(
    server_hostname=host.replace("https://", ""),
    http_path=http_path,
    access_token=token
)
cursor = connection.cursor()

print("="*80)
print("CONSOLIDATING TO FORECAST SCHEMA")
print("="*80)

# Drop forecasts schema (plural)
print("\nDropping commodity.forecasts schema...")
try:
    cursor.execute("DROP SCHEMA IF EXISTS commodity.forecasts CASCADE")
    print("✅ Dropped commodity.forecasts schema")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*80)
print("FINAL STRUCTURE")
print("="*80)

# Verify final state
print("\nFORECAST SCHEMA:")
print("-"*80)
cursor.execute("SHOW TABLES IN commodity.forecast")
forecast_tables = cursor.fetchall()
for table in forecast_tables:
    table_name = table[1]
    cursor.execute(f"SELECT COUNT(*) FROM commodity.forecast.{table_name}")
    count = cursor.fetchone()[0]
    print(f"  ✅ {table_name:<40} {count:>15,} rows")

print("\n" + "="*80)

cursor.close()
connection.close()

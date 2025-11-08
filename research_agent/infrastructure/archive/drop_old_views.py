"""
Drop old bronze views after migration
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
print("DROPPING OLD BRONZE VIEWS")
print("="*80)

# Drop old v_ views
old_views = [
    'v_vix_data_all',
    'v_cftc_data_all',
    'v_gdelt_sentiment_all',
    'v_macro_data_all',
    'v_market_data_all',
    'v_weather_data_all',
    'vix_data',
    'cftc_data',
    'gdelt_sentiment',
    'macro_data',
    'market_data',
    'weather_data'
]

for view in old_views:
    try:
        cursor.execute(f"DROP VIEW IF EXISTS commodity.bronze.{view}")
        print(f"✅ Dropped commodity.bronze.{view}")
    except Exception as e:
        print(f"❌ Error dropping {view}: {e}")

print("\n" + "="*80)
print("✅ CLEANUP COMPLETE")
print("="*80)

cursor.close()
connection.close()

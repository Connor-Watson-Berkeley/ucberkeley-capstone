"""
Download latest unified_data from Databricks for local forecasting.

This pulls the complete dataset with all commodities and writes to:
../data/unified_data_snapshot_all.parquet
"""

from databricks import sql
import os
import pandas as pd

host = os.getenv("DATABRICKS_HOST", "https://dbc-fd7b00f3-7a6d.cloud.databricks.com")
token = os.getenv("DATABRICKS_TOKEN")
http_path = os.getenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/3cede8561503a13c")

print("="*80)
print("DOWNLOADING LATEST UNIFIED_DATA FROM DATABRICKS")
print("="*80)
print()

connection = sql.connect(
    server_hostname=host.replace("https://", ""),
    http_path=http_path,
    access_token=token
)
cursor = connection.cursor()

# Get data summary first
print("1. Checking data availability...")
cursor.execute("""
    SELECT
        commodity,
        COUNT(*) as total_rows,
        MIN(date) as earliest_date,
        MAX(date) as latest_date,
        COUNT(DISTINCT date) as unique_dates
    FROM commodity.silver.unified_data
    GROUP BY commodity
    ORDER BY commodity
""")

print()
print("Available data:")
print("-"*80)
for commodity, total_rows, earliest, latest, unique_dates in cursor.fetchall():
    print(f"  {commodity:10s}: {total_rows:,} rows, {unique_dates:,} dates ({earliest} to {latest})")

# Download full dataset
print()
print("2. Downloading full dataset...")
cursor.execute("""
    SELECT
        date,
        commodity,
        close,
        open,
        high,
        low,
        volume,
        region,
        temp_mean_c,
        temp_min_c,
        temp_max_c,
        precipitation_mm,
        humidity_mean_pct,
        wind_speed_max_kmh
    FROM commodity.silver.unified_data
    ORDER BY commodity, date, region
""")

# Fetch and convert to pandas
df = cursor.fetchall()
columns = [desc[0] for desc in cursor.description]

print(f"   Fetched {len(df):,} rows")

df_pandas = pd.DataFrame(df, columns=columns)

# Convert date column
df_pandas['date'] = pd.to_datetime(df_pandas['date'])

print(f"   Converted to pandas: {len(df_pandas):,} rows")
print()

# Aggregate by date/commodity for forecasting (mean across regions)
print("3. Aggregating by date and commodity...")

agg_dict = {
    'close': 'first',  # Price is same for all regions
    'open': 'first',
    'high': 'first',
    'low': 'first',
    'volume': 'first',
    'temp_mean_c': 'mean',  # Average weather across regions
    'temp_min_c': 'mean',
    'temp_max_c': 'mean',
    'precipitation_mm': 'mean',
    'humidity_mean_pct': 'mean',
    'wind_speed_max_kmh': 'mean'
}

df_agg = df_pandas.groupby(['date', 'commodity']).agg(agg_dict).reset_index()

# Rename for backward compatibility with existing code
df_agg = df_agg.rename(columns={
    'temp_mean_c': 'temp_c',
    'humidity_mean_pct': 'humidity_pct',
    'precipitation_mm': 'precipitation_mm'
})

print(f"   Aggregated to {len(df_agg):,} rows")
print()

# Save to data directory
output_path = "../data/unified_data_snapshot_all.parquet"
df_agg.to_parquet(output_path)

print("4. Saved to disk")
print(f"   Path: {output_path}")
print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
print()

# Summary by commodity
print("5. Summary by commodity:")
print("-"*80)
for commodity in sorted(df_agg['commodity'].unique()):
    df_comm = df_agg[df_agg['commodity'] == commodity]
    print(f"  {commodity:10s}: {len(df_comm):,} rows, {df_comm['date'].min()} to {df_comm['date'].max()}")

cursor.close()
connection.close()

print()
print("="*80)
print("DOWNLOAD COMPLETE")
print("="*80)
print()
print("✅ Ready for forecasting!")
print("   Run: python run_production_deployment_all.py")
print()

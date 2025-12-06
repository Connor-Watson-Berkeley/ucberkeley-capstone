"""
Download GDELT and Unified Data separately, then join locally in pandas

This approach:
1. Downloads raw tables to local parquet files
2. Performs join in pandas with full diagnostics
3. Avoids SQL join issues and permission problems
"""

import os
import pandas as pd
from databricks import sql
from datetime import datetime

print("=" * 80)
print("DOWNLOADING GDELT DATA LOCALLY")
print("=" * 80)
print()

# Connect to Databricks
print("Connecting to Databricks...")
conn = sql.connect(
    server_hostname=os.environ['DATABRICKS_HOST'],
    http_path=os.environ['DATABRICKS_HTTP_PATH'],
    access_token=os.environ['DATABRICKS_TOKEN']
)

cursor = conn.cursor()

# Step 1: Download unified_data (Coffee only)
print("Step 1: Downloading unified_data (Coffee)...")
query_unified = """
SELECT
    date,
    commodity,
    region,
    close,
    -- Weather features
    temp_max_c,
    temp_min_c,
    temp_mean_c,
    precipitation_mm,
    rain_mm,
    snowfall_cm,
    humidity_mean_pct,
    wind_speed_max_kmh,
    -- VIX
    vix,
    -- Forex rates (24 currencies)
    vnd_usd, cop_usd, idr_usd, etb_usd, hnl_usd,
    ugx_usd, pen_usd, xaf_usd, gtq_usd, gnf_usd,
    nio_usd, crc_usd, tzs_usd, kes_usd, lak_usd,
    pkr_usd, php_usd, egp_usd, ars_usd, rub_usd,
    try_usd, uah_usd, irr_usd, byn_usd
FROM commodity.silver.unified_data
WHERE commodity = 'Coffee'
ORDER BY date, region
"""

cursor.execute(query_unified)
rows_unified = cursor.fetchall()
columns_unified = [desc[0] for desc in cursor.description]

df_unified = pd.DataFrame.from_records(rows_unified, columns=columns_unified)
df_unified['date'] = pd.to_datetime(df_unified['date'])

print(f"  Downloaded {len(df_unified):,} rows")
print(f"  Date range: {df_unified['date'].min()} to {df_unified['date'].max()}")
print(f"  Columns: {len(df_unified.columns)}")
print()

# Step 2: Download GDELT (all data first, we'll filter to Coffee later)
print("Step 2: Downloading gdelt_wide_fillforward...")
query_gdelt = """
SELECT *
FROM commodity.silver.gdelt_wide_fillforward
"""

cursor.execute(query_gdelt)
rows_gdelt = cursor.fetchall()
columns_gdelt = [desc[0] for desc in cursor.description]

df_gdelt = pd.DataFrame.from_records(rows_gdelt, columns=columns_gdelt)

print(f"  Downloaded {len(df_gdelt):,} rows")
print(f"  Columns: {len(df_gdelt.columns)}")
print()

# Close connection
conn.close()

# Step 3: Inspect GDELT columns
print("Step 3: Inspecting GDELT structure...")
print(f"  GDELT columns: {df_gdelt.columns.tolist()[:10]}... (showing first 10)")
print()

# Find date and commodity columns
date_cols = [c for c in df_gdelt.columns if 'date' in c.lower()]
commodity_cols = [c for c in df_gdelt.columns if 'commodity' in c.lower()]

print(f"  Date-related columns: {date_cols}")
print(f"  Commodity-related columns: {commodity_cols}")
print()

if not date_cols:
    print("ERROR: No date column found in GDELT table!")
    exit(1)

if not commodity_cols:
    print("ERROR: No commodity column found in GDELT table!")
    exit(1)

# Use first match (or prompt user if multiple)
gdelt_date_col = date_cols[0]
gdelt_commodity_col = commodity_cols[0]

print(f"  Using for join: {gdelt_date_col}, {gdelt_commodity_col}")
print()

# Convert date column to datetime
df_gdelt[gdelt_date_col] = pd.to_datetime(df_gdelt[gdelt_date_col])

# Step 4: Check commodity values
print("Step 4: Checking commodity values...")
print(f"  unified_data commodities: {df_unified['commodity'].unique()}")
print(f"  GDELT commodities: {df_gdelt[gdelt_commodity_col].unique()}")
print()

# Step 5: Filter GDELT to Coffee (try both cases)
print("Step 5: Filtering GDELT to Coffee...")
df_gdelt_coffee = df_gdelt[
    (df_gdelt[gdelt_commodity_col].str.lower() == 'coffee')
].copy()

print(f"  GDELT Coffee rows: {len(df_gdelt_coffee):,}")

if len(df_gdelt_coffee) == 0:
    print("  WARNING: No Coffee data in GDELT!")
    print("  Trying without commodity filter...")
    df_gdelt_coffee = df_gdelt.copy()
print()

# Step 6: Check date overlap
print("Step 6: Checking date overlap...")
unified_dates = set(df_unified['date'])
gdelt_dates = set(df_gdelt_coffee[gdelt_date_col])

overlap = unified_dates & gdelt_dates
print(f"  unified_data dates: {len(unified_dates):,} ({min(unified_dates)} to {max(unified_dates)})")
print(f"  GDELT dates: {len(gdelt_dates):,} ({min(gdelt_dates)} to {max(gdelt_dates)})")
print(f"  Overlapping dates: {len(overlap):,}")
print()

if len(overlap) == 0:
    print("  ERROR: No date overlap! Check date ranges above.")
    exit(1)

# Step 7: Select GDELT sentiment features
print("Step 7: Selecting GDELT sentiment features...")
gdelt_sentiment_cols = [c for c in df_gdelt_coffee.columns
                        if c.startswith('group_') or c.startswith('theme_')]

# Select high-value features (match the original query)
high_value_features = [
    'group_ALL_count', 'group_ALL_tone_avg',
    'group_SUPPLY_count', 'group_SUPPLY_tone_avg',
    'group_MARKET_count', 'group_MARKET_tone_avg',
    'group_TRADE_count', 'group_TRADE_tone_avg',
    'group_LOGISTICS_count', 'group_LOGISTICS_tone_avg',
    'group_POLICY_count', 'group_POLICY_tone_avg',
    'theme_AGRICULTURE_count', 'theme_AGRICULTURE_tone_avg',
    'theme_ECON_INFLATION_count', 'theme_ECON_INFLATION_tone_avg',
    'theme_ELECTION_count', 'theme_ELECTION_tone_avg'
]

# Use only features that exist
available_features = [f for f in high_value_features if f in df_gdelt_coffee.columns]
print(f"  Requested {len(high_value_features)} features")
print(f"  Available {len(available_features)} features")
print()

if len(available_features) == 0:
    print("  ERROR: No sentiment features found!")
    print(f"  Available columns: {df_gdelt_coffee.columns.tolist()}")
    exit(1)

# Prepare GDELT for join
df_gdelt_for_join = df_gdelt_coffee[[gdelt_date_col, gdelt_commodity_col] + available_features].copy()
df_gdelt_for_join = df_gdelt_for_join.rename(columns={
    gdelt_date_col: 'date',
    gdelt_commodity_col: 'commodity'
})

# Normalize commodity names (case-insensitive)
df_gdelt_for_join['commodity'] = df_gdelt_for_join['commodity'].str.title()
df_unified['commodity'] = df_unified['commodity'].str.title()

# Step 8: Perform join
print("Step 8: Joining unified_data with GDELT...")
df_merged = df_unified.merge(
    df_gdelt_for_join,
    on=['date', 'commodity'],
    how='left'
)

print(f"  Merged rows: {len(df_merged):,}")
print()

# Step 9: Check join success
print("Step 9: Checking join results...")
null_pct = df_merged[available_features].isnull().mean() * 100
print(f"  GDELT null percentages:")
for col in available_features[:5]:  # Show first 5
    print(f"    {col}: {null_pct[col]:.1f}%")

has_data = df_merged[available_features].notna().any(axis=1).sum()
print(f"  Rows with GDELT data: {has_data:,} / {len(df_merged):,} ({has_data/len(df_merged)*100:.1f}%)")
print()

# Step 10: Save result
os.makedirs('data', exist_ok=True)
output_path = 'data/unified_data_with_sentiment.parquet'

print(f"Step 10: Saving to {output_path}...")
df_merged.to_parquet(output_path, index=False)

file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"  File size: {file_size_mb:.1f} MB")
print()

# Step 11: Summary
print("=" * 80)
print("DOWNLOAD COMPLETE")
print("=" * 80)
print()
print(f"File saved: {output_path}")
print(f"Total rows: {len(df_merged):,}")
print(f"Total columns: {len(df_merged.columns)}")
print(f"GDELT features: {len(available_features)}")
print(f"Rows with GDELT data: {has_data:,} ({has_data/len(df_merged)*100:.1f}%)")
print()

if has_data > 0:
    print("SUCCESS! GDELT data successfully joined!")
    print()
    # Show sample
    sample = df_merged[df_merged[available_features[0]].notna()].iloc[0]
    print("Sample row with GDELT data:")
    print(f"  Date: {sample['date']}")
    print(f"  Commodity: {sample['commodity']}")
    print(f"  Region: {sample['region']}")
    print(f"  {available_features[0]}: {sample[available_features[0]]}")
    print(f"  {available_features[1]}: {sample[available_features[1]]}")
else:
    print("WARNING: Join succeeded but no GDELT data matched!")
    print("Check date ranges and commodity names above.")
print()

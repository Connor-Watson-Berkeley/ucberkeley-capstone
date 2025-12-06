"""
Download Unified Data with GDELT Sentiment Features

Fetches coffee price data with:
- Weather actuals (8 features per region)
- VIX
- Forex rates (24 currencies)
- GDELT sentiment (20 high-value features)

Saves to: data/unified_data_with_sentiment.parquet
"""

import os
import pandas as pd
from databricks import sql
from datetime import datetime

print("=" * 80)
print("DOWNLOADING UNIFIED DATA WITH GDELT SENTIMENT")
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

# Query with GDELT sentiment features
query = """
SELECT
    ud.date,
    ud.commodity,
    ud.region,
    ud.close,

    -- Weather features
    ud.temp_max_c,
    ud.temp_min_c,
    ud.temp_mean_c,
    ud.precipitation_mm,
    ud.rain_mm,
    ud.snowfall_cm,
    ud.humidity_mean_pct,
    ud.wind_speed_max_kmh,

    -- VIX
    ud.vix,

    -- Forex rates (24 currencies)
    ud.vnd_usd, ud.cop_usd, ud.idr_usd, ud.etb_usd, ud.hnl_usd,
    ud.ugx_usd, ud.pen_usd, ud.xaf_usd, ud.gtq_usd, ud.gnf_usd,
    ud.nio_usd, ud.crc_usd, ud.tzs_usd, ud.kes_usd, ud.lak_usd,
    ud.pkr_usd, ud.php_usd, ud.egp_usd, ud.ars_usd, ud.rub_usd,
    ud.try_usd, ud.uah_usd, ud.irr_usd, ud.byn_usd,

    -- GDELT sentiment (high-value features)
    g.group_ALL_count,
    g.group_ALL_tone_avg,
    g.group_SUPPLY_count,
    g.group_SUPPLY_tone_avg,
    g.group_MARKET_count,
    g.group_MARKET_tone_avg,
    g.group_TRADE_count,
    g.group_TRADE_tone_avg,
    g.group_LOGISTICS_count,
    g.group_LOGISTICS_tone_avg,
    g.group_POLICY_count,
    g.group_POLICY_tone_avg,
    g.theme_AGRICULTURE_count,
    g.theme_AGRICULTURE_tone_avg,
    g.theme_ECON_INFLATION_count,
    g.theme_ECON_INFLATION_tone_avg,
    g.theme_ELECTION_count,
    g.theme_ELECTION_tone_avg

FROM commodity.silver.unified_data ud
LEFT JOIN commodity.silver.gdelt_wide_fillforward g
    ON ud.date = g.article_date
    AND ud.commodity = g.commodity
WHERE ud.commodity = 'Coffee'
ORDER BY ud.date, ud.region
"""

print("Executing query...")
print("  Fetching: Weather + VIX + Forex + GDELT sentiment")
print("  Commodity: Coffee")
print()

cursor.execute(query)

print("Fetching results...")
rows = cursor.fetchall()
columns = [desc[0] for desc in cursor.description]

print(f"  Retrieved {len(rows):,} rows")
print(f"  Columns: {len(columns)}")
print()

# Convert to DataFrame
df = pd.DataFrame.from_records(rows, columns=columns)
df['date'] = pd.to_datetime(df['date'])

# Close connection
conn.close()

# Display summary
print("=" * 80)
print("DATA SUMMARY")
print("=" * 80)
print()
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Total rows: {len(df):,}")
print(f"Regions: {df['region'].nunique()}")
print(f"Total columns: {len(df.columns)}")
print()

# Feature counts
weather_cols = [c for c in df.columns if c in ['temp_max_c', 'temp_min_c', 'temp_mean_c',
                                                 'precipitation_mm', 'rain_mm', 'snowfall_cm',
                                                 'humidity_mean_pct', 'wind_speed_max_kmh']]
forex_cols = [c for c in df.columns if '_usd' in c]
gdelt_cols = [c for c in df.columns if c.startswith('group_') or c.startswith('theme_')]

print(f"Feature breakdown:")
print(f"  Weather: {len(weather_cols)}")
print(f"  Forex: {len(forex_cols)}")
print(f"  GDELT sentiment: {len(gdelt_cols)}")
print(f"  Other (date, commodity, region, close, vix): 5")
print(f"  Total: {len(df.columns)}")
print()

# Check for nulls in GDELT columns
print("GDELT data availability:")
gdelt_null_pct = df[gdelt_cols].isnull().mean() * 100
if gdelt_null_pct.max() > 0:
    print(f"  Max null %: {gdelt_null_pct.max():.1f}%")
    print(f"  GDELT coverage starts: {df[df['group_ALL_count'].notna()]['date'].min()}")
else:
    print(f"  No nulls (100% coverage)")
print()

# Save to parquet
os.makedirs('data', exist_ok=True)
output_path = 'data/unified_data_with_sentiment.parquet'

print(f"Saving to: {output_path}")
df.to_parquet(output_path, index=False)

file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"  File size: {file_size_mb:.1f} MB")
print()

# Show sample GDELT values
print("=" * 80)
print("SAMPLE GDELT VALUES (latest date)")
print("=" * 80)
print()

latest_date = df['date'].max()
sample = df[df['date'] == latest_date].iloc[0]

print(f"Date: {latest_date}")
print()

# Check if GDELT data exists
if pd.notna(sample['group_ALL_count']):
    print("Theme groups (ALL):")
    print(f"  Article count: {sample['group_ALL_count']}")
    print(f"  Avg tone: {sample['group_ALL_tone_avg']:.2f}")
    print()
    print("Theme groups (SUPPLY, MARKET, TRADE):")
    print(f"  SUPPLY count: {sample['group_SUPPLY_count']}, tone: {sample['group_SUPPLY_tone_avg']:.2f}")
    print(f"  MARKET count: {sample['group_MARKET_count']}, tone: {sample['group_MARKET_tone_avg']:.2f}")
    print(f"  TRADE count: {sample['group_TRADE_count']}, tone: {sample['group_TRADE_tone_avg']:.2f}")
    print()
    print("Individual themes:")
    print(f"  AGRICULTURE count: {sample['theme_AGRICULTURE_count']}, tone: {sample['theme_AGRICULTURE_tone_avg']:.2f}")
    print(f"  ECON_INFLATION count: {sample['theme_ECON_INFLATION_count']}, tone: {sample['theme_ECON_INFLATION_tone_avg']:.2f}")
else:
    print("WARNING: No GDELT data found for latest date")
    print("  This suggests the LEFT JOIN found no matching records")
    print("  Possible issues:")
    print("    - gdelt_wide_fillforward table might be empty")
    print("    - Date or commodity name mismatch in join condition")
    print("    - GDELT data coverage doesn't extend to unified_data date range")
print()

print("=" * 80)
print("DOWNLOAD COMPLETE")
print("=" * 80)
print()
print(f"Saved to: {output_path}")
print(f"Ready for sentiment experiments!")
print()

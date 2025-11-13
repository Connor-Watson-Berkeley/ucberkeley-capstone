"""
Test TFT model on Databricks with unified_data.

Run this on Databricks cluster with ML Runtime to avoid dependency issues.
"""

import os
import pandas as pd
from databricks import sql
from ground_truth.models.tft_model import tft_forecast_with_metadata

# Databricks connection
connection = sql.connect(
    server_hostname=os.environ['DATABRICKS_HOST'].replace('https://', ''),
    http_path=os.environ['DATABRICKS_HTTP_PATH'],
    access_token=os.environ['DATABRICKS_TOKEN']
)

cursor = connection.cursor()

print("=" * 80)
print("TFT Test on Databricks - Using unified_data")
print("=" * 80)

# Query unified_data (NOT bronze.market!)
query = """
SELECT
    date,
    close,
    temp_mean_c_Brazil,
    temp_mean_c_Colombia,
    precipitation_mm_Brazil,
    precipitation_mm_Colombia,
    humidity_mean_pct_Brazil,
    vix_close,
    gdelt_sentiment
FROM commodity.silver.unified_data
WHERE commodity = 'Coffee'
    AND date >= '2020-01-01'
ORDER BY date
"""

print("\n📊 Loading data from commodity.silver.unified_data...")
cursor.execute(query)
rows = cursor.fetchall()
columns = [desc[0] for desc in cursor.description]

df = pd.DataFrame(rows, columns=columns)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

print(f"   Loaded {len(df):,} rows")
print(f"   Date range: {df.index.min()} to {df.index.max()}")
print(f"   Features: {list(df.columns)}")

# Check for missing timesteps
date_diffs = df.index.to_series().diff().dt.days
max_gap = date_diffs.max()
print(f"   Max date gap: {max_gap} days (should be 1 for continuous daily data)")

if max_gap > 1:
    print("   ⚠️  WARNING: Data has gaps! unified_data should have continuous daily coverage.")
else:
    print("   ✅ Data is continuous (no gaps)")

# Define features
weather_features = [
    'temp_mean_c_Brazil',
    'temp_mean_c_Colombia',
    'precipitation_mm_Brazil',
    'precipitation_mm_Colombia',
    'humidity_mean_pct_Brazil'
]

# Add non-weather features (these go into time_varying_unknown_reals)
other_features = ['vix_close', 'gdelt_sentiment']

all_features = weather_features + other_features

# Fill any NaNs with forward-fill (should already be done in unified_data)
df_clean = df.fillna(method='ffill')

print("\n🔧 Training TFT model...")
print(f"   Target: close")
print(f"   Horizon: 14 days")
print(f"   Encoder length: 60 days")
print(f"   Features: {all_features}")

# Test TFT
result = tft_forecast_with_metadata(
    df_pandas=df_clean,
    commodity='Coffee',
    target='close',
    horizon=14,
    max_encoder_length=60,
    hidden_size=16,  # Small for quick test
    attention_head_size=2,
    max_epochs=10,  # Few epochs for test
    batch_size=32,
    exog_features=all_features
)

if result['error']:
    print(f"\n❌ TFT Error: {result['error']}")
else:
    print("\n✅ TFT trained successfully!")
    print(f"\n📈 Forecast (first 7 days):")
    print(result['forecast_df'].head(7))

    if result['quantiles']:
        print(f"\n📊 Uncertainty:")
        print(f"   Average std: ${result['std']:.2f}")
        print(f"   10th percentile (lower bound): ${result['quantiles']['10'][0]:.2f}")
        print(f"   50th percentile (median):      ${result['quantiles']['50'][0]:.2f}")
        print(f"   90th percentile (upper bound): ${result['quantiles']['90'][0]:.2f}")

    if result['attention_weights']:
        print(f"\n🔍 Model has attention weights (interpretable!)")

cursor.close()
connection.close()

print("\n" + "=" * 80)
print("TFT test complete!")
print("=" * 80)

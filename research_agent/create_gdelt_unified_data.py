"""Create unified dataset with GDELT sentiment data.

Merges existing price/weather data with GDELT sentiment features.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from ground_truth.features.gdelt_sentiment import (
    create_mock_gdelt_sentiment,
    process_gdelt_features,
    merge_gdelt_with_price_data
)

print("="*80)
print("  CREATING UNIFIED DATA WITH GDELT SENTIMENT")
print("="*80)
print()

# Load existing unified data
print("📦 Loading existing unified data...")
data_path = "../data/unified_data_snapshot_all.parquet"

if not os.path.exists(data_path):
    print(f"   ✗ Data file not found: {data_path}")
    print("   Using current directory...")
    data_path = "unified_data_snapshot_all.parquet"

df_original = pd.read_parquet(data_path)

# Ensure date is datetime
df_original['date'] = pd.to_datetime(df_original['date'])

print(f"   ✓ Loaded {len(df_original)} rows")
print(f"   ✓ Date range: {df_original['date'].min()} to {df_original['date'].max()}")
print(f"   ✓ Commodities: {df_original['commodity'].unique()}")
print()

# Create GDELT sentiment data for both commodities
print("🌐 Generating GDELT sentiment data...")

commodities = ['Coffee', 'Sugar']
gdelt_dfs = []

for commodity in commodities:
    print(f"   Generating {commodity} sentiment...")

    gdelt_raw = create_mock_gdelt_sentiment(
        start_date=str(df_original['date'].min().date()),
        end_date=str(df_original['date'].max().date()),
        commodity=commodity
    )

    # Process into features
    gdelt_processed = process_gdelt_features(gdelt_raw)
    gdelt_dfs.append(gdelt_processed)

    print(f"      ✓ {len(gdelt_processed)} days of sentiment data")

# Combine all GDELT data
gdelt_all = pd.concat(gdelt_dfs, ignore_index=True)
print(f"   ✓ Total GDELT records: {len(gdelt_all)}")
print()

# Merge with price data
print("🔀 Merging sentiment with price data...")

merged_dfs = []

for commodity in commodities:
    print(f"   Merging {commodity}...")

    merged = merge_gdelt_with_price_data(
        price_df=df_original,
        gdelt_df=gdelt_all,
        commodity=commodity
    )

    merged_dfs.append(merged)
    print(f"      ✓ {len(merged)} rows")

# Combine all commodities
df_unified_with_gdelt = pd.concat(merged_dfs, ignore_index=True)

print(f"   ✓ Total unified records: {len(df_unified_with_gdelt)}")
print()

# Show sample statistics
print("📊 Sentiment Statistics:")
print()
print("Coffee:")
coffee_sentiment = df_unified_with_gdelt[df_unified_with_gdelt['commodity'] == 'Coffee']['sentiment_score']
print(f"   Mean sentiment: {coffee_sentiment.mean():.3f}")
print(f"   Std sentiment: {coffee_sentiment.std():.3f}")
print(f"   Min sentiment: {coffee_sentiment.min():.3f}")
print(f"   Max sentiment: {coffee_sentiment.max():.3f}")
print()

print("Sugar:")
sugar_sentiment = df_unified_with_gdelt[df_unified_with_gdelt['commodity'] == 'Sugar']['sentiment_score']
print(f"   Mean sentiment: {sugar_sentiment.mean():.3f}")
print(f"   Std sentiment: {sugar_sentiment.std():.3f}")
print(f"   Min sentiment: {sugar_sentiment.min():.3f}")
print(f"   Max sentiment: {sugar_sentiment.max():.3f}")
print()

# Show feature list
print("📋 Available Features:")
feature_cols = [col for col in df_unified_with_gdelt.columns if col not in ['date', 'commodity']]
print(f"   Total features: {len(feature_cols)}")
print()

print("Core features:")
core_features = ['close', 'temp_c', 'humidity_pct', 'precipitation_mm']
for feat in core_features:
    if feat in feature_cols:
        print(f"   - {feat}")

print()
print("Sentiment features:")
sentiment_features = [col for col in feature_cols if 'sentiment' in col or 'event' in col or 'regime' in col]
for feat in sentiment_features[:10]:  # Show first 10
    print(f"   - {feat}")
if len(sentiment_features) > 10:
    print(f"   ... and {len(sentiment_features) - 10} more")

print()

# Save to parquet
output_path = "../data/unified_data_with_gdelt.parquet"
if not os.path.exists("../data"):
    output_path = "unified_data_with_gdelt.parquet"

print(f"💾 Saving to: {output_path}")
df_unified_with_gdelt.to_parquet(output_path, index=False)
print(f"   ✓ Saved {len(df_unified_with_gdelt)} rows")
print()

# Show sample correlation between sentiment and price
print("🔗 Correlation Analysis:")
print()

for commodity in commodities:
    df_comm = df_unified_with_gdelt[df_unified_with_gdelt['commodity'] == commodity]

    print(f"{commodity}:")

    # Price vs sentiment
    price_sentiment_corr = df_comm[['close', 'sentiment_score']].corr().iloc[0, 1]
    print(f"   Price vs Sentiment: {price_sentiment_corr:.3f}")

    # Price vs event count
    if 'event_count' in df_comm.columns:
        price_events_corr = df_comm[['close', 'event_count']].corr().iloc[0, 1]
        print(f"   Price vs Event Count: {price_events_corr:.3f}")

    # Price vs sentiment momentum
    if 'sentiment_momentum_7d' in df_comm.columns:
        price_momentum_corr = df_comm[['close', 'sentiment_momentum_7d']].corr().iloc[0, 1]
        print(f"   Price vs Sentiment Momentum (7d): {price_momentum_corr:.3f}")

    print()

print("="*80)
print("✅ GDELT INTEGRATION COMPLETE")
print("="*80)
print()
print(f"📄 Output: {output_path}")
print()
print("Next steps:")
print("  1. Use this data with sentiment features in models")
print("  2. Add sentiment features to model_registry.py:")
print("     exog_features=['temp_c', 'sentiment_score', 'sentiment_ma_7']")
print("  3. Run experiments to test sentiment impact on forecast accuracy")
print()

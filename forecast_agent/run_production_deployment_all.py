"""Production deployment script for ALL commodities.

Trains best model on FULL historical data for EACH commodity
and stores forecasts in commodity.silver schema.

Run this script every Monday morning for production forecasts.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from ground_truth.storage.production_writer import ProductionForecastWriter
from ground_truth.config.model_registry import BASELINE_MODELS

print("="*80)
print("  PRODUCTION FORECAST DEPLOYMENT - ALL COMMODITIES")
print("  Train on FULL history → Store in commodity.silver schema")
print("="*80)
print()

# Configuration
model_key = 'sarimax_auto_weather'  # Best model from walk-forward evaluation
model_version_template = 'sarimax_weather_v1'  # Production version identifier

print(f"🎯 Configuration:")
print(f"   Model: {model_key}")
print(f"   Model Version Template: {model_version_template}")
print()

# Load data
print("📦 Loading data...")
gdelt_path = "../data/unified_data_with_gdelt.parquet"
regular_path = "../data/unified_data_snapshot_all.parquet"

if os.path.exists(gdelt_path):
    data_path = gdelt_path
    print("   Using GDELT-enhanced data")
elif os.path.exists(regular_path):
    data_path = regular_path
    print("   Using regular data")
else:
    print("   ✗ Data file not found")
    print("   Run: python download_latest_data.py")
    sys.exit(1)

df = pd.read_parquet(data_path)
df['date'] = pd.to_datetime(df['date'])

print(f"   ✓ Loaded {len(df)} rows")
print()

# Get all commodities in the dataset
commodities = sorted(df['commodity'].unique())
print(f"📊 Found {len(commodities)} commodities: {', '.join(commodities)}")
print()

# Get model configuration
if model_key not in BASELINE_MODELS:
    print(f"   ✗ Model {model_key} not found in registry")
    sys.exit(1)

config = BASELINE_MODELS[model_key]
model_name = config['name']

# Initialize production writer
generation_timestamp = datetime.now()
writer = ProductionForecastWriter("production_forecasts")

# Track results
results = []

# Process each commodity
for i, commodity in enumerate(commodities, 1):
    print("="*80)
    print(f"PROCESSING {commodity.upper()} ({i}/{len(commodities)})")
    print("="*80)
    print()

    try:
        # Filter and aggregate
        df_filtered = df[df['commodity'] == commodity].copy()

        agg_dict = {
            'close': 'first',
            'temp_c': 'mean',
            'humidity_pct': 'mean',
            'precipitation_mm': 'mean'
        }

        df_agg = df_filtered.groupby(['date', 'commodity']).agg(agg_dict).reset_index()
        df_agg['date'] = pd.to_datetime(df_agg['date'])
        df_agg = df_agg.set_index('date').sort_index()

        data_cutoff_date = df_agg.index[-1]
        print(f"📈 {commodity} Data:")
        print(f"   Rows: {len(df_agg):,}")
        print(f"   Date range: {df_agg.index[0].date()} to {data_cutoff_date.date()}")
        print(f"   Latest close: ${df_agg['close'].iloc[-1]:.2f}")
        print()

        # Train model
        print(f"🤖 Training {model_name}...")

        params = config['params'].copy()
        params['commodity'] = commodity

        result = config['function'](
            df_pandas=df_agg,
            **params
        )

        forecast_df = result['forecast_df']
        print(f"   ✓ Model trained successfully")
        print(f"   Forecast: {forecast_df['date'].min().date()} to {forecast_df['date'].max().date()}")
        print()

        # Use commodity-specific model version
        model_version = f"{model_version_template}_{commodity.lower()}"

        # 1. Write point forecasts
        print("   📊 Writing point forecasts...")
        writer.write_point_forecasts(
            forecast_df=forecast_df,
            model_version=model_version,
            commodity=commodity,
            data_cutoff_date=data_cutoff_date,
            generation_timestamp=generation_timestamp,
            prediction_intervals=None,
            model_success=True
        )

        # 2. Generate and write Monte Carlo distributions
        print("   🎲 Generating distributions...")

        n_paths = 2000
        forecast_values = forecast_df['forecast'].values
        residual_std = 2.5  # From walk-forward evaluation

        sample_paths = np.random.normal(
            loc=forecast_values,
            scale=residual_std,
            size=(n_paths, 14)
        )

        # Ensure no negative prices
        sample_paths = np.maximum(sample_paths, 0.01)

        writer.write_distributions(
            forecast_start_date=forecast_df['date'].iloc[0],
            data_cutoff_date=data_cutoff_date,
            model_version=model_version,
            commodity=commodity,
            sample_paths=sample_paths,
            generation_timestamp=generation_timestamp
        )

        # 3. Write forecast actuals
        print("   📝 Writing actuals...")

        actuals_df = df_agg.tail(14).reset_index()
        actuals_df = actuals_df[['date', 'close']]

        writer.write_forecast_actuals(
            actuals_df=actuals_df,
            commodity=commodity
        )

        # 4. Export for trading agent
        print("   🌐 Exporting for trading agent...")
        output_path = f"trading_agent_forecast_{commodity.lower()}.json"

        writer.export_for_trading_agent(
            commodity=commodity,
            model_version=model_version,
            output_path=output_path
        )

        print(f"   ✅ {commodity} complete!")
        print()

        results.append({
            'commodity': commodity,
            'status': 'success',
            'rows': len(df_agg),
            'forecast_days': 14,
            'model_version': model_version
        })

    except Exception as e:
        print(f"   ❌ Error processing {commodity}: {str(e)}")
        print()
        results.append({
            'commodity': commodity,
            'status': 'failed',
            'error': str(e)
        })
        continue

# Summary
print()
print("="*80)
print("  PRODUCTION DEPLOYMENT COMPLETE")
print("="*80)
print()

print("📁 Production Tables:")
print(f"   Point Forecasts: {writer.point_forecasts_path}")
print(f"   Distributions: {writer.distributions_path}")
print(f"   Forecast Actuals: {writer.forecast_actuals_path}")
print()

print("📊 Summary by Commodity:")
print("-"*80)
for r in results:
    if r['status'] == 'success':
        print(f"   ✅ {r['commodity']:10s} - {r['rows']:,} days trained, {r['forecast_days']} days forecasted")
    else:
        print(f"   ❌ {r['commodity']:10s} - {r.get('error', 'Unknown error')}")

print()
print(f"🕐 Generation: {generation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📈 Forecast Horizon: 14 days")
print(f"🎲 Distribution Paths: 2,000 per commodity")
print()

successful = [r for r in results if r['status'] == 'success']
failed = [r for r in results if r['status'] == 'failed']

if failed:
    print(f"⚠️  {len(failed)} commodity(ies) failed - review errors above")
else:
    print(f"✅ All {len(successful)} commodities processed successfully!")

print()
print("📝 Next Steps:")
print("   1. Trading agent reads: trading_agent_forecast_<commodity>.json")
print("   2. Weekly retraining: Run this script every Monday")
print("   3. Monitor: Compare forecasts vs actuals in forecast_actuals table")
print()
print("="*80)

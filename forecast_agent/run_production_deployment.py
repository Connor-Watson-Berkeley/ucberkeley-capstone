"""Production deployment script for weekly retraining.

Trains best model on FULL historical data and stores forecasts
in commodity.silver schema (point_forecasts, distributions, forecast_actuals).

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
print("  PRODUCTION FORECAST DEPLOYMENT")
print("  Train on FULL history → Store in commodity.silver schema")
print("="*80)
print()

# Configuration
commodity = 'Coffee'
model_key = 'sarimax_auto_weather'  # Best model from walk-forward evaluation
model_version = 'sarimax_weather_v1'  # Production version identifier

print(f"🎯 Configuration:")
print(f"   Commodity: {commodity}")
print(f"   Model: {model_key}")
print(f"   Model Version: {model_version}")
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
    sys.exit(1)

df = pd.read_parquet(data_path)
df['date'] = pd.to_datetime(df['date'])

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
print(f"   ✓ Loaded {len(df_agg)} days")
print(f"   Data cutoff date: {data_cutoff_date.date()}")
print()

# Get model configuration
if model_key not in BASELINE_MODELS:
    print(f"   ✗ Model {model_key} not found in registry")
    sys.exit(1)

config = BASELINE_MODELS[model_key]
model_name = config['name']

print(f"🤖 Training {model_name} on FULL history...")
print(f"   Training days: {len(df_agg)}")
print(f"   Forecast horizon: 14 days")
print()

# Train model
try:
    params = config['params'].copy()
    params['commodity'] = commodity

    result = config['function'](
        df_pandas=df_agg,
        **params
    )

    forecast_df = result['forecast_df']
    print(f"   ✓ Model trained successfully")
    print(f"   Forecast dates: {forecast_df['date'].min().date()} to {forecast_df['date'].max().date()}")
    print()

except Exception as e:
    print(f"   ✗ Model training failed: {str(e)}")
    sys.exit(1)

# Initialize production writer
print("💾 Writing to production tables...")
generation_timestamp = datetime.now()
writer = ProductionForecastWriter("production_forecasts")

# 1. Write point forecasts (one row per day_ahead)
print()
print("   📊 Writing point_forecasts table...")
writer.write_point_forecasts(
    forecast_df=forecast_df,
    model_version=model_version,
    commodity=commodity,
    data_cutoff_date=data_cutoff_date,
    generation_timestamp=generation_timestamp,
    prediction_intervals=None,  # TODO: Add prediction intervals from model
    model_success=True
)

# 2. Generate and write Monte Carlo distributions
print()
print("   🎲 Generating Monte Carlo distributions...")

# Generate sample paths from forecast residuals
# In production, use model-specific uncertainty quantification
n_paths = 2000
forecast_values = forecast_df['forecast'].values

# Simple approach: Bootstrap from historical residuals
# TODO: Use model-specific prediction intervals or residual bootstrap
residual_std = 2.5  # From walk-forward evaluation MAE * 0.8 (approx std)

sample_paths = np.random.normal(
    loc=forecast_values,
    scale=residual_std,
    size=(n_paths, 14)
)

# Ensure no negative prices
sample_paths = np.maximum(sample_paths, 0.01)

print(f"   Generated {n_paths} sample paths")

writer.write_distributions(
    forecast_start_date=forecast_df['date'].iloc[0],
    data_cutoff_date=data_cutoff_date,
    model_version=model_version,
    commodity=commodity,
    sample_paths=sample_paths,
    generation_timestamp=generation_timestamp
)

# 3. Write forecast actuals (for last 14 days)
print()
print("   📝 Writing forecast_actuals table...")

# Get last 14 days of actuals for backtesting
actuals_df = df_agg.tail(14).reset_index()
actuals_df = actuals_df[['date', 'close']]

writer.write_forecast_actuals(
    actuals_df=actuals_df,
    commodity=commodity
)

# 4. Export for trading agent
print()
print("🌐 Exporting for trading agent...")
output_path = "trading_agent_forecast.json"

writer.export_for_trading_agent(
    commodity=commodity,
    model_version=model_version,
    output_path=output_path
)

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

print("📊 Forecast Summary:")
print(f"   Commodity: {commodity}")
print(f"   Model Version: {model_version}")
print(f"   Generation: {generation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   Data Cutoff: {data_cutoff_date.date()}")
print(f"   Forecast Horizon: 14 days")
print(f"   Point Forecasts: {len(forecast_df)} rows")
print(f"   Distribution Paths: {n_paths} paths")
print()

print("🔍 Sample Forecasts:")
print(forecast_df[['date', 'forecast']].head(7).to_string(index=False))
print()

print("✅ Ready for production consumption by trading agent!")
print()
print("📝 Next Steps:")
print("   1. Trading agent reads: trading_agent_forecast.json")
print("   2. Weekly retraining: Run this script every Monday")
print("   3. Monitor: Compare forecasts vs actuals in forecast_actuals table")
print("   4. Alert: If model_success = False, investigate model convergence")
print()
print("="*80)

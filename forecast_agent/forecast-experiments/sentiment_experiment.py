"""
GDELT Sentiment Experiment

Tests whether news sentiment features improve forecasting accuracy.

Compares:
- Baseline: Weather only
- With Sentiment: Weather + GDELT sentiment (18 features)

Models: Best performers from comprehensive experiments (TCN, NHiTS)
Horizons: 1-day and 7-day
Region: Aggregated (for speed)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# DARTS imports
from darts import TimeSeries
from darts.models import TCNModel, NHiTSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae

print("=" * 80)
print("GDELT SENTIMENT EXPERIMENT")
print("=" * 80)
print()

# Load cached data with sentiment
print("Loading data...")
df = pd.read_parquet('data/unified_data_with_sentiment.parquet')
print(f"Loaded {len(df):,} rows")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print()

# Check GDELT data availability
gdelt_cols = [c for c in df.columns if c.startswith('group_') or c.startswith('theme_')]
print(f"Found {len(gdelt_cols)} GDELT sentiment features")
print()

# Aggregate by date (all regions)
print("Aggregating by date...")
df_agg = df.groupby('date').agg({
    'close': 'mean',
    # Weather
    'temp_max_c': 'mean',
    'temp_min_c': 'mean',
    'temp_mean_c': 'mean',
    'precipitation_mm': 'mean',
    'rain_mm': 'mean',
    'snowfall_cm': 'mean',
    'humidity_mean_pct': 'mean',
    'wind_speed_max_kmh': 'mean',
    # VIX
    'vix': 'mean',
    # GDELT (already aggregated, but take mean anyway)
    **{col: 'mean' for col in gdelt_cols}
}).reset_index()

df_agg = df_agg.sort_values('date')
print(f"Aggregated to {len(df_agg)} days")

# Filter to dates where GDELT data exists (2021+)
print("Filtering to dates with GDELT coverage...")
df_agg = df_agg[df_agg[gdelt_cols[0]].notna()].copy()
print(f"After filtering: {len(df_agg)} days with GDELT data")
print(f"Date range: {df_agg['date'].min()} to {df_agg['date'].max()}")
print()

# Define feature sets
weather_features = [
    'temp_max_c', 'temp_min_c', 'temp_mean_c',
    'precipitation_mm', 'rain_mm', 'snowfall_cm',
    'humidity_mean_pct', 'wind_speed_max_kmh', 'vix'
]

sentiment_features = gdelt_cols
all_features_with_sentiment = weather_features + sentiment_features

print(f"Feature sets:")
print(f"  Weather only: {len(weather_features)} features")
print(f"  Weather + Sentiment: {len(all_features_with_sentiment)} features ({len(sentiment_features)} sentiment)")
print()

# Create TimeSeries
target = TimeSeries.from_dataframe(df_agg, time_col='date', value_cols='close')

weather_covariates = TimeSeries.from_dataframe(
    df_agg, time_col='date', value_cols=weather_features
)

all_covariates = TimeSeries.from_dataframe(
    df_agg, time_col='date', value_cols=all_features_with_sentiment
)

# Train/val split
train_size = int(len(target) * 0.8)
print(f"Train size: {train_size} days")
print(f"Val size: {len(target) - train_size} days")
print()

# Experiment configuration
horizons = [1, 7]
results = []

# Model configurations (using best from comprehensive experiments)
# Force CPU to avoid MPS float64 incompatibility
# Store as lambda functions to create fresh instances for each test
models_config = {
    'TCN': lambda: TCNModel(
        input_chunk_length=60,
        output_chunk_length=7,
        n_epochs=50,
        batch_size=32,
        random_state=42,
        force_reset=True,
        save_checkpoints=False,
        pl_trainer_kwargs={"accelerator": "cpu"}
    ),
    'NHiTS': lambda: NHiTSModel(
        input_chunk_length=60,
        output_chunk_length=7,
        n_epochs=50,
        batch_size=32,
        random_state=42,
        force_reset=True,
        save_checkpoints=False,
        pl_trainer_kwargs={"accelerator": "cpu"}
    )
}

print("=" * 80)
print("RUNNING EXPERIMENTS")
print("=" * 80)
print()

for horizon in horizons:
    print(f"\n{'='*80}")
    print(f"HORIZON: {horizon}-day")
    print(f"{'='*80}\n")

    for model_name, model_factory in models_config.items():

        # Test 1: Weather only (baseline)
        print(f"  {model_name} + Weather only...")
        model = model_factory()  # Create fresh model instance

        try:
            model.fit(series=target[:train_size], past_covariates=weather_covariates[:train_size])

            # Walk-forward validation
            historical_forecasts = model.historical_forecasts(
                series=target,
                past_covariates=weather_covariates,
                start=train_size,
                forecast_horizon=horizon,
                stride=horizon,
                retrain=False,
                verbose=False
            )

            val_target = target[train_size:]
            mape_score = mape(val_target, historical_forecasts)
            rmse_score = rmse(val_target, historical_forecasts)
            mae_score = mae(val_target, historical_forecasts)

            print(f"    MAPE: {mape_score:.2f}%, RMSE: {rmse_score:.2f}, MAE: {mae_score:.2f}")

            results.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model': model_name,
                'feature_set': 'weather_only',
                'num_features': len(weather_features),
                'horizon_days': horizon,
                'mape': float(mape_score),
                'rmse': float(rmse_score),
                'mae': float(mae_score),
                'success': True
            })

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model': model_name,
                'feature_set': 'weather_only',
                'num_features': len(weather_features),
                'horizon_days': horizon,
                'mape': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'success': False,
                'error': str(e)
            })

        # Test 2: Weather + Sentiment
        print(f"  {model_name} + Weather + Sentiment...")
        model = model_factory()  # Create fresh model instance

        try:
            model.fit(series=target[:train_size], past_covariates=all_covariates[:train_size])

            historical_forecasts = model.historical_forecasts(
                series=target,
                past_covariates=all_covariates,
                start=train_size,
                forecast_horizon=horizon,
                stride=horizon,
                retrain=False,
                verbose=False
            )

            val_target = target[train_size:]
            mape_score = mape(val_target, historical_forecasts)
            rmse_score = rmse(val_target, historical_forecasts)
            mae_score = mae(val_target, historical_forecasts)

            print(f"    MAPE: {mape_score:.2f}%, RMSE: {rmse_score:.2f}, MAE: {mae_score:.2f}")

            results.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model': model_name,
                'feature_set': 'weather_sentiment',
                'num_features': len(all_features_with_sentiment),
                'horizon_days': horizon,
                'mape': float(mape_score),
                'rmse': float(rmse_score),
                'mae': float(mae_score),
                'success': True
            })

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model': model_name,
                'feature_set': 'weather_sentiment',
                'num_features': len(all_features_with_sentiment),
                'horizon_days': horizon,
                'mape': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'success': False,
                'error': str(e)
            })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv('experiment_results_sentiment.csv', index=False)

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print()

df_success = df_results[df_results['success']]

if len(df_success) > 0:
    print("All results:")
    print(df_success[['model', 'feature_set', 'horizon_days', 'mape']].to_string(index=False))
    print()

    # Compare weather vs weather+sentiment
    print("Impact of Sentiment Features:")
    print()

    for model in df_success['model'].unique():
        for horizon in df_success['horizon_days'].unique():
            weather_only = df_success[
                (df_success['model'] == model) &
                (df_success['feature_set'] == 'weather_only') &
                (df_success['horizon_days'] == horizon)
            ]

            with_sentiment = df_success[
                (df_success['model'] == model) &
                (df_success['feature_set'] == 'weather_sentiment') &
                (df_success['horizon_days'] == horizon)
            ]

            if len(weather_only) > 0 and len(with_sentiment) > 0:
                baseline_mape = weather_only['mape'].values[0]
                sentiment_mape = with_sentiment['mape'].values[0]
                improvement = baseline_mape - sentiment_mape
                improvement_pct = (improvement / baseline_mape) * 100

                symbol = "✓" if improvement > 0 else "✗"
                print(f"{model} @ {int(horizon)}-day:")
                print(f"  Weather only: {baseline_mape:.2f}% MAPE")
                print(f"  + Sentiment:  {sentiment_mape:.2f}% MAPE")
                print(f"  Change: {improvement:+.2f}% ({improvement_pct:+.1f}%) {symbol}")
                print()

print(f"Results saved to: experiment_results_sentiment.csv")
print()

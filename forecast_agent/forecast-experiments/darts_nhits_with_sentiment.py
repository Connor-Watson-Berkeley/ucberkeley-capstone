"""
N-HiTS Experiment with GDELT Sentiment Features

Tests whether adding news sentiment improves forecast accuracy.
Compares baseline model (weather only) vs enhanced model (weather + sentiment).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from darts import TimeSeries
from darts.models import NHiTSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae
from load_gdelt_data import get_sentiment_feature_names

import warnings
warnings.filterwarnings('ignore')


def load_sentiment_data(data_file='data/unified_with_sentiment.parquet'):
    """Load merged price/weather/sentiment data."""
    df = pd.read_parquet(data_file)

    # Rename columns for consistency
    df = df.rename(columns={
        'close': 'price',
        'temp_mean_c': 'temperature_2m_mean',
        'precipitation_mm': 'precipitation_sum',
        'wind_speed_max_kmh': 'wind_speed_10m_max',
        'humidity_mean_pct': 'relative_humidity_2m_mean',
    })

    return df


def prepare_darts_series_with_sentiment(df, target_col='price', weather_cols=None, sentiment_cols=None):
    """Convert DataFrame to DARTS TimeSeries with weather and sentiment covariates."""
    df = df.set_index('date')

    # Create target series
    target_series = TimeSeries.from_dataframe(
        df,
        value_cols=target_col,
        freq='D'
    )

    # Create covariate series (weather + sentiment)
    covariate_cols = []
    if weather_cols:
        covariate_cols.extend(weather_cols)
    if sentiment_cols:
        covariate_cols.extend(sentiment_cols)

    covariate_series = None
    if covariate_cols:
        covariate_series = TimeSeries.from_dataframe(
            df,
            value_cols=covariate_cols,
            freq='D'
        )

    return target_series, covariate_series


def run_nhits_with_sentiment(
    include_sentiment=True,
    lookback_days=730,
    forecast_horizon=14,
    input_chunk_length=60,
    output_chunk_length=14,
    num_stacks=3,
    num_blocks=1,
    num_layers=2,
    layer_widths=512,
    batch_size=32,
    n_epochs=100,
    learning_rate=1e-3
):
    """
    Run N-HiTS experiment with optional sentiment features.

    Args:
        include_sentiment: If True, includes GDELT sentiment features
    """
    model_type = "N-HiTS with Sentiment" if include_sentiment else "N-HiTS (Baseline)"

    print("=" * 80)
    print(f"DARTS {model_type}")
    print("=" * 80)
    print(f"Lookback days: {lookback_days}")
    print(f"Forecast horizon: {forecast_horizon}")
    print(f"Sentiment features: {'YES' if include_sentiment else 'NO'}")
    print()

    # Load data
    print("Loading data...")
    df = load_sentiment_data()
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()

    # Define weather covariates (baseline)
    weather_covariates = [
        'temperature_2m_mean',
        'precipitation_sum',
        'wind_speed_10m_max',
        'relative_humidity_2m_mean',
        'temp_max_c',
        'temp_min_c',
        'rain_mm'
    ]

    # Get sentiment covariates
    sentiment_covariates = get_sentiment_feature_names() if include_sentiment else []

    # Prepare DARTS series
    print("Preparing DARTS TimeSeries...")
    target_series, covariate_series = prepare_darts_series_with_sentiment(
        df,
        target_col='price',
        weather_cols=weather_covariates,
        sentiment_cols=sentiment_covariates
    )

    print(f"Target series length: {len(target_series)}")
    if covariate_series:
        print(f"Covariate series length: {len(covariate_series)}")
        print(f"Number of covariate features: {covariate_series.n_components}")
        if include_sentiment:
            print(f"  Weather features: {len(weather_covariates)}")
            print(f"  Sentiment features: {len(sentiment_covariates)}")
    print()

    # Split into train/validation
    train_size = int(len(target_series) * 0.8)
    train_target = target_series[:train_size]
    val_target = target_series[train_size:]

    train_covariates = covariate_series[:train_size] if covariate_series else None
    val_covariates = covariate_series[train_size:] if covariate_series else None

    print(f"Train size: {len(train_target)} | Validation size: {len(val_target)}")
    print()

    # Scale data
    print("Scaling data...")
    target_scaler = Scaler()
    train_target_scaled = target_scaler.fit_transform(train_target)
    val_target_scaled = target_scaler.transform(val_target)

    covariate_scaler = None
    train_covariates_scaled = None
    val_covariates_scaled = None
    if train_covariates:
        covariate_scaler = Scaler()
        train_covariates_scaled = covariate_scaler.fit_transform(train_covariates)
        val_covariates_scaled = covariate_scaler.transform(val_covariates)
    print()

    # Initialize N-HiTS model
    print("Initializing N-HiTS model...")
    model = NHiTSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        batch_size=batch_size,
        n_epochs=n_epochs,
        optimizer_kwargs={'lr': learning_rate},
        random_state=42,
        force_reset=True,
        save_checkpoints=True,
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "callbacks": [],
        }
    )
    print()

    # Train model
    print("Training N-HiTS model...")
    print(f"Epochs: {n_epochs} | Batch size: {batch_size}")

    model.fit(
        series=train_target_scaled,
        past_covariates=train_covariates_scaled,
        val_series=val_target_scaled,
        val_past_covariates=val_covariates_scaled,
        verbose=True
    )
    print()
    print("Training completed!")
    print()

    # Make predictions
    print("Generating forecasts...")
    full_target_scaled = target_scaler.transform(target_series)
    full_covariates_scaled = covariate_scaler.transform(covariate_series) if covariate_series else None

    forecast_n = min(forecast_horizon, output_chunk_length)
    forecast_scaled = model.predict(
        n=forecast_n,
        series=full_target_scaled,
        past_covariates=full_covariates_scaled
    )

    forecast = target_scaler.inverse_transform(forecast_scaled)
    print()

    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_n = min(len(val_target), output_chunk_length)
    val_forecast_scaled = model.predict(
        n=val_n,
        series=train_target_scaled,
        past_covariates=full_covariates_scaled
    )
    val_forecast = target_scaler.inverse_transform(val_forecast_scaled)

    val_target_subset = val_target[:val_n]
    mape_score = mape(val_target_subset, val_forecast)
    rmse_score = rmse(val_target_subset, val_forecast)
    mae_score = mae(val_target_subset, val_forecast)

    print(f"Validation MAPE: {mape_score:.2f}%")
    print(f"Validation RMSE: ${rmse_score:.4f}")
    print(f"Validation MAE: ${mae_score:.4f}")
    print()

    # Print forecast summary
    print("=" * 80)
    print(f"Forecast Summary (next {forecast_n} days)")
    print("=" * 80)
    forecast_df = pd.DataFrame({
        'date': forecast.time_index,
        'forecast': forecast.values().flatten()
    })
    print(forecast_df.to_string(index=False))
    print()

    results = {
        'model_type': model_type,
        'include_sentiment': include_sentiment,
        'model': model,
        'forecast': forecast,
        'forecast_df': forecast_df,
        'val_forecast': val_forecast,
        'actual': val_target,
        'metrics': {
            'mape': mape_score,
            'rmse': rmse_score,
            'mae': mae_score
        },
        'num_features': covariate_series.n_components if covariate_series else 0,
        'scalers': {
            'target': target_scaler,
            'covariate': covariate_scaler
        }
    }

    return results


def compare_with_and_without_sentiment():
    """Run experiments comparing baseline vs sentiment-enhanced models."""

    print("\n" + "=" * 80)
    print("SENTIMENT FEATURE COMPARISON EXPERIMENT")
    print("=" * 80)
    print()

    # Run baseline (no sentiment)
    print("EXPERIMENT 1: Baseline (Weather Only)")
    print("-" * 80)
    baseline_results = run_nhits_with_sentiment(
        include_sentiment=False,
        n_epochs=100
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Enhanced (Weather + Sentiment)")
    print("-" * 80)
    sentiment_results = run_nhits_with_sentiment(
        include_sentiment=True,
        n_epochs=100
    )

    # Print comparison
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print()

    print(f"{'Model':<30} {'Features':<12} {'MAPE':<10} {'RMSE':<12} {'MAE':<12}")
    print("-" * 80)

    baseline = baseline_results['metrics']
    sentiment = sentiment_results['metrics']

    print(f"{'Baseline (Weather Only)':<30} {baseline_results['num_features']:<12} "
          f"{baseline['mape']:<10.2f} ${baseline['rmse']:<11.2f} ${baseline['mae']:<11.2f}")

    print(f"{'Enhanced (Weather+Sentiment)':<30} {sentiment_results['num_features']:<12} "
          f"{sentiment['mape']:<10.2f} ${sentiment['rmse']:<11.2f} ${sentiment['mae']:<11.2f}")

    print()
    print("=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)

    mape_improvement = ((baseline['mape'] - sentiment['mape']) / baseline['mape']) * 100
    rmse_improvement = ((baseline['rmse'] - sentiment['rmse']) / baseline['rmse']) * 100
    mae_improvement = ((baseline['mae'] - sentiment['mae']) / baseline['mae']) * 100

    if mape_improvement > 0:
        print(f"✅ Sentiment features IMPROVED accuracy by {mape_improvement:.1f}%")
        print(f"   MAPE: {baseline['mape']:.2f}% → {sentiment['mape']:.2f}%")
    else:
        print(f"⚠️  Sentiment features did NOT improve accuracy ({abs(mape_improvement):.1f}% worse)")

    print(f"\nRMSE improvement: {rmse_improvement:+.1f}%")
    print(f"MAE improvement: {mae_improvement:+.1f}%")

    print("\n" + "=" * 80)

    return baseline_results, sentiment_results


if __name__ == '__main__':
    import sys
    import os

    # Check if sentiment data exists
    sentiment_data_file = 'data/unified_with_sentiment.parquet'
    if not os.path.exists(sentiment_data_file):
        print(f"\n❌ Sentiment data not found: {sentiment_data_file}")
        print("\nPlease run first:")
        print("  python3 load_gdelt_data.py")
        print("\nThis will:")
        print("  1. Download GDELT sentiment data from Databricks")
        print("  2. Merge with unified price/weather data")
        print("  3. Create enhanced dataset with sentiment features")
        print()
        sys.exit(1)

    # Run comparison experiment
    baseline_results, sentiment_results = compare_with_and_without_sentiment()

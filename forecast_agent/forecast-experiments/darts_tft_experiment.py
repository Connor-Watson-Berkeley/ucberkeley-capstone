"""
DARTS Temporal Fusion Transformer (TFT) Experiment

This script trains a TFT model on coffee price data with weather covariates
using the DARTS library. TFT is ideal for multivariate forecasting with
attention mechanisms and built-in interpretability.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae
from darts.utils.likelihood_models import QuantileRegression
from load_local_data import load_local_data

import warnings
warnings.filterwarnings('ignore')


def load_data_from_local(commodity='Coffee', region=None, lookback_days=365):
    """
    Load commodity price and weather data from local parquet file.

    Args:
        commodity: Commodity name (default: Coffee)
        region: Region name (default: None, aggregates all regions)
        lookback_days: Number of days to look back (default: 365)

    Returns:
        DataFrame with price and weather features
    """
    # Load from local file
    df = load_local_data(
        commodity=commodity,
        region=region,
        lookback_days=lookback_days
    )

    # Rename columns to match expected names
    df = df.rename(columns={
        'close': 'price',
        'temp_mean_c': 'temperature_2m_mean',
        'precipitation_mm': 'precipitation_sum',
        'wind_speed_max_kmh': 'wind_speed_10m_max',
        'humidity_mean_pct': 'relative_humidity_2m_mean',
        'rain_mm': 'rain_mm'
    })

    return df


def prepare_darts_series(df, target_col='price', covariate_cols=None):
    """
    Convert DataFrame to DARTS TimeSeries objects.

    Args:
        df: DataFrame with datetime index
        target_col: Name of target column
        covariate_cols: List of covariate column names

    Returns:
        tuple: (target_series, covariate_series)
    """
    # Set datetime index
    df = df.set_index('date')

    # Create target series
    target_series = TimeSeries.from_dataframe(
        df,
        value_cols=target_col,
        freq='D'
    )

    # Create covariate series if provided
    covariate_series = None
    if covariate_cols:
        covariate_series = TimeSeries.from_dataframe(
            df,
            value_cols=covariate_cols,
            freq='D'
        )

    return target_series, covariate_series


def run_tft_experiment(
    lookback_days=365,
    forecast_horizon=14,
    input_chunk_length=60,
    output_chunk_length=14,
    hidden_size=64,
    lstm_layers=2,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=32,
    n_epochs=50,
    learning_rate=1e-3
):
    """
    Run TFT model experiment.

    Args:
        lookback_days: Days of historical data to use
        forecast_horizon: Number of days to forecast
        input_chunk_length: Input sequence length for model
        output_chunk_length: Output sequence length for model
        hidden_size: Hidden layer size
        lstm_layers: Number of LSTM layers
        num_attention_heads: Number of attention heads
        dropout: Dropout rate
        batch_size: Training batch size
        n_epochs: Number of training epochs
        learning_rate: Learning rate

    Returns:
        dict: Results with metrics and forecasts
    """
    print("=" * 80)
    print("DARTS TFT Experiment")
    print("=" * 80)
    print(f"Lookback days: {lookback_days}")
    print(f"Forecast horizon: {forecast_horizon}")
    print(f"Input chunk length: {input_chunk_length}")
    print(f"Output chunk length: {output_chunk_length}")
    print()

    # Load data
    print("Loading data from local file...")
    df = load_data_from_local(commodity='Coffee', region='Bahia_Brazil', lookback_days=lookback_days)
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()

    # Define covariates
    weather_covariates = [
        'temperature_2m_mean',
        'precipitation_sum',
        'wind_speed_10m_max',
        'relative_humidity_2m_mean',
        'temp_max_c',
        'temp_min_c',
        'rain_mm'
    ]

    # Prepare DARTS series
    print("Preparing DARTS TimeSeries...")
    target_series, covariate_series = prepare_darts_series(
        df,
        target_col='price',
        covariate_cols=weather_covariates
    )
    print(f"Target series length: {len(target_series)}")
    print(f"Covariate series length: {len(covariate_series)}")
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

    covariate_scaler = Scaler()
    train_covariates_scaled = covariate_scaler.fit_transform(train_covariates) if train_covariates else None
    val_covariates_scaled = covariate_scaler.transform(val_covariates) if val_covariates else None
    print()

    # Initialize TFT model
    print("Initializing TFT model...")
    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=n_epochs,
        optimizer_kwargs={'lr': learning_rate},
        likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),  # Probabilistic forecasting
        random_state=42,
        force_reset=True,
        save_checkpoints=True,
        pl_trainer_kwargs={
            "accelerator": "cpu",  # Use CPU to avoid MPS float64 issue
            "callbacks": [],
        }
    )
    print()

    # Train model
    print("Training TFT model...")
    print(f"Epochs: {n_epochs} | Batch size: {batch_size} | Learning rate: {learning_rate}")

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
    forecast_scaled = model.predict(
        n=forecast_horizon,
        series=train_target_scaled,
        past_covariates=train_covariates_scaled,
        num_samples=100  # For probabilistic forecasting
    )

    # Inverse transform predictions
    forecast = target_scaler.inverse_transform(forecast_scaled)
    print()

    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_forecast_scaled = model.predict(
        n=len(val_target),
        series=train_target_scaled,
        past_covariates=train_covariates_scaled,
        num_samples=100
    )
    val_forecast = target_scaler.inverse_transform(val_forecast_scaled)

    # Calculate metrics
    mape_score = mape(val_target, val_forecast)
    rmse_score = rmse(val_target, val_forecast)
    mae_score = mae(val_target, val_forecast)

    print(f"Validation MAPE: {mape_score:.2f}%")
    print(f"Validation RMSE: {rmse_score:.4f}")
    print(f"Validation MAE: {mae_score:.4f}")
    print()

    # Extract quantile forecasts
    forecast_median = forecast.quantile(0.5)
    forecast_lower = forecast.quantile(0.1)
    forecast_upper = forecast.quantile(0.9)

    # Print forecast summary
    print("=" * 80)
    print(f"Forecast Summary (next {forecast_horizon} days)")
    print("=" * 80)
    forecast_df = pd.DataFrame({
        'date': forecast.time_index,
        'forecast_median': forecast_median.values().flatten(),
        'forecast_lower_10': forecast_lower.values().flatten(),
        'forecast_upper_90': forecast_upper.values().flatten()
    })
    print(forecast_df.to_string(index=False))
    print()

    results = {
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
        'scalers': {
            'target': target_scaler,
            'covariate': covariate_scaler
        }
    }

    return results


if __name__ == '__main__':
    # Run experiment with default parameters
    results = run_tft_experiment(
        lookback_days=365,
        forecast_horizon=14,
        input_chunk_length=60,
        output_chunk_length=14,
        hidden_size=64,
        lstm_layers=2,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=32,
        n_epochs=50,
        learning_rate=1e-3
    )

    print("Experiment completed successfully!")
    print(f"Final validation MAPE: {results['metrics']['mape']:.2f}%")

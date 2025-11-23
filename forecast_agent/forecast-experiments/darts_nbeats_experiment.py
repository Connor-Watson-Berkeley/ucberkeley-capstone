"""
DARTS N-BEATS Experiment

This script trains an N-BEATS model on coffee price data using the DARTS library.
N-BEATS (Neural Basis Expansion Analysis for Time Series) is a pure deep learning
architecture that has achieved state-of-the-art results on time series forecasting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae
from load_local_data import load_local_data

import warnings
warnings.filterwarnings('ignore')


def load_data_from_local(commodity='Coffee', region=None, lookback_days=365):
    """
    Load commodity price data from local parquet file.

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


def run_nbeats_experiment(
    lookback_days=365,
    forecast_horizon=14,
    input_chunk_length=60,
    output_chunk_length=14,
    num_stacks=30,
    num_blocks=1,
    num_layers=4,
    layer_widths=256,
    expansion_coefficient_dim=5,
    batch_size=32,
    n_epochs=100,
    learning_rate=1e-3,
    generic_architecture=True
):
    """
    Run N-BEATS model experiment.

    Args:
        lookback_days: Days of historical data to use
        forecast_horizon: Number of days to forecast
        input_chunk_length: Input sequence length for model
        output_chunk_length: Output sequence length for model
        num_stacks: Number of stacks in the N-BEATS architecture
        num_blocks: Number of blocks per stack
        num_layers: Number of fully connected layers per block
        layer_widths: Width of fully connected layers
        expansion_coefficient_dim: Dimensionality of waveform generator parameters
        batch_size: Training batch size
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        generic_architecture: If True, uses generic architecture; if False, uses interpretable

    Returns:
        dict: Results with metrics and forecasts
    """
    print("=" * 80)
    print("DARTS N-BEATS Experiment")
    print("=" * 80)
    print(f"Architecture: {'Generic' if generic_architecture else 'Interpretable'}")
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

    # Define covariates (N-BEATS can use past covariates)
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
    if covariate_series:
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

    covariate_scaler = None
    train_covariates_scaled = None
    val_covariates_scaled = None
    if train_covariates:
        covariate_scaler = Scaler()
        train_covariates_scaled = covariate_scaler.fit_transform(train_covariates)
        val_covariates_scaled = covariate_scaler.transform(val_covariates)
    print()

    # Initialize N-BEATS model
    print("Initializing N-BEATS model...")
    print(f"  Stacks: {num_stacks}")
    print(f"  Blocks per stack: {num_blocks}")
    print(f"  Layers per block: {num_layers}")
    print(f"  Layer width: {layer_widths}")

    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        expansion_coefficient_dim=expansion_coefficient_dim,
        batch_size=batch_size,
        n_epochs=n_epochs,
        optimizer_kwargs={'lr': learning_rate},
        generic_architecture=generic_architecture,
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
    print("Training N-BEATS model...")
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

    # Make predictions (use full series for better context)
    print("Generating forecasts...")
    # For final forecast, use all available data
    full_target_scaled = target_scaler.transform(target_series)
    full_covariates_scaled = covariate_scaler.transform(covariate_series) if covariate_series else None

    # Only forecast up to output_chunk_length to avoid needing future covariates
    forecast_n = min(forecast_horizon, output_chunk_length)
    forecast_scaled = model.predict(
        n=forecast_n,
        series=full_target_scaled,
        past_covariates=full_covariates_scaled
    )

    # Inverse transform predictions
    forecast = target_scaler.inverse_transform(forecast_scaled)
    print()

    # Evaluate on validation set (limit prediction to available covariate length)
    print("Evaluating on validation set...")
    # Use the full series (train + val) for prediction, but only use available covariates
    full_target_scaled = target_scaler.transform(target_series)
    full_covariates_scaled = covariate_scaler.transform(covariate_series) if covariate_series else None

    # Predict only output_chunk_length to avoid needing future covariates
    val_n = min(len(val_target), output_chunk_length)
    val_forecast_scaled = model.predict(
        n=val_n,
        series=train_target_scaled,
        past_covariates=full_covariates_scaled
    )
    val_forecast = target_scaler.inverse_transform(val_forecast_scaled)

    # Calculate metrics on the predicted portion
    val_target_subset = val_target[:val_n]
    mape_score = mape(val_target_subset, val_forecast)
    rmse_score = rmse(val_target_subset, val_forecast)
    mae_score = mae(val_target_subset, val_forecast)

    print(f"Validation MAPE: {mape_score:.2f}% (on {val_n} days)")
    print(f"Validation RMSE: {rmse_score:.4f}")
    print(f"Validation MAE: {mae_score:.4f}")
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
    print(f"Note: Limited to {forecast_n} days (output_chunk_length) to avoid requiring future covariates")
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
    # Run experiment with default parameters (Generic architecture)
    print("\n" + "=" * 80)
    print("Running N-BEATS with Generic Architecture")
    print("=" * 80 + "\n")

    results_generic = run_nbeats_experiment(
        lookback_days=730,  # Use 2 years of data for better training
        forecast_horizon=14,
        input_chunk_length=60,
        output_chunk_length=14,
        num_stacks=30,
        num_blocks=1,
        num_layers=4,
        layer_widths=256,
        expansion_coefficient_dim=5,
        batch_size=32,
        n_epochs=100,
        learning_rate=1e-3,
        generic_architecture=True
    )

    print("\n" + "=" * 80)
    print("Experiment completed successfully!")
    print(f"Final validation MAPE: {results_generic['metrics']['mape']:.2f}%")
    print("=" * 80)

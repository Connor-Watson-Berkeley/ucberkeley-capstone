"""
Quick DARTS Test - Small N-BEATS experiment

This is a lightweight test to validate DARTS installation and basic functionality
with a smaller dataset and fewer epochs for faster execution.
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


def quick_test():
    """
    Quick test with 90 days of data and lightweight model.
    """
    print("=" * 80)
    print("DARTS Quick Test - N-BEATS Model")
    print("=" * 80)

    # Load small dataset (90 days) from local parquet
    print("Loading 90 days of Coffee price data from local file...")
    df = load_local_data(commodity='Coffee', lookback_days=90)

    # Rename close to price for consistency
    df = df.rename(columns={'close': 'price'})
    df = df.set_index('date')

    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print()

    # Create TimeSeries
    series = TimeSeries.from_dataframe(df, value_cols='price', freq='D')

    # Train/test split
    train = series[:70]  # 70 days for training
    test = series[70:]   # 20 days for testing

    print(f"Train: {len(train)} days | Test: {len(test)} days")
    print()

    # Scale
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    # Lightweight N-BEATS model (10 epochs only)
    print("Training lightweight N-BEATS model (10 epochs)...")
    model = NBEATSModel(
        input_chunk_length=30,
        output_chunk_length=10,
        num_stacks=10,  # Reduced from 30
        num_blocks=1,
        num_layers=2,   # Reduced from 4
        layer_widths=128,  # Reduced from 256
        n_epochs=10,    # Reduced from 100
        batch_size=16,
        random_state=42,
        force_reset=True,
        pl_trainer_kwargs={"accelerator": "cpu", "callbacks": []}  # Use CPU to avoid MPS float64 issue
    )

    model.fit(series=train_scaled, verbose=False)
    print("Training complete!")
    print()

    # Forecast
    forecast_scaled = model.predict(n=len(test))
    forecast = scaler.inverse_transform(forecast_scaled)

    # Metrics
    mape_score = mape(test, forecast)
    rmse_score = rmse(test, forecast)
    mae_score = mae(test, forecast)

    print("=" * 80)
    print("Results")
    print("=" * 80)
    print(f"MAPE: {mape_score:.2f}%")
    print(f"RMSE: ${rmse_score:.4f}")
    print(f"MAE: ${mae_score:.4f}")
    print("=" * 80)
    print()
    print("DARTS is working correctly!")

    return {
        'mape': mape_score,
        'rmse': rmse_score,
        'mae': mae_score
    }


if __name__ == '__main__':
    results = quick_test()

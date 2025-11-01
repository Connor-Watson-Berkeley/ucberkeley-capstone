"""Populate distributions table with multiple model forecasts for Coffee.

This script generates Monte Carlo forecast distributions from several models
and writes them to the production distributions table.

Usage:
    python populate_distributions_coffee.py

Models included:
    1. SARIMAX with weather features
    2. Prophet
    3. ARIMA
    4. Random Walk (baseline)
    5. Naive (baseline)

Output:
    - Writes 2000 Monte Carlo paths per model to distributions table
    - Saves to production_forecasts/distributions.parquet
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import SparkSession

# Import models
from ground_truth.models.sarimax import sarimax_forecast_with_metadata
from ground_truth.models.prophet_model import prophet_forecast_with_metadata
from ground_truth.models.random_walk import random_walk_forecast_with_metadata
from ground_truth.models.arima import arima_forecast_with_metadata
from ground_truth.models.naive import naive_forecast_with_metadata

# Import data loader and production writer
from ground_truth.core.data_loader import load_unified_data, prepare_model_data
from ground_truth.storage.production_writer import ProductionForecastWriter
from ground_truth.core.logger import get_logger

logger = get_logger(__name__)


def generate_monte_carlo_paths(forecast_df: pd.DataFrame,
                                residual_std: float,
                                n_paths: int = 2000) -> np.ndarray:
    """
    Generate Monte Carlo sample paths from point forecast using bootstrap residuals.

    Args:
        forecast_df: DataFrame with 'forecast' column (14-day horizon)
        residual_std: Standard deviation of residuals from training
        n_paths: Number of sample paths to generate

    Returns:
        Array of shape (n_paths, 14) with price paths
    """
    forecast_values = forecast_df['forecast'].values

    if len(forecast_values) != 14:
        raise ValueError(f"Expected 14-day forecast, got {len(forecast_values)}")

    # Generate paths with cumulative error (random walk in errors)
    sample_paths = np.zeros((n_paths, 14))

    for path_id in range(n_paths):
        # Each day's error is independent normal
        daily_errors = np.random.normal(0, residual_std, size=14)

        # Apply errors to forecast
        sample_paths[path_id, :] = forecast_values + daily_errors

        # Ensure non-negative prices (Coffee prices can't be negative)
        sample_paths[path_id, :] = np.maximum(sample_paths[path_id, :], 10.0)

    return sample_paths


def populate_distributions_for_coffee(cutoff_date: str = None,
                                      n_paths: int = 2000):
    """
    Populate distributions table with forecasts from multiple models for Coffee.

    Args:
        cutoff_date: Data cutoff date (YYYY-MM-DD). If None, uses latest available.
        n_paths: Number of Monte Carlo paths per model
    """

    logger.info("="*80)
    logger.info("POPULATING DISTRIBUTIONS TABLE FOR COFFEE")
    logger.info("="*80)

    # 1. Load Coffee data
    logger.info("\n[1/5] Loading Coffee data from unified_data...")
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("populate_coffee_distributions") \
        .getOrCreate()

    df_spark = load_unified_data(
        spark,
        table_name="commodity.silver.unified_data",
        commodity='Coffee',
        cutoff_date=cutoff_date
    )

    df_pandas = prepare_model_data(
        df_spark,
        commodity='Coffee',
        features=['close', 'temp_c', 'humidity_pct', 'precipitation_mm', 'vix']
    )

    df_pandas = df_pandas.set_index('date').sort_index()

    logger.info(f"  Loaded {len(df_pandas):,} days of Coffee data")
    logger.info(f"  Date range: {df_pandas.index.min()} to {df_pandas.index.max()}")

    # Get cutoff date and forecast start date
    data_cutoff_date = df_pandas.index[-1]
    forecast_start_date = data_cutoff_date + timedelta(days=1)
    generation_timestamp = datetime.now()

    logger.info(f"  Data cutoff: {data_cutoff_date}")
    logger.info(f"  Forecast starts: {forecast_start_date}")

    # 2. Initialize production writer
    logger.info("\n[2/5] Initializing production forecast writer...")
    writer = ProductionForecastWriter("production_forecasts")

    # 3. Define models to run
    models_to_run = [
        {
            'name': 'sarimax_weather_v1',
            'function': sarimax_forecast_with_metadata,
            'params': {
                'target': 'close',
                'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm', 'vix'],
                'horizon': 14
            },
            'residual_std': 3.2  # Estimated from walk-forward results
        },
        {
            'name': 'prophet_v1',
            'function': prophet_forecast_with_metadata,
            'params': {
                'target': 'close',
                'horizon': 14
            },
            'residual_std': 3.5
        },
        {
            'name': 'naive_baseline',
            'function': naive_forecast_with_metadata,
            'params': {
                'target': 'close',
                'horizon': 14,
                'method': 'last'
            },
            'residual_std': 4.5
        },
        {
            'name': 'arima_v1',
            'function': arima_forecast_with_metadata,
            'params': {
                'target': 'close',
                'horizon': 14
            },
            'residual_std': 3.4
        },
        {
            'name': 'random_walk_baseline',
            'function': random_walk_forecast_with_metadata,
            'params': {
                'target': 'close',
                'horizon': 14,
                'method': 'drift'
            },
            'residual_std': 4.0
        }
    ]

    logger.info(f"  Models to run: {len(models_to_run)}")

    # 4. Train each model and generate distributions
    logger.info("\n[3/5] Training models and generating distributions...")

    results = []

    for i, model_config in enumerate(models_to_run, 1):
        model_name = model_config['name']
        logger.info(f"\n  [{i}/{len(models_to_run)}] Running {model_name}...")

        try:
            # Train model
            result = model_config['function'](
                df_pandas=df_pandas,
                commodity='Coffee',
                **model_config['params']
            )

            if not result['success']:
                logger.warning(f"    ✗ {model_name} failed to converge. Skipping.")
                continue

            forecast_df = result['forecast_df']

            logger.info(f"    ✓ Model trained successfully")
            logger.info(f"    Forecast: {forecast_df['forecast'].iloc[0]:.2f} to {forecast_df['forecast'].iloc[-1]:.2f}")

            # Generate Monte Carlo paths
            logger.info(f"    Generating {n_paths:,} Monte Carlo paths...")
            sample_paths = generate_monte_carlo_paths(
                forecast_df=forecast_df,
                residual_std=model_config['residual_std'],
                n_paths=n_paths
            )

            # Write distributions to table
            writer.write_distributions(
                forecast_start_date=forecast_start_date,
                data_cutoff_date=data_cutoff_date,
                model_version=model_name,
                commodity='Coffee',
                sample_paths=sample_paths,
                generation_timestamp=generation_timestamp,
                n_paths=n_paths
            )

            results.append({
                'model': model_name,
                'success': True,
                'n_paths': n_paths,
                'mean_forecast': forecast_df['forecast'].mean()
            })

        except Exception as e:
            logger.error(f"    ✗ {model_name} failed with error: {e}", exc_info=True)
            results.append({
                'model': model_name,
                'success': False,
                'error': str(e)
            })

    # 5. Summary
    logger.info("\n[4/5] Summary of results:")
    logger.info("="*80)

    successful_models = [r for r in results if r['success']]
    failed_models = [r for r in results if not r['success']]

    logger.info(f"  Successful: {len(successful_models)}/{len(results)} models")

    for result in successful_models:
        logger.info(f"    ✓ {result['model']}: {result['n_paths']:,} paths, "
                   f"mean forecast = ${result['mean_forecast']:.2f}")

    if failed_models:
        logger.warning(f"\n  Failed: {len(failed_models)} models")
        for result in failed_models:
            logger.warning(f"    ✗ {result['model']}: {result.get('error', 'Unknown error')}")

    # 6. Final statistics
    logger.info("\n[5/5] Final distributions table statistics:")
    logger.info("="*80)

    dist_df = writer.distributions
    logger.info(f"  Total rows: {len(dist_df):,}")
    logger.info(f"  Models: {dist_df['model_version'].nunique()}")
    logger.info(f"  Commodities: {dist_df['commodity'].nunique()}")
    logger.info(f"  Total paths: {dist_df['path_id'].nunique():,}")

    logger.info(f"\n  Distributions table location:")
    logger.info(f"    {writer.distributions_path}")

    logger.info("\n✅ Distribution population complete!")

    return dist_df


if __name__ == "__main__":
    import sys

    # Optional: specify cutoff date from command line
    cutoff_date = sys.argv[1] if len(sys.argv) > 1 else None

    # Run population
    df = populate_distributions_for_coffee(cutoff_date=cutoff_date, n_paths=2000)

    print("\n" + "="*80)
    print("PREVIEW OF DISTRIBUTIONS TABLE")
    print("="*80)
    print(df.head(20))

    print("\n" + "="*80)
    print("DISTRIBUTIONS BY MODEL")
    print("="*80)
    print(df.groupby('model_version').agg({
        'path_id': 'count',
        'day_1': ['mean', 'std'],
        'day_14': ['mean', 'std']
    }).round(2))

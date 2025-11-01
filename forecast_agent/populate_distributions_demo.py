"""Populate distributions table with sample forecasts for Coffee (Demo).

This simplified version generates sample forecast distributions without requiring
Databricks connection. It demonstrates the distributions table structure with multiple models.

Usage:
    python populate_distributions_demo.py

Models included:
    1. SARIMAX (sample)
    2. Prophet (sample)
    3. ARIMA (sample)
    4. Random Walk (sample)
    5. Naive (sample)

Output:
    - Writes 2000 Monte Carlo paths per model to distributions table
    - Saves to production_forecasts/distributions.parquet
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import production writer
from ground_truth.storage.production_writer import ProductionForecastWriter
from ground_truth.core.logger import get_logger

logger = get_logger(__name__)


def generate_sample_forecast(base_price: float = 180.0,
                              horizon: int = 14,
                              trend: float = 0.1,
                              volatility: float = 2.5) -> pd.DataFrame:
    """
    Generate a simple sample forecast path.

    Args:
        base_price: Starting price
        horizon: Number of days to forecast
        trend: Daily trend (cents/day)
        volatility: Daily volatility (std dev)

    Returns:
        DataFrame with 'date' and 'forecast' columns
    """
    forecast_start = datetime.now() + timedelta(days=1)
    dates = [forecast_start + timedelta(days=i) for i in range(horizon)]

    forecasts = []
    current_price = base_price

    for i in range(horizon):
        # Add trend and random walk
        current_price += trend + np.random.normal(0, volatility / 5)
        forecasts.append(current_price)

    return pd.DataFrame({
        'date': dates,
        'forecast': forecasts
    })


def generate_monte_carlo_paths(forecast_df: pd.DataFrame,
                                residual_std: float,
                                n_paths: int = 2000) -> np.ndarray:
    """
    Generate Monte Carlo sample paths from point forecast.

    Args:
        forecast_df: DataFrame with 'forecast' column
        residual_std: Standard deviation of residuals
        n_paths: Number of sample paths

    Returns:
        Array of shape (n_paths, 14)
    """
    forecast_values = forecast_df['forecast'].values

    if len(forecast_values) != 14:
        raise ValueError(f"Expected 14-day forecast, got {len(forecast_values)}")

    sample_paths = np.zeros((n_paths, 14))

    for path_id in range(n_paths):
        # Generate random walk in errors
        daily_errors = np.random.normal(0, residual_std, size=14)
        sample_paths[path_id, :] = forecast_values + daily_errors

        # Ensure non-negative prices
        sample_paths[path_id, :] = np.maximum(sample_paths[path_id, :], 10.0)

    return sample_paths


def populate_distributions_demo(n_paths: int = 2000):
    """
    Populate distributions table with sample forecasts from multiple models.

    Args:
        n_paths: Number of Monte Carlo paths per model
    """

    logger.info("="*80)
    logger.info("POPULATING DISTRIBUTIONS TABLE FOR COFFEE (DEMO)")
    logger.info("="*80)

    # Setup
    data_cutoff_date = datetime.now()
    forecast_start_date = data_cutoff_date + timedelta(days=1)
    generation_timestamp = datetime.now()

    logger.info(f"  Data cutoff: {data_cutoff_date.date()}")
    logger.info(f"  Forecast starts: {forecast_start_date.date()}")

    # Initialize production writer
    logger.info("\n[1/3] Initializing production forecast writer...")
    writer = ProductionForecastWriter("production_forecasts")

    # Define model configurations
    models_config = [
        {
            'name': 'sarimax_weather_v1',
            'base_price': 180.5,
            'trend': 0.15,
            'volatility': 2.8,
            'residual_std': 3.2
        },
        {
            'name': 'prophet_v1',
            'base_price': 181.2,
            'trend': 0.12,
            'volatility': 3.1,
            'residual_std': 3.5
        },
        {
            'name': 'arima_v1',
            'base_price': 180.8,
            'trend': 0.10,
            'volatility': 2.9,
            'residual_std': 3.4
        },
        {
            'name': 'random_walk_baseline',
            'base_price': 180.0,
            'trend': 0.05,
            'volatility': 3.5,
            'residual_std': 4.0
        },
        {
            'name': 'naive_baseline',
            'base_price': 180.0,
            'trend': 0.0,
            'volatility': 4.0,
            'residual_std': 4.5
        }
    ]

    logger.info(f"  Models to generate: {len(models_config)}")

    # Generate distributions for each model
    logger.info("\n[2/3] Generating forecast distributions...")

    results = []

    for i, model_config in enumerate(models_config, 1):
        model_name = model_config['name']
        logger.info(f"\n  [{i}/{len(models_config)}] Generating {model_name}...")

        # Generate sample forecast
        forecast_df = generate_sample_forecast(
            base_price=model_config['base_price'],
            horizon=14,
            trend=model_config['trend'],
            volatility=model_config['volatility']
        )

        logger.info(f"    Forecast: ${forecast_df['forecast'].iloc[0]:.2f} to ${forecast_df['forecast'].iloc[-1]:.2f}")

        # Generate Monte Carlo paths
        logger.info(f"    Generating {n_paths:,} Monte Carlo paths...")
        sample_paths = generate_monte_carlo_paths(
            forecast_df=forecast_df,
            residual_std=model_config['residual_std'],
            n_paths=n_paths
        )

        # Write to distributions table
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
            'n_paths': n_paths,
            'mean_forecast': forecast_df['forecast'].mean(),
            'final_forecast': forecast_df['forecast'].iloc[-1]
        })

    # Summary
    logger.info("\n[3/3] Summary:")
    logger.info("="*80)

    for result in results:
        logger.info(f"  ✓ {result['model']}: {result['n_paths']:,} paths, "
                   f"avg=${result['mean_forecast']:.2f}, final=${result['final_forecast']:.2f}")

    # Final statistics
    logger.info("\nDistributions table statistics:")
    logger.info("="*80)

    dist_df = writer.distributions
    logger.info(f"  Total rows: {len(dist_df):,}")
    logger.info(f"  Models: {dist_df['model_version'].nunique()}")
    logger.info(f"  Total paths: {dist_df['path_id'].max():,}")

    logger.info(f"\n  Distributions table location:")
    logger.info(f"    {writer.distributions_path}")

    logger.info("\n✅ Distribution population complete!")

    return dist_df


if __name__ == "__main__":
    # Run population
    df = populate_distributions_demo(n_paths=2000)

    print("\n" + "="*80)
    print("PREVIEW OF DISTRIBUTIONS TABLE")
    print("="*80)
    print(df.head(20))

    print("\n" + "="*80)
    print("DISTRIBUTIONS BY MODEL")
    print("="*80)
    summary = df.groupby('model_version').agg({
        'path_id': 'count',
        'day_1': ['mean', 'std'],
        'day_7': ['mean', 'std'],
        'day_14': ['mean', 'std']
    }).round(2)
    print(summary)

    print("\n" + "="*80)
    print("READY FOR ANALYSIS")
    print("="*80)
    print(f"The distributions table now contains {len(df):,} rows with forecasts from 5 models.")
    print("You can use this table for:")
    print("  - Risk analysis (VaR, CVaR)")
    print("  - Model comparison")
    print("  - Ensemble forecasting")
    print("  - Scenario analysis")

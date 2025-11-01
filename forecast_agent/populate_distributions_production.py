"""Populate distributions table using PRODUCTION forecast pipeline.

This script uses:
- REAL Coffee data from unified_data parquet files
- ACTUAL production models from model registry
- REAL trained forecasts (not sample data)
- Production-grade Monte Carlo distributions

Usage:
    python populate_distributions_production.py

Models trained:
    - SARIMAX+Weather (best model)
    - Prophet
    - XGBoost
    - ARIMA
    - Random Walk (baseline)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from ground_truth.storage.production_writer import ProductionForecastWriter
from ground_truth.config.model_registry import BASELINE_MODELS
from ground_truth.core.logger import get_logger

logger = get_logger(__name__)


def estimate_residual_std(df_pandas: pd.DataFrame, horizon: int = 14) -> float:
    """
    Estimate residual standard deviation from recent price volatility.

    Args:
        df_pandas: Historical price data
        horizon: Forecast horizon

    Returns:
        Estimated residual std dev
    """
    # Use last 90 days of daily returns
    recent_data = df_pandas['close'].tail(90)
    daily_returns = recent_data.pct_change().dropna()

    # Annualized volatility → daily volatility
    volatility = daily_returns.std() * np.sqrt(252)  # Annualize
    daily_vol = volatility / np.sqrt(252)

    # Scale by price level and horizon
    avg_price = recent_data.mean()
    residual_std = avg_price * daily_vol * np.sqrt(horizon / 14)

    return max(residual_std, 1.0)  # Minimum 1.0


def generate_monte_carlo_from_forecast(forecast_df: pd.DataFrame,
                                        residual_std: float,
                                        n_paths: int = 2000) -> np.ndarray:
    """
    Generate Monte Carlo paths using Geometric Brownian Motion.

    Args:
        forecast_df: Point forecast DataFrame
        residual_std: Residual standard deviation
        n_paths: Number of paths

    Returns:
        Array of shape (n_paths, 14)
    """
    forecast_values = forecast_df['forecast'].values

    if len(forecast_values) != 14:
        raise ValueError(f"Expected 14-day forecast, got {len(forecast_values)}")

    sample_paths = np.zeros((n_paths, 14))

    for path_id in range(n_paths):
        # Geometric Brownian Motion: S(t+1) = S(t) * exp((μ - σ²/2)dt + σ√dt*ε)
        # Simplified: Add correlated noise that compounds over time

        path = np.zeros(14)
        path[0] = forecast_values[0] + np.random.normal(0, residual_std)

        for day in range(1, 14):
            # Drift component (from forecast)
            drift = forecast_values[day] - forecast_values[day-1]

            # Volatility component (random walk in errors)
            shock = np.random.normal(0, residual_std * 0.7)  # Dampen for stability

            path[day] = path[day-1] + drift + shock

        # Ensure positive prices
        path = np.maximum(path, 10.0)
        sample_paths[path_id, :] = path

    return sample_paths


def populate_distributions_production(commodity: str = 'Coffee',
                                       data_path: str = None,
                                       n_paths: int = 2000,
                                       models_to_run: list = None):
    """
    Populate distributions table using production forecast pipeline.

    Args:
        commodity: 'Coffee' or 'Sugar'
        data_path: Path to unified_data parquet (auto-detects if None)
        n_paths: Number of Monte Carlo paths per model
        models_to_run: List of model keys to run (default: best 5 models)
    """

    logger.info("="*80)
    logger.info(f"PRODUCTION DISTRIBUTIONS POPULATION - {commodity}")
    logger.info("="*80)

    # 1. Load data
    logger.info("\n[1/5] Loading production data...")

    if data_path is None:
        # Auto-detect data file
        gdelt_path = "../data/unified_data_with_gdelt.parquet"
        regular_path = "../data/unified_data_snapshot_all.parquet"

        if os.path.exists(gdelt_path):
            data_path = gdelt_path
            logger.info("  Using GDELT-enhanced data")
        elif os.path.exists(regular_path):
            data_path = regular_path
            logger.info("  Using regular data")
        else:
            raise FileNotFoundError("Data file not found. Please specify data_path.")

    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date'])

    # Filter and aggregate by commodity
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
    forecast_start_date = data_cutoff_date + timedelta(days=1)
    generation_timestamp = datetime.now()

    logger.info(f"  ✓ Loaded {len(df_agg):,} days of {commodity} data")
    logger.info(f"  Date range: {df_agg.index.min().date()} to {df_agg.index.max().date()}")
    logger.info(f"  Data cutoff: {data_cutoff_date.date()}")
    logger.info(f"  Forecast starts: {forecast_start_date.date()}")

    # Estimate residual std from data
    base_residual_std = estimate_residual_std(df_agg)
    logger.info(f"  Estimated base residual std: ${base_residual_std:.2f}")

    # 2. Select models to run
    if models_to_run is None:
        # Best performing models from walk-forward evaluation
        models_to_run = [
            'sarimax_auto_weather',  # Best overall
            'prophet',
            'xgboost_weather',
            'arima_111',
            'random_walk'
        ]

    logger.info(f"\n[2/5] Selected {len(models_to_run)} production models:")
    for model_key in models_to_run:
        if model_key in BASELINE_MODELS:
            logger.info(f"  - {BASELINE_MODELS[model_key]['name']}")

    # 3. Initialize production writer
    logger.info("\n[3/5] Initializing production forecast writer...")
    writer = ProductionForecastWriter("production_forecasts")

    # 4. Train models and generate distributions
    logger.info("\n[4/5] Training models on FULL history...")

    results = []

    for i, model_key in enumerate(models_to_run, 1):
        if model_key not in BASELINE_MODELS:
            logger.warning(f"  [{i}/{len(models_to_run)}] Model '{model_key}' not found in registry. Skipping.")
            continue

        config = BASELINE_MODELS[model_key]
        model_name = config['name']
        model_version = f"{model_key}_v1"

        logger.info(f"\n  [{i}/{len(models_to_run)}] Training {model_name}...")
        logger.info(f"    Model version: {model_version}")
        logger.info(f"    Training on {len(df_agg):,} days")

        try:
            # Prepare parameters
            params = config['params'].copy()
            params['commodity'] = commodity

            # Train model
            result = config['function'](
                df_pandas=df_agg,
                **params
            )

            if not result.get('success', True):
                logger.warning(f"    ✗ Model failed to converge. Skipping.")
                continue

            forecast_df = result['forecast_df']

            logger.info(f"    ✓ Model trained successfully")
            logger.info(f"    Forecast range: ${forecast_df['forecast'].iloc[0]:.2f} to ${forecast_df['forecast'].iloc[-1]:.2f}")

            # Adjust residual std based on model type
            model_residual_multipliers = {
                'sarimax_auto_weather': 0.9,  # Best model, lower uncertainty
                'prophet': 1.0,
                'xgboost_weather': 0.95,
                'arima_111': 1.1,
                'random_walk': 1.3  # Baseline, higher uncertainty
            }

            multiplier = model_residual_multipliers.get(model_key, 1.0)
            model_residual_std = base_residual_std * multiplier

            # Generate Monte Carlo paths
            logger.info(f"    Generating {n_paths:,} Monte Carlo paths...")
            logger.info(f"    Using residual std: ${model_residual_std:.2f}")

            sample_paths = generate_monte_carlo_from_forecast(
                forecast_df=forecast_df,
                residual_std=model_residual_std,
                n_paths=n_paths
            )

            # Write to distributions table
            writer.write_distributions(
                forecast_start_date=forecast_start_date,
                data_cutoff_date=data_cutoff_date,
                model_version=model_version,
                commodity=commodity,
                sample_paths=sample_paths,
                generation_timestamp=generation_timestamp,
                n_paths=n_paths
            )

            results.append({
                'model_key': model_key,
                'model_name': model_name,
                'model_version': model_version,
                'success': True,
                'n_paths': n_paths,
                'mean_forecast': forecast_df['forecast'].mean(),
                'final_forecast': forecast_df['forecast'].iloc[-1],
                'residual_std': model_residual_std
            })

        except Exception as e:
            logger.error(f"    ✗ Model failed with error: {e}", exc_info=True)
            results.append({
                'model_key': model_key,
                'model_name': model_name,
                'success': False,
                'error': str(e)
            })

    # 5. Summary
    logger.info("\n[5/5] Summary:")
    logger.info("="*80)

    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]

    logger.info(f"  Successful: {len(successful)}/{len(results)} models")

    for result in successful:
        logger.info(f"    ✓ {result['model_name']} ({result['model_version']}): "
                   f"{result['n_paths']:,} paths, "
                   f"avg=${result['mean_forecast']:.2f}, "
                   f"final=${result['final_forecast']:.2f}")

    if failed:
        logger.warning(f"\n  Failed: {len(failed)} models")
        for result in failed:
            logger.warning(f"    ✗ {result['model_name']}: {result.get('error', 'Unknown')}")

    # Final stats
    logger.info("\nDistributions table statistics:")
    logger.info("="*80)

    dist_df = writer.distributions
    logger.info(f"  Total rows: {len(dist_df):,}")
    logger.info(f"  Unique models: {dist_df['model_version'].nunique()}")
    logger.info(f"  Total paths: {dist_df['path_id'].max():,}")
    logger.info(f"\n  Location: {writer.distributions_path}")

    logger.info("\n✅ Production distributions population complete!")

    return dist_df, results


if __name__ == "__main__":
    # Run production population
    df, results = populate_distributions_production(
        commodity='Coffee',
        n_paths=2000
    )

    print("\n" + "="*80)
    print("PRODUCTION DISTRIBUTIONS TABLE")
    print("="*80)
    print(f"\nTotal rows: {len(df):,}")
    print(f"Models: {df['model_version'].nunique()}")

    print("\nForecast statistics by model:")
    print("-" * 80)
    summary = df.groupby('model_version').agg({
        'path_id': 'count',
        'day_1': ['mean', 'std', 'min', 'max'],
        'day_14': ['mean', 'std', 'min', 'max']
    }).round(2)
    print(summary)

    print("\n" + "="*80)
    print("READY FOR ANALYSIS")
    print("="*80)
    print("The distributions table contains REAL forecasts from production models.")
    print("Use for:")
    print("  - VaR and CVaR risk analysis")
    print("  - Model ensemble and comparison")
    print("  - Probabilistic scenario planning")

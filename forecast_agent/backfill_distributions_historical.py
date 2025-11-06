"""Backfill distributions table with historical forecasts for backtesting.

This script generates Monte Carlo distributions at multiple historical dates
using walk-forward methodology.

Usage:
    python backfill_distributions_historical.py --n_windows 30 --n_paths 2000

Outputs:
    - Distributions with actuals (path_id=0) for backtesting
    - Multiple forecast dates for VaR/CVaR evaluation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import warnings
import argparse
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from ground_truth.storage.production_writer import ProductionForecastWriter
from ground_truth.config.model_registry import BASELINE_MODELS
from ground_truth.core.logger import get_logger

logger = get_logger(__name__)


def estimate_residual_std(df_pandas: pd.DataFrame, horizon: int = 14) -> float:
    """Estimate residual standard deviation from recent volatility."""
    recent_data = df_pandas['close'].tail(90)
    daily_returns = recent_data.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    daily_vol = volatility / np.sqrt(252)
    avg_price = recent_data.mean()
    residual_std = avg_price * daily_vol * np.sqrt(horizon / 14)
    return max(residual_std, 1.0)


def generate_monte_carlo_from_forecast(forecast_df: pd.DataFrame,
                                        residual_std: float,
                                        n_paths: int = 2000) -> np.ndarray:
    """Generate Monte Carlo paths using Geometric Brownian Motion."""
    forecast_values = forecast_df['forecast'].values

    if len(forecast_values) != 14:
        raise ValueError(f"Expected 14-day forecast, got {len(forecast_values)}")

    sample_paths = np.zeros((n_paths, 14))

    for path_id in range(n_paths):
        path = np.zeros(14)
        path[0] = forecast_values[0] + np.random.normal(0, residual_std)

        for day in range(1, 14):
            drift = forecast_values[day] - forecast_values[day-1]
            shock = np.random.normal(0, residual_std * 0.7)
            path[day] = path[day-1] + drift + shock

        path = np.maximum(path, 10.0)
        sample_paths[path_id, :] = path

    return sample_paths


def backfill_distributions_historical(
    commodity: str = 'Coffee',
    data_path: str = None,
    n_windows: int = 30,
    n_paths: int = 2000,
    models_to_run: list = None,
    initial_train_days: int = 1095,
    step_days: int = 14
):
    """
    Backfill distributions table with historical forecasts.

    Args:
        commodity: 'Coffee' or 'Sugar'
        data_path: Path to unified_data parquet
        n_windows: Number of historical forecast dates
        n_paths: Monte Carlo paths per model per date
        models_to_run: List of model keys (default: production models)
        initial_train_days: Initial training window (3 years)
        step_days: Step size between windows (14 days)
    """

    logger.info("="*80)
    logger.info(f"BACKFILLING HISTORICAL DISTRIBUTIONS - {commodity}")
    logger.info("="*80)
    logger.info(f"  Windows: {n_windows}")
    logger.info(f"  Paths per model: {n_paths}")
    logger.info(f"  Initial training: {initial_train_days} days")
    logger.info(f"  Step size: {step_days} days")

    # Load data
    logger.info("\n[1/4] Loading data...")

    if data_path is None:
        gdelt_path = "../data/unified_data_with_gdelt.parquet"
        regular_path = "../data/unified_data_snapshot_all.parquet"

        if os.path.exists(gdelt_path):
            data_path = gdelt_path
        elif os.path.exists(regular_path):
            data_path = regular_path
        else:
            raise FileNotFoundError("Data file not found")

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

    logger.info(f"  ✓ Loaded {len(df_agg):,} days of {commodity} data")
    logger.info(f"  Date range: {df_agg.index.min().date()} to {df_agg.index.max().date()}")

    # Select models
    if models_to_run is None:
        models_to_run = [
            'sarimax_auto_weather',
            'prophet',
            'xgboost_weather',
            'arima_111',
            'random_walk'
        ]

    logger.info(f"\n[2/4] Selected {len(models_to_run)} models:")
    for model_key in models_to_run:
        if model_key in BASELINE_MODELS:
            logger.info(f"  - {BASELINE_MODELS[model_key]['name']}")

    # Initialize writer
    logger.info("\n[3/4] Initializing writer...")
    writer = ProductionForecastWriter("production_forecasts")

    # Walk-forward backfill
    logger.info(f"\n[4/4] Generating distributions for {n_windows} historical dates...")

    total_rows_written = 0

    for window_idx in range(n_windows):
        train_end_idx = initial_train_days + (window_idx * step_days)

        if train_end_idx >= len(df_agg):
            logger.warning(f"  Reached end of data at window {window_idx}")
            break

        # Split data
        df_train = df_agg.iloc[:train_end_idx]
        data_cutoff_date = df_train.index[-1]
        forecast_start_date = data_cutoff_date + timedelta(days=1)

        # Get actuals for next 14 days
        actuals_end_idx = min(train_end_idx + 14, len(df_agg))
        df_actuals = df_agg.iloc[train_end_idx:actuals_end_idx].copy()
        df_actuals = df_actuals.reset_index()
        df_actuals.columns = ['date', 'commodity', 'actual', 'temp_c', 'humidity_pct', 'precipitation_mm']

        logger.info(f"\n  Window {window_idx + 1}/{n_windows}")
        logger.info(f"    Train: {df_train.index.min().date()} to {data_cutoff_date.date()} ({len(df_train)} days)")
        logger.info(f"    Forecast: {forecast_start_date.date()}")
        logger.info(f"    Actuals: {len(df_actuals)} days available")

        generation_timestamp = datetime.now()
        base_residual_std = estimate_residual_std(df_train)

        window_rows = 0

        # Train each model
        for model_key in models_to_run:
            if model_key not in BASELINE_MODELS:
                continue

            config = BASELINE_MODELS[model_key]
            model_version = f"{model_key}_v1"

            try:
                params = config['params'].copy()
                params['commodity'] = commodity

                result = config['function'](df_pandas=df_train, **params)

                if not result.get('success', True):
                    logger.warning(f"      {config['name']}: Failed to converge")
                    continue

                forecast_df = result['forecast_df']

                # Adjust residual std
                model_residual_multipliers = {
                    'sarimax_auto_weather': 0.9,
                    'prophet': 1.0,
                    'xgboost_weather': 0.95,
                    'arima_111': 1.1,
                    'random_walk': 1.3
                }

                multiplier = model_residual_multipliers.get(model_key, 1.0)
                model_residual_std = base_residual_std * multiplier

                # Generate Monte Carlo paths
                sample_paths = generate_monte_carlo_from_forecast(
                    forecast_df=forecast_df,
                    residual_std=model_residual_std,
                    n_paths=n_paths
                )

                # Write distributions with actuals
                rows_written = writer.write_distributions(
                    forecast_start_date=forecast_start_date,
                    data_cutoff_date=data_cutoff_date,
                    model_version=model_version,
                    commodity=commodity,
                    sample_paths=sample_paths,
                    generation_timestamp=generation_timestamp,
                    n_paths=n_paths,
                    actuals_df=df_actuals if len(df_actuals) > 0 else None
                )

                window_rows += rows_written
                logger.info(f"      {config['name']}: {rows_written} rows (inc. actuals)")

            except Exception as e:
                logger.error(f"      {config['name']}: Error - {e}")
                continue

        total_rows_written += window_rows
        logger.info(f"    Window total: {window_rows} rows | Overall: {total_rows_written:,} rows")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("BACKFILL COMPLETE")
    logger.info("="*80)

    dist_df = writer.distributions
    logger.info(f"  Total rows: {len(dist_df):,}")
    logger.info(f"  Unique forecast dates: {dist_df['forecast_start_date'].nunique()}")
    logger.info(f"  Models: {dist_df['model_version'].nunique()}")
    logger.info(f"  Path ID range: {dist_df['path_id'].min()} to {dist_df['path_id'].max()}")

    # Check actuals
    actuals_count = (dist_df['is_actuals'] == True).sum()
    logger.info(f"  Actuals rows (path_id=0): {actuals_count}")

    logger.info(f"\n  Location: {writer.distributions_path}")
    logger.info("\n✅ Ready to upload to Databricks!")

    return dist_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backfill historical distributions')
    parser.add_argument('--commodity', type=str, default='Coffee', help='Coffee or Sugar')
    parser.add_argument('--n_windows', type=int, default=30, help='Number of historical dates')
    parser.add_argument('--n_paths', type=int, default=2000, help='Monte Carlo paths per model')
    parser.add_argument('--initial_train_days', type=int, default=1095, help='Initial training days (3 years)')
    parser.add_argument('--step_days', type=int, default=14, help='Step size between windows')

    args = parser.parse_args()

    df = backfill_distributions_historical(
        commodity=args.commodity,
        n_windows=args.n_windows,
        n_paths=args.n_paths,
        initial_train_days=args.initial_train_days,
        step_days=args.step_days
    )

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Review generated distributions:")
    print(f"   - Total rows: {len(df):,}")
    print(f"   - Forecast dates: {df['forecast_start_date'].nunique()}")
    print("")
    print("2. Upload to Databricks:")
    print("   python upload_distributions_to_databricks.py")

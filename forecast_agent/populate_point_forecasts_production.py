"""Populate point_forecasts table using PRODUCTION forecast pipeline.

This script generates point forecasts with prediction intervals for all production models.

Usage:
    python populate_point_forecasts_production.py

Outputs:
    - production_forecasts/point_forecasts.parquet (local)
    - commodity.forecast.point_forecasts (Databricks)
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


def populate_point_forecasts_production(commodity: str = 'Coffee',
                                         data_path: str = None,
                                         models_to_run: list = None):
    """
    Populate point_forecasts table using production forecast pipeline.

    Args:
        commodity: 'Coffee' or 'Sugar'
        data_path: Path to unified_data parquet (auto-detects if None)
        models_to_run: List of model keys to run (default: best 5 models)
    """

    logger.info("="*80)
    logger.info(f"PRODUCTION POINT FORECASTS POPULATION - {commodity}")
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

    # 4. Train models and generate point forecasts
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
            prediction_intervals = result.get('prediction_intervals', None)

            logger.info(f"    ✓ Model trained successfully")
            logger.info(f"    Forecast range: ${forecast_df['forecast'].iloc[0]:.2f} to ${forecast_df['forecast'].iloc[-1]:.2f}")

            # Write to point_forecasts table
            writer.write_point_forecasts(
                forecast_df=forecast_df,
                model_version=model_version,
                commodity=commodity,
                data_cutoff_date=data_cutoff_date,
                generation_timestamp=generation_timestamp,
                prediction_intervals=prediction_intervals,
                model_success=True,
                actuals_df=None  # No actuals for future dates
            )

            results.append({
                'model_key': model_key,
                'model_name': model_name,
                'model_version': model_version,
                'success': True,
                'mean_forecast': forecast_df['forecast'].mean(),
                'final_forecast': forecast_df['forecast'].iloc[-1]
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
                   f"avg=${result['mean_forecast']:.2f}, "
                   f"final=${result['final_forecast']:.2f}")

    if failed:
        logger.warning(f"\n  Failed: {len(failed)} models")
        for result in failed:
            logger.warning(f"    ✗ {result['model_name']}: {result.get('error', 'Unknown')}")

    # Final stats
    logger.info("\nPoint forecasts table statistics:")
    logger.info("="*80)

    pf_df = writer.point_forecasts
    logger.info(f"  Total rows: {len(pf_df):,}")
    logger.info(f"  Unique models: {pf_df['model_version'].nunique()}")
    logger.info(f"  Forecast dates: {pf_df['forecast_date'].min()} to {pf_df['forecast_date'].max()}")
    logger.info(f"\n  Location: {writer.point_forecasts_path}")

    logger.info("\n✅ Production point forecasts population complete!")

    return pf_df, results


if __name__ == "__main__":
    # Run production population
    df, results = populate_point_forecasts_production(
        commodity='Coffee'
    )

    print("\n" + "="*80)
    print("PRODUCTION POINT FORECASTS TABLE")
    print("="*80)
    print(f"\nTotal rows: {len(df):,}")
    print(f"Models: {df['model_version'].nunique()}")

    print("\nForecast statistics by model:")
    print("-" * 80)
    summary = df.groupby('model_version').agg({
        'forecast_mean': ['mean', 'std', 'min', 'max'],
        'day_ahead': ['min', 'max']
    }).round(2)
    print(summary)

    print("\n" + "="*80)
    print("READY FOR UPLOAD TO DATABRICKS")
    print("="*80)
    print("Next step: Run upload_point_forecasts_to_databricks.py")

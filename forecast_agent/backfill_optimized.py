"""
Optimized Backfill Script with In-Memory Data Loading

Key Optimizations:
1. Pre-load ALL historical data once (single DB query)
2. Slice data in memory for each forecast date (eliminates 2,872 DB queries)
3. Larger batch sizes for writes (100 instead of 50)
4. Progress tracking with ETA
5. Parallel-friendly (can run multiple date ranges simultaneously)

Performance: ~10-50x faster than backfill_rolling_window.py

Usage:
    # Optimized backfill for 2024
    python backfill_optimized.py --commodity Coffee --models naive --train-frequency semiannually --start-date 2024-01-01 --end-date 2024-12-31
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent))

from databricks import sql
from ground_truth.config.model_registry import BASELINE_MODELS
from utils.model_persistence import load_model
from utils.monte_carlo_simulation import generate_monte_carlo_paths
from ground_truth.models import naive, random_walk, arima, sarimax, xgboost_model, prophet_model


# Model predict functions (inference only)
MODEL_PREDICT_FUNCTIONS = {
    'naive': naive.naive_predict,
    'random_walk': random_walk.random_walk_predict,
    'arima_111': arima.arima_predict,
    'sarimax_auto': sarimax.sarimax_predict,
    'sarimax_auto_weather': sarimax.sarimax_predict,
    'sarimax_auto_weather_seasonal': sarimax.sarimax_predict,
    'xgboost': None,  # TODO
    'prophet': prophet_model.prophet_predict,
}


def get_training_dates(start_date: date, end_date: date, frequency: str) -> List[date]:
    """Generate training dates based on frequency."""
    training_dates = []
    current = start_date

    frequency_map = {
        'daily': timedelta(days=1),
        'weekly': timedelta(days=7),
        'biweekly': timedelta(days=14),
        'monthly': relativedelta(months=1),
        'quarterly': relativedelta(months=3),
        'semiannually': relativedelta(months=6),
        'annually': relativedelta(years=1)
    }

    if frequency not in frequency_map:
        raise ValueError(f"Unknown frequency: {frequency}")

    delta = frequency_map[frequency]

    while current <= end_date:
        training_dates.append(current)
        if isinstance(delta, timedelta):
            current = current + delta
        else:
            current = current + delta

    return training_dates


def load_all_historical_data(connection, commodity: str) -> pd.DataFrame:
    """
    Load ALL historical data once (single DB query).
    This eliminates the need for 2,872 separate queries.
    """
    cursor = connection.cursor()

    query = f"""
        SELECT
            date,
            close,
            open,
            high,
            low,
            volume,
            temp_mean_c,
            humidity_mean_pct,
            precipitation_mm,
            vix,
            cop_usd
        FROM commodity.silver.unified_data
        WHERE commodity = '{commodity}'
        ORDER BY date
    """

    print(f"   Loading all historical {commodity} data...")
    start_time = time.time()

    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    df = pd.DataFrame(rows, columns=columns)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    elapsed = time.time() - start_time
    print(f"   ✓ Loaded {len(df):,} days in {elapsed:.1f}s")

    cursor.close()
    return df


def get_existing_forecasts(connection, commodity: str, model_name: str) -> set:
    """Get set of forecast dates that already exist (for resume mode)."""
    cursor = connection.cursor()

    query = f"""
        SELECT DISTINCT forecast_start_date
        FROM commodity.forecast.distributions
        WHERE commodity = '{commodity}'
          AND model_version = '{model_name}'
          AND is_actuals = FALSE
    """

    cursor.execute(query)
    existing = {row[0] for row in cursor.fetchall()}
    cursor.close()

    return existing


def generate_sample_paths(
    point_forecast: pd.Series,
    fitted_model_dict: Dict,
    training_df: pd.DataFrame,
    num_paths: int = 2000
) -> np.ndarray:
    """
    Generate Monte Carlo paths using model-based simulation.

    This delegates to the centralized monte_carlo_simulation module which:
    - For SARIMA: Uses statsmodels' simulate() to generate from actual ARIMA process
    - For XGBoost: Uses GBM with estimated volatility
    - For other models: Uses appropriate stochastic process
    """
    horizon = len(point_forecast)

    # Create forecast DataFrame in expected format
    forecast_df = pd.DataFrame({
        'forecast': point_forecast.values
    })

    # Use centralized model-based simulation
    paths_list = generate_monte_carlo_paths(
        fitted_model=fitted_model_dict,
        forecast_df=forecast_df,
        n_paths=num_paths,
        horizon=horizon,
        training_df=training_df
    )

    # Convert list of dicts to numpy array
    paths = np.array([path['values'] for path in paths_list])

    return paths


def write_forecasts_batch(
    connection,
    batch_data: List[tuple],
    commodity: str,
    model_name: str
):
    """Write a batch of forecasts to database."""
    cursor = connection.cursor()

    insert_sql = """
    INSERT INTO commodity.forecast.distributions (
        forecast_start_date, forecast_end_date, commodity, model_version,
        path_id, is_actuals,
        day_1, day_2, day_3, day_4, day_5, day_6, day_7,
        day_8, day_9, day_10, day_11, day_12, day_13, day_14,
        year, month
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    cursor.executemany(insert_sql, batch_data)
    cursor.close()


def main():
    parser = argparse.ArgumentParser(description='Optimized backfill with in-memory data loading')
    parser.add_argument('--commodity', type=str, required=True, choices=['Coffee', 'Sugar'])
    parser.add_argument('--models', type=str, nargs='+', required=True)
    parser.add_argument('--train-frequency', type=str, required=True,
                        choices=['daily', 'weekly', 'biweekly', 'monthly', 'quarterly', 'semiannually', 'annually'])
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of forecasts to write per batch (default: 100)')
    parser.add_argument('--num-paths', type=int, default=2000,
                        help='Number of Monte Carlo paths (default: 2000)')

    args = parser.parse_args()

    # Load credentials
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
    DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")

    if not all([DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH]):
        print("ERROR: Missing Databricks credentials")
        sys.exit(1)

    print("=" * 80)
    print("OPTIMIZED BACKFILL - In-Memory Data Loading")
    print("=" * 80)
    print(f"Commodity: {args.commodity}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Training Frequency: {args.train_frequency}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 80)

    # Connect to Databricks
    print("\n📡 Connecting to Databricks...")
    connection = sql.connect(
        server_hostname=DATABRICKS_HOST.replace('https://', ''),
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )
    print("✅ Connected")

    # Load ALL historical data once (KEY OPTIMIZATION)
    print(f"\n📊 Phase 1: Load All Historical Data")
    full_data = load_all_historical_data(connection, args.commodity)

    # Determine date range
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    else:
        start_date = full_data.index[1095].date()  # 3 years from start

    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    else:
        end_date = full_data.index[-1].date()

    # Generate forecast dates
    forecast_dates = get_training_dates(start_date, end_date, args.train_frequency)
    print(f"\n📅 Phase 2: Generate {len(forecast_dates)} Forecasts")
    print(f"   Date range: {forecast_dates[0]} to {forecast_dates[-1]}")

    # Process each model
    for model_key in args.models:
        model_config = BASELINE_MODELS[model_key]
        model_name = model_config['name']

        print(f"\n{'='*80}")
        print(f"Model: {model_name} ({model_key})")
        print(f"{'='*80}")

        # Get existing forecasts (resume mode)
        existing = get_existing_forecasts(connection, args.commodity, model_name)
        print(f"   Found {len(existing)} existing forecasts - will skip")

        # Filter to only new dates
        new_dates = [d for d in forecast_dates if str(d) not in existing]
        print(f"   Generating {len(new_dates)} new forecasts...")

        if len(new_dates) == 0:
            print("   ⏩ All forecasts already exist - skipping")
            continue

        # Batch processing
        batch_data = []
        forecasts_generated = 0
        start_time = time.time()

        for idx, forecast_date in enumerate(new_dates, 1):
            # Slice data in memory (FAST - no DB query!)
            cutoff_date = pd.Timestamp(forecast_date)
            training_df = full_data[full_data.index <= cutoff_date].tail(90)  # Last 90 days

            if len(training_df) < 30:
                continue

            # Load pretrained model
            fitted_model = load_model(
                connection,
                args.commodity,
                model_name,
                str(forecast_date),
                model_version="v1.0"
            )

            if fitted_model is None:
                continue

            # Generate forecast (inference only - no training!)
            predict_fn = MODEL_PREDICT_FUNCTIONS.get(model_key)
            if predict_fn is None:
                continue

            try:
                forecast_df = predict_fn(fitted_model, horizon=14)

                # Generate Monte Carlo paths using model-based simulation
                paths = generate_sample_paths(
                    forecast_df['yhat'],
                    fitted_model,
                    training_df,
                    num_paths=args.num_paths
                )

                # Prepare batch insert data
                forecast_end_date = (pd.Timestamp(forecast_date) + timedelta(days=13)).date()
                year = forecast_date.year
                month = forecast_date.month

                for path_id in range(args.num_paths):
                    path_values = paths[path_id, :].tolist()

                    batch_data.append((
                        str(forecast_date), str(forecast_end_date),
                        args.commodity, model_name,
                        path_id, False,
                        *path_values,
                        year, month
                    ))

                forecasts_generated += 1

                # Write batch when full
                if len(batch_data) >= args.batch_size * args.num_paths:
                    write_forecasts_batch(connection, batch_data, args.commodity, model_name)

                    elapsed = time.time() - start_time
                    rate = forecasts_generated / elapsed
                    remaining = len(new_dates) - idx
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta_minutes = eta_seconds / 60

                    print(f"   Progress: {idx}/{len(new_dates)} forecasts "
                          f"({idx/len(new_dates)*100:.1f}%) | "
                          f"Rate: {rate:.1f} forecasts/sec | "
                          f"ETA: {eta_minutes:.1f} min")

                    batch_data = []

            except Exception as e:
                print(f"   ❌ Error on {forecast_date}: {e}")
                continue

        # Write remaining batch
        if batch_data:
            write_forecasts_batch(connection, batch_data, args.commodity, model_name)

        elapsed = time.time() - start_time
        print(f"\n✅ {model_name} Complete: {forecasts_generated} forecasts in {elapsed/60:.1f} minutes")
        if forecasts_generated > 0:
            print(f"   Average: {elapsed/forecasts_generated:.2f} seconds per forecast")

    connection.close()

    print("\n" + "=" * 80)
    print("BACKFILL COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

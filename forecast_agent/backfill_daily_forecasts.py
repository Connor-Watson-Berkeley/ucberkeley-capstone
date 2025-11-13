"""
Backfill daily forecast windows for trading agent backtesting.

Generates one forecast per day (instead of weekly) to enable realistic
trading backtests where a fresh forecast is available every single day.

Usage:
    # Backfill all missing dates
    python backfill_daily_forecasts.py --commodity Coffee --models sarimax_auto_weather_v1 xgboost_weather_v1

    # Backfill specific date range
    python backfill_daily_forecasts.py --commodity Coffee --start-date 2020-01-01 --end-date 2025-11-12

    # Resume from checkpoint
    python backfill_daily_forecasts.py --commodity Coffee --resume

    # Dry run to see what would be generated
    python backfill_daily_forecasts.py --commodity Coffee --dry-run

Strategy:
    - Expanding window: Each day trains on ALL data up to cutoff
    - Data leakage prevention: forecast_start_date > data_cutoff_date
    - Resumable: Checks existing dates in distributions table, skips them
    - Parallel: Can run multiple models concurrently via CLI args
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Tuple
import sys
import os
from pathlib import Path

# Add forecast_agent to path
sys.path.insert(0, str(Path(__file__).parent))

from databricks import sql
from ground_truth.config.model_registry import BASELINE_MODELS
from ground_truth.storage.production_writer import ProductionForecastWriter


def get_existing_forecast_dates(connection, commodity: str, model_version: str) -> List[date]:
    """
    Query existing forecast_start_dates from distributions table.

    Returns:
        List of dates that already have forecasts
    """
    cursor = connection.cursor()
    cursor.execute(f"""
        SELECT DISTINCT forecast_start_date
        FROM commodity.forecast.distributions
        WHERE commodity = '{commodity}'
          AND model_version = '{model_version}'
          AND is_actuals = FALSE
        ORDER BY forecast_start_date
    """)

    existing = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return existing


def get_date_range_to_backfill(
    connection,
    commodity: str,
    model_version: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    min_training_days: int = 365 * 3
) -> List[date]:
    """
    Calculate which dates need forecasts generated.

    Args:
        connection: Databricks connection
        commodity: 'Coffee' or 'Sugar'
        model_version: Model identifier
        start_date: Earliest forecast date (default: first viable date with min training)
        end_date: Latest forecast date (default: yesterday)
        min_training_days: Minimum training days required (default: 3 years)

    Returns:
        List of dates to generate forecasts for
    """
    cursor = connection.cursor()

    # Get earliest available data date
    cursor.execute(f"""
        SELECT MIN(date) as earliest_date
        FROM commodity.bronze.market
        WHERE commodity = '{commodity}'
    """)
    earliest_data_date = cursor.fetchall()[0][0]

    # Calculate first viable forecast date (earliest_data + min_training_days)
    if start_date is None:
        start_date = earliest_data_date + timedelta(days=min_training_days)

    # Default end date is yesterday (don't forecast today since market may still be open)
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    # Get existing forecast dates
    existing = set(get_existing_forecast_dates(connection, commodity, model_version))

    # Generate all dates in range
    all_dates = []
    current = start_date
    while current <= end_date:
        if current not in existing:
            all_dates.append(current)
        current += timedelta(days=1)

    cursor.close()

    print(f"\n📅 Date Range Analysis:")
    print(f"  Earliest data: {earliest_data_date}")
    print(f"  First viable forecast: {start_date} (after {min_training_days} days training)")
    print(f"  End date: {end_date}")
    print(f"  Total dates in range: {(end_date - start_date).days + 1}")
    print(f"  Already have forecasts: {len(existing)}")
    print(f"  Need to generate: {len(all_dates)}")

    return all_dates


def load_training_data(
    connection,
    commodity: str,
    cutoff_date: date
) -> pd.DataFrame:
    """
    Load all available data up to (and including) cutoff_date.

    This is the expanding window approach - each day gets more training data.

    Args:
        connection: Databricks connection
        commodity: 'Coffee' or 'Sugar'
        cutoff_date: Last date to include in training

    Returns:
        DataFrame with columns: [date, close, ...features]
    """
    cursor = connection.cursor()

    # Load market data with unified_data features
    # TODO: Update this query to use commodity.silver.unified_data once weather_v2 is integrated
    query = f"""
        SELECT
            date,
            close,
            open,
            high,
            low,
            volume
        FROM commodity.bronze.market
        WHERE commodity = '{commodity}'
          AND date <= '{cutoff_date}'
        ORDER BY date
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    df = pd.DataFrame(rows, columns=columns)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    cursor.close()
    return df


def generate_forecast_for_date(
    forecast_start_date: date,
    training_df: pd.DataFrame,
    model_config: Dict,
    commodity: str,
    n_paths: int = 2000,
    forecast_horizon: int = 14
) -> Dict:
    """
    Generate forecast for a single date.

    Args:
        forecast_start_date: First day of forecast window
        training_df: Training data (all data up to cutoff)
        model_config: Model configuration from registry
        commodity: 'Coffee' or 'Sugar'
        n_paths: Number of Monte Carlo paths
        forecast_horizon: Days to forecast

    Returns:
        Dict with forecast results ready for distributions table
    """
    # Get model function and params
    model_fn = model_config['function']
    model_params = model_config['params'].copy()

    # Override horizon if needed
    model_params['horizon'] = forecast_horizon

    try:
        # Call model function with commodity parameter
        result = model_fn(df_pandas=training_df, commodity=commodity, **model_params)

        forecast_df = result['forecast_df']

        # Generate Monte Carlo paths for uncertainty quantification
        # Use forecast std if available, otherwise estimate from historical volatility
        if 'std' in result:
            forecast_std = result['std']
        else:
            # Estimate std from recent price changes
            returns = training_df['close'].pct_change().dropna()
            daily_std = returns.std()
            forecast_std = training_df['close'].iloc[-1] * daily_std

        # Generate paths
        paths = []
        for path_id in range(1, n_paths + 1):
            # Add random noise to point forecast
            noise = np.random.normal(0, forecast_std, len(forecast_df))
            path_forecast = forecast_df['forecast'].values + noise
            paths.append({
                'path_id': path_id,
                'values': path_forecast.tolist()
            })

        return {
            'success': True,
            'forecast_df': forecast_df,
            'paths': paths,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'forecast_df': None,
            'paths': None,
            'error': str(e)
        }


def write_to_distributions_table(
    connection,
    forecast_start_date: date,
    data_cutoff_date: date,
    paths: List[Dict],
    model_version: str,
    commodity: str
):
    """
    Write forecast paths to commodity.forecast.distributions table.

    Schema:
        path_id (int): 1-2000 for forecasts, 0 for actuals
        forecast_start_date (date): First day of 14-day window
        data_cutoff_date (date): Last training date
        generation_timestamp (timestamp): When generated
        model_version (string): Model identifier
        commodity (string): Coffee or Sugar
        day_1 to day_14 (float): Forecasted prices
        is_actuals (boolean): FALSE for forecasts
        has_data_leakage (boolean): Should be FALSE
    """
    cursor = connection.cursor()

    generation_timestamp = datetime.now()

    # Build INSERT statement with batch insert (more efficient than row-by-row)
    # Use VALUES clause with multiple rows
    values_rows = []

    for path in paths:
        path_id = path['path_id']
        values = path['values']

        # Create day values, pad if less than 14 days
        day_values = []
        for i in range(14):
            if i < len(values):
                day_values.append(f"{values[i]:.2f}")
            else:
                day_values.append("NULL")

        # Format row for INSERT
        has_data_leakage = 1 if forecast_start_date <= data_cutoff_date else 0
        row_sql = f"({path_id}, '{forecast_start_date}', '{data_cutoff_date}', '{generation_timestamp}', '{model_version}', '{commodity}', {', '.join(day_values)}, FALSE, {has_data_leakage})"
        values_rows.append(row_sql)

    # Batch insert in chunks of 500 rows (Databricks limit)
    chunk_size = 500
    for i in range(0, len(values_rows), chunk_size):
        chunk = values_rows[i:i+chunk_size]

        insert_sql = f"""
        INSERT INTO commodity.forecast.distributions
        (path_id, forecast_start_date, data_cutoff_date, generation_timestamp,
         model_version, commodity, day_1, day_2, day_3, day_4, day_5, day_6, day_7,
         day_8, day_9, day_10, day_11, day_12, day_13, day_14, is_actuals, has_data_leakage)
        VALUES {', '.join(chunk)}
        """

        cursor.execute(insert_sql)

    cursor.close()

    print(f"  ✅ Wrote {len(paths)} paths to distributions table")


def backfill_daily_forecasts(
    commodity: str,
    model_versions: List[str],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    resume: bool = True,
    dry_run: bool = False,
    databricks_host: str = None,
    databricks_token: str = None,
    databricks_http_path: str = None
):
    """
    Main backfill orchestration.

    Args:
        commodity: 'Coffee' or 'Sugar'
        model_versions: List of model keys from BASELINE_MODELS
        start_date: First forecast date (default: auto-calculate)
        end_date: Last forecast date (default: yesterday)
        resume: Skip dates that already exist (default: True)
        dry_run: Print plan without executing (default: False)
        databricks_host: Databricks workspace URL
        databricks_token: Databricks access token
        databricks_http_path: Warehouse HTTP path
    """
    # Load credentials from environment if not provided
    if not databricks_host:
        databricks_host = os.getenv('DATABRICKS_HOST')
    if not databricks_token:
        databricks_token = os.getenv('DATABRICKS_TOKEN')
    if not databricks_http_path:
        databricks_http_path = os.getenv('DATABRICKS_HTTP_PATH')

    # Connect to Databricks
    connection = sql.connect(
        server_hostname=databricks_host.replace('https://', ''),
        http_path=databricks_http_path,
        access_token=databricks_token
    )

    print(f"\n{'='*80}")
    print(f"Daily Forecast Backfill - {commodity}")
    print(f"{'='*80}")
    print(f"Models: {', '.join(model_versions)}")
    print(f"Resume mode: {resume}")
    print(f"Dry run: {dry_run}")

    for model_version in model_versions:
        print(f"\n🔧 Processing model: {model_version}")

        # Get model config
        if model_version not in BASELINE_MODELS:
            print(f"  ⚠️  Model '{model_version}' not found in registry, skipping")
            continue

        model_config = BASELINE_MODELS[model_version]

        # Get dates to backfill
        dates_to_process = get_date_range_to_backfill(
            connection, commodity, model_version, start_date, end_date
        )

        if len(dates_to_process) == 0:
            print(f"  ✅ All dates already have forecasts for {model_version}")
            continue

        if dry_run:
            print(f"\n  🔍 DRY RUN - Would generate {len(dates_to_process)} forecasts:")
            print(f"     First: {dates_to_process[0]}")
            print(f"     Last: {dates_to_process[-1]}")
            continue

        # Process each date
        success_count = 0
        error_count = 0

        for i, forecast_date in enumerate(dates_to_process, 1):
            cutoff_date = forecast_date - timedelta(days=1)

            print(f"\n  [{i}/{len(dates_to_process)}] Forecast for {forecast_date}")
            print(f"     Training cutoff: {cutoff_date}")

            # Load training data (expanding window)
            training_df = load_training_data(connection, commodity, cutoff_date)
            print(f"     Training samples: {len(training_df)} days")

            # Generate forecast
            result = generate_forecast_for_date(
                forecast_date,
                training_df,
                model_config,
                commodity
            )

            if not result['success']:
                print(f"     ❌ Error: {result['error']}")
                error_count += 1
                continue

            # Write to distributions table
            write_to_distributions_table(
                connection,
                forecast_date,
                cutoff_date,
                result['paths'],
                model_version,
                commodity
            )

            success_count += 1

            # Progress update every 50 forecasts
            if i % 50 == 0:
                print(f"\n  📊 Progress: {i}/{len(dates_to_process)} ({100*i/len(dates_to_process):.1f}%)")
                print(f"     Successful: {success_count}, Errors: {error_count}")

        print(f"\n  ✅ Completed {model_version}")
        print(f"     Successful: {success_count}/{len(dates_to_process)}")
        print(f"     Errors: {error_count}/{len(dates_to_process)}")

    connection.close()

    print(f"\n{'='*80}")
    print(f"Backfill Complete")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Backfill daily forecast windows')
    parser.add_argument('--commodity', required=True, choices=['Coffee', 'Sugar'],
                       help='Commodity to backfill')
    parser.add_argument('--models', nargs='+',
                       help='Model versions to backfill (default: all models)')
    parser.add_argument('--start-date', type=lambda s: datetime.strptime(s, '%Y-%m-%d').date(),
                       help='First forecast date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=lambda s: datetime.strptime(s, '%Y-%m-%d').date(),
                       help='Last forecast date (YYYY-MM-DD)')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Skip dates that already exist (default: True)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                       help='Regenerate all dates even if they exist')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without executing')

    args = parser.parse_args()

    # Default to all models if not specified
    if args.models is None:
        args.models = list(BASELINE_MODELS.keys())

    backfill_daily_forecasts(
        commodity=args.commodity,
        model_versions=args.models,
        start_date=args.start_date,
        end_date=args.end_date,
        resume=args.resume,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()

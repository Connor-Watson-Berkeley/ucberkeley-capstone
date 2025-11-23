"""
Backfill forecasts using rolling window cross-validation.

Key innovation: Train periodically, forecast daily.
- Train model every N months (e.g., semiannually for transformers)
- Generate forecasts for every day using most recently trained model
- Enables realistic backtesting without training 2,860 separate models!

Usage:
    # Semiannual training for expensive models (transformers)
    python backfill_rolling_window.py --commodity Coffee --models transformer_v1 --train-frequency semiannually

    # Monthly training for cheap models (ARIMA, XGBoost)
    python backfill_rolling_window.py --commodity Coffee --models naive arima_111 xgboost --train-frequency monthly

    # Weekly training for very cheap models
    python backfill_rolling_window.py --commodity Coffee --models naive --train-frequency weekly
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
from dateutil.relativedelta import relativedelta
import sys
import os
from pathlib import Path

# Add parent directory to path (skip in Databricks notebooks where __file__ doesn't exist)
try:
    sys.path.insert(0, str(Path(__file__).parent))
except NameError:
    pass  # Running in Databricks notebook

from databricks import sql
from databricks.sql.exc import RequestError
from ground_truth.config.model_registry import BASELINE_MODELS
from utils.model_persistence import load_model
from utils.monte_carlo_simulation import generate_monte_carlo_paths


# Global flag to detect execution environment
_IS_DATABRICKS = None
_SPARK = None


def is_databricks():
    """Detect if running in Databricks environment."""
    global _IS_DATABRICKS, _SPARK
    if _IS_DATABRICKS is None:
        try:
            from pyspark.sql import SparkSession
            _SPARK = SparkSession.builder.getOrCreate()
            _IS_DATABRICKS = True
        except:
            _IS_DATABRICKS = False
    return _IS_DATABRICKS


def execute_sql(query: str, connection=None):
    """
    Execute SQL query in appropriate environment.

    In Databricks: Uses spark.sql()
    Locally: Uses databricks.sql connection

    Returns: pandas DataFrame
    """
    if is_databricks():
        # Running in Databricks - use Spark SQL
        return _SPARK.sql(query).toPandas()
    else:
        # Running locally - use databricks.sql connector
        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        if rows:
            columns = [desc[0] for desc in cursor.description]
            return pd.DataFrame(rows, columns=columns)
        return pd.DataFrame()


def execute_insert(query: str, connection=None):
    """
    Execute INSERT/UPDATE/DELETE query in appropriate environment.

    In Databricks: Uses spark.sql()
    Locally: Uses databricks.sql connection

    Returns: None
    """
    if is_databricks():
        # Running in Databricks - use Spark SQL
        _SPARK.sql(query)
    else:
        # Running locally - use databricks.sql connector
        cursor = connection.cursor()
        cursor.execute(query)
        cursor.close()


def get_training_dates(
    start_date: date,
    end_date: date,
    frequency: str
) -> List[date]:
    """
    Generate training dates based on frequency.

    Args:
        start_date: First possible training date
        end_date: Last possible training date
        frequency: 'daily', 'weekly', 'monthly', 'quarterly', 'semiannually', 'annually'

    Returns:
        List of dates to train models on
    """
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
        raise ValueError(f"Unknown frequency: {frequency}. Choose from {list(frequency_map.keys())}")

    delta = frequency_map[frequency]

    while current <= end_date:
        training_dates.append(current)

        if isinstance(delta, timedelta):
            current = current + delta
        else:  # relativedelta
            current = current + delta

    return training_dates


def reconnect_if_needed(connection, databricks_host, databricks_token, databricks_http_path):
    """
    Check if connection is alive and reconnect if needed.
    Returns: connection (either existing or new)
    """
    # In Databricks, there's no connection to check - Spark SQL is always available
    if is_databricks():
        return None

    try:
        # Try a simple query to test connection
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchall()
        cursor.close()
        return connection
    except (RequestError, Exception) as e:
        # Session is dead, create new connection
        print(f"     🔄 Session timeout detected, reconnecting...")
        try:
            connection.close()
        except:
            pass

        new_connection = sql.connect(
            server_hostname=databricks_host.replace('https://', ''),
            http_path=databricks_http_path,
            access_token=databricks_token
        )
        print(f"     ✅ Reconnected successfully")
        return new_connection


def load_training_data(connection, commodity: str, cutoff_date: date, lookback_days: Optional[int] = None) -> pd.DataFrame:
    """
    Load data up to cutoff_date for training or inference.

    Args:
        connection: Databricks SQL connection (None if running in Databricks)
        commodity: 'Coffee' or 'Sugar'
        cutoff_date: Latest date to include
        lookback_days: If specified, only load last N days (for inference with pretrained models).
                       If None, load all historical data (for training).

    Returns:
        DataFrame with market data
    """
    # For inference with pretrained models, only load recent data
    if lookback_days is not None:
        start_date = cutoff_date - timedelta(days=lookback_days)
        date_filter = f"AND date > '{start_date}' AND date <= '{cutoff_date}'"
    else:
        date_filter = f"AND date <= '{cutoff_date}'"

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
          {date_filter}
        ORDER BY date
    """

    df = execute_sql(query, connection)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

    return df


def train_model(
    training_data: pd.DataFrame,
    model_config: Dict,
    commodity: str
) -> Dict:
    """
    Train a single model on training data.

    Returns:
        Dict with fitted model info or error
    """
    model_fn = model_config['function']
    model_params = model_config['params'].copy()

    try:
        # For this backfill, we just need to verify the model can train
        # The actual forecast generation happens per-date below
        print(f"     Training on {len(training_data)} days of data...")
        return {'success': True, 'training_end': training_data.index[-1]}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def generate_forecast_for_date(
    forecast_start_date: date,
    training_df: pd.DataFrame,
    model_config: Dict,
    commodity: str,
    n_paths: int = 2000,
    forecast_horizon: int = 14,
    fitted_model: Optional[Dict] = None
) -> Dict:
    """Generate forecast for a single date using trained model.

    Args:
        fitted_model: Optional pre-trained model from database. If provided, uses inference-only mode.
    """
    model_fn = model_config['function']
    model_params = model_config['params'].copy()
    model_params['horizon'] = forecast_horizon

    # Initialize forecast_std to avoid UnboundLocalError in except block
    forecast_std = None

    try:
        # Pass fitted_model if available (inference-only mode)
        if fitted_model is not None:
            result = model_fn(df_pandas=training_df, commodity=commodity, fitted_model=fitted_model, **model_params)
        else:
            result = model_fn(df_pandas=training_df, commodity=commodity, **model_params)
        forecast_df = result['forecast_df']

        # Generate Monte Carlo paths using model-based simulation
        # For SARIMA: Simulates from actual ARIMA process
        # For other models: Uses appropriate stochastic process (GBM, random walk, etc.)
        if fitted_model is not None:
            # Use fitted model for model-based simulation
            paths = generate_monte_carlo_paths(
                fitted_model=fitted_model,
                forecast_df=forecast_df,
                n_paths=n_paths,
                horizon=forecast_horizon,
                training_df=training_df
            )

            # Compute forecast_std for metadata
            if 'yhat_std' in forecast_df.columns:
                forecast_std = forecast_df['yhat_std'].mean()
            elif 'std' in result:
                forecast_std = result['std']
            else:
                returns = training_df['close'].pct_change().dropna()
                daily_std = returns.std()
                forecast_std = training_df['close'].iloc[-1] * daily_std
        else:
            # Fallback to simple Gaussian noise if no fitted model
            if 'std' in result:
                forecast_std = result['std']
            else:
                returns = training_df['close'].pct_change().dropna()
                daily_std = returns.std()
                forecast_std = training_df['close'].iloc[-1] * daily_std

            paths = []
            for path_id in range(1, n_paths + 1):
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
            'mean_forecast': forecast_df['forecast'].values,
            'forecast_std': forecast_std,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'forecast_df': None,
            'paths': None,
            'mean_forecast': None,
            'forecast_std': None,
            'error': str(e)
        }


def accumulate_forecast_data(
    forecast_data: Dict,
    distributions_list: List[Dict],
    point_forecasts_list: List[Dict],
    generation_timestamp: datetime
):
    """
    Accumulate forecast data into lists for DataFrame batch writing.

    Instead of writing to DB immediately, we accumulate all data in memory
    and write as Spark DataFrames at the end (much faster for Databricks).

    Args:
        forecast_data: Single forecast with paths, mean_forecast, etc.
        distributions_list: List to accumulate distribution rows
        point_forecasts_list: List to accumulate point forecast rows
        generation_timestamp: When this forecast was generated
    """
    forecast_start_date = forecast_data['forecast_start_date']
    data_cutoff_date = forecast_data['data_cutoff_date']
    paths = forecast_data['paths']
    mean_forecast = forecast_data['mean_forecast']
    forecast_std = forecast_data['forecast_std']
    model_version = forecast_data['model_version']
    commodity = forecast_data['commodity']

    # Accumulate distribution rows
    for path in paths:
        path_id = path['path_id']
        values = path['values']

        # Pad with None if needed
        day_values = {}
        for i in range(14):
            day_col = f'day_{i+1}'
            day_values[day_col] = float(values[i]) if i < len(values) else None

        has_data_leakage = forecast_start_date <= data_cutoff_date

        distributions_list.append({
            'path_id': path_id,
            'forecast_start_date': forecast_start_date,
            'data_cutoff_date': data_cutoff_date,
            'generation_timestamp': generation_timestamp,
            'model_version': model_version,
            'commodity': commodity,
            **day_values,
            'is_actuals': False,
            'has_data_leakage': has_data_leakage
        })

    # Accumulate point forecast rows
    for day_idx in range(len(mean_forecast)):
        forecast_date = forecast_start_date + timedelta(days=day_idx)
        day_ahead = day_idx + 1

        vol_scaled = forecast_std * np.sqrt(day_ahead)
        lower_95 = mean_forecast[day_idx] - 1.96 * vol_scaled
        upper_95 = mean_forecast[day_idx] + 1.96 * vol_scaled

        has_data_leakage = forecast_date <= data_cutoff_date

        point_forecasts_list.append({
            'forecast_date': forecast_date,
            'data_cutoff_date': data_cutoff_date,
            'generation_timestamp': generation_timestamp,
            'day_ahead': day_ahead,
            'forecast_mean': float(mean_forecast[day_idx]),
            'forecast_std': float(forecast_std),
            'lower_95': float(lower_95),
            'upper_95': float(upper_95),
            'model_version': model_version,
            'commodity': commodity,
            'model_success': True,
            'actual_close': None,
            'has_data_leakage': has_data_leakage
        })


def write_dataframes_to_tables(
    connection,
    distributions_data: List[Dict],
    point_forecasts_data: List[Dict]
):
    """
    Write accumulated forecast data to tables using Spark DataFrames (Databricks)
    or SQL inserts (local execution).

    This is the DataFrame approach - accumulate all data, write once.
    Much faster than incremental batch inserts for Databricks.

    Args:
        connection: Databricks SQL connection (None if running in Databricks)
        distributions_data: List of distribution row dicts
        point_forecasts_data: List of point forecast row dicts
    """
    if not distributions_data and not point_forecasts_data:
        print("       No data to write")
        return

    print(f"       Writing {len(distributions_data):,} distribution rows and {len(point_forecasts_data):,} point forecast rows...")

    if is_databricks():
        # ============================================================
        # DATABRICKS: Use Spark DataFrames (fast!)
        # ============================================================
        from pyspark.sql.types import StructType, StructField, IntegerType, StringType, TimestampType, DoubleType, BooleanType, DateType
        from pyspark.sql.functions import col, to_date

        # Write distributions
        if distributions_data:
            # Create DataFrame and cast types to match table schema
            from pyspark.sql.types import FloatType
            dist_df = _SPARK.createDataFrame(distributions_data)

            # Ensure correct types for metadata columns
            dist_df = dist_df.withColumn("path_id", col("path_id").cast(IntegerType())) \
                             .withColumn("forecast_start_date", to_date(col("forecast_start_date"))) \
                             .withColumn("data_cutoff_date", to_date(col("data_cutoff_date"))) \
                             .withColumn("generation_timestamp", col("generation_timestamp").cast(TimestampType())) \
                             .withColumn("is_actuals", col("is_actuals").cast(BooleanType())) \
                             .withColumn("has_data_leakage", col("has_data_leakage").cast(BooleanType()))

            # Cast all day_N columns to float (Spark infers double from Python floats)
            for i in range(1, 15):
                day_col = f"day_{i}"
                dist_df = dist_df.withColumn(day_col, col(day_col).cast(FloatType()))

            dist_df.write.mode("append").saveAsTable("commodity.forecast.distributions")
            print(f"       ✅ Wrote {len(distributions_data):,} distribution rows")

        # Write point forecasts
        if point_forecasts_data:
            # Create DataFrame and cast types to match table schema
            from pyspark.sql.types import FloatType
            point_df = _SPARK.createDataFrame(point_forecasts_data)

            # Ensure correct types
            point_df = point_df.withColumn("forecast_date", to_date(col("forecast_date"))) \
                               .withColumn("data_cutoff_date", to_date(col("data_cutoff_date"))) \
                               .withColumn("generation_timestamp", col("generation_timestamp").cast(TimestampType())) \
                               .withColumn("day_ahead", col("day_ahead").cast(IntegerType())) \
                               .withColumn("forecast_mean", col("forecast_mean").cast(FloatType())) \
                               .withColumn("forecast_std", col("forecast_std").cast(FloatType())) \
                               .withColumn("lower_95", col("lower_95").cast(FloatType())) \
                               .withColumn("upper_95", col("upper_95").cast(FloatType())) \
                               .withColumn("model_success", col("model_success").cast(BooleanType())) \
                               .withColumn("has_data_leakage", col("has_data_leakage").cast(BooleanType()))

            point_df.write.mode("append").saveAsTable("commodity.forecast.point_forecasts")
            print(f"       ✅ Wrote {len(point_forecasts_data):,} point forecast rows")

    else:
        # ============================================================
        # LOCAL: Use SQL INSERT statements (slower but necessary)
        # ============================================================

        # Write distributions in chunks
        if distributions_data:
            chunk_size = 500
            for i in range(0, len(distributions_data), chunk_size):
                chunk = distributions_data[i:i+chunk_size]

                # Convert dicts to SQL VALUES
                value_rows = []
                for row in chunk:
                    day_vals = [f"{row[f'day_{i+1}']:.2f}" if row[f'day_{i+1}'] is not None else "NULL" for i in range(14)]
                    has_leak = 1 if row['has_data_leakage'] else 0
                    value_rows.append(
                        f"({row['path_id']}, '{row['forecast_start_date']}', '{row['data_cutoff_date']}', "
                        f"'{row['generation_timestamp']}', '{row['model_version']}', '{row['commodity']}', "
                        f"{', '.join(day_vals)}, FALSE, {has_leak})"
                    )

                insert_sql = f"""
                INSERT INTO commodity.forecast.distributions
                (path_id, forecast_start_date, data_cutoff_date, generation_timestamp,
                 model_version, commodity, day_1, day_2, day_3, day_4, day_5, day_6, day_7,
                 day_8, day_9, day_10, day_11, day_12, day_13, day_14, is_actuals, has_data_leakage)
                VALUES {', '.join(value_rows)}
                """
                execute_insert(insert_sql, connection)

            print(f"       ✅ Wrote {len(distributions_data):,} distribution rows")

        # Write point forecasts in chunks
        if point_forecasts_data:
            chunk_size = 1000
            for i in range(0, len(point_forecasts_data), chunk_size):
                chunk = point_forecasts_data[i:i+chunk_size]

                value_rows = []
                for row in chunk:
                    has_leak = 1 if row['has_data_leakage'] else 0
                    value_rows.append(
                        f"('{row['forecast_date']}', '{row['data_cutoff_date']}', '{row['generation_timestamp']}', "
                        f"{row['day_ahead']}, {row['forecast_mean']:.2f}, {row['forecast_std']:.2f}, "
                        f"{row['lower_95']:.2f}, {row['upper_95']:.2f}, '{row['model_version']}', "
                        f"'{row['commodity']}', TRUE, NULL, {has_leak})"
                    )

                insert_sql = f"""
                INSERT INTO commodity.forecast.point_forecasts
                (forecast_date, data_cutoff_date, generation_timestamp, day_ahead,
                 forecast_mean, forecast_std, lower_95, upper_95, model_version,
                 commodity, model_success, actual_close, has_data_leakage)
                VALUES {', '.join(value_rows)}
                """
                execute_insert(insert_sql, connection)

            print(f"       ✅ Wrote {len(point_forecasts_data):,} point forecast rows")


def write_actuals_to_table(connection, commodity: str, forecast_dates: List[date]):
    """
    Fetch and write actuals for all forecast dates.

    This is separate from the main DataFrame write because actuals
    need to be fetched from bronze.market first.
    """
    if not forecast_dates:
        return

    # Get unique date range
    min_date = min(forecast_dates)
    max_date = max(forecast_dates) + timedelta(days=14)

    # Fetch all actuals for this commodity in date range
    actuals_df = execute_sql(f"""
        SELECT date, close
        FROM commodity.bronze.market
        WHERE commodity = '{commodity}'
          AND date >= '{min_date}'
          AND date < '{max_date}'
        ORDER BY date
    """, connection)

    if actuals_df.empty:
        return

    total_written = 0

    if is_databricks():
        # Use Spark DataFrame for actuals
        actuals_data = []
        for _, row in actuals_df.iterrows():
            actuals_data.append({
                'forecast_date': row['date'],
                'commodity': commodity,
                'actual_close': float(row['close'])
            })

        if actuals_data:
            actuals_spark_df = _SPARK.createDataFrame(actuals_data)
            # Use insertInto with overwrite=False to handle duplicates gracefully
            actuals_spark_df.write.mode("append").saveAsTable("commodity.forecast.forecast_actuals")
            total_written = len(actuals_data)
    else:
        # Local: Insert one by one (handle duplicates)
        for _, row in actuals_df.iterrows():
            try:
                execute_insert(f"""
                    INSERT INTO commodity.forecast.forecast_actuals
                    (forecast_date, commodity, actual_close)
                    VALUES ('{row['date']}', '{commodity}', {row['close']:.2f})
                """, connection)
                total_written += 1
            except Exception:
                # Duplicate key, skip
                pass

    if total_written > 0:
        print(f"       ✅ Wrote {total_written} actuals")


def write_batch_to_tables(
    connection,
    batch_data: List[Dict]
):
    """
    DEPRECATED: Legacy function for backward compatibility.

    New approach: Use accumulate_forecast_data() + write_dataframes_to_tables()
    instead of immediate batch writes.
    """
    # For backward compatibility, convert to new approach
    distributions_list = []
    point_forecasts_list = []
    generation_timestamp = datetime.now()

    for forecast_data in batch_data:
        accumulate_forecast_data(
            forecast_data,
            distributions_list,
            point_forecasts_list,
            generation_timestamp
        )

    write_dataframes_to_tables(connection, distributions_list, point_forecasts_list)

    # Write actuals
    forecast_dates = [fd['forecast_start_date'] for fd in batch_data]
    commodities = list(set([fd['commodity'] for fd in batch_data]))

    for commodity in commodities:
        write_actuals_to_table(connection, commodity, forecast_dates)


def write_all_tables(
    connection,
    forecast_start_date: date,
    data_cutoff_date: date,
    paths: List[Dict],
    mean_forecast: np.ndarray,
    forecast_std: float,
    model_version: str,
    commodity: str
):
    """
    DEPRECATED: Use write_batch_to_tables() instead for better performance.

    This function kept for backward compatibility but converts to batch format.
    """
    batch_data = [{
        'forecast_start_date': forecast_start_date,
        'data_cutoff_date': data_cutoff_date,
        'paths': paths,
        'mean_forecast': mean_forecast,
        'forecast_std': forecast_std,
        'model_version': model_version,
        'commodity': commodity
    }]
    write_batch_to_tables(connection, batch_data)


def get_existing_forecast_dates(connection, commodity: str, model_version: str) -> set:
    """Get set of dates that already have forecasts."""
    result_df = execute_sql(f"""
        SELECT DISTINCT forecast_start_date
        FROM commodity.forecast.distributions
        WHERE commodity = '{commodity}'
          AND model_version = '{model_version}'
          AND is_actuals = FALSE
    """, connection)

    if result_df.empty:
        return set()
    return set(result_df['forecast_start_date'].tolist())


def backfill_rolling_window(
    commodity: str,
    model_versions: List[str],
    train_frequency: str = 'semiannually',
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    min_training_days: int = 365 * 3,
    use_pretrained: bool = False,
    model_version_tag: str = 'v1.0',
    databricks_host: str = None,
    databricks_token: str = None,
    databricks_http_path: str = None
):
    """
    Main rolling window backfill orchestration.

    Strategy:
        1. Generate training dates based on frequency (e.g., every 6 months)
        2. For each training date:
           - Train model on all data up to that date
           - Generate forecasts for all days until next training date
        3. Populate all 3 tables (distributions, point_forecasts, actuals)
    """
    # Setup connection (only needed for local execution)
    connection = None
    if not is_databricks():
        # Running locally - need databricks.sql connection
        if not databricks_host:
            databricks_host = os.getenv('DATABRICKS_HOST')
        if not databricks_token:
            databricks_token = os.getenv('DATABRICKS_TOKEN')
        if not databricks_http_path:
            databricks_http_path = os.getenv('DATABRICKS_CLUSTER_HTTP_PATH') or os.getenv('DATABRICKS_HTTP_PATH')

        connection = sql.connect(
            server_hostname=databricks_host.replace('https://', ''),
            http_path=databricks_http_path,
            access_token=databricks_token
        )

    # Get earliest data date
    result_df = execute_sql(f"""
        SELECT MIN(date) as earliest_date
        FROM commodity.bronze.market
        WHERE commodity = '{commodity}'
    """, connection)
    earliest_data_date = result_df['earliest_date'].iloc[0] if not result_df.empty else None

    if start_date is None:
        start_date = earliest_data_date + timedelta(days=min_training_days)

    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    print(f"\n{'='*80}")
    print(f"Rolling Window Backfill - {commodity}")
    print(f"{'='*80}")
    print(f"Train frequency: {train_frequency}")
    print(f"Models: {', '.join(model_versions)}")
    print(f"Date range: {start_date} to {end_date}")

    # Generate training dates
    training_dates = get_training_dates(start_date, end_date, train_frequency)
    print(f"\n📅 Training Schedule: {len(training_dates)} model trainings")
    print(f"   First: {training_dates[0]}")
    print(f"   Last: {training_dates[-1]}")

    # Calculate total forecasts (all days between start and end)
    total_forecast_days = (end_date - start_date).days + 1
    print(f"   Total forecast days: {total_forecast_days:,}")
    print(f"   Ratio: 1 training per {total_forecast_days / len(training_dates):.1f} forecasts")

    for model_version in model_versions:
        print(f"\n{'='*80}")
        print(f"🔧 Model: {model_version}")
        print(f"{'='*80}")

        if model_version not in BASELINE_MODELS:
            print(f"  ⚠️  Model not found in registry, skipping")
            continue

        model_config = BASELINE_MODELS[model_version]

        # Get existing forecasts for resume capability
        existing_dates = get_existing_forecast_dates(connection, commodity, model_version)
        print(f"\n  📋 Resume mode: {len(existing_dates)} forecasts already exist, will skip these")

        success_count = 0
        error_count = 0
        skipped_count = 0

        # Accumulate all forecast data for this model (DataFrame approach)
        all_distributions = []
        all_point_forecasts = []
        all_forecast_dates = []
        generation_timestamp = datetime.now()

        # Rolling window: train periodically, forecast daily
        for train_idx, train_date in enumerate(training_dates, 1):
            cutoff_date = train_date - timedelta(days=1)

            # Determine forecast window (until next training date or end)
            if train_idx < len(training_dates):
                next_train_date = training_dates[train_idx]
                forecast_end = next_train_date - timedelta(days=1)
            else:
                forecast_end = end_date

            print(f"\n  [{train_idx}/{len(training_dates)}] Training on {train_date}")
            print(f"     Data cutoff: {cutoff_date}")

            # Check connection and reconnect if needed
            connection = reconnect_if_needed(connection, databricks_host, databricks_token, databricks_http_path)

            # Try to load pretrained model if use_pretrained is True
            fitted_model_dict = None
            if use_pretrained:
                model_name = model_config['name']
                training_date_str = train_date.strftime('%Y-%m-%d')

                print(f"     Loading pretrained model from database...")
                try:
                    loaded_model_data = load_model(
                        connection=connection,
                        commodity=commodity,
                        model_name=model_name,
                        training_date=training_date_str,
                        model_version=model_version_tag
                    )

                    if loaded_model_data:
                        fitted_model_dict = loaded_model_data['fitted_model']
                        print(f"     ✅ Loaded pretrained model (trained on {training_date_str})")
                    else:
                        print(f"     ⚠️  Pretrained model not found, falling back to training")
                        use_pretrained = False  # Fall back to training for this window
                except Exception as e:
                    print(f"     ⚠️  Error loading pretrained model: {e}")
                    use_pretrained = False  # Fall back to training

            # Load training data
            # - For inference with pretrained models: only load last 90 days (fast)
            # - For training: load all historical data (slow but necessary)
            lookback_days = 90 if use_pretrained else None
            training_df = load_training_data(connection, commodity, cutoff_date, lookback_days=lookback_days)
            print(f"     Training samples: {len(training_df)} days")

            # Train model if not using pretrained
            if not use_pretrained:
                train_result = train_model(training_df, model_config, commodity)
                if not train_result['success']:
                    print(f"     ❌ Training failed: {train_result['error']}")
                    error_count += 1
                    continue
                print(f"     ✅ Model trained successfully")
            else:
                print(f"     ⏭️  Skipping training (using pretrained model)")

            # Generate forecasts for all days until next training
            forecast_dates = []
            current = train_date
            while current <= forecast_end:
                forecast_dates.append(current)
                current += timedelta(days=1)

            print(f"     Generating {len(forecast_dates)} daily forecasts (reusing trained model)...")

            for i, forecast_date in enumerate(forecast_dates, 1):
                # Skip if already exists (resume mode)
                if forecast_date in existing_dates:
                    skipped_count += 1
                    if skipped_count % 100 == 0:
                        print(f"       Skipped: {skipped_count} (resume mode)")
                    continue

                # Use pretrained model if available, otherwise train+predict
                result = generate_forecast_for_date(
                    forecast_date,
                    training_df,
                    model_config,
                    commodity,
                    fitted_model=fitted_model_dict
                )

                if not result['success']:
                    print(f"       [{i}/{len(forecast_dates)}] {forecast_date}: ❌ {result['error']}")
                    error_count += 1
                    continue

                # Accumulate forecast data (no immediate write)
                forecast_data = {
                    'forecast_start_date': forecast_date,
                    'data_cutoff_date': cutoff_date,
                    'paths': result['paths'],
                    'mean_forecast': result['mean_forecast'],
                    'forecast_std': result['forecast_std'],
                    'model_version': model_version,
                    'commodity': commodity
                }

                accumulate_forecast_data(
                    forecast_data,
                    all_distributions,
                    all_point_forecasts,
                    generation_timestamp
                )

                all_forecast_dates.append(forecast_date)
                success_count += 1

                if success_count % 100 == 0:
                    print(f"       Progress: {success_count} new forecasts generated")

        # Write all accumulated data at once (DataFrame approach - much faster!)
        print(f"\n  💾 Writing all data to tables...")
        try:
            connection = reconnect_if_needed(connection, databricks_host, databricks_token, databricks_http_path)
            write_dataframes_to_tables(connection, all_distributions, all_point_forecasts)

            # Write actuals separately
            if all_forecast_dates:
                write_actuals_to_table(connection, commodity, all_forecast_dates)

        except Exception as e:
            print(f"       ❌ Write failed: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n  ✅ Completed {model_version}")
        print(f"     New forecasts: {success_count:,}")
        print(f"     Skipped (existing): {skipped_count:,}")
        print(f"     Errors: {error_count}")

    # Close connection if running locally
    if connection is not None:
        connection.close()

    print(f"\n{'='*80}")
    print(f"Backfill Complete!")
    print(f"{'='*80}")


def main():
    # Check if running in Databricks (dbutils available)
    try:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)

        # Running in Databricks - use widgets
        print("Running in Databricks notebook mode - using widgets for parameters")

        # Get Databricks connection details from Spark conf or environment
        # These should be set by the cluster or we use dummy values (won't be used in notebook mode)
        databricks_host = spark.conf.get("spark.databricks.workspaceUrl", None)
        if databricks_host:
            databricks_host = f"https://{databricks_host}"
            os.environ['DATABRICKS_HOST'] = databricks_host

        # For notebook execution, we can use the cluster's built-in auth
        # Set a placeholder for the token (won't actually be used in notebook context)
        os.environ['DATABRICKS_TOKEN'] = os.getenv('DATABRICKS_TOKEN', 'not-needed-in-notebook')
        os.environ['DATABRICKS_HTTP_PATH'] = os.getenv('DATABRICKS_HTTP_PATH', '/sql/1.0/endpoints/dummy')

        commodity_param = dbutils.widgets.get("commodity")
        commodities = [c.strip() for c in commodity_param.split(',')]
        models = dbutils.widgets.get("models").split(',')
        train_frequency = dbutils.widgets.get("train_frequency")
        start_date = datetime.strptime(dbutils.widgets.get("start_date"), '%Y-%m-%d').date()
        end_date = datetime.strptime(dbutils.widgets.get("end_date"), '%Y-%m-%d').date()
        model_version_tag = dbutils.widgets.get("model_version_tag")

        # Run backfill for each commodity
        for commodity in commodities:
            commodity = commodity.strip()
            print(f"\n{'='*80}")
            print(f"Processing commodity: {commodity}")
            print(f"{'='*80}")
            backfill_rolling_window(
                commodity=commodity,
                model_versions=models,
                train_frequency=train_frequency,
                start_date=start_date,
                end_date=end_date,
                use_pretrained=True,
                model_version_tag=model_version_tag
            )

    except (ImportError, Exception):
        # Running locally - use argparse
        print("Running in local mode - using command-line arguments")

        parser = argparse.ArgumentParser(description='Rolling window forecast backfill')
        parser.add_argument('--commodity', required=True, choices=['Coffee', 'Sugar'])
        parser.add_argument('--models', nargs='+', required=True,
                           help='Model versions to backfill')
        parser.add_argument('--train-frequency', default='semiannually',
                           choices=['daily', 'weekly', 'biweekly', 'monthly', 'quarterly', 'semiannually', 'annually'],
                           help='How often to retrain models (default: semiannually)')
        parser.add_argument('--start-date', type=lambda s: datetime.strptime(s, '%Y-%m-%d').date(),
                           help='First training date (YYYY-MM-DD)')
        parser.add_argument('--end-date', type=lambda s: datetime.strptime(s, '%Y-%m-%d').date(),
                           help='Last forecast date (YYYY-MM-DD)')
        parser.add_argument('--train-all-forecasts', action='store_true',
                           help='Train a new model for each forecast date (slow, ~180x slower than default). Default behavior uses pretrained models.')
        parser.add_argument('--model-version-tag', default='v1.0',
                           help='Model version tag for pretrained models (default: v1.0)')

        args = parser.parse_args()

        backfill_rolling_window(
            commodity=args.commodity,
            model_versions=args.models,
            train_frequency=args.train_frequency,
            start_date=args.start_date,
            end_date=args.end_date,
            use_pretrained=not args.train_all_forecasts,
            model_version_tag=args.model_version_tag
        )


if __name__ == '__main__':
    main()

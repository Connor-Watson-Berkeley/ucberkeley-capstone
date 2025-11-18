"""
Spark-Parallelized Forecast Backfill for Databricks

Distributes forecast generation across cluster nodes for massive speedup.

Architecture:
    1. Create DataFrame of all (commodity, model, date) tuples to forecast
    2. Broadcast pretrained models to all workers
    3. Use mapPartitions to run forecasts in parallel
    4. Batch write results to Delta tables

Performance:
    - Sequential: ~40-120 hours for 143,750 forecasts
    - Spark (100 cores): ~1-3 hours

Usage (Databricks notebook):
    %run ./backfill_rolling_window_spark

    # Run full backfill
    backfill_all_models_spark(
        commodities=['Coffee', 'Sugar'],
        models=['naive', 'xgboost', 'sarimax_auto_weather'],
        train_frequency='semiannually',
        start_date='2018-01-01',
        end_date='2025-11-17',
        num_partitions=200  # Adjust based on cluster size
    )

Usage (standalone):
    spark-submit backfill_rolling_window_spark.py \
        --commodities Coffee Sugar \
        --models naive xgboost \
        --train-frequency semiannually \
        --start-date 2018-01-01 \
        --end-date 2025-11-17 \
        --num-partitions 200
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Tuple, Optional, Iterator
from dateutil.relativedelta import relativedelta
import sys
import os
from pathlib import Path

# Spark imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).parent))

from ground_truth.config.model_registry import BASELINE_MODELS
from utils.model_persistence import load_model_from_json
from utils.monte_carlo_simulation import generate_monte_carlo_paths


# ============================================================================
# Schema Definitions
# ============================================================================

FORECAST_TASK_SCHEMA = StructType([
    StructField("commodity", StringType(), False),
    StructField("model_version", StringType(), False),
    StructField("forecast_start_date", DateType(), False),
    StructField("data_cutoff_date", DateType(), False),
])

DISTRIBUTION_SCHEMA = StructType([
    StructField("path_id", IntegerType(), False),
    StructField("forecast_start_date", DateType(), False),
    StructField("data_cutoff_date", DateType(), False),
    StructField("generation_timestamp", TimestampType(), False),
    StructField("model_version", StringType(), False),
    StructField("commodity", StringType(), False),
    StructField("day_1", DoubleType(), True),
    StructField("day_2", DoubleType(), True),
    StructField("day_3", DoubleType(), True),
    StructField("day_4", DoubleType(), True),
    StructField("day_5", DoubleType(), True),
    StructField("day_6", DoubleType(), True),
    StructField("day_7", DoubleType(), True),
    StructField("day_8", DoubleType(), True),
    StructField("day_9", DoubleType(), True),
    StructField("day_10", DoubleType(), True),
    StructField("day_11", DoubleType(), True),
    StructField("day_12", DoubleType(), True),
    StructField("day_13", DoubleType(), True),
    StructField("day_14", DoubleType(), True),
    StructField("is_actuals", BooleanType(), False),
    StructField("has_data_leakage", BooleanType(), False),
])


# ============================================================================
# Helper Functions
# ============================================================================

def get_training_dates(
    start_date: date,
    end_date: date,
    frequency: str
) -> List[date]:
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
        else:  # relativedelta
            current = current + delta

    return training_dates


def get_model_for_date(
    forecast_date: date,
    training_dates: List[date]
) -> date:
    """Find the most recent training date for a forecast date."""
    valid_training_dates = [td for td in training_dates if td <= forecast_date]
    if not valid_training_dates:
        raise ValueError(f"No training date found for {forecast_date}")
    return max(valid_training_dates)


# ============================================================================
# Spark UDFs and Worker Functions
# ============================================================================

def generate_forecast_partition(
    partition: Iterator[Tuple],
    pretrained_models: Dict[Tuple, dict],
    model_configs: Dict[str, dict]
) -> Iterator[Dict]:
    """
    Worker function to generate forecasts for a partition of tasks.

    Runs on executor nodes. Each partition processes a batch of forecast tasks.

    Args:
        partition: Iterator of (commodity, model_version, forecast_start_date, data_cutoff_date)
        pretrained_models: Broadcasted dict of {(commodity, model, training_date): fitted_model}
        model_configs: Model registry configs

    Yields:
        Dict with forecast results (distributions table rows)
    """
    from pyspark.sql import SparkSession

    # Get Spark session on worker
    spark = SparkSession.builder.getOrCreate()

    for row in partition:
        commodity = row[0]
        model_version = row[1]
        forecast_start_date = row[2]
        data_cutoff_date = row[3]

        try:
            # Load data from unified_data (up to data_cutoff_date)
            df_spark = spark.sql(f"""
                SELECT date, close, temp_c, humidity_pct, precipitation_mm, vix, cop_usd
                FROM commodity.silver.unified_data
                WHERE commodity = '{commodity}'
                  AND date <= '{data_cutoff_date}'
                ORDER BY date
            """)

            df_pandas = df_spark.toPandas()
            df_pandas['date'] = pd.to_datetime(df_pandas['date'])
            df_pandas = df_pandas.set_index('date')

            if len(df_pandas) < 365:  # Minimum training data
                continue

            # Get pretrained model
            model_config = model_configs.get(model_version)
            if not model_config:
                continue

            # Find which training date to use
            training_dates = pretrained_models.get('training_dates', {}).get((commodity, model_version), [])
            if not training_dates:
                continue

            training_date = get_model_for_date(forecast_start_date, training_dates)
            model_key = (commodity, model_version, training_date)

            fitted_model = pretrained_models.get(model_key)
            if not fitted_model:
                continue

            # Generate forecast using pretrained model
            forecast_function = model_config['function']
            result = forecast_function(
                df_pandas,
                commodity=commodity,
                fitted_model=fitted_model,
                **model_config.get('params', {})
            )

            forecast_df = result['forecast_df']

            # Generate Monte Carlo paths
            mean_forecast = forecast_df['yhat'].values[:14]
            forecast_std = forecast_df.get('yhat_std', pd.Series([3.0] * 14)).values[0]

            paths = generate_monte_carlo_paths(
                mean_forecast=mean_forecast,
                forecast_std=forecast_std,
                n_paths=2000
            )

            # Yield distribution rows
            generation_timestamp = datetime.now()

            for path_id, path in enumerate(paths):
                yield {
                    'path_id': path_id,
                    'forecast_start_date': forecast_start_date,
                    'data_cutoff_date': data_cutoff_date,
                    'generation_timestamp': generation_timestamp,
                    'model_version': model_version,
                    'commodity': commodity,
                    'day_1': float(path[0]) if len(path) > 0 else None,
                    'day_2': float(path[1]) if len(path) > 1 else None,
                    'day_3': float(path[2]) if len(path) > 2 else None,
                    'day_4': float(path[3]) if len(path) > 3 else None,
                    'day_5': float(path[4]) if len(path) > 4 else None,
                    'day_6': float(path[5]) if len(path) > 5 else None,
                    'day_7': float(path[6]) if len(path) > 6 else None,
                    'day_8': float(path[7]) if len(path) > 7 else None,
                    'day_9': float(path[8]) if len(path) > 8 else None,
                    'day_10': float(path[9]) if len(path) > 9 else None,
                    'day_11': float(path[10]) if len(path) > 10 else None,
                    'day_12': float(path[11]) if len(path) > 11 else None,
                    'day_13': float(path[12]) if len(path) > 12 else None,
                    'day_14': float(path[13]) if len(path) > 13 else None,
                    'is_actuals': False,
                    'has_data_leakage': False,
                }

        except Exception as e:
            print(f"Error forecasting {commodity}/{model_version}/{forecast_start_date}: {e}")
            continue


# ============================================================================
# Main Backfill Function
# ============================================================================

def backfill_all_models_spark(
    commodities: List[str] = ['Coffee', 'Sugar'],
    models: List[str] = None,
    train_frequency: str = 'semiannually',
    start_date: str = '2018-01-01',
    end_date: str = '2025-11-17',
    num_partitions: int = 200,
    batch_size: int = 10000
):
    """
    Parallel backfill using Spark.

    Args:
        commodities: List of commodities to backfill
        models: List of model versions (None = all models)
        train_frequency: How often models were trained
        start_date: First forecast date
        end_date: Last forecast date
        num_partitions: Number of Spark partitions (higher = more parallelism)
        batch_size: Rows to write per batch
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    if models is None:
        models = list(BASELINE_MODELS.keys())

    start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()

    print(f"{'='*80}")
    print(f"Spark Parallel Backfill")
    print(f"{'='*80}")
    print(f"Commodities: {commodities}")
    print(f"Models: {models}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Train Frequency: {train_frequency}")
    print(f"Partitions: {num_partitions}")
    print(f"{'='*80}\n")

    # Step 1: Generate all forecast tasks
    print("📋 Step 1: Generating forecast tasks...")

    training_dates_dict = {}
    all_forecast_dates = pd.date_range(start_dt, end_dt, freq='D')

    tasks = []
    for commodity in commodities:
        for model in models:
            # Get training dates for this model
            training_dates = get_training_dates(start_dt, end_dt, train_frequency)
            training_dates_dict[(commodity, model)] = training_dates

            # Create forecast task for each date
            for forecast_date in all_forecast_dates:
                tasks.append({
                    'commodity': commodity,
                    'model_version': model,
                    'forecast_start_date': forecast_date.date(),
                    'data_cutoff_date': forecast_date.date(),
                })

    print(f"   ✓ Created {len(tasks):,} forecast tasks")

    # Step 2: Load pretrained models
    print(f"\n📦 Step 2: Loading pretrained models...")

    pretrained_models = {}
    pretrained_models['training_dates'] = training_dates_dict

    for commodity in commodities:
        for model in models:
            for training_date in training_dates_dict.get((commodity, model), []):
                try:
                    fitted_model = load_model(
                        spark,
                        commodity=commodity,
                        model_version=model,
                        training_date=training_date,
                        train_frequency=train_frequency
                    )

                    if fitted_model:
                        model_key = (commodity, model, training_date)
                        pretrained_models[model_key] = fitted_model
                        print(f"   ✓ Loaded {commodity}/{model}/{training_date}")

                except Exception as e:
                    print(f"   ⚠️  Could not load {commodity}/{model}/{training_date}: {e}")

    print(f"   ✓ Loaded {len(pretrained_models) - 1} pretrained models")

    # Step 3: Broadcast models to all workers
    print(f"\n📡 Step 3: Broadcasting models to cluster...")
    broadcasted_models = spark.sparkContext.broadcast(pretrained_models)
    broadcasted_configs = spark.sparkContext.broadcast(BASELINE_MODELS)
    print(f"   ✓ Models broadcasted to all executors")

    # Step 4: Create Spark DataFrame of tasks
    print(f"\n🔧 Step 4: Creating Spark DataFrame ({num_partitions} partitions)...")
    tasks_df = spark.createDataFrame(tasks, schema=FORECAST_TASK_SCHEMA)
    tasks_df = tasks_df.repartition(num_partitions)
    print(f"   ✓ DataFrame created with {tasks_df.count():,} rows")

    # Step 5: Run parallel forecasts
    print(f"\n🚀 Step 5: Running parallel forecasts across cluster...")
    print(f"   (This will take 1-3 hours depending on cluster size)")

    def forecast_wrapper(partition):
        return generate_forecast_partition(
            partition,
            broadcasted_models.value,
            broadcasted_configs.value
        )

    # Execute forecasts in parallel
    results_rdd = tasks_df.rdd.mapPartitions(forecast_wrapper)
    results_df = spark.createDataFrame(results_rdd, schema=DISTRIBUTION_SCHEMA)

    # Step 6: Write results to Delta table
    print(f"\n💾 Step 6: Writing results to commodity.forecast.distributions...")

    results_df.write \
        .format("delta") \
        .mode("append") \
        .option("mergeSchema", "false") \
        .saveAsTable("commodity.forecast.distributions")

    final_count = results_df.count()
    print(f"   ✓ Wrote {final_count:,} forecast paths")

    # Summary
    print(f"\n{'='*80}")
    print(f"✅ Backfill Complete!")
    print(f"{'='*80}")
    print(f"Total forecasts: {len(tasks):,}")
    print(f"Total paths: {final_count:,}")
    print(f"{'='*80}\n")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Spark-parallelized forecast backfill')
    parser.add_argument('--commodities', nargs='+', default=['Coffee', 'Sugar'],
                       help='Commodities to backfill')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Model versions (default: all models)')
    parser.add_argument('--train-frequency', default='semiannually',
                       choices=['daily', 'weekly', 'biweekly', 'monthly', 'quarterly', 'semiannually', 'annually'])
    parser.add_argument('--start-date', type=str, default='2018-01-01',
                       help='First forecast date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-11-17',
                       help='Last forecast date (YYYY-MM-DD)')
    parser.add_argument('--num-partitions', type=int, default=200,
                       help='Number of Spark partitions (adjust based on cluster size)')
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='Rows to write per batch')

    args = parser.parse_args()

    backfill_all_models_spark(
        commodities=args.commodities,
        models=args.models,
        train_frequency=args.train_frequency,
        start_date=args.start_date,
        end_date=args.end_date,
        num_partitions=args.num_partitions,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()

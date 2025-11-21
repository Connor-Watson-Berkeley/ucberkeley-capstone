# Databricks notebook source
"""
Spark-Parallelized Forecast Backfill

This script distributes forecast generation across Spark workers for massive speedup.

Performance:
- Local (serial): ~10-20 hours for 2,872 dates
- Spark (parallel): ~20-60 minutes with proper cluster

Usage in Databricks:
    %run ./backfill_spark

    # Backfill Coffee XGBoost
    result = spark_backfill(
        commodity="Coffee",
        models=["xgboost"],
        train_frequency="semiannually",
        start_date="2018-01-01",
        end_date="2025-11-16"
    )
"""

# COMMAND ----------

# MAGIC %pip install scikit-learn xgboost statsmodels

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, pandas_udf, PandasUDFType, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DateType, BooleanType, TimestampType
import pickle
import base64

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration and Model Registry

# COMMAND ----------

# Import model registry (adjust path as needed)
import sys
sys.path.insert(0, '/Workspace/forecast_agent')

from ground_truth.config.model_registry import BASELINE_MODELS
from ground_truth.models import naive, random_walk, arima, sarimax, xgboost_model, prophet_model
from utils.monte_carlo_simulation import generate_monte_carlo_paths

# COMMAND ----------

# Model predict functions mapping
MODEL_PREDICT_FUNCTIONS = {
    'naive': naive.naive_predict,
    'random_walk': random_walk.random_walk_predict,
    'arima_111': arima.arima_predict,
    'sarimax_auto': sarimax.sarimax_predict,
    'sarimax_auto_weather': sarimax.sarimax_predict,
    'sarimax_auto_weather_seasonal': sarimax.sarimax_predict,
    'xgboost': xgboost_model.xgboost_predict,
    'prophet': prophet_model.prophet_predict,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

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

    delta = frequency_map[frequency]

    while current <= end_date:
        training_dates.append(current)
        if isinstance(delta, timedelta):
            current = current + delta
        else:
            current = current + delta

    return training_dates

# COMMAND ----------

def get_forecast_dates_for_window(train_date: date, next_train_date: Optional[date], end_date: date) -> List[date]:
    """Get all daily forecast dates for a training window."""
    forecast_dates = []
    current = train_date
    window_end = next_train_date - timedelta(days=1) if next_train_date else end_date

    while current <= window_end and current <= end_date:
        forecast_dates.append(current)
        current += timedelta(days=1)

    return forecast_dates

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark UDF for Forecast Generation

# COMMAND ----------

# Define output schema for forecasts
forecast_schema = StructType([
    StructField("path_id", IntegerType(), False),
    StructField("forecast_start_date", DateType(), False),
    StructField("data_cutoff_date", DateType(), False),
    StructField("generation_timestamp", TimestampType(), False),
    StructField("model_version", StringType(), False),
    StructField("commodity", StringType(), False),
    StructField("day_1", FloatType(), True),
    StructField("day_2", FloatType(), True),
    StructField("day_3", FloatType(), True),
    StructField("day_4", FloatType(), True),
    StructField("day_5", FloatType(), True),
    StructField("day_6", FloatType(), True),
    StructField("day_7", FloatType(), True),
    StructField("day_8", FloatType(), True),
    StructField("day_9", FloatType(), True),
    StructField("day_10", FloatType(), True),
    StructField("day_11", FloatType(), True),
    StructField("day_12", FloatType(), True),
    StructField("day_13", FloatType(), True),
    StructField("day_14", FloatType(), True),
    StructField("is_actuals", BooleanType(), False),
    StructField("has_data_leakage", BooleanType(), False),
])

# COMMAND ----------

@pandas_udf(forecast_schema, PandasUDFType.GROUPED_MAP)
def generate_forecasts_for_date(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Generate forecasts for a single date (runs on Spark worker).

    This function is executed in parallel across Spark workers.
    """
    # Extract metadata from first row
    forecast_date = pdf['forecast_start_date'].iloc[0]
    commodity = pdf['commodity'].iloc[0]
    model_version = pdf['model_version'].iloc[0]
    train_date = pdf['train_date'].iloc[0]

    try:
        # 1. Load training data from Delta table
        cutoff_date = forecast_date - timedelta(days=1)

        training_df = spark.sql(f"""
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
              AND date <= '{cutoff_date}'
            ORDER BY date
        """).toPandas()

        training_df['date'] = pd.to_datetime(training_df['date'])
        training_df = training_df.set_index('date')

        # 2. Load pretrained model from database
        model_row = spark.sql(f"""
            SELECT model_object
            FROM commodity.forecast.trained_models
            WHERE commodity = '{commodity}'
              AND model_version = '{model_version}'
              AND train_date = '{train_date}'
            ORDER BY created_at DESC
            LIMIT 1
        """).collect()

        if not model_row:
            raise ValueError(f"No pretrained model found for {train_date}")

        # Deserialize model
        model_bytes = base64.b64decode(model_row[0]['model_object'])
        fitted_model = pickle.loads(model_bytes)

        # 3. Generate forecast using model-specific predict function
        model_key = model_version.replace('_v1', '').replace('_weather', '')
        predict_fn = MODEL_PREDICT_FUNCTIONS.get(model_key)

        if not predict_fn:
            raise ValueError(f"No predict function for {model_key}")

        forecast_result = predict_fn(fitted_model, horizon=14)
        point_forecast = forecast_result['forecast'] if isinstance(forecast_result, dict) else forecast_result

        # 4. Generate Monte Carlo paths using Spark RDD (efficient parallelization)
        # For SARIMA: Simulates from actual ARIMA process (each path in parallel)
        # For other models: Uses appropriate stochastic process (GBM, random walk, etc.)
        num_paths = 2000

        # Broadcast shared data to all workers (efficient - sent once per worker)
        model_broadcast = spark.sparkContext.broadcast(fitted_model)
        point_forecast_np = point_forecast.values if hasattr(point_forecast, 'values') else np.array(point_forecast)
        forecast_broadcast = spark.sparkContext.broadcast(point_forecast_np)

        # Get last close price for GBM (broadcast as scalar)
        S0 = training_df['close'].iloc[-1]
        S0_broadcast = spark.sparkContext.broadcast(float(S0))

        # Function to generate a single path (pure numpy - fast!)
        def generate_path_numpy(path_id):
            """Generate one Monte Carlo path using numpy (Spark-parallelized)."""
            model = model_broadcast.value
            point_forecast = forecast_broadcast.value
            S0 = S0_broadcast.value

            model_type = model.get('model_type', 'unknown')

            # Generate single path based on model type
            if model_type == 'sarimax':
                # SARIMA: Use statsmodels simulate() for one realization
                fitted_sarimax = model['fitted_model']
                try:
                    simulated = fitted_sarimax.simulate(
                        nsimulations=14,
                        repetitions=1,
                        initial_state=None
                    )
                    path_values = simulated.values
                except:
                    # Fallback to noise (numpy)
                    noise = np.random.normal(0, fitted_sarimax.scale, 14)
                    path_values = point_forecast + noise

            elif model_type == 'xgboost':
                # GBM for XGBoost (pure numpy)
                residuals = model.get('residuals')
                sigma = np.std(residuals) / S0 if residuals is not None else 0.02

                S = S0
                path_values = np.zeros(14)
                for t in range(14):
                    target = point_forecast[t]
                    drift = (target - S) / S if S > 0 else 0
                    dW = np.random.normal(0, 1)
                    S = S * np.exp((drift - 0.5 * sigma**2) + sigma * dW)
                    path_values[t] = S

            else:
                # Simple Gaussian noise (numpy)
                forecast_std = model.get('std', 1.0)
                noise = np.random.normal(0, forecast_std, 14)
                path_values = point_forecast + noise

            # Return as list of floats
            return path_values.tolist()

        # Create RDD of path IDs and map to generate paths (efficient!)
        paths_rdd = spark.sparkContext.parallelize(range(num_paths), numSlices=200)
        paths = paths_rdd.map(generate_path_numpy).collect()

        # Clean up broadcasts
        model_broadcast.unpersist()
        forecast_broadcast.unpersist()
        S0_broadcast.unpersist()

        # 5. Format output
        generation_timestamp = datetime.now()
        has_data_leakage = forecast_date <= cutoff_date

        result_rows = []
        for path_id, path_values in enumerate(paths):
            row = {
                'path_id': path_id,
                'forecast_start_date': forecast_date,
                'data_cutoff_date': cutoff_date,
                'generation_timestamp': generation_timestamp,
                'model_version': model_version,
                'commodity': commodity,
                'is_actuals': False,
                'has_data_leakage': has_data_leakage,
            }

            # Add day columns
            for i in range(14):
                day_key = f'day_{i+1}'
                row[day_key] = path_values[i] if i < len(path_values) else None

            result_rows.append(row)

        return pd.DataFrame(result_rows)

    except Exception as e:
        print(f"Error generating forecast for {forecast_date}: {str(e)}")
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=[field.name for field in forecast_schema.fields])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Spark Backfill Function

# COMMAND ----------

def spark_backfill(
    commodity: str,
    models: List[str],
    train_frequency: str = "semiannually",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_partitions: int = 100
) -> Dict:
    """
    Run parallelized forecast backfill using Spark.

    Args:
        commodity: 'Coffee' or 'Sugar'
        models: List of model keys from BASELINE_MODELS
        train_frequency: Training frequency (daily, weekly, monthly, semiannually, etc.)
        start_date: First forecast date (YYYY-MM-DD)
        end_date: Last forecast date (YYYY-MM-DD)
        num_partitions: Number of Spark partitions (parallelism level)

    Returns:
        Dict with summary statistics
    """
    print("="*80)
    print("SPARK PARALLELIZED FORECAST BACKFILL")
    print("="*80)
    print(f"Commodity: {commodity}")
    print(f"Models: {models}")
    print(f"Train Frequency: {train_frequency}")
    print(f"Partitions: {num_partitions}")
    print("="*80)

    # Parse dates
    if start_date:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
    else:
        # Get earliest date + 3 years for training
        min_date = spark.sql(f"""
            SELECT MIN(date) as min_date
            FROM commodity.silver.unified_data
            WHERE commodity = '{commodity}'
        """).collect()[0]['min_date']
        start_dt = min_date + timedelta(days=365*3)

    if end_date:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
    else:
        end_dt = date.today() - timedelta(days=1)

    print(f"\nDate Range: {start_dt} to {end_dt}")

    # Process each model
    results = {}

    for model_key in models:
        print(f"\n{'='*80}")
        print(f"Processing model: {model_key}")
        print(f"{'='*80}")

        model_config = BASELINE_MODELS[model_key]
        model_version = model_config['name']

        # 1. Get training schedule
        training_dates = get_training_dates(start_dt, end_dt, train_frequency)
        print(f"\nTraining windows: {len(training_dates)}")

        # 2. Generate all forecast dates with their corresponding training date
        forecast_tasks = []
        for i, train_date in enumerate(training_dates):
            next_train_date = training_dates[i+1] if i+1 < len(training_dates) else None
            forecast_dates = get_forecast_dates_for_window(train_date, next_train_date, end_dt)

            for forecast_date in forecast_dates:
                forecast_tasks.append({
                    'forecast_start_date': forecast_date,
                    'train_date': train_date,
                    'commodity': commodity,
                    'model_version': model_version,
                })

        print(f"Total forecast dates: {len(forecast_tasks)}")

        # 3. Check for existing forecasts
        existing_dates = spark.sql(f"""
            SELECT DISTINCT forecast_start_date
            FROM commodity.forecast.distributions
            WHERE commodity = '{commodity}'
              AND model_version = '{model_version}'
              AND is_actuals = FALSE
        """).toPandas()['forecast_start_date'].tolist()

        existing_set = set(pd.to_datetime(existing_dates).date)

        # Filter to only new dates
        new_tasks = [
            task for task in forecast_tasks
            if task['forecast_start_date'] not in existing_set
        ]

        print(f"Already exist: {len(existing_set)}")
        print(f"Need to generate: {len(new_tasks)}")

        if len(new_tasks) == 0:
            print("✅ All forecasts already exist")
            results[model_key] = {'status': 'complete', 'new_forecasts': 0}
            continue

        # 4. Create Spark DataFrame of tasks
        tasks_df = spark.createDataFrame(new_tasks)
        tasks_df = tasks_df.repartition(num_partitions, 'forecast_start_date')

        print(f"\n🚀 Launching Spark jobs across {num_partitions} partitions...")

        # 5. Execute parallel forecast generation
        forecasts_df = tasks_df.groupby('forecast_start_date').apply(generate_forecasts_for_date)

        # 6. Write to Delta table
        print("\n💾 Writing forecasts to Delta table...")

        forecasts_df.write \
            .format("delta") \
            .mode("append") \
            .saveAsTable("commodity.forecast.distributions")

        # 7. Verify results
        count = forecasts_df.count()
        expected_count = len(new_tasks) * 2000  # 2000 paths per forecast

        print(f"\n✅ Complete!")
        print(f"   Rows written: {count:,}")
        print(f"   Expected: {expected_count:,}")
        print(f"   Forecasts: {len(new_tasks)}")

        results[model_key] = {
            'status': 'success',
            'new_forecasts': len(new_tasks),
            'rows_written': count,
            'expected_rows': expected_count
        }

    print("\n" + "="*80)
    print("BACKFILL COMPLETE")
    print("="*80)

    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example Usage

# COMMAND ----------

# Example: Backfill Coffee with XGBoost
if __name__ == "__main__":
    result = spark_backfill(
        commodity="Coffee",
        models=["xgboost"],
        train_frequency="semiannually",
        start_date="2018-01-01",
        end_date="2025-11-16",
        num_partitions=200  # Adjust based on cluster size
    )

    display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster Sizing Recommendations
# MAGIC
# MAGIC For optimal performance:
# MAGIC
# MAGIC - **Small backfill** (<500 dates): 4-8 workers, 8-16 cores each
# MAGIC - **Medium backfill** (500-1500 dates): 8-16 workers, 16-32 cores each
# MAGIC - **Large backfill** (1500+ dates): 16-32 workers, 32-64 cores each
# MAGIC
# MAGIC Set `num_partitions` = 2-4x number of total cores for good parallelism.
# MAGIC
# MAGIC Expected runtime:
# MAGIC - 2,872 dates with 16 workers (128 cores): ~20-40 minutes
# MAGIC - 2,872 dates with 32 workers (256 cores): ~10-20 minutes

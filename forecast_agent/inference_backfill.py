#!/usr/bin/env python3
"""
Inference-Only Backfill

Pure inference script that:
1. Queries pretrained models from database
2. Generates forecasts for missing dates
3. NO training logic
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path for imports
try:
    sys.path.insert(0, str(Path(__file__).parent))
except NameError:
    pass  # Running in Databricks notebook

# Databricks notebook parameter support
try:
    # dbutils is a built-in object in Databricks notebooks (not importable)
    commodity = dbutils.widgets.get("commodity")
    models = dbutils.widgets.get("models")
    start_date = dbutils.widgets.get("start_date")
    end_date = dbutils.widgets.get("end_date")
    model_version_tag = dbutils.widgets.get("model_version_tag")
    print("Running in Databricks notebook mode - using widgets for parameters")
except NameError:
    # Local execution - use argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", required=True, help="Comma-separated: Coffee,Sugar")
    parser.add_argument("--models", required=True, help="Comma-separated: naive,xgboost")
    parser.add_argument("--start_date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--model_version_tag", default="v1.0", help="Model version")
    args = parser.parse_args()
    commodity = args.commodity
    models = args.models
    start_date = args.start_date
    end_date = args.end_date
    model_version_tag = args.model_version_tag

# Import utilities
from utils.model_persistence import load_model, is_databricks
from ground_truth.models.naive import NaiveForecast
from ground_truth.models.xgboost_model import XGBoostForecast

# Spark SQL setup for Databricks
if is_databricks():
    from pyspark.sql import SparkSession
    from pyspark.sql.types import *
    from pyspark.sql.functions import col, to_date
    spark = SparkSession.builder.getOrCreate()
    connection = None
else:
    # Local execution - use databricks.sql
    from databricks import sql
    connection = sql.connect(
        server_hostname=os.environ['DATABRICKS_HOST'].replace('https://', ''),
        http_path=os.environ['DATABRICKS_HTTP_PATH'],
        access_token=os.environ['DATABRICKS_TOKEN']
    )
    spark = None


def get_pretrained_models(commodity_name, model_name, model_version):
    """Query all pretrained models for commodity/model combination."""
    query = f"""
    SELECT
        model_id,
        commodity,
        model_name,
        model_version,
        training_cutoff_date,
        training_samples,
        training_start_date
    FROM commodity.forecast.trained_models
    WHERE commodity = '{commodity_name}'
      AND model_name = '{model_name}'
      AND model_version = '{model_version}'
    ORDER BY training_cutoff_date
    """

    if is_databricks():
        return spark.sql(query).toPandas()
    else:
        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        return pd.DataFrame(rows, columns=columns)


def get_existing_forecasts(commodity_name, model_name):
    """Get dates that already have forecasts."""
    query = f"""
    SELECT DISTINCT forecast_start_date
    FROM commodity.forecast.distributions
    WHERE commodity = '{commodity_name}'
      AND model_name = '{model_name}'
    """

    if is_databricks():
        result_df = spark.sql(query).toPandas()
        return set(pd.to_datetime(result_df['forecast_start_date']).dt.date) if not result_df.empty else set()
    else:
        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        return set(row[0] for row in rows) if rows else set()


def load_data_for_forecast(commodity_name, forecast_date):
    """Load historical data up to forecast_date for inference."""
    query = f"""
    SELECT date, price_close, volume
    FROM commodity.silver.unified_data
    WHERE commodity = '{commodity_name}'
      AND date < '{forecast_date}'
    ORDER BY date
    """

    if is_databricks():
        df = spark.sql(query).toPandas()
    else:
        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = ['date', 'price_close', 'volume']
        cursor.close()
        df = pd.DataFrame(rows, columns=columns)

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    return df


def generate_forecast_from_model(fitted_model, model_name, historical_data, forecast_horizon=14):
    """Generate forecast using loaded model."""
    if model_name.lower() == 'naive':
        forecaster = NaiveForecast(horizon=forecast_horizon)
        forecaster.fitted_model = fitted_model
    elif model_name.lower() == 'xgboost':
        forecaster = XGBoostForecast(horizon=forecast_horizon)
        forecaster.fitted_model = fitted_model
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Generate forecast
    forecast_data = forecaster.predict(historical_data)
    return forecast_data


def create_distribution_record(commodity_name, model_name, forecast_start_date, data_cutoff_date,
                                 forecast_data, generation_timestamp, path_id):
    """Create distribution record from forecast."""
    # Extract 14-day distribution
    distributions = {}
    for day_offset in range(1, 15):
        forecast_date = forecast_start_date + timedelta(days=day_offset-1)
        if forecast_date in forecast_data.index:
            distributions[f'day_{day_offset}'] = float(forecast_data.loc[forecast_date, 'forecast_mean'])
        else:
            distributions[f'day_{day_offset}'] = None

    return {
        'path_id': path_id,
        'commodity': commodity_name,
        'model_name': model_name,
        'forecast_start_date': forecast_start_date,
        'data_cutoff_date': data_cutoff_date,
        'generation_timestamp': generation_timestamp,
        'is_actuals': False,
        'has_data_leakage': False,
        **distributions
    }


def create_point_forecast_records(commodity_name, model_name, forecast_data, generation_timestamp, path_id):
    """Create point forecast records."""
    records = []
    for date, row in forecast_data.iterrows():
        records.append({
            'path_id': path_id,
            'commodity': commodity_name,
            'model_name': model_name,
            'forecast_date': date,
            'forecast_mean': float(row['forecast_mean']),
            'forecast_std': float(row.get('forecast_std', 0.0)),
            'lower_95': float(row.get('lower_95', row['forecast_mean'])),
            'upper_95': float(row.get('upper_95', row['forecast_mean'])),
            'generation_timestamp': generation_timestamp
        })
    return records


def write_forecasts_to_tables(distributions_data, point_forecasts_data):
    """Write forecast data to tables."""
    if is_databricks():
        # Write distributions
        if distributions_data:
            dist_df = spark.createDataFrame(distributions_data)

            # Cast types
            dist_df = dist_df.withColumn("path_id", col("path_id").cast(IntegerType())) \
                             .withColumn("forecast_start_date", to_date(col("forecast_start_date"))) \
                             .withColumn("data_cutoff_date", to_date(col("data_cutoff_date"))) \
                             .withColumn("generation_timestamp", col("generation_timestamp").cast(TimestampType())) \
                             .withColumn("is_actuals", col("is_actuals").cast(BooleanType())) \
                             .withColumn("has_data_leakage", col("has_data_leakage").cast(BooleanType()))

            # Cast day columns
            for i in range(1, 15):
                dist_df = dist_df.withColumn(f"day_{i}", col(f"day_{i}").cast(FloatType()))

            dist_df.write.mode("append").saveAsTable("commodity.forecast.distributions")
            print(f"  ✅ Wrote {len(distributions_data)} distribution records")

        # Write point forecasts
        if point_forecasts_data:
            point_df = spark.createDataFrame(point_forecasts_data)
            point_df = point_df.withColumn("forecast_date", to_date(col("forecast_date"))) \
                               .withColumn("forecast_mean", col("forecast_mean").cast(FloatType())) \
                               .withColumn("forecast_std", col("forecast_std").cast(FloatType())) \
                               .withColumn("lower_95", col("lower_95").cast(FloatType())) \
                               .withColumn("upper_95", col("upper_95").cast(FloatType()))

            point_df.write.mode("append").saveAsTable("commodity.forecast.point_forecasts")
            print(f"  ✅ Wrote {len(point_forecasts_data)} point forecast records")
    else:
        # Local execution - batch insert
        cursor = connection.cursor()

        # Insert distributions
        if distributions_data:
            # TODO: Implement batch insert for local execution
            pass

        # Insert point forecasts
        if point_forecasts_data:
            # TODO: Implement batch insert for local execution
            pass

        cursor.close()


def main():
    print("="*80)
    print("INFERENCE-ONLY BACKFILL")
    print("="*80)
    print(f"\nCommodities: {commodity}")
    print(f"Models: {models}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Model version: {model_version_tag}")
    print()

    # Parse inputs
    commodities = [c.strip() for c in commodity.split(',')]
    model_list = [m.strip() for m in models.split(',')]
    start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()

    # Generate all dates in range
    all_dates = pd.date_range(start_dt, end_dt, freq='D')

    generation_timestamp = datetime.now()
    path_id = 1  # Simple incrementing ID

    for commodity_name in commodities:
        print(f"\n{'='*80}")
        print(f"Processing commodity: {commodity_name}")
        print('='*80)

        for model_name in model_list:
            print(f"\n🔧 Model: {model_name}")
            print('='*80)

            # Get pretrained models
            pretrained_models = get_pretrained_models(commodity_name, model_name, model_version_tag)

            if pretrained_models.empty:
                print(f"  ⚠️  No pretrained models found for {commodity_name} / {model_name}")
                continue

            print(f"  Found {len(pretrained_models)} pretrained models")
            print(f"  Training dates: {pretrained_models['training_cutoff_date'].min()} to {pretrained_models['training_cutoff_date'].max()}")

            # Get existing forecasts to skip
            existing_forecasts = get_existing_forecasts(commodity_name, model_name)
            print(f"  Existing forecasts: {len(existing_forecasts)}")

            # Accumulate forecast data
            all_distributions = []
            all_point_forecasts = []
            forecasts_generated = 0

            # For each model, generate forecasts for dates after its training cutoff
            for idx, model_row in pretrained_models.iterrows():
                training_cutoff_date = pd.to_datetime(model_row['training_cutoff_date']).date()
                model_id = model_row['model_id']

                print(f"\n  [{idx+1}/{len(pretrained_models)}] Model: {model_id}")
                print(f"     Training cutoff: {training_cutoff_date}")

                # Load model from database
                loaded_data = load_model(
                    connection=connection,
                    commodity=commodity_name,
                    model_name=model_name,
                    training_date=str(training_cutoff_date),
                    model_version=model_version_tag
                )

                if not loaded_data:
                    print(f"     ⚠️  Failed to load model from database")
                    continue

                fitted_model = loaded_data['fitted_model']
                print(f"     ✅ Loaded pretrained model")

                # Generate forecasts for dates after training cutoff
                forecast_start = training_cutoff_date + timedelta(days=1)

                # Find next model's training date or use end_dt
                next_model_idx = idx + 1
                if next_model_idx < len(pretrained_models):
                    next_training_date = pd.to_datetime(pretrained_models.iloc[next_model_idx]['training_cutoff_date']).date()
                    forecast_end = min(next_training_date, end_dt)
                else:
                    forecast_end = end_dt

                # Generate forecasts for this window
                forecast_dates = pd.date_range(forecast_start, forecast_end, freq='D')

                # Filter out existing forecasts
                dates_to_forecast = [d.date() for d in forecast_dates if d.date() not in existing_forecasts]

                if not dates_to_forecast:
                    print(f"     All forecasts already exist (skip)")
                    continue

                print(f"     Generating {len(dates_to_forecast)} forecasts...")

                # Load historical data
                historical_data = load_data_for_forecast(commodity_name, forecast_start)

                # Generate forecast
                forecast_data = generate_forecast_from_model(fitted_model, model_name, historical_data)

                # Create records
                for forecast_date in dates_to_forecast:
                    forecast_date_dt = pd.Timestamp(forecast_date)

                    # Distribution record
                    dist_record = create_distribution_record(
                        commodity_name, model_name, forecast_date_dt, training_cutoff_date,
                        forecast_data, generation_timestamp, path_id
                    )
                    all_distributions.append(dist_record)

                    # Point forecast records (14 days)
                    point_records = create_point_forecast_records(
                        commodity_name, model_name, forecast_data.loc[forecast_date:forecast_date+timedelta(days=13)],
                        generation_timestamp, path_id
                    )
                    all_point_forecasts.extend(point_records)

                    path_id += 1
                    forecasts_generated += 1

                print(f"     ✅ Generated {len(dates_to_forecast)} forecasts")

            # Write all forecasts
            print(f"\n  💾 Writing forecasts to tables...")
            write_forecasts_to_tables(all_distributions, all_point_forecasts)

            print(f"\n  ✅ Completed {model_name}")
            print(f"     New forecasts: {forecasts_generated}")
            print(f"     Skipped (existing): {len(existing_forecasts)}")

    print("\n" + "="*80)
    print("✅ INFERENCE BACKFILL COMPLETE")
    print("="*80)

    if connection:
        connection.close()


if __name__ == "__main__":
    main()

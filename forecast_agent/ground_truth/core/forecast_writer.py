"""Forecast writer for Delta tables.

Writes forecasts to commodity.silver.point_forecasts and distributions tables.
Includes data leakage detection.
"""

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, current_timestamp
from datetime import datetime


def write_point_forecast(spark: SparkSession, forecast_df: pd.DataFrame,
                         model_name: str, commodity: str,
                         training_end: pd.Timestamp,
                         parameters: dict = None,
                         table_name: str = "commodity.silver.point_forecasts") -> None:
    """
    Write point forecast to Delta table with metadata.

    Args:
        spark: Active Spark session
        forecast_df: Forecast DataFrame (date, forecast, lower_80, upper_80, lower_95, upper_95)
        model_name: Model identifier (e.g., 'ARIMA(1,1,1)', 'Naive')
        commodity: 'Coffee' or 'Sugar'
        training_end: Last date in training data
        parameters: Model parameters dict (optional)
        table_name: Delta table name

    Data leakage check: Verifies forecast_start > training_end

    Example:
        write_point_forecast(
            spark,
            forecast_df,
            model_name='SARIMAX(1,1,1)',
            commodity='Coffee',
            training_end=pd.Timestamp('2024-01-01')
        )
    """
    # Data leakage check
    forecast_start = pd.Timestamp(forecast_df['date'].iloc[0])
    if forecast_start <= training_end:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! "
            f"Forecast starts {forecast_start} but training ends {training_end}. "
            f"Forecast must start AFTER training end."
        )

    # Add metadata columns
    forecast_df_out = forecast_df.copy()
    forecast_df_out['model_name'] = model_name
    forecast_df_out['commodity'] = commodity
    forecast_df_out['training_end'] = training_end
    forecast_df_out['created_at'] = datetime.now()

    # Add parameters as JSON string if provided
    if parameters:
        import json
        forecast_df_out['parameters'] = json.dumps(parameters)
    else:
        forecast_df_out['parameters'] = None

    # Rename 'forecast' to 'point_forecast' for schema consistency
    forecast_df_out = forecast_df_out.rename(columns={'forecast': 'point_forecast'})

    # Convert to Spark DataFrame
    forecast_spark = spark.createDataFrame(forecast_df_out)

    # Write to Delta table (append mode)
    forecast_spark.write \
        .format("delta") \
        .mode("append") \
        .saveAsTable(table_name)

    print(f"✓ Wrote {len(forecast_df)} forecasts to {table_name}")
    print(f"  Model: {model_name}, Commodity: {commodity}")
    print(f"  Forecast period: {forecast_df['date'].iloc[0]} to {forecast_df['date'].iloc[-1]}")


def write_distribution_forecast(spark: SparkSession, paths_df: pd.DataFrame,
                                 model_name: str, commodity: str,
                                 training_end: pd.Timestamp,
                                 parameters: dict = None,
                                 table_name: str = "commodity.silver.distributions") -> None:
    """
    Write distribution forecast (Monte Carlo paths) to Delta table.

    Args:
        spark: Active Spark session
        paths_df: Distribution DataFrame (date, path_id, price)
        model_name: Model identifier
        commodity: 'Coffee' or 'Sugar'
        training_end: Last date in training data
        parameters: Model parameters dict (optional)
        table_name: Delta table name

    Expected schema:
        - date: forecast date
        - path_id: 1-2000 (Monte Carlo path number)
        - price: simulated price for this path
        - model_name, commodity, training_end, parameters, created_at: metadata

    Example:
        # Generate 2000 Monte Carlo paths
        paths_df = generate_monte_carlo_paths(forecast_df, n_paths=2000)
        write_distribution_forecast(spark, paths_df, ...)
    """
    # Data leakage check
    forecast_start = pd.Timestamp(paths_df['date'].min())
    if forecast_start <= training_end:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! "
            f"Forecast starts {forecast_start} but training ends {training_end}."
        )

    # Add metadata
    paths_df_out = paths_df.copy()
    paths_df_out['model_name'] = model_name
    paths_df_out['commodity'] = commodity
    paths_df_out['training_end'] = training_end
    paths_df_out['created_at'] = datetime.now()

    if parameters:
        import json
        paths_df_out['parameters'] = json.dumps(parameters)
    else:
        paths_df_out['parameters'] = None

    # Convert to Spark DataFrame
    paths_spark = spark.createDataFrame(paths_df_out)

    # Write to Delta table (append mode)
    paths_spark.write \
        .format("delta") \
        .mode("append") \
        .saveAsTable(table_name)

    n_paths = paths_df['path_id'].nunique()
    print(f"✓ Wrote {len(paths_df)} distribution samples ({n_paths} paths) to {table_name}")


def write_forecast_actuals(spark: SparkSession, actuals_df: pd.DataFrame,
                            commodity: str,
                            table_name: str = "commodity.silver.forecast_actuals") -> None:
    """
    Write realized prices (actuals) for forecast evaluation.

    Args:
        spark: Active Spark session
        actuals_df: Actuals DataFrame (date, close)
        commodity: 'Coffee' or 'Sugar'
        table_name: Delta table name

    Note: This is a separate table from forecasts to avoid path_id=0 confusion.

    Example:
        # After forecast period passes, record actuals
        actuals_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', '2024-01-14'),
            'close': [actual_prices...]
        })
        write_forecast_actuals(spark, actuals_df, commodity='Coffee')
    """
    actuals_df_out = actuals_df.copy()
    actuals_df_out['commodity'] = commodity
    actuals_df_out['created_at'] = datetime.now()

    # Convert to Spark
    actuals_spark = spark.createDataFrame(actuals_df_out)

    # Write to Delta table (append mode, or merge if date+commodity already exists)
    actuals_spark.write \
        .format("delta") \
        .mode("append") \
        .saveAsTable(table_name)

    print(f"✓ Wrote {len(actuals_df)} actuals to {table_name}")
    print(f"  Commodity: {commodity}")
    print(f"  Date range: {actuals_df['date'].min()} to {actuals_df['date'].max()}")


def generate_monte_carlo_paths(forecast_df: pd.DataFrame, n_paths: int = 2000,
                                 method: str = 'normal') -> pd.DataFrame:
    """
    Generate Monte Carlo paths from point forecast + confidence intervals.

    Args:
        forecast_df: Point forecast with CI (date, forecast, lower_95, upper_95)
        n_paths: Number of paths to generate (default: 2000)
        method: 'normal' (assumes normal distribution)

    Returns:
        DataFrame with columns: date, path_id, price

    Example:
        paths = generate_monte_carlo_paths(forecast_df, n_paths=2000)
        # Returns ~28,000 rows (14 days × 2000 paths)
    """
    import numpy as np

    paths_list = []

    for _, row in forecast_df.iterrows():
        date = row['date']
        mean = row['forecast']

        # Estimate std dev from 95% CI
        # 95% CI = mean ± 1.96*std → std = (upper_95 - mean) / 1.96
        std = (row['upper_95'] - mean) / 1.96

        # Generate n_paths samples from normal distribution
        samples = np.random.normal(mean, std, n_paths)

        for path_id, price in enumerate(samples, start=1):
            paths_list.append({
                'date': date,
                'path_id': path_id,
                'price': price
            })

    return pd.DataFrame(paths_list)

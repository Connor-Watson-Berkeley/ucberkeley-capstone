"""Data loader for forecast models.

Loads unified_data from Databricks, applies feature engineering, converts to pandas.
"""

from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
import pandas as pd
from ground_truth.features import aggregators


def load_unified_data(spark: SparkSession, table_name: str = "commodity.silver.unified_data",
                      commodity: str = 'Coffee', cutoff_date: str = None) -> SparkDataFrame:
    """
    Load unified data from Databricks Delta table.

    Args:
        spark: Active Spark session
        table_name: Delta table name
        commodity: 'Coffee' or 'Sugar'
        cutoff_date: Optional - for backtesting, filters data <= cutoff_date

    Returns:
        Spark DataFrame with unified data

    Example:
        df = load_unified_data(spark, commodity='Coffee', cutoff_date='2024-01-01')
    """
    # Load from Delta table
    df = spark.table(table_name)

    # Filter by commodity
    df = df.filter(f"commodity = '{commodity}'")

    # Filter by cutoff_date if provided (for backtesting)
    if cutoff_date:
        df = df.filter(f"date <= '{cutoff_date}'")

    return df.orderBy("date")


def prepare_model_data(df_spark: SparkDataFrame, commodity: str,
                       features: list, aggregation_method: str = 'mean',
                       production_weights: dict = None,
                       cutoff_date: str = None) -> pd.DataFrame:
    """
    Prepare data for model training - aggregate regions and convert to pandas.

    Args:
        df_spark: Unified data (Spark)
        commodity: 'Coffee' or 'Sugar'
        features: List of feature names
        aggregation_method: 'mean', 'weighted', or 'pivot'
        production_weights: For weighted aggregation (optional)
        cutoff_date: Optional - for backtesting

    Returns:
        Pandas DataFrame with DatetimeIndex, ready for model training

    Example:
        df_pandas = prepare_model_data(
            df_spark,
            commodity='Coffee',
            features=['close', 'temp_c', 'humidity_pct'],
            aggregation_method='mean'
        )
    """
    # Apply regional aggregation
    if aggregation_method == 'mean':
        df_agg = aggregators.aggregate_regions_mean(
            df_spark, commodity, features, cutoff_date
        )
    elif aggregation_method == 'weighted':
        df_agg = aggregators.aggregate_regions_weighted(
            df_spark, commodity, features, cutoff_date, production_weights
        )
    elif aggregation_method == 'pivot':
        df_agg = aggregators.pivot_regions_as_features(
            df_spark, commodity, features, cutoff_date
        )
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    # Convert to pandas
    df_pandas = df_agg.toPandas()

    # Set date as index
    df_pandas['date'] = pd.to_datetime(df_pandas['date'])
    df_pandas = df_pandas.set_index('date').sort_index()

    return df_pandas


def load_and_prepare(spark: SparkSession, commodity: str, features: list,
                     aggregation_method: str = 'mean',
                     cutoff_date: str = None,
                     production_weights: dict = None) -> pd.DataFrame:
    """
    One-stop function: Load unified data and prepare for modeling.

    Args:
        spark: Active Spark session
        commodity: 'Coffee' or 'Sugar'
        features: List of feature names
        aggregation_method: 'mean', 'weighted', or 'pivot'
        cutoff_date: Optional - for backtesting
        production_weights: For weighted aggregation (optional)

    Returns:
        Pandas DataFrame ready for model training

    Example:
        df = load_and_prepare(
            spark,
            commodity='Coffee',
            features=['close', 'temp_c', 'humidity_pct'],
            cutoff_date='2024-01-01'
        )
    """
    # Load from Delta
    df_spark = load_unified_data(spark, commodity=commodity, cutoff_date=cutoff_date)

    # Prepare for modeling
    df_pandas = prepare_model_data(
        df_spark, commodity, features,
        aggregation_method, production_weights, cutoff_date
    )

    return df_pandas

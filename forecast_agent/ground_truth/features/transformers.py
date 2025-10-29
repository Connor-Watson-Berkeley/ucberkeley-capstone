"""Time-based feature transformations for commodity forecasting.

These functions add engineered features like lags, diffs, and rolling statistics.
Useful for XGBoost, LSTM, and other ML models that benefit from explicit features.

Note: Classical time series models (ARIMA, SARIMAX) don't need these - they model
temporal dependencies internally.
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import lag, lead, col, avg, stddev, when


def add_lags(df_spark: DataFrame, features: list, lags: list = [1, 7, 14, 30],
             cutoff_date: str = None) -> DataFrame:
    """
    Add lagged features - past values as predictors.

    Use case: XGBoost, LSTM

    Args:
        df_spark: Input data
        features: Features to lag (e.g., ['close', 'temp_c'])
        lags: List of lag periods (e.g., [1, 7, 14] = yesterday, last week, 2 weeks ago)
        cutoff_date: Optional - for backtesting

    Returns:
        DataFrame with new lag columns: close_lag_1, close_lag_7, temp_c_lag_14, etc.

    Example:
        close on 2024-01-15 = 167.50
        → close_lag_1 on 2024-01-16 = 167.50 (yesterday's close)
        → close_lag_7 on 2024-01-22 = 167.50 (last week's close)

    Use: XGBoost can learn "if yesterday's price was X, today will be Y"
    """
    if cutoff_date:
        df_spark = df_spark.filter(f"date <= '{cutoff_date}'")

    # Define window for lagging
    window = Window.partitionBy("commodity").orderBy("date")

    # Add lag columns
    for feature in features:
        if feature in df_spark.columns:
            for lag_period in lags:
                lag_col_name = f"{feature}_lag_{lag_period}"
                df_spark = df_spark.withColumn(
                    lag_col_name,
                    lag(col(feature), lag_period).over(window)
                )

    return df_spark


def add_rolling_stats(df_spark: DataFrame, features: list, windows: list = [7, 30, 90],
                      cutoff_date: str = None) -> DataFrame:
    """
    Add rolling statistics - moving averages and standard deviations.

    Use case: XGBoost (captures trends and volatility)

    Args:
        df_spark: Input data
        features: Features to compute rolling stats on
        windows: Window sizes (e.g., [7, 30] = weekly, monthly)
        cutoff_date: Optional - for backtesting

    Returns:
        DataFrame with rolling mean and std columns

    Example:
        close_rolling_mean_7: 7-day moving average
        close_rolling_std_30: 30-day volatility (std dev)

    Use: XGBoost can learn from trends (rising prices) and volatility (high std = uncertainty)
    """
    if cutoff_date:
        df_spark = df_spark.filter(f"date <= '{cutoff_date}'")

    for feature in features:
        if feature not in df_spark.columns:
            continue

        for window_size in windows:
            # Define window
            window = Window.partitionBy("commodity") \
                .orderBy("date") \
                .rowsBetween(-window_size + 1, 0)

            # Rolling mean
            mean_col = f"{feature}_rolling_mean_{window_size}"
            df_spark = df_spark.withColumn(mean_col, avg(col(feature)).over(window))

            # Rolling std dev
            std_col = f"{feature}_rolling_std_{window_size}"
            df_spark = df_spark.withColumn(std_col, stddev(col(feature)).over(window))

    return df_spark


def add_differences(df_spark: DataFrame, features: list, periods: list = [1, 7],
                    cutoff_date: str = None) -> DataFrame:
    """
    Add price changes - differences from past values.

    Use case: XGBoost (models often work better with changes than levels)

    Args:
        df_spark: Input data
        features: Features to difference
        periods: Periods to difference over (e.g., [1, 7] = daily change, weekly change)
        cutoff_date: Optional - for backtesting

    Returns:
        DataFrame with difference columns

    Example:
        close=167.50 today, close=165.00 yesterday
        → close_diff_1 = 2.50 (daily change)

        close=167.50 today, close=160.00 last week
        → close_diff_7 = 7.50 (weekly change)

    Use: XGBoost learns from momentum (positive change = bullish)
    """
    if cutoff_date:
        df_spark = df_spark.filter(f"date <= '{cutoff_date}'")

    window = Window.partitionBy("commodity").orderBy("date")

    for feature in features:
        if feature not in df_spark.columns:
            continue

        for period in periods:
            diff_col = f"{feature}_diff_{period}"
            df_spark = df_spark.withColumn(
                diff_col,
                col(feature) - lag(col(feature), period).over(window)
            )

    return df_spark


def add_date_features(df_spark: DataFrame, cutoff_date: str = None) -> DataFrame:
    """
    Add calendar features - day of week, month, etc.

    Use case: XGBoost, LSTM (captures seasonality)

    Args:
        df_spark: Input data
        cutoff_date: Optional - for backtesting

    Returns:
        DataFrame with date features: day_of_week, month, quarter, day_of_year

    Example:
        date=2024-01-15
        → day_of_week=1 (Monday)
        → month=1 (January)
        → quarter=1 (Q1)
        → day_of_year=15

    Use: XGBoost can learn seasonal patterns (e.g., coffee harvest cycles)
    """
    from pyspark.sql.functions import dayofweek, month, quarter, dayofyear

    if cutoff_date:
        df_spark = df_spark.filter(f"date <= '{cutoff_date}'")

    df_spark = df_spark.withColumn("day_of_week", dayofweek(col("date")))
    df_spark = df_spark.withColumn("month", month(col("date")))
    df_spark = df_spark.withColumn("quarter", quarter(col("date")))
    df_spark = df_spark.withColumn("day_of_year", dayofyear(col("date")))

    return df_spark


def add_interaction_features(df_spark: DataFrame, feature_pairs: list,
                             cutoff_date: str = None) -> DataFrame:
    """
    Add interaction features - products of feature pairs.

    Use case: XGBoost (captures feature interactions)

    Args:
        df_spark: Input data
        feature_pairs: List of tuples (feature1, feature2) to multiply
        cutoff_date: Optional - for backtesting

    Returns:
        DataFrame with interaction columns

    Example:
        feature_pairs=[('temp_c', 'precipitation_mm')]
        → Creates: temp_c_x_precipitation_mm

    Use: XGBoost can learn "hot + rainy = good for coffee growth"
    """
    if cutoff_date:
        df_spark = df_spark.filter(f"date <= '{cutoff_date}'")

    for feat1, feat2 in feature_pairs:
        if feat1 in df_spark.columns and feat2 in df_spark.columns:
            interaction_col = f"{feat1}_x_{feat2}"
            df_spark = df_spark.withColumn(
                interaction_col,
                col(feat1) * col(feat2)
            )

    return df_spark


# Composite function: Apply all transformations at once
def engineer_all_features(df_spark: DataFrame, commodity: str, base_features: list,
                          cutoff_date: str = None) -> DataFrame:
    """
    Apply all feature engineering transformations - convenience function.

    Use case: XGBoost with full feature set

    Args:
        df_spark: Input data
        commodity: 'Coffee' or 'Sugar'
        base_features: Base features (e.g., ['close', 'temp_c'])
        cutoff_date: Optional - for backtesting

    Returns:
        DataFrame with all engineered features

    Features added:
        - Lags: 1, 7, 14, 30 days
        - Rolling stats: 7, 30, 90 day windows
        - Differences: 1, 7 day changes
        - Date features: day of week, month, etc.

    Note: Creates MANY features! Use for XGBoost, not ARIMA.
    """
    df_spark = df_spark.filter(f"commodity = '{commodity}'")

    if cutoff_date:
        df_spark = df_spark.filter(f"date <= '{cutoff_date}'")

    # Add all transformations
    df_spark = add_lags(df_spark, base_features, lags=[1, 7, 14, 30])
    df_spark = add_rolling_stats(df_spark, base_features, windows=[7, 30, 90])
    df_spark = add_differences(df_spark, base_features, periods=[1, 7])
    df_spark = add_date_features(df_spark)

    return df_spark

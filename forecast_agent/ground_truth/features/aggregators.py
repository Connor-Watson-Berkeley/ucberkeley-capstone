"""Regional aggregation strategies for commodity forecasting.

Each function transforms regional data (weather by region) into model-ready features.
Models can choose their aggregation strategy based on their needs:
- ARIMA/SARIMAX: Simple mean (aggregate_regions_mean)
- LSTM/Transformers: Each region as feature (pivot_regions_as_features)
- XGBoost: Weighted by production (aggregate_regions_weighted)
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import avg, sum as spark_sum, col, first


def aggregate_regions_mean(df_spark: DataFrame, commodity: str, features: list, cutoff_date: str = None) -> DataFrame:
    """
    Average weather across all regions - simple baseline.

    Use case: ARIMA, SARIMAX (simple models)

    Args:
        df_spark: Unified data (date, commodity, region grain)
        commodity: 'Coffee' or 'Sugar'
        features: List of feature names (e.g., ['close', 'temp_c', 'humidity_pct'])
        cutoff_date: Optional - filter data <= cutoff_date (for backtesting)

    Returns:
        DataFrame with (date, commodity) grain, regional features averaged

    Example:
        Input:  date=2024-01-15, region=Colombia, temp_c=25
                date=2024-01-15, region=Vietnam, temp_c=30
        Output: date=2024-01-15, temp_c=27.5 (average)
    """
    # Filter by cutoff if provided
    if cutoff_date:
        df_spark = df_spark.filter(f"date <= '{cutoff_date}'")

    # Filter by commodity
    df_spark = df_spark.filter(f"commodity = '{commodity}'")

    # Identify which features are regional (need aggregation)
    regional_features = ['temp_c', 'humidity_pct', 'precipitation_mm']

    # Build aggregation expressions
    agg_exprs = []

    for feature in features:
        if feature in regional_features:
            # Average across regions
            agg_exprs.append(avg(feature).alias(feature))
        else:
            # Non-regional features (close, vix, etc.) - just take first (same for all regions)
            agg_exprs.append(first(feature).alias(feature))

    # Group by date, commodity
    df_agg = df_spark.groupBy("date", "commodity").agg(*agg_exprs)

    return df_agg.orderBy("date")


def aggregate_regions_weighted(df_spark: DataFrame, commodity: str, features: list,
                                 cutoff_date: str = None, production_weights: dict = None) -> DataFrame:
    """
    Weight regions by production volume - more sophisticated.

    Use case: SARIMAX, XGBoost (when production data available)

    Args:
        df_spark: Unified data
        commodity: 'Coffee' or 'Sugar'
        features: List of feature names
        cutoff_date: Optional - for backtesting
        production_weights: Dict mapping region -> weight (e.g., {'Colombia': 0.3, 'Brazil': 0.4})

    Returns:
        DataFrame with production-weighted regional averages

    Example:
        Colombia (weight=0.6): temp=25
        Vietnam (weight=0.4): temp=30
        Weighted avg: 0.6*25 + 0.4*30 = 27

    Note:
        If production_weights not provided, falls back to simple mean.
    """
    if cutoff_date:
        df_spark = df_spark.filter(f"date <= '{cutoff_date}'")

    df_spark = df_spark.filter(f"commodity = '{commodity}'")

    # If no weights provided, fall back to simple mean
    if not production_weights:
        return aggregate_regions_mean(df_spark, commodity, features)

    # Add production weight column
    # (In future: could join from production_data table)
    from pyspark.sql.functions import when, lit

    weight_expr = None
    for region, weight in production_weights.items():
        if weight_expr is None:
            weight_expr = when(col("region") == region, lit(weight))
        else:
            weight_expr = weight_expr.when(col("region") == region, lit(weight))

    weight_expr = weight_expr.otherwise(lit(0.0))

    df_weighted = df_spark.withColumn("production_weight", weight_expr)

    # Calculate weighted averages
    regional_features = ['temp_c', 'humidity_pct', 'precipitation_mm']
    agg_exprs = []

    for feature in features:
        if feature in regional_features:
            # Weighted average
            agg_exprs.append(
                (spark_sum(col(feature) * col("production_weight")) /
                 spark_sum("production_weight")).alias(feature)
            )
        else:
            agg_exprs.append(first(feature).alias(feature))

    df_agg = df_weighted.groupBy("date", "commodity").agg(*agg_exprs)

    return df_agg.orderBy("date")


def pivot_regions_as_features(df_spark: DataFrame, commodity: str, features: list,
                                cutoff_date: str = None) -> DataFrame:
    """
    Each region becomes a separate feature column - for LSTM/Transformers.

    Use case: LSTM, Transformers (can handle high dimensionality)

    Args:
        df_spark: Unified data
        commodity: 'Coffee' or 'Sugar'
        features: List of base features (e.g., ['close', 'temp_c'])
        cutoff_date: Optional - for backtesting

    Returns:
        DataFrame with columns like: close, temp_c_colombia, temp_c_vietnam, humidity_pct_colombia, ...

    Example:
        Input:  date=2024-01-15, region=Colombia, temp_c=25, humidity=80
                date=2024-01-15, region=Vietnam, temp_c=30, humidity=75
        Output: date=2024-01-15, temp_c_colombia=25, temp_c_vietnam=30,
                humidity_pct_colombia=80, humidity_pct_vietnam=75

    Note:
        This creates many features! Only use with models that can handle it.
        For 65 regions × 3 weather features = 195 features.
    """
    if cutoff_date:
        df_spark = df_spark.filter(f"date <= '{cutoff_date}'")

    df_spark = df_spark.filter(f"commodity = '{commodity}'")

    # Get list of regions
    regions = df_spark.select("region").distinct().rdd.flatMap(lambda x: x).collect()

    # Pivot regional features
    regional_features = ['temp_c', 'humidity_pct', 'precipitation_mm']

    # Start with non-regional features (same across all regions)
    non_regional = [f for f in features if f not in regional_features]
    df_pivot = df_spark.groupBy("date", "commodity")

    # Add non-regional features
    agg_exprs = [first(f).alias(f) for f in non_regional]

    # Pivot each regional feature
    for feature in features:
        if feature in regional_features:
            for region in regions:
                # Create column: temp_c_colombia, temp_c_vietnam, etc.
                region_clean = region.lower().replace(" ", "_").replace("-", "_")
                col_name = f"{feature}_{region_clean}"

                agg_exprs.append(
                    first(when(col("region") == region, col(feature))).alias(col_name)
                )

    df_pivot = df_pivot.agg(*agg_exprs)

    return df_pivot.orderBy("date")


# Future: Add more aggregation strategies
# def aggregate_regions_by_distance(df_spark, commodity, features, origin_region, cutoff_date=None):
#     """Weight regions by distance from origin - for spatial modeling."""
#     pass

# def aggregate_regions_hierarchical(df_spark, commodity, features, cutoff_date=None):
#     """Group regions by country, then aggregate - for hierarchical forecasting."""
#     pass

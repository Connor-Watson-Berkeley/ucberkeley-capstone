"""
Forecast Data Loader

Loads forecast distribution data from Unity Catalog table:
commodity.forecast.distributions

This module provides functions to query forecast data by commodity and model,
and transform it into the prediction matrices format expected by the
backtesting engine.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime


def get_available_commodities(connection) -> List[str]:
    """
    Get list of available commodities from distributions table.

    Args:
        connection: Databricks SQL connection

    Returns:
        List of commodity names (e.g., ['Coffee', 'Sugar'])
    """
    cursor = connection.cursor()
    cursor.execute("""
        SELECT DISTINCT commodity
        FROM commodity.forecast.distributions
        ORDER BY commodity
    """)
    commodities = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return commodities


def get_available_models(commodity: str, connection) -> List[str]:
    """
    Query distinct model versions for a specific commodity.

    Args:
        commodity: str - 'Coffee' or 'Sugar' (case-sensitive)
        connection: Databricks SQL connection

    Returns:
        List of model_version strings sorted alphabetically

    Example:
        >>> get_available_models('Coffee', conn)
        ['arima_111_v1', 'prophet_v1', 'sarimax_auto_weather_v1', ...]
    """
    cursor = connection.cursor()
    cursor.execute("""
        SELECT DISTINCT model_version
        FROM commodity.forecast.distributions
        WHERE commodity = %s
        ORDER BY model_version
    """, (commodity,))

    models = [row[0] for row in cursor.fetchall()]
    cursor.close()

    return models


def load_forecast_distributions(
    commodity: str,
    model_version: str,
    connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Load forecast distribution paths for a specific commodity and model.

    Args:
        commodity: str - 'Coffee' or 'Sugar' (case-sensitive)
        model_version: str - Model identifier (e.g., 'sarimax_auto_weather_v1')
        connection: Databricks SQL connection
        start_date: Optional[str] - Filter forecasts >= this date (YYYY-MM-DD)
        end_date: Optional[str] - Filter forecasts <= this date (YYYY-MM-DD)
        limit: Optional[int] - Limit number of rows returned (for testing)

    Returns:
        DataFrame with columns:
            - path_id: int (1-2000 for forecast paths, 0 for actuals)
            - forecast_start_date: date
            - data_cutoff_date: date
            - generation_timestamp: timestamp
            - model_version: str
            - commodity: str
            - day_1 through day_14: float (predicted prices)
            - is_actuals: bool
            - has_data_leakage: bool
    """
    # Build query
    query = """
        SELECT
            path_id,
            forecast_start_date,
            data_cutoff_date,
            generation_timestamp,
            model_version,
            commodity,
            day_1, day_2, day_3, day_4, day_5, day_6, day_7,
            day_8, day_9, day_10, day_11, day_12, day_13, day_14,
            is_actuals,
            has_data_leakage
        FROM commodity.forecast.distributions
        WHERE commodity = %s
          AND model_version = %s
    """

    params = [commodity, model_version]

    # Add date filters if provided
    if start_date:
        query += " AND forecast_start_date >= %s"
        params.append(start_date)

    if end_date:
        query += " AND forecast_start_date <= %s"
        params.append(end_date)

    # Add ordering
    query += " ORDER BY forecast_start_date, path_id"

    # Add limit if provided
    if limit:
        query += f" LIMIT {limit}"

    # Execute query
    cursor = connection.cursor()
    cursor.execute(query, params)

    # Fetch results and convert to DataFrame
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    cursor.close()

    df = pd.DataFrame(rows, columns=columns)

    # Convert date columns
    if len(df) > 0:
        df['forecast_start_date'] = pd.to_datetime(df['forecast_start_date'])
        df['data_cutoff_date'] = pd.to_datetime(df['data_cutoff_date'])
        df['generation_timestamp'] = pd.to_datetime(df['generation_timestamp'])

    return df


def load_forecast_distributions_all_models(
    commodity: str,
    connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load forecast distributions for all models of a commodity.

    Args:
        commodity: str - 'Coffee' or 'Sugar'
        connection: Databricks SQL connection
        start_date: Optional[str] - Filter forecasts >= this date
        end_date: Optional[str] - Filter forecasts <= this date

    Returns:
        Dict mapping model_version to DataFrame

    Example:
        >>> results = load_forecast_distributions_all_models('Coffee', conn)
        >>> results.keys()
        dict_keys(['arima_111_v1', 'prophet_v1', 'sarimax_auto_weather_v1', ...])
    """
    models = get_available_models(commodity, connection)

    results = {}
    for model in models:
        print(f"Loading {commodity} - {model}...")
        df = load_forecast_distributions(
            commodity=commodity,
            model_version=model,
            connection=connection,
            start_date=start_date,
            end_date=end_date
        )
        results[model] = df
        print(f"  ✓ Loaded {len(df):,} rows")

    return results


def transform_to_prediction_matrices(
    df: pd.DataFrame,
    forecast_horizon: int = 14
) -> Dict[datetime, np.ndarray]:
    """
    Transform distribution DataFrame into prediction matrices format.

    The backtesting engine expects a dict mapping dates to prediction matrices:
        {date: np.ndarray of shape (N_paths, H_days)}

    Where:
        - N_paths = number of Monte Carlo simulation paths (typically 2000)
        - H_days = forecast horizon (typically 14 days)

    Args:
        df: DataFrame from load_forecast_distributions()
        forecast_horizon: Number of days ahead (default 14)

    Returns:
        Dict mapping forecast_start_date to numpy array of predictions

    Example:
        >>> df = load_forecast_distributions('Coffee', 'prophet_v1', conn)
        >>> matrices = transform_to_prediction_matrices(df)
        >>> matrices[pd.Timestamp('2020-01-03')].shape
        (2000, 14)
    """
    if len(df) == 0:
        return {}

    # Filter to only forecast paths (exclude actuals where path_id = 0)
    forecast_df = df[df['is_actuals'] == False].copy()

    # Get day columns
    day_cols = [f'day_{i}' for i in range(1, forecast_horizon + 1)]

    # Group by forecast_start_date
    prediction_matrices = {}

    for forecast_date, group in forecast_df.groupby('forecast_start_date'):
        # Extract prediction values for all paths
        # Each row is one simulation path
        predictions = group[day_cols].values

        # Store as numpy array: shape = (n_paths, horizon_days)
        prediction_matrices[forecast_date] = predictions

    return prediction_matrices


def get_data_summary(commodity: str, model_version: str, connection) -> Dict:
    """
    Get summary statistics for a commodity/model combination.

    Args:
        commodity: str - Commodity name
        model_version: str - Model identifier
        connection: Databricks SQL connection

    Returns:
        Dict with summary statistics
    """
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total_rows,
            COUNT(DISTINCT path_id) as unique_paths,
            COUNT(DISTINCT forecast_start_date) as forecast_dates,
            MIN(forecast_start_date) as earliest_date,
            MAX(forecast_start_date) as latest_date,
            SUM(CASE WHEN is_actuals THEN 1 ELSE 0 END) as actuals_rows,
            SUM(CASE WHEN has_data_leakage THEN 1 ELSE 0 END) as leakage_rows
        FROM commodity.forecast.distributions
        WHERE commodity = %s
          AND model_version = %s
    """, (commodity, model_version))

    row = cursor.fetchone()
    cursor.close()

    return {
        'commodity': commodity,
        'model_version': model_version,
        'total_rows': row[0],
        'unique_paths': row[1],
        'forecast_dates': row[2],
        'earliest_date': row[3],
        'latest_date': row[4],
        'actuals_rows': row[5],
        'leakage_rows': row[6]
    }


def validate_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate data quality of loaded forecast distributions.

    Args:
        df: DataFrame from load_forecast_distributions()

    Returns:
        Dict with validation results
    """
    results = {
        'total_rows': len(df),
        'has_data': len(df) > 0,
        'has_leakage': False,
        'missing_values': {},
        'date_range': None,
        'path_count': 0,
        'issues': []
    }

    if len(df) == 0:
        results['issues'].append('No data loaded')
        return results

    # Check for data leakage
    if df['has_data_leakage'].any():
        results['has_leakage'] = True
        results['issues'].append(f"{df['has_data_leakage'].sum()} rows with data leakage")

    # Check for missing values in prediction columns
    day_cols = [f'day_{i}' for i in range(1, 15)]
    for col in day_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                results['missing_values'][col] = null_count
                results['issues'].append(f"{col}: {null_count} missing values")

    # Date range
    results['date_range'] = (
        df['forecast_start_date'].min(),
        df['forecast_start_date'].max()
    )

    # Path count
    results['path_count'] = df['path_id'].nunique()

    return results


# Utility function for pretty printing summaries
def print_data_summary(summary: Dict):
    """Print formatted data summary."""
    print(f"\n{'='*60}")
    print(f"DATA SUMMARY: {summary['commodity']} - {summary['model_version']}")
    print(f"{'='*60}")
    print(f"Total Rows:        {summary['total_rows']:,}")
    print(f"Unique Paths:      {summary['unique_paths']:,}")
    print(f"Forecast Dates:    {summary['forecast_dates']:,}")
    print(f"Date Range:        {summary['earliest_date']} to {summary['latest_date']}")
    print(f"Actuals Rows:      {summary['actuals_rows']:,}")
    print(f"Leakage Rows:      {summary['leakage_rows']:,}")
    print(f"{'='*60}\n")

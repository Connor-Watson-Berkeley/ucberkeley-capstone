"""
Feature Engineering Pipeline for Multi-Region and GDELT Data

Supports flexible data aggregation strategies for different model types:
- 'aggregate': Aggregate across regions/themes (for traditional models)
- 'pivot': Create region-specific/theme-specific columns (for SARIMAX, XGBoost)
- 'all': Keep raw multi-row or vector format (for neural models)

Also supports normalization for neural networks (StandardScaler, MinMaxScaler).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def prepare_data_for_model(
    raw_data: pd.DataFrame,
    commodity: str,
    region_strategy: str = 'aggregate',
    gdelt_strategy: Optional[str] = None,
    gdelt_themes: Optional[List[str]] = None,
    feature_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Prepare data for model training/prediction with flexible aggregation.

    Args:
        raw_data: Raw data from unified_data (may have multiple regions per date)
        commodity: Commodity name ('Coffee' or 'Sugar')
        region_strategy: How to handle region data
            - 'aggregate': Average across regions (e.g., temp_mean_c = AVG(temp_mean_c))
            - 'pivot': Create region-specific columns (e.g., brazil_temp_mean_c, colombia_temp_mean_c)
            - 'all': Keep multi-row format (for neural models)
        gdelt_strategy: How to handle GDELT theme data (if None, uses region_strategy)
            - 'aggregate': Average across themes
            - 'pivot': Create theme-specific columns (e.g., supply_tone_avg, logistics_tone_avg)
            - 'select': Use specific themes only (requires gdelt_themes parameter)
        gdelt_themes: List of GDELT themes to use when gdelt_strategy='select'
            e.g., ['SUPPLY', 'LOGISTICS', 'MARKET']
        feature_columns: Optional list of specific feature columns to include

    Returns:
        Processed DataFrame ready for model training/prediction
    """
    if gdelt_strategy is None:
        gdelt_strategy = region_strategy

    # Filter by commodity
    df = raw_data[raw_data['commodity'] == commodity].copy()

    # Strategy: 'all' - Keep raw multi-row format
    if region_strategy == 'all':
        # For neural models - just return filtered data
        if feature_columns:
            return df[['date', 'commodity', 'region'] + feature_columns].set_index('date')
        return df.set_index('date')

    # Strategy: 'aggregate' - Average across regions
    elif region_strategy == 'aggregate':
        # Group by date and aggregate
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude columns we don't want to aggregate
        exclude_cols = ['commodity', 'region']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        agg_dict = {col: 'mean' for col in numeric_cols}

        df_agg = df.groupby('date').agg(agg_dict).reset_index()
        df_agg['commodity'] = commodity

        # Handle GDELT themes
        if gdelt_strategy == 'aggregate':
            # Average across GDELT themes (if present)
            df_agg = _aggregate_gdelt_themes(df_agg)
        elif gdelt_strategy == 'select' and gdelt_themes:
            # Keep only specific themes
            df_agg = _select_gdelt_themes(df_agg, gdelt_themes)

        if feature_columns:
            keep_cols = ['date'] + [col for col in feature_columns if col in df_agg.columns]
            df_agg = df_agg[keep_cols]

        return df_agg.set_index('date')

    # Strategy: 'pivot' - Create region-specific columns
    elif region_strategy == 'pivot':
        # Pivot region data to create region-specific columns
        df_pivot = _pivot_regions(df, feature_columns)

        # Handle GDELT themes
        if gdelt_strategy == 'pivot':
            # GDELT is already in wide format (one row per date)
            # Just keep theme-specific columns
            pass
        elif gdelt_strategy == 'select' and gdelt_themes:
            # Keep only specific themes
            df_pivot = _select_gdelt_themes(df_pivot, gdelt_themes)
        elif gdelt_strategy == 'aggregate':
            # Average across GDELT themes
            df_pivot = _aggregate_gdelt_themes(df_pivot)

        return df_pivot.set_index('date')

    else:
        raise ValueError(f"Invalid region_strategy: {region_strategy}. Use 'aggregate', 'pivot', or 'all'")


def _pivot_regions(df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Pivot region data to create region-specific columns.

    Example:
        Input (multi-row):
            date       region    temp_mean_c  humidity_mean_pct
            2020-01-01 Brazil    25.0         70.0
            2020-01-01 Colombia  20.0         80.0

        Output (single-row):
            date       brazil_temp_mean_c  brazil_humidity_mean_pct  colombia_temp_mean_c  colombia_humidity_mean_pct
            2020-01-01 25.0               70.0                      20.0                  80.0
    """
    if 'region' not in df.columns:
        # Already pivoted or no region column
        return df

    # Determine which columns to pivot
    if feature_columns:
        pivot_cols = [col for col in feature_columns if col in df.columns and col not in ['date', 'commodity', 'region', 'close']]
    else:
        # Pivot all numeric columns except target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        pivot_cols = [col for col in numeric_cols if col not in ['close']]

    # Also include target column (close)
    pivot_cols_with_target = pivot_cols + ['close']

    # Get unique regions
    regions = df['region'].unique()

    # Create pivoted dataframe
    dfs = []
    for region in regions:
        region_df = df[df['region'] == region].copy()

        # Rename columns with region prefix
        region_name = region.lower().replace(' ', '_')
        rename_dict = {col: f"{region_name}_{col}" for col in pivot_cols_with_target}
        region_df = region_df.rename(columns=rename_dict)

        # Keep only date and renamed columns
        keep_cols = ['date'] + list(rename_dict.values())
        region_df = region_df[keep_cols]

        dfs.append(region_df)

    # Merge all region dataframes
    df_pivot = dfs[0]
    for region_df in dfs[1:]:
        df_pivot = df_pivot.merge(region_df, on='date', how='outer')

    return df_pivot


def _aggregate_gdelt_themes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate GDELT themes by averaging across theme-specific columns.

    Example:
        Input:
            group_SUPPLY_tone_avg    group_LOGISTICS_tone_avg    group_TRADE_tone_avg
            0.5                      0.3                         0.4

        Output:
            gdelt_tone_avg_all
            0.4  (average of 0.5, 0.3, 0.4)
    """
    # Find all GDELT theme columns
    gdelt_cols = [col for col in df.columns if col.startswith('group_')]

    if not gdelt_cols:
        return df

    # Group by metric type (tone_avg, tone_positive, etc.)
    metric_groups = {}
    for col in gdelt_cols:
        # Extract metric name (e.g., 'tone_avg' from 'group_SUPPLY_tone_avg')
        parts = col.split('_')
        if len(parts) >= 3:
            metric = '_'.join(parts[2:])  # e.g., 'tone_avg'
            if metric not in metric_groups:
                metric_groups[metric] = []
            metric_groups[metric].append(col)

    # Create aggregated columns
    for metric, cols in metric_groups.items():
        df[f'gdelt_{metric}_all'] = df[cols].mean(axis=1)

    # Drop original theme columns
    df = df.drop(columns=gdelt_cols)

    return df


def _select_gdelt_themes(df: pd.DataFrame, themes: List[str]) -> pd.DataFrame:
    """
    Keep only specific GDELT themes.

    Args:
        df: DataFrame with GDELT columns
        themes: List of theme names (e.g., ['SUPPLY', 'LOGISTICS', 'MARKET'])

    Returns:
        DataFrame with only selected theme columns
    """
    # Find columns for selected themes
    keep_cols = []
    for col in df.columns:
        if not col.startswith('group_'):
            keep_cols.append(col)
        else:
            # Check if column belongs to selected themes
            for theme in themes:
                if f'group_{theme}_' in col:
                    keep_cols.append(col)
                    break

    return df[keep_cols]


def get_feature_columns_from_config(model_config: Dict) -> List[str]:
    """
    Extract feature column names from model configuration.

    Args:
        model_config: Model configuration dict from model_registry.py

    Returns:
        List of feature column names
    """
    params = model_config.get('params', {})

    # Get exogenous features
    exog_features = params.get('exog_features', [])

    # Get target column
    target = params.get('target', 'close')

    # Combine
    feature_cols = [target] + exog_features

    return feature_cols


def normalize_data(
    df: pd.DataFrame,
    method: str = 'standard',
    target_column: str = 'close',
    exclude_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Normalize data for neural networks.

    Args:
        df: DataFrame to normalize
        method: Normalization method ('standard' or 'minmax')
            - 'standard': StandardScaler (mean=0, std=1)
            - 'minmax': MinMaxScaler (range 0-1)
        target_column: Target column name (will be scaled separately)
        exclude_columns: Columns to exclude from normalization (e.g., date, categorical features)

    Returns:
        Tuple of (normalized DataFrame, scaler_dict for inverse transform)
    """
    if exclude_columns is None:
        exclude_columns = []

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Separate target from features
    feature_cols = [col for col in numeric_cols if col != target_column and col not in exclude_columns]

    # Initialize scalers
    if method == 'standard':
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
    elif method == 'minmax':
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
    else:
        raise ValueError(f"Invalid normalization method: {method}. Use 'standard' or 'minmax'")

    # Create copy for normalized data
    df_norm = df.copy()

    # Normalize features
    if feature_cols:
        df_norm[feature_cols] = feature_scaler.fit_transform(df[feature_cols])

    # Normalize target separately
    if target_column in df.columns:
        df_norm[[target_column]] = target_scaler.fit_transform(df[[target_column]])

    # Store scalers for inverse transform
    scaler_dict = {
        'method': method,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_cols': feature_cols,
        'target_column': target_column
    }

    return df_norm, scaler_dict


def denormalize_data(
    df_norm: pd.DataFrame,
    scaler_dict: Dict,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Denormalize (inverse transform) normalized data.

    Args:
        df_norm: Normalized DataFrame
        scaler_dict: Scaler dictionary from normalize_data()
        columns: Specific columns to denormalize (if None, denormalizes all)

    Returns:
        Denormalized DataFrame
    """
    df_denorm = df_norm.copy()

    # Denormalize features
    if columns is None or any(col in columns for col in scaler_dict['feature_cols']):
        feature_cols = scaler_dict['feature_cols']
        feature_cols_to_denorm = [col for col in feature_cols if col in df_denorm.columns]
        if feature_cols_to_denorm:
            df_denorm[feature_cols_to_denorm] = scaler_dict['feature_scaler'].inverse_transform(
                df_denorm[feature_cols_to_denorm]
            )

    # Denormalize target
    target_col = scaler_dict['target_column']
    if (columns is None or target_col in columns) and target_col in df_denorm.columns:
        df_denorm[[target_col]] = scaler_dict['target_scaler'].inverse_transform(
            df_denorm[[target_col]]
        )

    return df_denorm


def denormalize_forecast(
    forecast_values: np.ndarray,
    scaler_dict: Dict
) -> np.ndarray:
    """
    Denormalize forecast predictions (specifically for target variable).

    Args:
        forecast_values: Normalized forecast values (1D or 2D array)
        scaler_dict: Scaler dictionary from normalize_data()

    Returns:
        Denormalized forecast values
    """
    target_scaler = scaler_dict['target_scaler']

    # Reshape if 1D
    if forecast_values.ndim == 1:
        forecast_values = forecast_values.reshape(-1, 1)
        denorm_values = target_scaler.inverse_transform(forecast_values).flatten()
    else:
        denorm_values = target_scaler.inverse_transform(forecast_values)

    return denorm_values


# Example usage patterns:
"""
# Pattern 1: Traditional models with aggregated data (no normalization)
df_prepared = prepare_data_for_model(
    raw_data=unified_data,
    commodity='Coffee',
    region_strategy='aggregate',
    gdelt_strategy='aggregate'
)
# Result: One row per date, all regions averaged, all themes averaged

# Pattern 2: SARIMAX/XGBoost with region-specific columns
df_prepared = prepare_data_for_model(
    raw_data=unified_data,
    commodity='Coffee',
    region_strategy='pivot',
    gdelt_strategy='select',
    gdelt_themes=['SUPPLY', 'LOGISTICS', 'MARKET']
)
# Result: One row per date, brazil_temp_mean_c, colombia_temp_mean_c, etc.
#         Plus: group_SUPPLY_tone_avg, group_LOGISTICS_tone_avg, etc.

# Pattern 3: Neural models with full vectors and normalization
df_prepared = prepare_data_for_model(
    raw_data=unified_data,
    commodity='Coffee',
    region_strategy='all',
    gdelt_strategy='all'
)

# Normalize for LSTM
df_norm, scaler_dict = normalize_data(
    df=df_prepared,
    method='standard',  # or 'minmax'
    target_column='close'
)

# Train LSTM with normalized data
# lstm_model.fit(df_norm, ...)

# Forecast (normalized output)
forecast_norm = lstm_model.predict(...)

# Denormalize forecast
forecast_actual = denormalize_forecast(forecast_norm, scaler_dict)

# Pattern 4: Feature subset with specific columns
df_prepared = prepare_data_for_model(
    raw_data=unified_data,
    commodity='Coffee',
    region_strategy='aggregate',
    feature_columns=['close', 'temp_mean_c', 'humidity_mean_pct', 'precipitation_mm']
)
# Result: Only specified columns, regions aggregated
"""

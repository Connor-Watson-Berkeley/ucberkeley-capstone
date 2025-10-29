"""Random Walk with drift forecaster.

Forecast = last value + average daily drift * days ahead

Slightly better than naive - accounts for trends.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def random_walk_forecast(df_pandas: pd.DataFrame, target: str = 'close',
                         horizon: int = 14, lookback_days: int = 30) -> pd.DataFrame:
    """
    Random Walk with drift - adds average trend to last value.

    Args:
        df_pandas: Training data with DatetimeIndex
        target: Target column name
        horizon: Forecast days ahead
        lookback_days: Days to calculate drift from

    Returns:
        DataFrame with forecast and confidence intervals

    Example:
        Last close = 167.50
        Average daily change (last 30 days) = +0.20
        Forecast day 1 = 167.50 + 0.20 = 167.70
        Forecast day 2 = 167.50 + 2*0.20 = 167.90
        ...
    """
    # Get last value
    last_value = df_pandas[target].iloc[-1]
    last_date = df_pandas.index[-1]

    # Calculate drift (average daily change)
    recent_data = df_pandas[target].iloc[-lookback_days:]
    daily_changes = recent_data.diff().dropna()
    drift = daily_changes.mean()

    # Calculate volatility for confidence intervals
    daily_vol = daily_changes.std()

    # Create future dates
    future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                  periods=horizon, freq='D')

    # Build forecast
    forecasts = []
    lower_80_list = []
    upper_80_list = []
    lower_95_list = []
    upper_95_list = []

    for days_ahead in range(1, horizon + 1):
        # Point forecast: last value + drift * days
        point_forecast = last_value + drift * days_ahead

        # Confidence intervals: expand with sqrt(time)
        vol_scaled = daily_vol * np.sqrt(days_ahead)

        forecasts.append(point_forecast)
        lower_80_list.append(point_forecast - 1.28 * vol_scaled)
        upper_80_list.append(point_forecast + 1.28 * vol_scaled)
        lower_95_list.append(point_forecast - 1.96 * vol_scaled)
        upper_95_list.append(point_forecast + 1.96 * vol_scaled)

    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': forecasts,
        'lower_80': lower_80_list,
        'upper_80': upper_80_list,
        'lower_95': lower_95_list,
        'upper_95': upper_95_list
    })

    return forecast_df


def random_walk_forecast_with_metadata(df_pandas: pd.DataFrame, commodity: str,
                                        target: str = 'close', horizon: int = 14,
                                        lookback_days: int = 30,
                                        cutoff_date: str = None) -> dict:
    """
    Random Walk forecast with full metadata for model registry.

    Args:
        df_pandas: Training data
        commodity: 'Coffee' or 'Sugar'
        target: Target column
        horizon: Forecast days
        lookback_days: Days for drift calculation
        cutoff_date: Optional - for backtesting

    Returns:
        Dict with forecast and metadata
    """
    # Filter by cutoff if provided
    if cutoff_date:
        df_pandas = df_pandas[df_pandas.index <= cutoff_date]

    # Calculate drift for metadata
    recent_data = df_pandas[target].iloc[-lookback_days:]
    drift = recent_data.diff().dropna().mean()

    # Generate forecast
    forecast_df = random_walk_forecast(df_pandas, target, horizon, lookback_days)

    # Add metadata
    return {
        'forecast_df': forecast_df,
        'model_name': 'RandomWalk',
        'commodity': commodity,
        'parameters': {
            'method': 'random_walk_with_drift',
            'target': target,
            'horizon': horizon,
            'lookback_days': lookback_days,
            'estimated_drift': float(drift)
        },
        'training_end': df_pandas.index[-1],
        'forecast_start': forecast_df['date'].iloc[0],
        'forecast_end': forecast_df['date'].iloc[-1]
    }

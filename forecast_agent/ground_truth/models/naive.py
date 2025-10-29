"""Naive persistence forecaster - simplest baseline.

Forecast = last observed value, held constant for entire horizon.

Use case: Baseline comparison - every model should beat this!
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def naive_forecast(df_pandas: pd.DataFrame, target: str = 'close',
                   horizon: int = 14) -> pd.DataFrame:
    """
    Naive persistence forecast - last value repeated for horizon.

    Args:
        df_pandas: Training data with DatetimeIndex
        target: Target column name (default: 'close')
        horizon: Forecast days ahead

    Returns:
        DataFrame with columns: [date, forecast, lower_80, upper_80, lower_95, upper_95]

    Example:
        Last close = 167.50
        Forecast: All 14 days = 167.50

    Confidence intervals: Use historical volatility
    """
    # Get last value
    last_value = df_pandas[target].iloc[-1]
    last_date = df_pandas.index[-1]

    # Calculate historical volatility for confidence intervals
    returns = df_pandas[target].pct_change().dropna()
    daily_vol = returns.std()

    # Create future dates
    future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                  periods=horizon, freq='D')

    # Build forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': last_value,
    })

    # Confidence intervals: expand with sqrt(time)
    # 80% CI: ±1.28 std devs, 95% CI: ±1.96 std devs
    for i, days_ahead in enumerate(range(1, horizon + 1)):
        vol_scaled = last_value * daily_vol * np.sqrt(days_ahead)

        forecast_df.loc[i, 'lower_80'] = last_value - 1.28 * vol_scaled
        forecast_df.loc[i, 'upper_80'] = last_value + 1.28 * vol_scaled
        forecast_df.loc[i, 'lower_95'] = last_value - 1.96 * vol_scaled
        forecast_df.loc[i, 'upper_95'] = last_value + 1.96 * vol_scaled

    return forecast_df


def naive_forecast_with_metadata(df_pandas: pd.DataFrame, commodity: str,
                                  target: str = 'close', horizon: int = 14,
                                  cutoff_date: str = None) -> dict:
    """
    Naive forecast with full metadata for model registry.

    Args:
        df_pandas: Training data
        commodity: 'Coffee' or 'Sugar'
        target: Target column
        horizon: Forecast days
        cutoff_date: Optional - for backtesting

    Returns:
        Dict with keys: forecast_df, model_name, parameters, training_end
    """
    # Filter by cutoff if provided
    if cutoff_date:
        df_pandas = df_pandas[df_pandas.index <= cutoff_date]

    # Generate forecast
    forecast_df = naive_forecast(df_pandas, target, horizon)

    # Add metadata
    return {
        'forecast_df': forecast_df,
        'model_name': 'Naive',
        'commodity': commodity,
        'parameters': {
            'method': 'persistence',
            'target': target,
            'horizon': horizon
        },
        'training_end': df_pandas.index[-1],
        'forecast_start': forecast_df['date'].iloc[0],
        'forecast_end': forecast_df['date'].iloc[-1]
    }

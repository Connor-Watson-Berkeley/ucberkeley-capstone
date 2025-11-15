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


def random_walk_train(df_pandas: pd.DataFrame, target: str = 'close',
                      lookback_days: int = 30) -> dict:
    """
    Train random walk model (captures state needed for forecasting).

    For random walk, "training" means capturing:
    - Last value
    - Drift (average daily change)
    - Volatility (for confidence intervals)

    Args:
        df_pandas: Training data with DatetimeIndex
        target: Target column name
        lookback_days: Days to calculate drift from

    Returns:
        Dict containing fitted model state
    """
    last_value = df_pandas[target].iloc[-1]
    last_date = df_pandas.index[-1]

    # Calculate drift (average daily change)
    recent_data = df_pandas[target].iloc[-lookback_days:]
    daily_changes = recent_data.diff().dropna()
    drift = daily_changes.mean()

    # Calculate volatility for confidence intervals
    daily_vol = daily_changes.std()

    return {
        'last_value': last_value,
        'drift': drift,
        'daily_vol': daily_vol,
        'last_date': last_date,
        'target': target,
        'lookback_days': lookback_days,
        'model_type': 'random_walk'
    }


def random_walk_predict(fitted_model: dict, horizon: int = 14) -> pd.DataFrame:
    """
    Generate forecast using fitted random walk model.

    Args:
        fitted_model: Dict returned by random_walk_train()
        horizon: Forecast days ahead

    Returns:
        DataFrame with columns: [date, forecast, lower_80, upper_80, lower_95, upper_95]
    """
    last_value = fitted_model['last_value']
    last_date = fitted_model['last_date']
    drift = fitted_model['drift']
    daily_vol = fitted_model['daily_vol']

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
                                        cutoff_date: str = None,
                                        fitted_model: dict = None) -> dict:
    """
    Random Walk forecast with full metadata for model registry.

    Can either train+predict (if fitted_model is None) or just predict
    (if fitted_model is provided).

    Args:
        df_pandas: Training data (only used if fitted_model is None)
        commodity: 'Coffee' or 'Sugar'
        target: Target column
        horizon: Forecast days
        lookback_days: Days for drift calculation
        cutoff_date: Optional - for backtesting
        fitted_model: Optional - pre-trained model from random_walk_train()

    Returns:
        Dict with forecast and metadata
    """
    # If no fitted model provided, train one
    if fitted_model is None:
        # Filter by cutoff if provided
        if cutoff_date:
            df_pandas = df_pandas[df_pandas.index <= cutoff_date]

        # Train model
        fitted_model = random_walk_train(df_pandas, target, lookback_days)

    # Generate forecast using fitted model
    forecast_df = random_walk_predict(fitted_model, horizon)

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
            'estimated_drift': float(fitted_model['drift'])
        },
        'training_end': fitted_model['last_date'],
        'forecast_start': forecast_df['date'].iloc[0],
        'forecast_end': forecast_df['date'].iloc[-1],
        'fitted_model': fitted_model  # Return fitted model for reuse!
    }

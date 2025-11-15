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


def naive_train(df_pandas: pd.DataFrame, target: str = 'close') -> dict:
    """
    Train naive model (captures state needed for forecasting).

    For naive model, "training" just means capturing the last value
    and historical volatility.

    Args:
        df_pandas: Training data with DatetimeIndex
        target: Target column name

    Returns:
        Dict containing fitted model state (last_value, daily_vol, last_date, target)
    """
    last_value = df_pandas[target].iloc[-1]
    last_date = df_pandas.index[-1]

    # Calculate historical volatility
    returns = df_pandas[target].pct_change().dropna()
    daily_vol = returns.std()

    return {
        'last_value': last_value,
        'daily_vol': daily_vol,
        'last_date': last_date,
        'target': target,
        'model_type': 'naive'
    }


def naive_predict(fitted_model: dict, horizon: int = 14) -> pd.DataFrame:
    """
    Generate forecast using fitted naive model.

    Args:
        fitted_model: Dict returned by naive_train()
        horizon: Forecast days ahead

    Returns:
        DataFrame with columns: [date, forecast, lower_80, upper_80, lower_95, upper_95]
    """
    last_value = fitted_model['last_value']
    last_date = fitted_model['last_date']
    daily_vol = fitted_model['daily_vol']

    # Create future dates
    future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                  periods=horizon, freq='D')

    # Build forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': last_value,
    })

    # Confidence intervals: expand with sqrt(time)
    for i, days_ahead in enumerate(range(1, horizon + 1)):
        vol_scaled = last_value * daily_vol * np.sqrt(days_ahead)

        forecast_df.loc[i, 'lower_80'] = last_value - 1.28 * vol_scaled
        forecast_df.loc[i, 'upper_80'] = last_value + 1.28 * vol_scaled
        forecast_df.loc[i, 'lower_95'] = last_value - 1.96 * vol_scaled
        forecast_df.loc[i, 'upper_95'] = last_value + 1.96 * vol_scaled

    return forecast_df


def naive_forecast_with_metadata(df_pandas: pd.DataFrame, commodity: str,
                                  target: str = 'close', horizon: int = 14,
                                  cutoff_date: str = None,
                                  fitted_model: dict = None) -> dict:
    """
    Naive forecast with full metadata for model registry.

    Can either train+predict (if fitted_model is None) or just predict
    (if fitted_model is provided).

    Args:
        df_pandas: Training data (only used if fitted_model is None)
        commodity: 'Coffee' or 'Sugar'
        target: Target column
        horizon: Forecast days
        cutoff_date: Optional - for backtesting
        fitted_model: Optional - pre-trained model from naive_train()

    Returns:
        Dict with keys: forecast_df, model_name, parameters, training_end, fitted_model
    """
    # If no fitted model provided, train one
    if fitted_model is None:
        # Filter by cutoff if provided
        if cutoff_date:
            df_pandas = df_pandas[df_pandas.index <= cutoff_date]

        # Train model
        fitted_model = naive_train(df_pandas, target)

    # Generate forecast using fitted model
    forecast_df = naive_predict(fitted_model, horizon)

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
        'training_end': fitted_model['last_date'],
        'forecast_start': forecast_df['date'].iloc[0],
        'forecast_end': forecast_df['date'].iloc[-1],
        'fitted_model': fitted_model  # Return fitted model for reuse!
    }

"""Covariate projection strategies for forecast horizon.

When forecasting 14 days ahead, exogenous variables (weather, etc.) need values
for the entire forecast period. These functions handle that projection.

Key principle: Only use past data (no future knowledge).
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def none_needed(df_pandas: pd.DataFrame, features: list, horizon: int = 14) -> pd.DataFrame:
    """
    No covariate projection needed - for pure ARIMA models.

    Use case: ARIMA without exogenous variables

    Args:
        df_pandas: Training data (already converted from Spark)
        features: Feature list (unused)
        horizon: Forecast days (unused)

    Returns:
        None (no covariates needed)
    """
    return None


def persist_last_value(df_pandas: pd.DataFrame, features: list, horizon: int = 14) -> pd.DataFrame:
    """
    Roll forward most recent values - simple baseline (prototype approach).

    Use case: SARIMAX baseline, LSTM

    Args:
        df_pandas: Training data with index as date
        features: List of covariate features to project
        horizon: Number of days to forecast

    Returns:
        DataFrame with projected covariates (horizon rows)

    Example:
        Last known: temp_c=25, humidity=80
        Projection:  day 1-14 all have temp_c=25, humidity=80

    Pros: Simple, stable
    Cons: Unrealistic (weather doesn't stay constant for 14 days)
    """
    # Get last known values
    last_row = df_pandas.iloc[-1]

    # Create future dates
    last_date = df_pandas.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')

    # Replicate last values for all future dates
    projected = pd.DataFrame(index=future_dates)

    for feature in features:
        if feature in df_pandas.columns and feature != 'close':  # Don't project target
            projected[feature] = last_row[feature]

    return projected


def seasonal_average(df_pandas: pd.DataFrame, features: list, horizon: int = 14,
                     lookback_years: int = 3) -> pd.DataFrame:
    """
    Use historical average for same calendar period - captures seasonality.

    Use case: SARIMAX (better than persistence)

    Args:
        df_pandas: Training data with DatetimeIndex
        features: Covariate features to project
        horizon: Forecast days
        lookback_years: How many past years to average

    Returns:
        DataFrame with seasonal averages

    Example:
        Forecasting Jan 15, 2024:
        - Look at Jan 15 in 2021, 2022, 2023
        - Average their temp_c, humidity_pct
        - Use that as Jan 15, 2024 projection

    Pros: Captures seasonality
    Cons: Ignores recent trends, climate change
    """
    last_date = df_pandas.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')

    projected = pd.DataFrame(index=future_dates)

    for feature in features:
        if feature not in df_pandas.columns or feature == 'close':
            continue

        projected_values = []

        for future_date in future_dates:
            # Get same day-of-year from past N years
            historical_values = []

            for year_back in range(1, lookback_years + 1):
                historical_date = future_date - pd.DateOffset(years=year_back)

                # Try exact date, then +/- 1 day (for leap years, etc.)
                for offset in [0, -1, 1]:
                    check_date = historical_date + timedelta(days=offset)
                    if check_date in df_pandas.index:
                        val = df_pandas.loc[check_date, feature]
                        if pd.notna(val):
                            historical_values.append(val)
                        break

            # Average historical values
            if historical_values:
                projected_values.append(np.mean(historical_values))
            else:
                # Fall back to last known value if no historical data
                projected_values.append(df_pandas[feature].iloc[-1])

        projected[feature] = projected_values

    return projected


def linear_trend(df_pandas: pd.DataFrame, features: list, horizon: int = 14,
                 lookback_days: int = 30) -> pd.DataFrame:
    """
    Fit linear trend on recent data, extrapolate forward - for trending variables.

    Use case: SARIMAX (advanced)

    Args:
        df_pandas: Training data
        features: Covariates to project
        horizon: Forecast days
        lookback_days: Days to fit trend on

    Returns:
        DataFrame with trend-based projections

    Example:
        Last 30 days: temp increasing by 0.5°C per day
        Projection: Continue that trend forward

    Pros: Captures recent trends (e.g., warming temperatures)
    Cons: Can extrapolate unrealistically, overshoot

    Note: Use with caution - trends can reverse!
    """
    from scipy import stats

    last_date = df_pandas.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')

    # Get recent data for trend fitting
    recent_data = df_pandas.iloc[-lookback_days:]

    projected = pd.DataFrame(index=future_dates)

    for feature in features:
        if feature not in df_pandas.columns or feature == 'close':
            continue

        # Fit linear regression
        y = recent_data[feature].values
        x = np.arange(len(y))

        # Handle NaN
        mask = ~np.isnan(y)
        if mask.sum() < 5:  # Need at least 5 points
            # Fall back to last value
            projected[feature] = df_pandas[feature].iloc[-1]
            continue

        slope, intercept, r_val, p_val, std_err = stats.linregress(x[mask], y[mask])

        # Project forward
        future_x = np.arange(len(y), len(y) + horizon)
        future_y = slope * future_x + intercept

        projected[feature] = future_y

    return projected


def weather_forecast_api(df_pandas: pd.DataFrame, features: list, horizon: int = 14) -> pd.DataFrame:
    """
    Use actual 14-day weather forecast from API - FUTURE implementation.

    Use case: All models (when available) - HIGH IMPACT improvement

    Args:
        df_pandas: Training data
        features: Features to project
        horizon: Forecast days

    Returns:
        DataFrame with actual weather forecasts

    Raises:
        NotImplementedError: Weather API not yet integrated

    TODO:
        - Integrate OpenWeather API or Weather.gov
        - Cache forecasts to avoid repeated API calls
        - Handle API failures gracefully (fall back to seasonal_average)
        - Cost: ~$200/month for commercial use

    See: agent_instructions/FUTURE_IMPROVEMENTS.md for details
    """
    raise NotImplementedError(
        "Weather API not yet integrated. "
        "See FUTURE_IMPROVEMENTS.md for implementation plan. "
        "Use 'persist_last_value' or 'seasonal_average' for now."
    )


# Helper function for model selection
def get_projection_function(method: str):
    """
    Get projection function by name.

    Args:
        method: 'none', 'persist', 'seasonal', 'linear', 'weather_api'

    Returns:
        Projection function

    Example:
        fn = get_projection_function('seasonal')
        projected = fn(df, features=['temp_c', 'humidity'], horizon=14)
    """
    functions = {
        'none': none_needed,
        'persist': persist_last_value,
        'seasonal': seasonal_average,
        'linear': linear_trend,
        'weather_api': weather_forecast_api
    }

    if method not in functions:
        raise ValueError(f"Unknown projection method: {method}. Choose from {list(functions.keys())}")

    return functions[method]

"""ARIMA forecaster - classical time series model.

ARIMA(p,d,q) = AutoRegressive Integrated Moving Average
- p: AR order (lagged values)
- d: Differencing order (remove trends)
- q: MA order (lagged errors)

Use case: Univariate baseline (no exogenous variables)
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA


def arima_forecast(df_pandas: pd.DataFrame, target: str = 'close',
                   order: tuple = (1, 1, 1), horizon: int = 14) -> pd.DataFrame:
    """
    ARIMA forecast - classical time series model.

    Args:
        df_pandas: Training data with DatetimeIndex
        target: Target column name
        order: (p, d, q) tuple - AR, differencing, MA orders
        horizon: Forecast days ahead

    Returns:
        DataFrame with forecast and confidence intervals

    Example:
        ARIMA(1,1,1):
        - p=1: Use yesterday's value
        - d=1: First differencing (remove trend)
        - q=1: Use yesterday's forecast error
    """
    # Extract target series
    y = df_pandas[target]
    last_date = df_pandas.index[-1]

    # Fit ARIMA model
    model = ARIMA(y, order=order)
    fitted = model.fit()

    # Generate forecast
    forecast_result = fitted.forecast(steps=horizon)
    forecast_ci = fitted.get_forecast(steps=horizon).conf_int(alpha=0.2)  # 80% CI
    forecast_ci_95 = fitted.get_forecast(steps=horizon).conf_int(alpha=0.05)  # 95% CI

    # Create future dates
    future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                  periods=horizon, freq='D')

    # Build forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': forecast_result.values,
        'lower_80': forecast_ci.iloc[:, 0].values,
        'upper_80': forecast_ci.iloc[:, 1].values,
        'lower_95': forecast_ci_95.iloc[:, 0].values,
        'upper_95': forecast_ci_95.iloc[:, 1].values
    })

    return forecast_df


def arima_forecast_with_metadata(df_pandas: pd.DataFrame, commodity: str,
                                  target: str = 'close', order: tuple = (1, 1, 1),
                                  horizon: int = 14, cutoff_date: str = None) -> dict:
    """
    ARIMA forecast with full metadata for model registry.

    Args:
        df_pandas: Training data
        commodity: 'Coffee' or 'Sugar'
        target: Target column
        order: (p, d, q) ARIMA order
        horizon: Forecast days
        cutoff_date: Optional - for backtesting

    Returns:
        Dict with forecast, model diagnostics, and metadata
    """
    # Filter by cutoff if provided
    if cutoff_date:
        df_pandas = df_pandas[df_pandas.index <= cutoff_date]

    # Extract target series
    y = df_pandas[target]

    # Fit ARIMA model
    model = ARIMA(y, order=order)
    fitted = model.fit()

    # Generate forecast
    forecast_df = arima_forecast(df_pandas, target, order, horizon)

    # Extract model diagnostics
    p, d, q = order
    aic = fitted.aic
    bic = fitted.bic

    # Add metadata
    return {
        'forecast_df': forecast_df,
        'model_name': f'ARIMA({p},{d},{q})',
        'commodity': commodity,
        'parameters': {
            'method': 'arima',
            'target': target,
            'order': order,
            'p': p,
            'd': d,
            'q': q,
            'horizon': horizon,
            'aic': float(aic),
            'bic': float(bic)
        },
        'fitted_model': fitted,  # For diagnostics
        'training_end': df_pandas.index[-1],
        'forecast_start': forecast_df['date'].iloc[0],
        'forecast_end': forecast_df['date'].iloc[-1]
    }

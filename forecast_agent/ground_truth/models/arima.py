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


def arima_train(df_pandas: pd.DataFrame, target: str = 'close',
                order: tuple = (1, 1, 1)) -> dict:
    """
    Train ARIMA model.

    Args:
        df_pandas: Training data with DatetimeIndex
        target: Target column name
        order: (p, d, q) ARIMA order

    Returns:
        Dict containing fitted ARIMA model and metadata
    """
    # Extract target series
    y = df_pandas[target]

    # Fit ARIMA model
    model = ARIMA(y, order=order)
    fitted = model.fit()

    return {
        'fitted_model': fitted,  # statsmodels ARIMAResults object
        'last_date': df_pandas.index[-1],
        'target': target,
        'order': order,
        'aic': float(fitted.aic),
        'bic': float(fitted.bic),
        'model_type': 'arima'
    }


def arima_predict(fitted_model_dict: dict, horizon: int = 14) -> pd.DataFrame:
    """
    Generate forecast using fitted ARIMA model.

    Args:
        fitted_model_dict: Dict returned by arima_train()
        horizon: Forecast days ahead

    Returns:
        DataFrame with forecast and confidence intervals
    """
    fitted = fitted_model_dict['fitted_model']
    last_date = fitted_model_dict['last_date']

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
                                  horizon: int = 14, cutoff_date: str = None,
                                  fitted_model: dict = None) -> dict:
    """
    ARIMA forecast with full metadata for model registry.

    Can either train+predict (if fitted_model is None) or just predict
    (if fitted_model is provided).

    Args:
        df_pandas: Training data (only used if fitted_model is None)
        commodity: 'Coffee' or 'Sugar'
        target: Target column
        order: (p, d, q) ARIMA order
        horizon: Forecast days
        cutoff_date: Optional - for backtesting
        fitted_model: Optional - pre-trained model from arima_train()

    Returns:
        Dict with forecast, model diagnostics, and metadata
    """
    # If no fitted model provided, train one
    if fitted_model is None:
        # Filter by cutoff if provided
        if cutoff_date:
            df_pandas = df_pandas[df_pandas.index <= cutoff_date]

        # Train model
        fitted_model = arima_train(df_pandas, target, order)

    # Generate forecast using fitted model
    forecast_df = arima_predict(fitted_model, horizon)

    # Extract model diagnostics
    p, d, q = order
    aic = fitted_model['aic']
    bic = fitted_model['bic']

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
        'fitted_model': fitted_model,  # Return fitted model for reuse!
        'training_end': fitted_model['last_date'],
        'forecast_start': forecast_df['date'].iloc[0],
        'forecast_end': forecast_df['date'].iloc[-1]
    }

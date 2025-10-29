"""Statsforecast models - AutoARIMA, AutoETS, and more.

Fast, automated time series forecasting.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from statsforecast import StatsForecast
    from statsforecast.models import (
        AutoARIMA, AutoETS, AutoCES, AutoTheta,
        Naive, SeasonalNaive, SimpleExponentialSmoothing,
        Holt, HoltWinters
    )
    STATSFORECAST_AVAILABLE = True
except ImportError:
    STATSFORECAST_AVAILABLE = False
    print("⚠️  statsforecast not available. Install with: pip install statsforecast")


def statsforecast_model_forecast(df_pandas: pd.DataFrame, model_class,
                                 target: str = 'close', horizon: int = 14,
                                 season_length: int = 7) -> dict:
    """
    Generic statsforecast model wrapper.

    Args:
        df_pandas: Training data with DatetimeIndex
        model_class: Statsforecast model class (e.g., AutoETS)
        target: Target column
        horizon: Forecast horizon
        season_length: Seasonality period (7 for weekly)

    Returns:
        Dict with forecast_df and model
    """
    if not STATSFORECAST_AVAILABLE:
        raise ImportError("statsforecast not installed")

    # Prepare data (needs 'unique_id', 'ds', 'y')
    df_model = pd.DataFrame({
        'unique_id': 'Coffee',  # Single series
        'ds': df_pandas.index,
        'y': df_pandas[target].values
    })

    # Initialize model
    model = StatsForecast(
        models=[model_class(season_length=season_length)],
        freq='D',
        n_jobs=-1
    )

    # Fit and forecast
    model.fit(df_model)
    forecast = model.predict(h=horizon)

    # Build output DataFrame
    last_train_date = df_pandas.index[-1]
    forecast_dates = pd.date_range(
        start=last_train_date + timedelta(days=1),
        periods=horizon,
        freq='D'
    )

    # Get model name from class
    model_col = [col for col in forecast.columns if col not in ['unique_id', 'ds']][0]

    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecast': forecast[model_col].values,
        'lower_80': np.nan,  # Most models don't provide intervals
        'upper_80': np.nan,
        'lower_95': np.nan,
        'upper_95': np.nan
    })

    return {
        'forecast_df': forecast_df,
        'model': model
    }


def auto_arima_forecast_with_metadata(df_pandas: pd.DataFrame, commodity: str,
                                      target: str = 'close', horizon: int = 14,
                                      cutoff_date: str = None) -> dict:
    """AutoARIMA with automatic parameter selection."""
    if cutoff_date:
        df_pandas = df_pandas[df_pandas.index <= cutoff_date]

    result = statsforecast_model_forecast(df_pandas, AutoARIMA, target, horizon)

    return {
        'forecast_df': result['forecast_df'],
        'model_name': 'AutoARIMA',
        'commodity': commodity,
        'parameters': {
            'method': 'auto_arima_statsforecast',
            'target': target,
            'horizon': horizon
        },
        'model': result['model'],
        'training_end': df_pandas.index[-1],
        'forecast_start': result['forecast_df']['date'].iloc[0],
        'forecast_end': result['forecast_df']['date'].iloc[-1]
    }


def auto_ets_forecast_with_metadata(df_pandas: pd.DataFrame, commodity: str,
                                    target: str = 'close', horizon: int = 14,
                                    cutoff_date: str = None) -> dict:
    """AutoETS - Exponential smoothing with automatic model selection."""
    if cutoff_date:
        df_pandas = df_pandas[df_pandas.index <= cutoff_date]

    result = statsforecast_model_forecast(df_pandas, AutoETS, target, horizon)

    return {
        'forecast_df': result['forecast_df'],
        'model_name': 'AutoETS',
        'commodity': commodity,
        'parameters': {
            'method': 'auto_ets',
            'target': target,
            'horizon': horizon,
            'description': 'Exponential Smoothing State Space Model'
        },
        'model': result['model'],
        'training_end': df_pandas.index[-1],
        'forecast_start': result['forecast_df']['date'].iloc[0],
        'forecast_end': result['forecast_df']['date'].iloc[-1]
    }


def holt_winters_forecast_with_metadata(df_pandas: pd.DataFrame, commodity: str,
                                        target: str = 'close', horizon: int = 14,
                                        cutoff_date: str = None) -> dict:
    """Holt-Winters - Triple exponential smoothing."""
    if cutoff_date:
        df_pandas = df_pandas[df_pandas.index <= cutoff_date]

    result = statsforecast_model_forecast(df_pandas, HoltWinters, target, horizon, season_length=7)

    return {
        'forecast_df': result['forecast_df'],
        'model_name': 'HoltWinters',
        'commodity': commodity,
        'parameters': {
            'method': 'holt_winters',
            'target': target,
            'horizon': horizon,
            'description': 'Triple Exponential Smoothing (trend + seasonality)'
        },
        'model': result['model'],
        'training_end': df_pandas.index[-1],
        'forecast_start': result['forecast_df']['date'].iloc[0],
        'forecast_end': result['forecast_df']['date'].iloc[-1]
    }


def auto_theta_forecast_with_metadata(df_pandas: pd.DataFrame, commodity: str,
                                      target: str = 'close', horizon: int = 14,
                                      cutoff_date: str = None) -> dict:
    """AutoTheta - Theta method for forecasting."""
    if cutoff_date:
        df_pandas = df_pandas[df_pandas.index <= cutoff_date]

    result = statsforecast_model_forecast(df_pandas, AutoTheta, target, horizon)

    return {
        'forecast_df': result['forecast_df'],
        'model_name': 'AutoTheta',
        'commodity': commodity,
        'parameters': {
            'method': 'auto_theta',
            'target': target,
            'horizon': horizon,
            'description': 'Theta method - winner of M3 competition'
        },
        'model': result['model'],
        'training_end': df_pandas.index[-1],
        'forecast_start': result['forecast_df']['date'].iloc[0],
        'forecast_end': result['forecast_df']['date'].iloc[-1]
    }

"""Prophet forecaster from Meta/Facebook.

Prophet handles seasonality, holidays, and trend changes automatically.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


def prophet_forecast(df_pandas: pd.DataFrame, target: str = 'close',
                     exog_features: list = None, horizon: int = 14,
                     weekly_seasonality: bool = True,
                     yearly_seasonality: bool = True) -> dict:
    """
    Prophet forecast with optional exogenous variables.

    Args:
        df_pandas: Training data with DatetimeIndex
        target: Target column
        exog_features: List of regressors (optional)
        horizon: Forecast days ahead
        weekly_seasonality: Enable weekly patterns
        yearly_seasonality: Enable yearly patterns

    Returns:
        Dict with forecast_df and Prophet model
    """
    # Prepare data for Prophet (needs 'ds' and 'y' columns)
    prophet_df = pd.DataFrame({
        'ds': df_pandas.index,
        'y': df_pandas[target].values
    })

    # Initialize Prophet
    model = Prophet(
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,  # Flexibility in trend changes
        seasonality_prior_scale=10.0
    )

    # Add exogenous regressors
    if exog_features:
        for feature in exog_features:
            if feature in df_pandas.columns:
                prophet_df[feature] = df_pandas[feature].values
                model.add_regressor(feature)

    # Fit model
    model.fit(prophet_df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=horizon, freq='D')

    # Add exogenous features for forecast period (use last known values)
    if exog_features:
        for feature in exog_features:
            if feature in df_pandas.columns:
                # Extend with last known value
                last_value = df_pandas[feature].iloc[-1]
                future[feature] = df_pandas[feature].reindex(future['ds'], method='ffill').fillna(last_value).values

    # Generate forecast
    forecast = model.predict(future)

    # Extract only future period
    last_train_date = df_pandas.index[-1]
    forecast_future = forecast[forecast['ds'] > last_train_date].reset_index(drop=True)

    # Build output DataFrame
    forecast_df = pd.DataFrame({
        'date': pd.to_datetime(forecast_future['ds']),
        'forecast': forecast_future['yhat'].values,
        'lower_80': forecast_future['yhat_lower'].values,  # Prophet gives ~95% CI by default
        'upper_80': forecast_future['yhat_upper'].values,
        'lower_95': forecast_future['yhat_lower'].values,
        'upper_95': forecast_future['yhat_upper'].values
    })

    return {
        'forecast_df': forecast_df,
        'model': model,
        'forecast_components': forecast
    }


def prophet_forecast_with_metadata(df_pandas: pd.DataFrame, commodity: str,
                                    target: str = 'close',
                                    exog_features: list = None,
                                    horizon: int = 14,
                                    cutoff_date: str = None) -> dict:
    """
    Prophet forecast with full metadata for model registry.

    Args:
        df_pandas: Training data
        commodity: 'Coffee' or 'Sugar'
        target: Target column
        exog_features: List of regressors
        horizon: Forecast days
        cutoff_date: Optional - for backtesting

    Returns:
        Dict with forecast and metadata
    """
    # Filter by cutoff if provided
    if cutoff_date:
        df_pandas = df_pandas[df_pandas.index <= cutoff_date]

    # Generate forecast
    result = prophet_forecast(df_pandas, target, exog_features, horizon)

    # Build metadata
    model_name = 'Prophet'
    if exog_features:
        model_name += f'+{len(exog_features)}exog'

    return {
        'forecast_df': result['forecast_df'],
        'model_name': model_name,
        'commodity': commodity,
        'parameters': {
            'method': 'prophet',
            'target': target,
            'horizon': horizon,
            'exog_features': exog_features,
            'weekly_seasonality': True,
            'yearly_seasonality': True
        },
        'model': result['model'],
        'training_end': df_pandas.index[-1],
        'forecast_start': result['forecast_df']['date'].iloc[0],
        'forecast_end': result['forecast_df']['date'].iloc[-1]
    }

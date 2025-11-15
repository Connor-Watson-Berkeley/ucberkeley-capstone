"""SARIMAX forecaster - ARIMA with exogenous variables and seasonality.

SARIMAX(p,d,q)(P,D,Q,s) = Seasonal ARIMA with eXogenous variables
- (p,d,q): Non-seasonal ARIMA orders
- (P,D,Q,s): Seasonal ARIMA orders (s = seasonal period)
- Exogenous: Weather covariates (temp, humidity, precipitation)

Use case: Sophisticated baseline with weather data
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima


def sarimax_forecast(df_pandas: pd.DataFrame, target: str = 'close',
                     exog_features: list = None, order: tuple = None,
                     seasonal_order: tuple = (0, 0, 0, 0), horizon: int = 14,
                     exog_forecast: pd.DataFrame = None) -> pd.DataFrame:
    """
    SARIMAX forecast with exogenous variables.

    Args:
        df_pandas: Training data with DatetimeIndex
        target: Target column name
        exog_features: List of exogenous features (e.g., ['temp_c', 'humidity_pct'])
        order: (p, d, q) tuple - if None, uses auto_arima to find best
        seasonal_order: (P, D, Q, s) tuple for seasonality
        horizon: Forecast days ahead
        exog_forecast: Projected exogenous variables for forecast period

    Returns:
        DataFrame with forecast and confidence intervals

    Example:
        SARIMAX(1,1,1)(0,0,0,0) with temp_c, humidity_pct:
        - Uses past prices AND weather to predict future prices
        - Needs projected weather for forecast period
    """
    # Extract target and exogenous variables
    y = df_pandas[target]
    X = df_pandas[exog_features] if exog_features else None
    last_date = df_pandas.index[-1]

    # Auto-fit if order not specified
    if order is None:
        print(f"Auto-fitting SARIMAX order...")
        auto_model = auto_arima(
            y, X=X,
            seasonal=False,  # Disable for now (can enable with s=365 for annual)
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=3, max_q=3, max_d=2,
            trace=False
        )
        order = auto_model.order
        print(f"  Best order: {order}")

    # Fit SARIMAX model
    model = SARIMAX(y, exog=X, order=order, seasonal_order=seasonal_order)
    fitted = model.fit(disp=False)

    # Generate forecast
    # IMPORTANT: SARIMAX needs exogenous variables for forecast period
    forecast_result = fitted.forecast(steps=horizon, exog=exog_forecast[exog_features] if exog_forecast is not None and exog_features else None)
    forecast_obj = fitted.get_forecast(steps=horizon, exog=exog_forecast[exog_features] if exog_forecast is not None and exog_features else None)
    forecast_ci = forecast_obj.conf_int(alpha=0.2)  # 80% CI
    forecast_ci_95 = forecast_obj.conf_int(alpha=0.05)  # 95% CI

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


def sarimax_train(df_pandas: pd.DataFrame, target: str = 'close',
                  exog_features: list = None, order: tuple = None,
                  seasonal_order: tuple = (0, 0, 0, 0),
                  covariate_projection_method: str = 'persist') -> dict:
    """
    Train SARIMAX model.

    Args:
        df_pandas: Training data with DatetimeIndex
        target: Target column name
        exog_features: List of exogenous features
        order: (p, d, q) - if None, auto-fits
        seasonal_order: (P, D, Q, s)
        covariate_projection_method: Method to project exogenous variables

    Returns:
        Dict containing fitted SARIMAX model and metadata
    """
    # Extract target and exogenous variables
    y = df_pandas[target]
    X = df_pandas[exog_features] if exog_features else None

    # Auto-fit if order not specified
    fitted_order = order
    if order is None:
        auto_model = auto_arima(
            y, X=X,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=3, max_q=3, max_d=2,
            trace=False
        )
        fitted_order = auto_model.order

    # Fit SARIMAX model
    model = SARIMAX(y, exog=X, order=fitted_order, seasonal_order=seasonal_order)
    fitted = model.fit(disp=False)

    return {
        'fitted_model': fitted,  # statsmodels SARIMAXResults object
        'last_date': df_pandas.index[-1],
        'target': target,
        'order': fitted_order,
        'seasonal_order': seasonal_order,
        'exog_features': exog_features,
        'covariate_projection_method': covariate_projection_method,
        'aic': float(fitted.aic),
        'bic': float(fitted.bic),
        'auto_fitted': (order is None),
        'model_type': 'sarimax'
    }


def sarimax_predict(fitted_model_dict: dict, horizon: int = 14,
                    exog_data_for_projection: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generate forecast using fitted SARIMAX model.

    Args:
        fitted_model_dict: Dict returned by sarimax_train()
        horizon: Forecast days ahead
        exog_data_for_projection: Historical data needed for projecting exogenous variables

    Returns:
        DataFrame with forecast and confidence intervals
    """
    fitted = fitted_model_dict['fitted_model']
    last_date = fitted_model_dict['last_date']
    exog_features = fitted_model_dict['exog_features']
    covariate_projection_method = fitted_model_dict['covariate_projection_method']

    # Project exogenous variables if needed
    exog_forecast = None
    if exog_features and exog_data_for_projection is not None:
        from ground_truth.features.covariate_projection import get_projection_function
        projection_fn = get_projection_function(covariate_projection_method)
        exog_forecast = projection_fn(exog_data_for_projection, exog_features, horizon)

    # Generate forecast
    forecast_result = fitted.forecast(
        steps=horizon,
        exog=exog_forecast[exog_features] if exog_forecast is not None and exog_features else None
    )
    forecast_obj = fitted.get_forecast(
        steps=horizon,
        exog=exog_forecast[exog_features] if exog_forecast is not None and exog_features else None
    )
    forecast_ci = forecast_obj.conf_int(alpha=0.2)  # 80% CI
    forecast_ci_95 = forecast_obj.conf_int(alpha=0.05)  # 95% CI

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


def sarimax_forecast_with_metadata(df_pandas: pd.DataFrame, commodity: str,
                                    target: str = 'close',
                                    exog_features: list = None,
                                    covariate_projection_method: str = 'persist',
                                    order: tuple = None,
                                    seasonal_order: tuple = (0, 0, 0, 0),
                                    horizon: int = 14,
                                    cutoff_date: str = None,
                                    fitted_model: dict = None) -> dict:
    """
    SARIMAX forecast with full metadata for model registry.

    Can either train+predict (if fitted_model is None) or just predict
    (if fitted_model is provided).

    Args:
        df_pandas: Training data (used for training OR exog projection)
        commodity: 'Coffee' or 'Sugar'
        target: Target column
        exog_features: List of exogenous features
        covariate_projection_method: 'persist', 'seasonal', 'linear', or 'weather_api'
        order: (p, d, q) - if None, auto-fits
        seasonal_order: (P, D, Q, s)
        horizon: Forecast days
        cutoff_date: Optional - for backtesting
        fitted_model: Optional - pre-trained model from sarimax_train()

    Returns:
        Dict with forecast, model diagnostics, and metadata
    """
    # If no fitted model provided, train one
    if fitted_model is None:
        # Filter by cutoff if provided
        if cutoff_date:
            df_pandas = df_pandas[df_pandas.index <= cutoff_date]

        # Train model
        fitted_model = sarimax_train(
            df_pandas, target, exog_features, order, seasonal_order,
            covariate_projection_method
        )

    # Generate forecast using fitted model
    # Pass df_pandas for exog projection (needed even during inference)
    forecast_df = sarimax_predict(fitted_model, horizon, df_pandas)

    # Extract model diagnostics
    fitted_order = fitted_model['order']
    seasonal_order_val = fitted_model['seasonal_order']
    p, d, q = fitted_order
    P, D, Q, s = seasonal_order_val
    aic = fitted_model['aic']
    bic = fitted_model['bic']

    # Build model name
    model_name = f'SARIMAX({p},{d},{q})({P},{D},{Q},{s})'
    if exog_features:
        model_name += f'+{len(exog_features)}exog'

    # Add metadata
    return {
        'forecast_df': forecast_df,
        'model_name': model_name,
        'commodity': commodity,
        'parameters': {
            'method': 'sarimax',
            'target': target,
            'order': fitted_order,
            'seasonal_order': seasonal_order_val,
            'p': p, 'd': d, 'q': q,
            'P': P, 'D': D, 'Q': Q, 's': s,
            'exog_features': exog_features,
            'covariate_projection': covariate_projection_method,
            'horizon': horizon,
            'aic': float(aic),
            'bic': float(bic),
            'auto_fitted': fitted_model['auto_fitted']
        },
        'fitted_model': fitted_model,  # Return fitted model for reuse!
        'training_end': fitted_model['last_date'],
        'forecast_start': forecast_df['date'].iloc[0],
        'forecast_end': forecast_df['date'].iloc[-1]
    }

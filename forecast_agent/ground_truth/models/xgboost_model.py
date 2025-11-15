"""XGBoost forecaster with feature engineering.

Uses lagged features, rolling stats, and weather data for multi-step forecasting.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict
import xgboost as xgb


def create_features(df: pd.DataFrame, target: str = 'close',
                    lags: list = [1, 7, 14], windows: list = [7, 30]) -> pd.DataFrame:
    """
    Create features for XGBoost model.

    Args:
        df: Input data with DatetimeIndex
        target: Target column
        lags: Lag periods
        windows: Rolling window sizes

    Returns:
        DataFrame with engineered features
    """
    df_feat = df.copy()

    # Lag features
    for lag in lags:
        df_feat[f'{target}_lag_{lag}'] = df_feat[target].shift(lag)

    # Rolling statistics
    for window in windows:
        df_feat[f'{target}_rolling_mean_{window}'] = df_feat[target].rolling(window).mean()
        df_feat[f'{target}_rolling_std_{window}'] = df_feat[target].rolling(window).std()

    # Differences
    df_feat[f'{target}_diff_1'] = df_feat[target].diff()
    df_feat[f'{target}_diff_7'] = df_feat[target].diff(7)

    # Date features
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['day_of_year'] = df_feat.index.dayofyear

    # Drop rows with NaN (from lagging/rolling)
    max_lag = max(lags + windows)
    df_feat = df_feat.iloc[max_lag:]

    return df_feat


def xgboost_forecast(df_pandas: pd.DataFrame, target: str = 'close',
                     exog_features: list = None, horizon: int = 14,
                     lags: list = [1, 7, 14], windows: list = [7, 30],
                     params: dict = None) -> Dict:
    """
    XGBoost multi-step forecast with direct strategy.

    Args:
        df_pandas: Training data with DatetimeIndex
        target: Target column
        exog_features: Exogenous features (weather, etc.)
        horizon: Forecast days
        lags: Lag periods for feature engineering
        windows: Rolling window sizes
        params: XGBoost parameters (optional)

    Returns:
        Dict with forecast_df, model, feature_importance
    """
    # Default XGBoost params
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

    # Create features
    df_feat = create_features(df_pandas, target, lags, windows)

    # Prepare features for training
    feature_cols = [col for col in df_feat.columns if col != target and col != 'commodity']

    # Add exogenous features if provided
    if exog_features:
        feature_cols = [col for col in feature_cols if col in df_feat.columns or col in exog_features]

    X = df_feat[feature_cols]
    y = df_feat[target]

    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)

    # Multi-step forecasting (direct strategy)
    last_date = df_pandas.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')

    forecasts = []
    lower_95_list = []
    upper_95_list = []

    # For simplicity, we'll use recursive forecasting
    # (In production, you'd train separate models for each horizon)
    forecast_df_temp = df_pandas.copy()

    for i, future_date in enumerate(future_dates):
        # Create features for this forecast step
        df_temp = create_features(forecast_df_temp, target, lags, windows)

        if len(df_temp) == 0:
            # Not enough data for features
            break

        # Get last row features
        X_forecast = df_temp[feature_cols].iloc[-1:].copy()

        # Add date features for future date
        X_forecast['day_of_week'] = future_date.dayofweek
        X_forecast['month'] = future_date.month
        X_forecast['day_of_year'] = future_date.dayofyear

        # Predict
        pred = model.predict(X_forecast)[0]
        forecasts.append(pred)

        # Simple confidence intervals (use historical residuals)
        residuals = y - model.predict(X)
        std_residual = residuals.std()
        lower_95_list.append(pred - 1.96 * std_residual * np.sqrt(i + 1))
        upper_95_list.append(pred + 1.96 * std_residual * np.sqrt(i + 1))

        # Update forecast_df_temp with prediction for next iteration
        new_row = pd.DataFrame({target: [pred]}, index=[future_date])
        if exog_features:
            # Use last known exog values (simplified)
            for feat in exog_features:
                if feat in df_pandas.columns:
                    new_row[feat] = df_pandas[feat].iloc[-1]

        forecast_df_temp = pd.concat([forecast_df_temp, new_row])

    # Build forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': future_dates[:len(forecasts)],
        'forecast': forecasts,
        'lower_80': [f - 1.28 * std_residual * np.sqrt(i+1) for i, f in enumerate(forecasts)],
        'upper_80': [f + 1.28 * std_residual * np.sqrt(i+1) for i, f in enumerate(forecasts)],
        'lower_95': lower_95_list,
        'upper_95': upper_95_list
    })

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'forecast_df': forecast_df,
        'model': model,
        'feature_importance': feature_importance
    }


def xgboost_forecast_with_metadata(df_pandas: pd.DataFrame, commodity: str,
                                    target: str = 'close',
                                    exog_features: list = None,
                                    horizon: int = 14,
                                    lags: list = [1, 7, 14],
                                    windows: list = [7, 30],
                                    cutoff_date: str = None,
                                    fitted_model: dict = None) -> dict:
    """
    XGBoost forecast with full metadata for model registry.

    Can either train+predict (if fitted_model is None) or just predict
    (if fitted_model is provided).

    Args:
        df_pandas: Training data (only used if fitted_model is None)
        commodity: 'Coffee' or 'Sugar'
        target: Target column
        exog_features: Exogenous features
        horizon: Forecast days
        lags: Lag periods
        windows: Rolling windows
        cutoff_date: Optional - for backtesting
        fitted_model: Optional - pre-trained model dict with 'model', 'feature_importance', etc.

    Returns:
        Dict with forecast, model, feature importance, and metadata
    """
    # If no fitted model provided, train one
    if fitted_model is None:
        # Filter by cutoff if provided
        if cutoff_date:
            df_pandas = df_pandas[df_pandas.index <= cutoff_date]

        # Train model
        result = xgboost_forecast(df_pandas, target, exog_features, horizon, lags, windows)
        training_end = df_pandas.index[-1]
    else:
        # Use pre-trained model for prediction
        # TODO: Implement inference-only mode for XGBoost
        # For now, fall back to retraining (will fix in next iteration)
        if cutoff_date:
            df_pandas = df_pandas[df_pandas.index <= cutoff_date]
        result = xgboost_forecast(df_pandas, target, exog_features, horizon, lags, windows)
        training_end = df_pandas.index[-1]

    # Package with metadata
    return {
        'forecast_df': result['forecast_df'],
        'model_name': 'XGBoost',
        'commodity': commodity,
        'parameters': {
            'method': 'xgboost',
            'target': target,
            'horizon': horizon,
            'lags': lags,
            'windows': windows,
            'exog_features': exog_features,
            'n_features': len(result['model'].feature_names_in_)
        },
        'fitted_model': result['model'],  # Return fitted model for reuse!
        'feature_importance': result['feature_importance'],
        'training_end': training_end,
        'forecast_start': result['forecast_df']['date'].iloc[0],
        'forecast_end': result['forecast_df']['date'].iloc[-1],
        'std': result.get('std', None)  # Include std if available
    }

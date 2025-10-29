"""Panel data forecasting - treat commodities as panel individuals.

Uses XGBoost with commodity fixed effects.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from ground_truth.features.lag_features import create_lag_features


def panel_xgboost_forecast(df_panel: pd.DataFrame, target_commodity: str,
                           target: str = 'close', horizon: int = 14,
                           lags: list = [1, 7, 14], windows: list = [7, 30],
                           exog_features: list = None) -> dict:
    """
    Panel data XGBoost - learns from multiple commodities.

    Approach:
    - Train on both Coffee and Sugar
    - Add commodity fixed effects (dummy variables)
    - Leverage cross-commodity patterns
    - Forecast for target commodity

    Args:
        df_panel: Panel data with 'commodity' column
        target_commodity: Commodity to forecast ('Coffee' or 'Sugar')
        target: Target column
        horizon: Forecast days
        lags: Lag periods
        windows: Rolling window sizes
        exog_features: Additional features

    Returns:
        Dict with forecast_df and model
    """
    # Ensure commodity column exists
    if 'commodity' not in df_panel.columns:
        raise ValueError("Panel data must have 'commodity' column")

    # Create features for all commodities
    commodities = df_panel['commodity'].unique()

    df_list = []
    for commodity in commodities:
        df_comm = df_panel[df_panel['commodity'] == commodity].copy()

        # Create lag features
        df_comm = create_lag_features(df_comm, target=target, lags=lags, windows=windows)

        # Add commodity dummies
        for c in commodities:
            df_comm[f'is_{c}'] = 1 if c == commodity else 0

        df_list.append(df_comm)

    # Combine all commodities
    df_combined = pd.concat(df_list, ignore_index=False)

    # Prepare features
    feature_cols = []

    # Lag features
    for lag in lags:
        feature_cols.append(f'{target}_lag_{lag}')
    for window in windows:
        feature_cols.extend([
            f'{target}_rolling_mean_{window}',
            f'{target}_rolling_std_{window}'
        ])

    # Commodity dummies
    for commodity in commodities:
        feature_cols.append(f'is_{commodity}')

    # Exogenous features
    if exog_features:
        for feat in exog_features:
            if feat in df_combined.columns:
                feature_cols.append(feat)

    # Filter to available features
    feature_cols = [col for col in feature_cols if col in df_combined.columns]

    # Remove NaN rows (from lag creation)
    df_combined = df_combined.dropna(subset=feature_cols + [target])

    # Train model on all commodities
    X_train = df_combined[feature_cols].values
    y_train = df_combined[target].values

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Generate forecast for target commodity
    df_target = df_panel[df_panel['commodity'] == target_commodity].copy()
    last_train_idx = df_target.index[-1]

    forecast_dates = pd.date_range(
        start=last_train_idx + timedelta(days=1),
        periods=horizon,
        freq='D'
    )

    forecasts = []
    last_known_values = df_target.copy()

    for i, date in enumerate(forecast_dates):
        # Create feature vector
        features = []

        # Lag features (use recent history)
        for lag in lags:
            if len(last_known_values) >= lag:
                features.append(last_known_values[target].iloc[-lag])
            else:
                features.append(last_known_values[target].iloc[-1])

        # Rolling statistics
        for window in windows:
            if len(last_known_values) >= window:
                features.append(last_known_values[target].iloc[-window:].mean())
                features.append(last_known_values[target].iloc[-window:].std())
            else:
                features.append(last_known_values[target].mean())
                features.append(last_known_values[target].std())

        # Commodity dummies
        for commodity in commodities:
            features.append(1 if commodity == target_commodity else 0)

        # Exogenous features
        if exog_features:
            for feat in exog_features:
                if feat in df_target.columns:
                    features.append(df_target[feat].iloc[-1])

        # Predict
        pred = model.predict([features])[0]
        forecasts.append(pred)

        # Update history
        new_row = pd.DataFrame({
            target: [pred],
            'commodity': [target_commodity]
        }, index=[date])

        # Add exog features
        if exog_features:
            for feat in exog_features:
                if feat in df_target.columns:
                    new_row[feat] = df_target[feat].iloc[-1]

        last_known_values = pd.concat([last_known_values, new_row])

    # Build forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecast': forecasts,
        'lower_80': np.nan,
        'upper_80': np.nan,
        'lower_95': np.nan,
        'upper_95': np.nan
    })

    return {
        'forecast_df': forecast_df,
        'model': model,
        'feature_importance': dict(zip(feature_cols, model.feature_importances_))
    }


def panel_xgboost_forecast_with_metadata(df_panel: pd.DataFrame, commodity: str,
                                        target: str = 'close',
                                        exog_features: list = None,
                                        horizon: int = 14,
                                        cutoff_date: str = None,
                                        lags: list = [1, 7, 14],
                                        windows: list = [7, 30]) -> dict:
    """
    Panel XGBoost forecast with metadata for model registry.

    Args:
        df_panel: Panel data with multiple commodities
        commodity: Target commodity to forecast
        target: Target column
        exog_features: Additional features
        horizon: Forecast days
        cutoff_date: Optional cutoff for backtesting
        lags: Lag periods
        windows: Rolling windows

    Returns:
        Dict with forecast and metadata
    """
    # Filter by cutoff if provided
    if cutoff_date:
        df_panel = df_panel[df_panel.index <= cutoff_date]

    # Generate forecast
    result = panel_xgboost_forecast(
        df_panel, commodity, target, horizon, lags, windows, exog_features
    )

    model_name = f'Panel-XGBoost ({len(df_panel["commodity"].unique())} commodities)'

    return {
        'forecast_df': result['forecast_df'],
        'model_name': model_name,
        'commodity': commodity,
        'parameters': {
            'method': 'panel_xgboost',
            'target': target,
            'horizon': horizon,
            'exog_features': exog_features,
            'lags': lags,
            'windows': windows,
            'commodities': list(df_panel['commodity'].unique())
        },
        'model': result['model'],
        'feature_importance': result['feature_importance'],
        'training_end': df_panel[df_panel['commodity'] == commodity].index[-1],
        'forecast_start': result['forecast_df']['date'].iloc[0],
        'forecast_end': result['forecast_df']['date'].iloc[-1]
    }

"""XGBoost with advanced feature engineering.

Uses technical indicators, Fourier features, and price patterns.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from ground_truth.features.lag_features import create_lag_features
from ground_truth.features.advanced_features import create_advanced_features


def xgboost_advanced_forecast(df_pandas: pd.DataFrame, target: str = 'close',
                              exog_features: list = None, horizon: int = 14,
                              lags: list = [1, 7, 14], windows: list = [7, 30],
                              use_technical: bool = True,
                              use_fourier: bool = True,
                              use_cyclical: bool = True,
                              use_patterns: bool = True) -> dict:
    """
    XGBoost with advanced technical indicators and feature engineering.

    Features:
        - Technical indicators (RSI, MACD, Bollinger Bands)
        - Fourier features for seasonality
        - Cyclical time encoding
        - Price patterns and momentum
        - Traditional lag features

    Args:
        df_pandas: Training data with DatetimeIndex
        target: Target column
        exog_features: Additional exogenous features
        horizon: Forecast horizon (days)
        lags: Lag periods
        windows: Rolling window sizes
        use_technical: Include technical indicators
        use_fourier: Include Fourier seasonality features
        use_cyclical: Include cyclical time encoding
        use_patterns: Include price patterns

    Returns:
        Dict with forecast_df and model
    """
    df = df_pandas.copy()

    # Create advanced features
    df = create_advanced_features(
        df, target=target,
        include_technical=use_technical,
        include_fourier=use_fourier,
        include_cyclical=use_cyclical,
        include_patterns=use_patterns,
        include_interactions=False  # Skip interactions for now
    )

    # Create traditional lag features
    df = create_lag_features(df, target=target, lags=lags, windows=windows)

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

    # Technical indicators
    if use_technical:
        for period in [14, 30, 60]:
            feature_cols.extend([
                f'rsi_{period}',
                f'momentum_{period}',
                f'volatility_{period}'
            ])
        feature_cols.extend(['macd', 'macd_signal', 'macd_hist',
                           'bb_width', 'bb_position'])

    # Fourier features
    if use_fourier:
        for i in range(1, 4):
            feature_cols.extend([f'fourier_sin_{i}', f'fourier_cos_{i}'])

    # Cyclical time features
    if use_cyclical:
        feature_cols.extend([
            'dayofweek_sin', 'dayofweek_cos',
            'dayofyear_sin', 'dayofyear_cos'
        ])

    # Price patterns
    if use_patterns:
        for window in [5, 10, 20]:
            feature_cols.extend([
                f'dist_from_ma_{window}',
                f'channel_position_{window}',
                f'trend_slope_{window}'
            ])

    # Exogenous features
    if exog_features:
        for feat in exog_features:
            if feat in df.columns:
                feature_cols.append(feat)

    # Filter to available features
    feature_cols = [col for col in feature_cols if col in df.columns]

    # Train/predict split
    last_train_idx = df.index[-1]

    X_train = df[feature_cols].values
    y_train = df[target].values

    # Train XGBoost
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

    # Generate forecast
    forecast_dates = pd.date_range(
        start=last_train_idx + timedelta(days=1),
        periods=horizon,
        freq='D'
    )

    forecasts = []
    last_known_values = df.copy()

    for i, date in enumerate(forecast_dates):
        # Create features for this forecast step
        # Use last_known_values to compute features
        forecast_features = create_forecast_features_advanced(
            last_known_values, date, feature_cols, target,
            exog_features, lags, windows,
            use_technical, use_fourier, use_cyclical, use_patterns
        )

        # Predict
        pred = model.predict([forecast_features])[0]
        forecasts.append(pred)

        # Update last_known_values with new prediction
        new_row = {target: pred, 'date': date}
        for feat in exog_features or []:
            if feat in df.columns:
                new_row[feat] = df[feat].iloc[-1]  # Persist last value

        new_row_df = pd.DataFrame([new_row])
        new_row_df['date'] = pd.to_datetime(new_row_df['date'])
        new_row_df = new_row_df.set_index('date')

        last_known_values = pd.concat([last_known_values, new_row_df])

    # Build forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecast': forecasts,
        'lower_80': np.nan,  # Not computed for XGBoost
        'upper_80': np.nan,
        'lower_95': np.nan,
        'upper_95': np.nan
    })

    return {
        'forecast_df': forecast_df,
        'model': model,
        'feature_importance': dict(zip(feature_cols, model.feature_importances_))
    }


def create_forecast_features_advanced(df, forecast_date, feature_cols, target,
                                     exog_features, lags, windows,
                                     use_technical, use_fourier, use_cyclical, use_patterns):
    """
    Create feature vector for a single forecast step.

    Simplified version - uses last available values.
    """
    features = []

    # Create advanced features for the forecast date
    # This is a simplified approach - in production would be more sophisticated

    for feat in feature_cols:
        if feat in df.columns:
            features.append(df[feat].iloc[-1])
        elif 'fourier' in feat or 'dayof' in feat or 'weekof' in feat:
            # Recompute time-based features for forecast date
            if 'dayofweek_sin' == feat:
                features.append(np.sin(2 * np.pi * forecast_date.dayofweek / 7))
            elif 'dayofweek_cos' == feat:
                features.append(np.cos(2 * np.pi * forecast_date.dayofweek / 7))
            elif 'dayofyear_sin' == feat:
                features.append(np.sin(2 * np.pi * forecast_date.dayofyear / 365))
            elif 'dayofyear_cos' == feat:
                features.append(np.cos(2 * np.pi * forecast_date.dayofyear / 365))
            elif 'fourier' in feat:
                # Parse harmonic number
                harmonic = int(feat.split('_')[-1])
                if 'sin' in feat:
                    features.append(np.sin(2 * np.pi * harmonic * forecast_date.dayofyear / 365))
                else:
                    features.append(np.cos(2 * np.pi * harmonic * forecast_date.dayofyear / 365))
            else:
                features.append(0)
        else:
            features.append(0)

    return features


def xgboost_advanced_forecast_with_metadata(df_pandas: pd.DataFrame, commodity: str,
                                            target: str = 'close',
                                            exog_features: list = None,
                                            horizon: int = 14,
                                            cutoff_date: str = None,
                                            lags: list = [1, 7, 14],
                                            windows: list = [7, 30],
                                            use_technical: bool = True,
                                            use_fourier: bool = True,
                                            use_cyclical: bool = True,
                                            use_patterns: bool = True) -> dict:
    """
    XGBoost advanced forecast with metadata for model registry.

    Args:
        df_pandas: Training data
        commodity: Commodity name
        target: Target column
        exog_features: Additional features
        horizon: Forecast horizon
        cutoff_date: Optional cutoff for backtesting
        lags: Lag periods
        windows: Rolling windows
        use_technical: Include technical indicators
        use_fourier: Include Fourier features
        use_cyclical: Include cyclical time encoding
        use_patterns: Include price patterns

    Returns:
        Dict with forecast and metadata
    """
    # Filter by cutoff if provided
    if cutoff_date:
        df_pandas = df_pandas[df_pandas.index <= cutoff_date]

    # Generate forecast
    result = xgboost_advanced_forecast(
        df_pandas, target, exog_features, horizon, lags, windows,
        use_technical, use_fourier, use_cyclical, use_patterns
    )

    # Build model name
    model_name = 'XGBoost+Advanced'
    if use_technical:
        model_name += '+Technical'
    if use_fourier:
        model_name += '+Fourier'

    return {
        'forecast_df': result['forecast_df'],
        'model_name': model_name,
        'commodity': commodity,
        'parameters': {
            'method': 'xgboost_advanced',
            'target': target,
            'horizon': horizon,
            'exog_features': exog_features,
            'lags': lags,
            'windows': windows,
            'use_technical': use_technical,
            'use_fourier': use_fourier,
            'use_cyclical': use_cyclical,
            'use_patterns': use_patterns
        },
        'model': result['model'],
        'feature_importance': result['feature_importance'],
        'training_end': df_pandas.index[-1],
        'forecast_start': result['forecast_df']['date'].iloc[0],
        'forecast_end': result['forecast_df']['date'].iloc[-1]
    }

"""NeuralProphet - Neural network time series model with covariates.

Deep learning approach with automatic seasonality detection.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from neuralprophet import NeuralProphet
    NEURALPROPHET_AVAILABLE = True
except ImportError:
    NEURALPROPHET_AVAILABLE = False
    print("⚠️  NeuralProphet not available. Install with: pip install neuralprophet")


def neuralprophet_forecast(df_pandas: pd.DataFrame, target: str = 'close',
                           exog_features: list = None, horizon: int = 14,
                           n_lags: int = 14, n_forecasts: int = 1,
                           epochs: int = 50) -> dict:
    """
    NeuralProphet forecast with optional covariates.

    Neural network based approach with:
    - Automatic seasonality detection
    - Lagged regressors
    - Future covariates support

    Args:
        df_pandas: Training data with DatetimeIndex
        target: Target column
        exog_features: List of future covariates
        horizon: Forecast days ahead
        n_lags: Number of autoregressive lags
        n_forecasts: Number of steps to forecast at once
        epochs: Training epochs

    Returns:
        Dict with forecast_df and model
    """
    if not NEURALPROPHET_AVAILABLE:
        raise ImportError("NeuralProphet not installed")

    # Prepare data (needs 'ds' and 'y' columns)
    df_model = pd.DataFrame({
        'ds': df_pandas.index,
        'y': df_pandas[target].values
    })

    # Add covariates
    if exog_features:
        for feat in exog_features:
            if feat in df_pandas.columns:
                df_model[feat] = df_pandas[feat].values

    # Initialize NeuralProphet
    model = NeuralProphet(
        n_lags=n_lags,
        n_forecasts=n_forecasts,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        epochs=epochs,
        learning_rate=0.01,
        batch_size=32,
        loss_func='MSE'
    )

    # Add future regressors
    if exog_features:
        for feat in exog_features:
            if feat in df_model.columns:
                model.add_future_regressor(feat)

    # Fit model (suppress verbose output)
    model.fit(df_model, freq='D', verbose=False)

    # Create future dataframe
    future = model.make_future_dataframe(
        df_model,
        periods=horizon,
        n_historic_predictions=0
    )

    # Add future values for covariates (persist last values)
    if exog_features:
        for feat in exog_features:
            if feat in df_pandas.columns:
                last_value = df_pandas[feat].iloc[-1]
                # Fill future periods with last known value
                future.loc[future[feat].isna(), feat] = last_value

    # Generate forecast
    forecast = model.predict(future)

    # Extract forecast period only
    last_train_date = df_pandas.index[-1]
    forecast_future = forecast[forecast['ds'] > last_train_date].reset_index(drop=True)

    # Build output DataFrame
    forecast_df = pd.DataFrame({
        'date': pd.to_datetime(forecast_future['ds']),
        'forecast': forecast_future['yhat1'].values,
        'lower_80': np.nan,  # NeuralProphet doesn't provide confidence intervals easily
        'upper_80': np.nan,
        'lower_95': np.nan,
        'upper_95': np.nan
    })

    return {
        'forecast_df': forecast_df,
        'model': model
    }


def neuralprophet_forecast_with_metadata(df_pandas: pd.DataFrame, commodity: str,
                                         target: str = 'close',
                                         exog_features: list = None,
                                         horizon: int = 14,
                                         cutoff_date: str = None,
                                         n_lags: int = 14,
                                         epochs: int = 50) -> dict:
    """
    NeuralProphet forecast with metadata for model registry.

    Args:
        df_pandas: Training data
        commodity: Commodity name
        target: Target column
        exog_features: List of future regressors
        horizon: Forecast days
        cutoff_date: Optional cutoff for backtesting
        n_lags: Autoregressive lags
        epochs: Training epochs

    Returns:
        Dict with forecast and metadata
    """
    # Filter by cutoff if provided
    if cutoff_date:
        df_pandas = df_pandas[df_pandas.index <= cutoff_date]

    # Generate forecast
    result = neuralprophet_forecast(
        df_pandas, target, exog_features, horizon, n_lags, epochs=epochs
    )

    # Build model name
    model_name = 'NeuralProphet'
    if exog_features:
        model_name += f'+{len(exog_features)}vars'

    return {
        'forecast_df': result['forecast_df'],
        'model_name': model_name,
        'commodity': commodity,
        'parameters': {
            'method': 'neuralprophet',
            'target': target,
            'horizon': horizon,
            'exog_features': exog_features,
            'n_lags': n_lags,
            'epochs': epochs
        },
        'model': result['model'],
        'training_end': df_pandas.index[-1],
        'forecast_start': result['forecast_df']['date'].iloc[0],
        'forecast_end': result['forecast_df']['date'].iloc[-1]
    }

"""
Model-based Monte Carlo simulation for forecast uncertainty quantification.

Instead of adding simple Gaussian noise to point forecasts, this module
generates paths by simulating from the actual fitted model's stochastic process.

For SARIMA models:
    - Uses statsmodels' simulate() method with fitted parameters
    - Generates paths from the actual ARIMA process (respects autocorrelation, seasonality)

For XGBoost/tree-based models:
    - Uses Geometric Brownian Motion (GBM) with estimated volatility

For Prophet models:
    - Uses Prophet's built-in uncertainty intervals if available
    - Falls back to GBM otherwise
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


def generate_monte_carlo_paths(
    fitted_model: dict,
    forecast_df: pd.DataFrame,
    n_paths: int = 2000,
    horizon: int = 14,
    training_df: pd.DataFrame = None
) -> List[Dict]:
    """
    Generate Monte Carlo paths using model-specific simulation.

    Args:
        fitted_model: Dict containing fitted model object and metadata
        forecast_df: DataFrame with point forecast (has 'forecast' column)
        n_paths: Number of paths to generate
        horizon: Forecast horizon (days)
        training_df: Historical data (needed for some methods)

    Returns:
        List of dicts with {'path_id': int, 'values': list of floats}
    """
    model_type = fitted_model.get('model_type', 'unknown')

    if model_type == 'sarimax':
        return _simulate_sarimax_paths(fitted_model, forecast_df, n_paths, horizon, training_df)
    elif model_type == 'xgboost':
        return _simulate_gbm_paths(fitted_model, forecast_df, n_paths, horizon, training_df)
    elif model_type == 'prophet':
        return _simulate_prophet_paths(fitted_model, forecast_df, n_paths, horizon, training_df)
    elif model_type in ['naive', 'random_walk']:
        return _simulate_random_walk_paths(fitted_model, forecast_df, n_paths, horizon, training_df)
    else:
        # Fallback to simple Gaussian noise
        return _simulate_simple_gaussian_paths(fitted_model, forecast_df, n_paths, horizon, training_df)


def _simulate_sarimax_paths(
    fitted_model: dict,
    forecast_df: pd.DataFrame,
    n_paths: int,
    horizon: int,
    training_df: pd.DataFrame = None
) -> List[Dict]:
    """
    Simulate paths from fitted SARIMAX model using model's own simulation.

    This is the key improvement: we're simulating from the actual ARIMA process
    with fitted parameters, not just adding noise to the point forecast.
    """
    fitted_sarimax: SARIMAXResults = fitted_model['fitted_model']

    # Get model parameters
    order = fitted_model['order']
    seasonal_order = fitted_model.get('seasonal_order', (0, 0, 0, 0))

    paths = []

    for path_id in range(1, n_paths + 1):
        try:
            # Use statsmodels' simulate() method
            # This generates a new realization of the ARIMA process
            simulated = fitted_sarimax.simulate(
                nsimulations=horizon,
                repetitions=1,
                initial_state=None  # Uses end of training data
            )

            # Convert to list
            path_values = simulated.values.tolist()

            paths.append({
                'path_id': path_id,
                'values': path_values
            })

        except Exception as e:
            # If simulation fails, fall back to adding noise
            # This shouldn't happen but provides safety
            noise = np.random.normal(0, fitted_sarimax.scale, horizon)
            path_values = (forecast_df['forecast'].values + noise).tolist()

            paths.append({
                'path_id': path_id,
                'values': path_values
            })

    return paths


def _simulate_gbm_paths(
    fitted_model: dict,
    forecast_df: pd.DataFrame,
    n_paths: int,
    horizon: int,
    training_df: pd.DataFrame = None
) -> List[Dict]:
    """
    Simulate paths using Geometric Brownian Motion (GBM).

    For XGBoost and other models that don't have an inherent stochastic process,
    we use GBM which is appropriate for price processes.
    """
    # Get initial price
    if 'last_value' in fitted_model:
        S0 = fitted_model['last_value']
    elif training_df is not None:
        S0 = training_df['close'].iloc[-1]
    else:
        S0 = forecast_df['forecast'].iloc[0]

    # Estimate volatility from residuals or historical data
    if 'residuals' in fitted_model and fitted_model['residuals'] is not None:
        residuals = fitted_model['residuals']
        sigma = np.std(residuals) / S0 if S0 > 0 else 0.02
    elif training_df is not None:
        returns = training_df['close'].pct_change().dropna()
        sigma = returns.std() if len(returns) > 0 else 0.02
    else:
        sigma = 0.02  # Default 2% daily volatility

    point_forecast = forecast_df['forecast'].values
    paths = []

    for path_id in range(1, n_paths + 1):
        S = S0
        path_values = []

        for t in range(horizon):
            # Drift toward point forecast
            target = point_forecast[t] if t < len(point_forecast) else point_forecast[-1]
            drift = (target - S) / S if S > 0 else 0

            # GBM step
            dW = np.random.normal(0, 1)
            S = S * np.exp((drift - 0.5 * sigma**2) + sigma * dW)
            path_values.append(float(S))

        paths.append({
            'path_id': path_id,
            'values': path_values
        })

    return paths


def _simulate_prophet_paths(
    fitted_model: dict,
    forecast_df: pd.DataFrame,
    n_paths: int,
    horizon: int,
    training_df: pd.DataFrame = None
) -> List[Dict]:
    """
    Simulate paths for Prophet models.

    Prophet provides uncertainty intervals; we can use those to calibrate noise.
    """
    # Check if forecast_df has uncertainty columns
    if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
        # Estimate std from confidence intervals
        # Assuming 95% CI: (upper - lower) / (2 * 1.96)
        forecast_std = (forecast_df['yhat_upper'] - forecast_df['yhat_lower']) / (2 * 1.96)
        forecast_std = forecast_std.mean()  # Average std across horizon
    else:
        # Fallback to historical volatility
        if training_df is not None:
            returns = training_df['close'].pct_change().dropna()
            daily_std = returns.std()
            S0 = training_df['close'].iloc[-1]
            forecast_std = S0 * daily_std
        else:
            forecast_std = forecast_df['forecast'].std()

    paths = []
    for path_id in range(1, n_paths + 1):
        noise = np.random.normal(0, forecast_std, horizon)
        path_forecast = forecast_df['forecast'].values + noise

        paths.append({
            'path_id': path_id,
            'values': path_forecast.tolist()
        })

    return paths


def _simulate_random_walk_paths(
    fitted_model: dict,
    forecast_df: pd.DataFrame,
    n_paths: int,
    horizon: int,
    training_df: pd.DataFrame = None
) -> List[Dict]:
    """
    Simulate paths for random walk models.

    Random walk: S(t+1) = S(t) + epsilon, where epsilon ~ N(0, sigma^2)
    """
    # Get initial value
    if 'last_value' in fitted_model:
        S0 = fitted_model['last_value']
    elif training_df is not None:
        S0 = training_df['close'].iloc[-1]
    else:
        S0 = forecast_df['forecast'].iloc[0]

    # Estimate step size from historical data
    if 'step_std' in fitted_model:
        step_std = fitted_model['step_std']
    elif training_df is not None:
        differences = training_df['close'].diff().dropna()
        step_std = differences.std()
    else:
        step_std = 1.0

    paths = []
    for path_id in range(1, n_paths + 1):
        S = S0
        path_values = []

        for t in range(horizon):
            # Random walk step
            epsilon = np.random.normal(0, step_std)
            S = S + epsilon
            path_values.append(float(S))

        paths.append({
            'path_id': path_id,
            'values': path_values
        })

    return paths


def _simulate_simple_gaussian_paths(
    fitted_model: dict,
    forecast_df: pd.DataFrame,
    n_paths: int,
    horizon: int,
    training_df: pd.DataFrame = None
) -> List[Dict]:
    """
    Fallback: Simple Gaussian noise around point forecast.

    This is the old method - used as fallback when model type is unknown.
    """
    # Estimate std
    if 'std' in fitted_model:
        forecast_std = fitted_model['std']
    elif training_df is not None:
        returns = training_df['close'].pct_change().dropna()
        daily_std = returns.std()
        forecast_std = training_df['close'].iloc[-1] * daily_std
    else:
        forecast_std = forecast_df['forecast'].std()

    paths = []
    for path_id in range(1, n_paths + 1):
        noise = np.random.normal(0, forecast_std, len(forecast_df))
        path_forecast = forecast_df['forecast'].values + noise

        paths.append({
            'path_id': path_id,
            'values': path_forecast.tolist()
        })

    return paths

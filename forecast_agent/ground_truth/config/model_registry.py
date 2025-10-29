"""Model registry - configuration for all forecast models.

Defines model configurations for training and evaluation.
"""

from ground_truth.models import naive, random_walk, arima, sarimax


# Model registry: List of all baseline models to train
BASELINE_MODELS = {
    'naive': {
        'name': 'Naive',
        'function': naive.naive_forecast_with_metadata,
        'params': {
            'target': 'close',
            'horizon': 14
        },
        'description': 'Last value persistence - simplest baseline'
    },

    'random_walk': {
        'name': 'RandomWalk',
        'function': random_walk.random_walk_forecast_with_metadata,
        'params': {
            'target': 'close',
            'horizon': 14,
            'lookback_days': 30
        },
        'description': 'Random walk with drift from last 30 days'
    },

    'arima_111': {
        'name': 'ARIMA(1,1,1)',
        'function': arima.arima_forecast_with_metadata,
        'params': {
            'target': 'close',
            'order': (1, 1, 1),
            'horizon': 14
        },
        'description': 'Classical ARIMA with fixed order'
    },

    'sarimax_auto': {
        'name': 'SARIMAX(auto)',
        'function': sarimax.sarimax_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': None,  # No exogenous variables
            'order': None,  # Auto-fit
            'horizon': 14
        },
        'description': 'Auto-fitted SARIMAX without weather covariates'
    },

    'sarimax_auto_weather': {
        'name': 'SARIMAX(auto)+Weather',
        'function': sarimax.sarimax_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm'],
            'covariate_projection_method': 'persist',  # Roll forward last values
            'order': None,  # Auto-fit
            'horizon': 14
        },
        'description': 'Auto-fitted SARIMAX with weather covariates (persisted)'
    },

    'sarimax_auto_weather_seasonal': {
        'name': 'SARIMAX(auto)+Weather(seasonal)',
        'function': sarimax.sarimax_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm'],
            'covariate_projection_method': 'seasonal',  # Historical averages
            'order': None,  # Auto-fit
            'horizon': 14
        },
        'description': 'Auto-fitted SARIMAX with seasonal weather projection'
    },
}


# Commodity-specific configurations
COMMODITY_CONFIGS = {
    'Coffee': {
        'features': ['close', 'temp_c', 'humidity_pct', 'precipitation_mm'],
        'aggregation_method': 'mean',
        'production_weights': None  # Could add: {'Colombia': 0.3, 'Brazil': 0.4, ...}
    },
    'Sugar': {
        'features': ['close', 'temp_c', 'humidity_pct', 'precipitation_mm'],
        'aggregation_method': 'mean',
        'production_weights': None
    }
}


# Evaluation configuration
EVALUATION_CONFIG = {
    'walk_forward': {
        'initial_training_days': 365 * 3,  # 3 years initial training
        'forecast_horizon': 14,  # 14-day forecasts
        'step_size': 7,  # Retrain weekly
        'n_windows': 104  # ~2 years of weekly backtests
    },
    'metrics': {
        'primary': 'mae',  # Main metric for model selection
        'secondary': ['rmse', 'mape', 'directional_accuracy']
    },
    'statistical_tests': {
        'alpha': 0.05,  # Significance level
        'use_diebold_mariano': True,  # Preferred for time series
        'use_binomial_direction': True,  # Test directional accuracy
        'regression_threshold_std': 2.0  # Alert if MAE > 2 std above mean
    }
}


def get_model_config(model_key: str) -> dict:
    """
    Get configuration for a specific model.

    Args:
        model_key: Model key from BASELINE_MODELS

    Returns:
        Model configuration dict

    Raises:
        KeyError: If model_key not found

    Example:
        config = get_model_config('sarimax_auto_weather')
        forecast_fn = config['function']
        params = config['params']
    """
    if model_key not in BASELINE_MODELS:
        available = list(BASELINE_MODELS.keys())
        raise KeyError(
            f"Model '{model_key}' not found in registry. "
            f"Available models: {available}"
        )

    return BASELINE_MODELS[model_key]


def get_commodity_config(commodity: str) -> dict:
    """
    Get configuration for a commodity.

    Args:
        commodity: 'Coffee' or 'Sugar'

    Returns:
        Commodity configuration dict

    Example:
        config = get_commodity_config('Coffee')
        features = config['features']
    """
    if commodity not in COMMODITY_CONFIGS:
        raise ValueError(
            f"Commodity '{commodity}' not found. "
            f"Available: {list(COMMODITY_CONFIGS.keys())}"
        )

    return COMMODITY_CONFIGS[commodity]


def list_models() -> list:
    """Return list of all available model keys."""
    return list(BASELINE_MODELS.keys())


def print_model_summary():
    """Print summary of all registered models."""
    print("="*60)
    print("MODEL REGISTRY")
    print("="*60)

    for key, config in BASELINE_MODELS.items():
        print(f"\n{key}:")
        print(f"  Name: {config['name']}")
        print(f"  Description: {config['description']}")
        print(f"  Parameters: {config['params']}")

    print(f"\nTotal models: {len(BASELINE_MODELS)}")

"""Model registry - configuration for all forecast models.

Defines model configurations for training and evaluation.
"""

from ground_truth.models import naive, random_walk, arima, sarimax, xgboost_model, prophet_model
try:
    from ground_truth.models import neuralprophet_model, panel_model, statsforecast_models
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    neuralprophet_model = None
    panel_model = None
    statsforecast_models = None

try:
    from ground_truth.models import tft_model
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False
    tft_model = None


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
        'name': 'ARIMA(auto)',
        'function': sarimax.sarimax_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': None,  # No exogenous variables = ARIMA not SARIMAX
            'order': None,  # Auto-fit (typically selects (0,1,0) = naive)
            'horizon': 14
        },
        'description': 'Auto-fitted ARIMA (no exogenous vars) - often reduces to naive'
    },

    'sarimax_auto_weather': {
        'name': 'SARIMAX+Weather',
        'function': sarimax.sarimax_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_mean_c', 'humidity_mean_pct', 'precipitation_mm'],
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
            'exog_features': ['temp_mean_c', 'humidity_mean_pct', 'precipitation_mm'],
            'covariate_projection_method': 'seasonal',  # Historical averages
            'order': None,  # Auto-fit
            'horizon': 14
        },
        'description': 'Auto-fitted SARIMAX with seasonal weather projection'
    },

    'xgboost': {
        'name': 'XGBoost',
        'function': xgboost_model.xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': None,
            'horizon': 14,
            'lags': [1, 7, 14],
            'windows': [7, 30]
        },
        'description': 'XGBoost with engineered features (lags, rolling stats)'
    },

    'xgboost_weather': {
        'name': 'XGBoost+Weather',
        'function': xgboost_model.xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm'],
            'horizon': 14,
            'lags': [1, 7, 14],
            'windows': [7, 30]
        },
        'description': 'XGBoost with weather covariates and engineered features'
    },

    'xgboost_deep_lags': {
        'name': 'XGBoost+DeepLags',
        'function': xgboost_model.xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': None,
            'horizon': 14,
            'lags': [1, 2, 3, 7, 14, 21, 30],  # More lags
            'windows': [7, 14, 30, 60]  # More windows
        },
        'description': 'XGBoost with deep lag structure (7 lags, 4 windows)'
    },

    'xgboost_weather_deep': {
        'name': 'XGBoost+Weather+Deep',
        'function': xgboost_model.xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm'],
            'horizon': 14,
            'lags': [1, 2, 3, 7, 14, 21, 30],
            'windows': [7, 14, 30, 60]
        },
        'description': 'XGBoost with weather + deep feature engineering'
    },

    'prophet': {
        'name': 'Prophet',
        'function': prophet_model.prophet_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': None,
            'horizon': 14
        },
        'description': 'Meta Prophet with automatic seasonality detection'
    },

    'prophet_weather': {
        'name': 'Prophet+Weather',
        'function': prophet_model.prophet_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm'],
            'horizon': 14
        },
        'description': 'Meta Prophet with weather regressors and seasonality'
    },

    'xgboost_sentiment': {
        'name': 'XGBoost+Sentiment',
        'function': xgboost_model.xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['sentiment_score', 'sentiment_ma_7', 'sentiment_ma_14',
                            'sentiment_momentum_7d', 'event_count'],
            'horizon': 14,
            'lags': [1, 7, 14],
            'windows': [7, 30]
        },
        'description': 'XGBoost with GDELT sentiment features only'
    },

    'xgboost_weather_sentiment': {
        'name': 'XGBoost+Weather+Sentiment',
        'function': xgboost_model.xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm',
                            'sentiment_score', 'sentiment_ma_7', 'sentiment_ma_14',
                            'sentiment_momentum_7d', 'event_count'],
            'horizon': 14,
            'lags': [1, 7, 14],
            'windows': [7, 30]
        },
        'description': 'XGBoost with weather + GDELT sentiment features'
    },

    'xgboost_full_features': {
        'name': 'XGBoost+Full',
        'function': xgboost_model.xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm',
                            'sentiment_score', 'sentiment_ma_7', 'sentiment_ma_14', 'sentiment_ma_30',
                            'sentiment_momentum_1d', 'sentiment_momentum_7d',
                            'event_count', 'positive_ratio', 'negative_ratio'],
            'horizon': 14,
            'lags': [1, 2, 3, 7, 14, 21, 30],
            'windows': [7, 14, 30, 60]
        },
        'description': 'XGBoost with all features: weather, sentiment, deep lags'
    },

    'xgboost_ultra_deep': {
        'name': 'XGBoost+UltraDeep',
        'function': xgboost_model.xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm'],
            'horizon': 14,
            'lags': [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30],  # 11 lags
            'windows': [3, 5, 7, 10, 14, 21, 30, 60, 90]  # 9 windows
        },
        'description': 'XGBoost with ultra-deep lag structure (11 lags, 9 windows)'
    },

    'xgboost_minimal': {
        'name': 'XGBoost+Minimal',
        'function': xgboost_model.xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c'],
            'horizon': 14,
            'lags': [1, 7],  # Just 2 lags
            'windows': [7]  # Just 1 window
        },
        'description': 'XGBoost with minimal features (temp only, 2 lags, 1 window)'
    },

    'xgboost_short_term': {
        'name': 'XGBoost+ShortTerm',
        'function': xgboost_model.xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm'],
            'horizon': 14,
            'lags': [1, 2, 3],  # Very short-term lags
            'windows': [3, 5, 7]  # Short windows
        },
        'description': 'XGBoost optimized for short-term patterns'
    },

    'xgboost_long_term': {
        'name': 'XGBoost+LongTerm',
        'function': xgboost_model.xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm'],
            'horizon': 14,
            'lags': [7, 14, 21, 30, 60, 90],  # Long-term lags
            'windows': [30, 60, 90]  # Long windows
        },
        'description': 'XGBoost optimized for long-term patterns'
    },

    # ========================================================================
    # MODELS WITH VIX AND EXCHANGE RATES (Previously Missing!)
    # ========================================================================

    'sarimax_market': {
        'name': 'SARIMAX+Market',
        'function': sarimax.sarimax_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['vix'],  # Market volatility indicator
            'covariate_projection_method': 'persist',
            'order': None,  # Auto-fit
            'horizon': 14
        },
        'description': 'SARIMAX with VIX (market volatility) - was missing!'
    },

    'sarimax_weather_market': {
        'name': 'SARIMAX+Weather+VIX',
        'function': sarimax.sarimax_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm', 'vix'],
            'covariate_projection_method': 'persist',
            'order': None,  # Auto-fit
            'horizon': 14
        },
        'description': 'SARIMAX with weather + VIX'
    },

    'sarimax_colombian_trader': {
        'name': 'SARIMAX+COP',
        'function': sarimax.sarimax_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm', 'cop_usd'],
            'covariate_projection_method': 'persist',
            'order': None,  # Auto-fit
            'horizon': 14
        },
        'description': 'SARIMAX with COP/USD (critical for Colombian trader use case)'
    },

    'xgboost_market': {
        'name': 'XGBoost+VIX',
        'function': xgboost_model.xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['vix'],
            'horizon': 14,
            'lags': [1, 7, 14],
            'windows': [7, 30]
        },
        'description': 'XGBoost with VIX only'
    },

    'xgboost_weather_market': {
        'name': 'XGBoost+Weather+VIX',
        'function': xgboost_model.xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm', 'vix'],
            'horizon': 14,
            'lags': [1, 7, 14],
            'windows': [7, 30]
        },
        'description': 'XGBoost with weather + VIX'
    },

    'xgboost_colombian_trader': {
        'name': 'XGBoost+COP',
        'function': xgboost_model.xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm', 'cop_usd', 'vix'],
            'horizon': 14,
            'lags': [1, 2, 3, 7, 14, 21, 30],
            'windows': [7, 14, 30, 60]
        },
        'description': 'XGBoost with weather + COP/USD + VIX (full Colombian trader model)'
    },
}

# Add advanced models if available
if ADVANCED_MODELS_AVAILABLE and neuralprophet_model is not None:
    BASELINE_MODELS['neuralprophet'] = {
        'name': 'NeuralProphet',
        'function': neuralprophet_model.neuralprophet_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': None,
            'horizon': 14,
            'n_lags': 14,
            'epochs': 100
        },
        'description': 'Neural network time series with deep learning'
    }

    BASELINE_MODELS['neuralprophet_weather'] = {
        'name': 'NeuralProphet+Weather',
        'function': neuralprophet_model.neuralprophet_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm'],
            'horizon': 14,
            'n_lags': 14,
            'epochs': 100
        },
        'description': 'NeuralProphet with weather covariates'
    }

    BASELINE_MODELS['neuralprophet_deep'] = {
        'name': 'NeuralProphet+Deep',
        'function': neuralprophet_model.neuralprophet_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'humidity_pct', 'precipitation_mm'],
            'horizon': 14,
            'n_lags': 30,  # More lags
            'epochs': 200  # More training
        },
        'description': 'NeuralProphet with deep lags and extended training'
    }

# Add statsforecast models if available
if ADVANCED_MODELS_AVAILABLE and statsforecast_models is not None:
    BASELINE_MODELS['auto_arima_stats'] = {
        'name': 'AutoARIMA (statsforecast)',
        'function': statsforecast_models.auto_arima_forecast_with_metadata,
        'params': {
            'target': 'close',
            'horizon': 14
        },
        'description': 'Automated ARIMA from statsforecast (fast)'
    }

    BASELINE_MODELS['auto_ets'] = {
        'name': 'AutoETS',
        'function': statsforecast_models.auto_ets_forecast_with_metadata,
        'params': {
            'target': 'close',
            'horizon': 14
        },
        'description': 'Exponential smoothing with automatic model selection'
    }

    BASELINE_MODELS['holt_winters'] = {
        'name': 'Holt-Winters',
        'function': statsforecast_models.holt_winters_forecast_with_metadata,
        'params': {
            'target': 'close',
            'horizon': 14
        },
        'description': 'Triple exponential smoothing (trend + seasonality)'
    }

    BASELINE_MODELS['auto_theta'] = {
        'name': 'AutoTheta',
        'function': statsforecast_models.auto_theta_forecast_with_metadata,
        'params': {
            'target': 'close',
            'horizon': 14
        },
        'description': 'Theta method - M3 competition winner'
    }

# Add Temporal Fusion Transformer if available
if TFT_AVAILABLE and tft_model is not None:
    BASELINE_MODELS['tft'] = {
        'name': 'TFT',
        'function': tft_model.tft_forecast_with_metadata,
        'params': {
            'target': 'close',
            'horizon': 14,
            'max_encoder_length': 60,
            'hidden_size': 32,
            'attention_head_size': 4,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'max_epochs': 30,
            'batch_size': 32
        },
        'description': 'Temporal Fusion Transformer - state-of-the-art deep learning for time series'
    }

    BASELINE_MODELS['tft_weather'] = {
        'name': 'TFT+Weather',
        'function': tft_model.tft_forecast_with_metadata,
        'params': {
            'target': 'close',
            'horizon': 14,
            'max_encoder_length': 60,
            'hidden_size': 32,
            'attention_head_size': 4,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'max_epochs': 30,
            'batch_size': 32,
            'exog_features': ['temp_max_c', 'temp_min_c', 'temp_mean_c',
                            'precipitation_mm', 'humidity_mean_pct']
        },
        'description': 'TFT with weather covariates and attention mechanisms'
    }

    BASELINE_MODELS['tft_full'] = {
        'name': 'TFT+Full',
        'function': tft_model.tft_forecast_with_metadata,
        'params': {
            'target': 'close',
            'horizon': 14,
            'max_encoder_length': 90,  # Longer lookback
            'hidden_size': 64,  # Larger model
            'attention_head_size': 8,  # More attention heads
            'dropout': 0.1,
            'learning_rate': 0.001,
            'max_epochs': 50,
            'batch_size': 16,  # Smaller batch for larger model
            'exog_features': ['temp_max_c', 'temp_min_c', 'temp_mean_c',
                            'precipitation_mm', 'humidity_mean_pct',
                            'vix', 'sentiment_score', 'event_count',
                            # Top coffee producer forex (7/8 available, BRL missing)
                            'cop_usd', 'vnd_usd', 'idr_usd', 'etb_usd',
                            'hnl_usd', 'ugx_usd', 'pen_usd']
        },
        'description': 'TFT with ALL features: weather, market, sentiment, forex (best performance)'
    }

    BASELINE_MODELS['tft_forex'] = {
        'name': 'TFT+Forex',
        'function': tft_model.tft_forecast_with_metadata,
        'params': {
            'target': 'close',
            'horizon': 14,
            'max_encoder_length': 60,
            'hidden_size': 32,
            'attention_head_size': 4,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'max_epochs': 30,
            'batch_size': 32,
            # Focus on top producer currencies
            'exog_features': ['cop_usd', 'vnd_usd', 'idr_usd']
        },
        'description': 'TFT with top 3 producer forex rates (Colombia, Vietnam, Indonesia)'
    }

    BASELINE_MODELS['tft_ensemble'] = {
        'name': 'TFT Ensemble (5 models)',
        'function': tft_model.tft_ensemble_forecast,
        'params': {
            'target': 'close',
            'horizon': 14,
            'max_encoder_length': 60,
            'hidden_size': 32,
            'attention_head_size': 4,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'max_epochs': 30,
            'batch_size': 32,
            'n_models': 5,
            'exog_features': ['temp_max_c', 'temp_min_c', 'temp_mean_c',
                            'precipitation_mm', 'humidity_mean_pct',
                            # Add key forex rates
                            'cop_usd', 'vnd_usd', 'idr_usd']
        },
        'description': 'Ensemble of 5 TFT models with weather + forex for robustness'
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

"""Model pipeline interface for standardized model training and inference.

Provides a clean interface that works with the cross-validator.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np


class ModelPipeline(ABC):
    """Base class for forecast model pipelines.
    
    Provides a standardized interface that works seamlessly with TimeSeriesCrossValidator.
    """
    
    def __init__(self, model_name: str, **params):
        """
        Initialize model pipeline.
        
        Args:
            model_name: Human-readable model name
            **params: Model-specific parameters
        """
        self.model_name = model_name
        self.params = params
        self.is_fitted = False
        self.fitted_model = None
    
    @abstractmethod
    def fit(self, train_df: pd.DataFrame, target_col: str = 'close', **kwargs) -> 'ModelPipeline':
        """
        Fit model on training data.
        
        Args:
            train_df: Training data with datetime index
            target_col: Target column name
            **kwargs: Additional fit parameters
            
        Returns:
            self (for method chaining)
        """
        pass
    
    @abstractmethod
    def predict(self,
               horizon: int = 14,
               **kwargs) -> pd.DataFrame:
        """
        Generate forecast.
        
        Args:
            horizon: Forecast horizon (days)
            **kwargs: Additional prediction parameters
            
        Returns:
            DataFrame with columns ['date', 'forecast']
        """
        pass
    
    def __call__(self, train_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Convenience method: fit and predict in one call.
        
        This allows ModelPipeline to work directly with TimeSeriesCrossValidator.
        
        Args:
            train_df: Training data
            **kwargs: Parameters passed to fit() and predict()
            
        Returns:
            Forecast DataFrame
        """
        target_col = kwargs.pop('target_col', 'close')
        horizon = kwargs.pop('horizon', 14)
        
        self.fit(train_df, target_col=target_col, **kwargs)
        return self.predict(horizon=horizon, **kwargs)
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        return self.params.copy()
    
    def set_params(self, **params) -> 'ModelPipeline':
        """Set model parameters."""
        self.params.update(params)
        return self


class NaivePipeline(ModelPipeline):
    """Naive forecast pipeline (last value persistence)."""
    
    def __init__(self, **params):
        super().__init__("Naive", **params)
        self.last_value = None
        self.last_date = None
    
    def fit(self, train_df: pd.DataFrame, target_col: str = 'close', **kwargs) -> 'ModelPipeline':
        self.last_value = train_df[target_col].iloc[-1]
        self.last_date = train_df.index[-1]
        self.is_fitted = True
        return self
    
    def predict(self, horizon: int = 14, **kwargs) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecast_dates = pd.date_range(
            start=self.last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        return pd.DataFrame({
            'date': forecast_dates,
            'forecast': [self.last_value] * horizon
        })


class RandomWalkPipeline(ModelPipeline):
    """Random walk with drift pipeline."""
    
    def __init__(self, lookback_days: int = 30, **params):
        super().__init__("RandomWalk", lookback_days=lookback_days, **params)
        self.drift = None
        self.last_value = None
        self.last_date = None
    
    def fit(self, train_df: pd.DataFrame, target_col: str = 'close', **kwargs) -> 'ModelPipeline':
        lookback = self.params.get('lookback_days', 30)
        
        # Calculate drift from recent data
        recent_data = train_df[target_col].iloc[-lookback:]
        self.drift = (recent_data.iloc[-1] - recent_data.iloc[0]) / len(recent_data)
        
        self.last_value = train_df[target_col].iloc[-1]
        self.last_date = train_df.index[-1]
        self.is_fitted = True
        return self
    
    def predict(self, horizon: int = 14, **kwargs) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecast_dates = pd.date_range(
            start=self.last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        # Random walk: last_value + cumulative drift
        forecasts = [self.last_value + (i + 1) * self.drift for i in range(horizon)]
        
        return pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecasts
        })


def create_model_from_registry(model_key: str, **kwargs) -> ModelPipeline:
    """
    Create ModelPipeline from model registry key.
    
    This is a bridge to existing model registry - converts registry models
    to ModelPipeline interface.
    
    FAIL-OPEN: If model can't be loaded (missing packages), raises ImportError
    which should be caught by calling code.
    
    Args:
        model_key: Key from BASELINE_MODELS registry
        **kwargs: Additional parameters
        
    Returns:
        ModelPipeline instance
        
    Raises:
        ImportError: If required packages not available
        ValueError: If model_key not found
    """
    # Try to import model registry (may fail if ground_truth not available)
    try:
        from ground_truth.config.model_registry import BASELINE_MODELS
    except ImportError as e:
        # Fall back to basic models only
        if model_key == 'naive':
            return NaivePipeline(**kwargs)
        elif model_key == 'random_walk':
            return RandomWalkPipeline(lookback_days=30, **kwargs)
        else:
            raise ImportError(f"Model registry not available and {model_key} not in basic models. Error: {e}")
    
    if model_key not in BASELINE_MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(BASELINE_MODELS.keys())}")
    
    config = BASELINE_MODELS[model_key]
    
    # Simple implementations (no dependencies)
    if model_key == 'naive':
        return NaivePipeline(**kwargs)
    elif model_key == 'random_walk':
        lookback = config['params'].get('lookback_days', 30)
        return RandomWalkPipeline(lookback_days=lookback, **kwargs)
    else:
        # For other models, wrap the existing function (may require packages)
        # ImportError will be raised if packages not available - caller should catch
        return WrappedModelPipeline(model_key, config, **kwargs)


class WrappedModelPipeline(ModelPipeline):
    """Wrapper to convert existing model functions to ModelPipeline interface."""
    
    def __init__(self, model_key: str, model_config: Dict, **params):
        model_name = model_config.get('name', model_key)
        super().__init__(model_name, **params)
        self.model_key = model_key
        self.model_config = model_config
        self.model_fn = model_config['function']
    
    def fit(self, train_df: pd.DataFrame, target_col: str = 'close', **kwargs) -> 'ModelPipeline':
        # For wrapped models, we'll fit during predict (lazy fitting)
        self.train_df = train_df
        self.target_col = target_col
        self.is_fitted = True
        return self
    
    def predict(self, horizon: int = 14, **kwargs) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare parameters
        params = self.model_config['params'].copy()
        params.update(self.params)
        params.update(kwargs)
        params['horizon'] = horizon
        params['target'] = self.target_col
        
        # Call existing model function
        result = self.model_fn(df_pandas=self.train_df, **params)
        
        # Extract forecast DataFrame
        if isinstance(result, dict):
            forecast_df = result.get('forecast_df', result.get('forecast'))
        else:
            forecast_df = result
        
        # Ensure proper format
        if not isinstance(forecast_df, pd.DataFrame):
            raise ValueError("Model function must return DataFrame")
        
        # Ensure 'date' column exists
        if 'date' not in forecast_df.columns and isinstance(forecast_df.index, pd.DatetimeIndex):
            forecast_df = forecast_df.reset_index()
        
        return forecast_df[['date', 'forecast']]


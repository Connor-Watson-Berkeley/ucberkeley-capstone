"""Base forecaster class to reduce tech debt and standardize model interface.

All forecasting models should inherit from this base class to ensure consistent API.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models.

    This class defines the standard interface that all forecasting models must implement.
    It ensures consistency across ARIMA, SARIMAX, XGBoost, Prophet, and other models.
    """

    def __init__(self, model_name: str, model_version: str):
        """Initialize base forecaster.

        Args:
            model_name: Human-readable model name (e.g., 'SARIMAX+Weather')
            model_version: Model version identifier (e.g., 'sarimax_weather_v1')
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.is_fitted = False
        self.data_cutoff_date = None

    @abstractmethod
    def fit(self,
            df: pd.DataFrame,
            target_col: str = 'close',
            exog_features: Optional[List[str]] = None,
            **kwargs) -> 'BaseForecaster':
        """Fit the model on training data.

        Args:
            df: Training data (pandas DataFrame with datetime index)
            target_col: Name of target column to forecast
            exog_features: List of exogenous feature columns
            **kwargs: Model-specific parameters

        Returns:
            self (fitted forecaster)
        """
        pass

    @abstractmethod
    def forecast(self,
                 horizon: int = 14,
                 exog_future: Optional[pd.DataFrame] = None,
                 **kwargs) -> Dict:
        """Generate forecast for specified horizon.

        Args:
            horizon: Number of periods to forecast
            exog_future: Future exogenous variables (for models that use them)
            **kwargs: Model-specific forecast parameters

        Returns:
            Dictionary with:
                - 'forecast_df': DataFrame with columns ['date', 'forecast']
                - 'prediction_intervals': Dict with 'lower_95', 'upper_95', 'std' (optional)
                - 'model_success': Boolean indicating if model converged
                - 'metadata': Additional model-specific metadata
        """
        pass

    @abstractmethod
    def forecast_with_intervals(self,
                                horizon: int = 14,
                                exog_future: Optional[pd.DataFrame] = None,
                                confidence_level: float = 0.95,
                                **kwargs) -> Dict:
        """Generate forecast with prediction intervals.

        Args:
            horizon: Number of periods to forecast
            exog_future: Future exogenous variables
            confidence_level: Confidence level for intervals (default 0.95)
            **kwargs: Model-specific parameters

        Returns:
            Dictionary with same structure as forecast() plus prediction intervals
        """
        pass

    def generate_sample_paths(self,
                             horizon: int = 14,
                             n_paths: int = 2000,
                             exog_future: Optional[pd.DataFrame] = None,
                             **kwargs) -> np.ndarray:
        """Generate Monte Carlo sample paths for distributions table.

        Args:
            horizon: Forecast horizon
            n_paths: Number of sample paths to generate
            exog_future: Future exogenous variables
            **kwargs: Model-specific parameters

        Returns:
            Array of shape (n_paths, horizon) with forecasted values

        Note:
            Default implementation uses normal distribution around point forecast.
            Override for model-specific implementations (e.g., bootstrap for ARIMA).
        """
        # Get point forecast
        forecast_result = self.forecast(horizon=horizon, exog_future=exog_future, **kwargs)
        forecast_values = forecast_result['forecast_df']['forecast'].values

        # Get std if available
        if 'prediction_intervals' in forecast_result and forecast_result['prediction_intervals']:
            std_values = forecast_result['prediction_intervals'].get('std')
            if std_values is None:
                # Estimate from confidence intervals
                lower = forecast_result['prediction_intervals']['lower_95']
                upper = forecast_result['prediction_intervals']['upper_95']
                std_values = [(u - l) / 3.92 for u, l in zip(upper, lower)]  # 95% CI = ±1.96*std
        else:
            # Use residual std as fallback (if available)
            std_values = [self._estimate_residual_std()] * horizon

        # Generate paths
        sample_paths = np.zeros((n_paths, horizon))
        for i in range(horizon):
            sample_paths[:, i] = np.random.normal(
                loc=forecast_values[i],
                scale=std_values[i] if isinstance(std_values, list) else std_values,
                size=n_paths
            )

        return sample_paths

    def _estimate_residual_std(self) -> float:
        """Estimate residual standard deviation (override in subclasses)."""
        return 2.5  # Default fallback

    def get_metadata(self) -> Dict:
        """Get model metadata for storage.

        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'is_fitted': self.is_fitted,
            'data_cutoff_date': str(self.data_cutoff_date) if self.data_cutoff_date else None
        }

    def validate_fitted(self):
        """Raise error if model not fitted."""
        if not self.is_fitted:
            raise ValueError(f"Model {self.model_name} must be fitted before forecasting")


class StatisticalForecaster(BaseForecaster):
    """Base class for statistical models (ARIMA, SARIMAX, ETS, etc.)."""

    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)
        self.residuals = None

    def _estimate_residual_std(self) -> float:
        """Estimate residual std from fitted model."""
        if self.residuals is not None:
            return float(np.std(self.residuals))
        return 2.5


class MLForecaster(BaseForecaster):
    """Base class for ML models (XGBoost, LightGBM, etc.)."""

    def __init__(self, model_name: str, model_version: str):
        super().__init__(model_name, model_version)
        self.feature_importance = None

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance (if available)."""
        return self.feature_importance


# Example usage template
USAGE_EXAMPLE = """
# Example: Implementing SARIMAX Forecaster

from ground_truth.core.base_forecaster import StatisticalForecaster
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAXForecaster(StatisticalForecaster):
    def __init__(self, model_name: str = 'SARIMAX', model_version: str = 'sarimax_v1'):
        super().__init__(model_name, model_version)
        self.order = (1, 1, 1)
        self.seasonal_order = (1, 1, 1, 7)

    def fit(self, df, target_col='close', exog_features=None, **kwargs):
        self.data_cutoff_date = df.index[-1]

        y = df[target_col]
        exog = df[exog_features] if exog_features else None

        self.model = SARIMAX(
            y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order
        )

        self.fitted_model = self.model.fit(disp=False)
        self.residuals = self.fitted_model.resid
        self.is_fitted = True

        return self

    def forecast(self, horizon=14, exog_future=None, **kwargs):
        self.validate_fitted()

        forecast_result = self.fitted_model.forecast(steps=horizon, exog=exog_future)

        forecast_dates = pd.date_range(
            start=self.data_cutoff_date + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'
        )

        return {
            'forecast_df': pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecast_result
            }),
            'model_success': True,
            'metadata': self.get_metadata()
        }

    def forecast_with_intervals(self, horizon=14, exog_future=None, confidence_level=0.95, **kwargs):
        self.validate_fitted()

        forecast_obj = self.fitted_model.get_forecast(steps=horizon, exog=exog_future)
        forecast_mean = forecast_obj.predicted_mean
        forecast_ci = forecast_obj.conf_int(alpha=1-confidence_level)

        forecast_dates = pd.date_range(
            start=self.data_cutoff_date + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'
        )

        return {
            'forecast_df': pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecast_mean
            }),
            'prediction_intervals': {
                'lower_95': forecast_ci.iloc[:, 0].tolist(),
                'upper_95': forecast_ci.iloc[:, 1].tolist(),
                'std': forecast_obj.se_mean.tolist()
            },
            'model_success': True,
            'metadata': self.get_metadata()
        }
"""

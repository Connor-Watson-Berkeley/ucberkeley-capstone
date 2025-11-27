"""Evaluator for forecast model metrics.

Inspired by DS261 FlightDelayEvaluator - simple and focused.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats


class ForecastEvaluator:
    """Evaluates forecast model performance with standard metrics.
    
    Similar to DS261 FlightDelayEvaluator but tailored for forecasting.
    """
    
    def __init__(self,
                 target_col: str = 'close',
                 prediction_col: str = 'forecast'):
        """
        Initialize evaluator.
        
        Args:
            target_col: Name of actual values column
            prediction_col: Name of forecast values column
        """
        self.target_col = target_col
        self.prediction_col = prediction_col
    
    def calculate_mae(self, actuals: np.ndarray, forecasts: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        mask = ~(np.isnan(actuals) | np.isnan(forecasts))
        if not np.any(mask):
            return np.nan
        return np.mean(np.abs(actuals[mask] - forecasts[mask]))
    
    def calculate_rmse(self, actuals: np.ndarray, forecasts: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        mask = ~(np.isnan(actuals) | np.isnan(forecasts))
        if not np.any(mask):
            return np.nan
        return np.sqrt(np.mean((actuals[mask] - forecasts[mask])**2))
    
    def calculate_mape(self, actuals: np.ndarray, forecasts: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        mask = ~(np.isnan(actuals) | np.isnan(forecasts)) & (actuals != 0)
        if not np.any(mask):
            return np.nan
        return np.mean(np.abs((actuals[mask] - forecasts[mask]) / actuals[mask])) * 100
    
    def calculate_directional_accuracy(self,
                                      actuals: np.ndarray,
                                      forecasts: np.ndarray,
                                      from_day0: bool = True) -> float:
        """
        Calculate directional accuracy.
        
        Args:
            actuals: Actual values
            forecasts: Forecast values
            from_day0: If True, compare to day 0 (trading signal).
                      If False, compare day-to-day direction.
        
        Returns:
            Percentage of correct directional predictions
        """
        if len(actuals) < 2 or len(forecasts) < 2:
            return 0.0
        
        if from_day0:
            # Compare each day to day 0 (trading signal quality)
            day0_actual = actuals[0]
            day0_forecast = forecasts[0]
            
            correct = 0
            total = 0
            
            for i in range(1, len(actuals)):
                actual_higher = actuals[i] > day0_actual
                forecast_higher = forecasts[i] > day0_forecast
                
                if actual_higher == forecast_higher:
                    correct += 1
                total += 1
            
            return (correct / total * 100) if total > 0 else 0.0
        else:
            # Day-to-day directional accuracy
            correct = 0
            total = 0
            
            for i in range(1, len(actuals)):
                actual_dir = actuals[i] > actuals[i-1]
                forecast_dir = forecasts[i] > forecasts[i-1]
                
                if actual_dir == forecast_dir:
                    correct += 1
                total += 1
            
            return (correct / total * 100) if total > 0 else 0.0
    
    def evaluate(self,
                actuals: np.ndarray,
                forecasts: np.ndarray) -> Dict:
        """
        Calculate all metrics.
        
        Args:
            actuals: Array of actual values
            forecasts: Array of forecast values
            
        Returns:
            Dictionary with all metrics
        """
        # Remove nulls
        mask = ~(np.isnan(actuals) | np.isnan(forecasts))
        actuals_clean = actuals[mask]
        forecasts_clean = forecasts[mask]
        
        if len(actuals_clean) == 0:
            return {
                'mae': np.nan,
                'rmse': np.nan,
                'mape': np.nan,
                'dir_day0': np.nan,
                'dir': np.nan,
                'n_samples': 0
            }
        
        return {
            'mae': self.calculate_mae(actuals_clean, forecasts_clean),
            'rmse': self.calculate_rmse(actuals_clean, forecasts_clean),
            'mape': self.calculate_mape(actuals_clean, forecasts_clean),
            'dir_day0': self.calculate_directional_accuracy(actuals_clean, forecasts_clean, from_day0=True),
            'dir': self.calculate_directional_accuracy(actuals_clean, forecasts_clean, from_day0=False),
            'n_samples': len(actuals_clean)
        }
    
    def evaluate_dataframe(self,
                          df: pd.DataFrame,
                          target_col: Optional[str] = None,
                          prediction_col: Optional[str] = None) -> Dict:
        """
        Evaluate forecasts from DataFrame.
        
        Args:
            df: DataFrame with actual and forecast columns
            target_col: Override default target column
            prediction_col: Override default prediction column
            
        Returns:
            Dictionary with all metrics
        """
        target = target_col or self.target_col
        prediction = prediction_col or self.prediction_col
        
        if target not in df.columns or prediction not in df.columns:
            raise ValueError(f"Columns {target} and/or {prediction} not found in DataFrame")
        
        actuals = df[target].values
        forecasts = df[prediction].values
        
        return self.evaluate(actuals, forecasts)


"""Cross-validator for time-series walk-forward validation.

Inspired by DS261 FlightDelayCV - clean orchestration of folds and evaluation.
"""

from typing import Dict, List, Tuple, Callable, Optional
import pandas as pd
import numpy as np

from .data_loader import TimeSeriesDataLoader
from .evaluator import ForecastEvaluator


class TimeSeriesCrossValidator:
    """Orchestrates walk-forward cross-validation for time-series models.
    
    Similar to DS261 FlightDelayCV but designed for forecasting use cases.
    """
    
    def __init__(self,
                 data_loader: TimeSeriesDataLoader,
                 evaluator: ForecastEvaluator,
                 folds: List[Tuple[pd.DataFrame, pd.DataFrame]]):
        """
        Initialize cross-validator.
        
        Args:
            data_loader: TimeSeriesDataLoader instance
            evaluator: ForecastEvaluator instance
            folds: List of (train_df, test_df) tuples
        """
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.folds = folds
        
        self.metrics = []
        self.models = []
        self.test_metric = None
        self.test_model = None
    
    def fit(self,
           model_fn: Callable,
           model_params: Dict,
           target_col: str = 'close',
           horizon: int = 14) -> pd.DataFrame:
        """
        Fit model on CV folds (excluding test fold if present).
        
        Args:
            model_fn: Model function that takes (train_df, **params) and returns forecast
            model_params: Parameters to pass to model function
            target_col: Target column name
            horizon: Forecast horizon (days)
            
        Returns:
            DataFrame with metrics per fold, plus mean/std summary rows
        """
        # CV folds only (exclude last fold if it's the test fold)
        cv_folds = self.folds[:-1] if len(self.folds) > 1 else self.folds
        
        for i, (train_df, val_df) in enumerate(cv_folds):
            try:
                # Train and forecast
                result = model_fn(train_df, **model_params)
                
                # Extract forecast
                if isinstance(result, dict):
                    forecast_df = result.get('forecast_df', result.get('forecast'))
                else:
                    forecast_df = result
                
                # Ensure DataFrame format
                if not isinstance(forecast_df, pd.DataFrame):
                    raise ValueError("Model function must return DataFrame or dict with 'forecast_df' key")
                
                # Get actuals from validation set
                val_dates = val_df.index[:horizon]
                actuals = val_df.loc[val_dates, target_col].values
                
                # Match forecast dates
                if 'date' in forecast_df.columns:
                    forecast_df = forecast_df.set_index('date')
                forecast_values = forecast_df.loc[val_dates, 'forecast'].values[:len(actuals)]
                
                # Evaluate
                metrics = self.evaluator.evaluate(actuals, forecast_values)
                metrics['fold_id'] = i + 1
                metrics['train_start'] = train_df.index[0]
                metrics['train_end'] = train_df.index[-1]
                metrics['test_start'] = val_df.index[0]
                metrics['test_end'] = val_df.index[min(horizon-1, len(val_df)-1)]
                
                self.metrics.append(metrics)
                self.models.append(result)
                
            except Exception as e:
                print(f"   ✗ Fold {i+1} failed: {str(e)[:100]}")
                continue
        
        # Create summary DataFrame
        if not self.metrics:
            return pd.DataFrame()
        
        metrics_df = pd.DataFrame(self.metrics)
        
        # Add summary rows
        summary = metrics_df[['mae', 'rmse', 'mape', 'dir_day0', 'dir']].agg(['mean', 'std'])
        metrics_df = pd.concat([metrics_df, summary], ignore_index=True)
        
        return metrics_df
    
    def evaluate_test(self,
                     model_fn: Callable,
                     model_params: Dict,
                     target_col: str = 'close',
                     horizon: int = 14) -> Dict:
        """
        Evaluate on final test fold (if present).
        
        Args:
            model_fn: Model function
            model_params: Model parameters
            target_col: Target column name
            horizon: Forecast horizon
            
        Returns:
            Dictionary with test metrics
        """
        if not self.folds:
            raise ValueError("No folds available")
        
        # Last fold is test fold
        train_df, test_df = self.folds[-1]
        
        try:
            # Train on all training data
            result = model_fn(train_df, **model_params)
            
            # Extract forecast
            if isinstance(result, dict):
                forecast_df = result.get('forecast_df', result.get('forecast'))
            else:
                forecast_df = result
            
            if not isinstance(forecast_df, pd.DataFrame):
                raise ValueError("Model function must return DataFrame or dict with 'forecast_df' key")
            
            # Get actuals
            test_dates = test_df.index[:horizon]
            actuals = test_df.loc[test_dates, target_col].values
            
            # Match forecast
            if 'date' in forecast_df.columns:
                forecast_df = forecast_df.set_index('date')
            forecast_values = forecast_df.loc[test_dates, 'forecast'].values[:len(actuals)]
            
            # Evaluate
            self.test_metric = self.evaluator.evaluate(actuals, forecast_values)
            self.test_model = result
            
            return self.test_metric
            
        except Exception as e:
            print(f"   ✗ Test fold failed: {str(e)[:100]}")
            return {}
    
    @staticmethod
    def compare_models(cv1: 'TimeSeriesCrossValidator',
                      cv2: 'TimeSeriesCrossValidator',
                      name1: str = "Model 1",
                      name2: str = "Model 2") -> Dict:
        """
        Compare two cross-validation runs.
        
        Args:
            cv1: First cross-validator results
            cv2: Second cross-validator results
            name1: Name of first model
            name2: Name of second model
            
        Returns:
            Comparison dictionary
        """
        if not cv1.metrics or not cv2.metrics:
            return {}
        
        m1 = pd.DataFrame(cv1.metrics)
        m2 = pd.DataFrame(cv2.metrics)
        
        return {
            'model1_name': name1,
            'model2_name': name2,
            'model1_mae_mean': m1['mae'].mean(),
            'model2_mae_mean': m2['mae'].mean(),
            'mae_improvement_pct': ((m2['mae'].mean() - m1['mae'].mean()) / m2['mae'].mean() * 100),
            'model1_dir_day0_mean': m1['dir_day0'].mean(),
            'model2_dir_day0_mean': m2['dir_day0'].mean(),
        }


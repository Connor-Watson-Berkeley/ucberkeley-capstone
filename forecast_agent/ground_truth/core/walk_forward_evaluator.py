"""Walk-forward backtesting evaluator with statistical tests.

Implements:
- Walk-forward validation across multiple 14-day windows
- Aggregate metrics across all windows
- White noise residual tests (Ljung-Box)
- Statistical significance tests (Diebold-Mariano vs Random Walk)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')


def diebold_mariano_test(errors1: np.ndarray, errors2: np.ndarray, horizon: int = 1) -> Dict:
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Tests null hypothesis that two forecasts have equal accuracy.

    Args:
        errors1: Forecast errors from model 1 (e.g., candidate model)
        errors2: Forecast errors from model 2 (e.g., random walk benchmark)
        horizon: Forecast horizon (for adjusting standard errors)

    Returns:
        Dict with test statistic, p-value, and interpretation
    """
    # Squared errors (for MSE comparison)
    d = errors1**2 - errors2**2

    # Mean of loss differential
    mean_d = np.mean(d)

    # Variance of loss differential (accounting for autocorrelation)
    n = len(d)
    var_d = np.var(d, ddof=1) / n

    # Adjust for multi-step forecasts (Harvey et al 1997 correction)
    var_d = var_d * (1 + (horizon - 1) * 2 / n)

    # Test statistic
    if var_d > 0:
        dm_stat = mean_d / np.sqrt(var_d)
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    else:
        dm_stat = 0.0
        p_value = 1.0

    # Interpretation
    if p_value < 0.05:
        if dm_stat < 0:
            conclusion = "Model 1 significantly better than Model 2 (p<0.05)"
        else:
            conclusion = "Model 2 significantly better than Model 1 (p<0.05)"
    else:
        conclusion = "No significant difference in accuracy (p>=0.05)"

    return {
        'dm_statistic': dm_stat,
        'p_value': p_value,
        'conclusion': conclusion,
        'mean_loss_diff': mean_d
    }


def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> Dict:
    """
    Ljung-Box test for white noise residuals.

    Tests null hypothesis that residuals are white noise (no autocorrelation).

    Args:
        residuals: Forecast residuals
        lags: Number of lags to test

    Returns:
        Dict with test statistics, p-values, and interpretation
    """
    if len(residuals) < lags + 1:
        return {
            'test_statistic': None,
            'p_value': None,
            'is_white_noise': None,
            'conclusion': 'Not enough data for Ljung-Box test'
        }

    # Run Ljung-Box test
    lb_result = acorr_ljungbox(residuals, lags=lags, return_df=False)

    # Get results at max lag
    test_stat = lb_result[0][-1]  # Q statistic at last lag
    p_value = lb_result[1][-1]     # p-value at last lag

    # Interpretation
    is_white_noise = p_value > 0.05

    if is_white_noise:
        conclusion = f"Residuals are white noise (p={p_value:.4f} > 0.05) - Good!"
    else:
        conclusion = f"Residuals show autocorrelation (p={p_value:.4f} < 0.05) - Model may be misspecified"

    return {
        'test_statistic': test_stat,
        'p_value': p_value,
        'is_white_noise': is_white_noise,
        'conclusion': conclusion,
        'lags_tested': lags
    }


def residual_diagnostics(actuals: np.ndarray, forecasts: np.ndarray) -> Dict:
    """
    Comprehensive residual diagnostics.

    Args:
        actuals: Actual values
        forecasts: Forecast values

    Returns:
        Dict with residual statistics and tests
    """
    residuals = actuals - forecasts

    # Basic statistics
    diagnostics = {
        'mean': np.mean(residuals),
        'std': np.std(residuals, ddof=1),
        'skewness': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals)
    }

    # Normality test (Jarque-Bera)
    if len(residuals) >= 8:
        jb_stat, jb_p = stats.jarque_bera(residuals)
        diagnostics['jarque_bera_stat'] = jb_stat
        diagnostics['jarque_bera_p'] = jb_p
        diagnostics['is_normal'] = jb_p > 0.05

    # White noise test (Ljung-Box)
    lb_lags = min(10, len(residuals) // 2)
    if lb_lags > 0:
        lb_result = ljung_box_test(residuals, lags=lb_lags)
        diagnostics.update({
            'ljung_box_stat': lb_result['test_statistic'],
            'ljung_box_p': lb_result['p_value'],
            'is_white_noise': lb_result['is_white_noise'],
            'ljung_box_conclusion': lb_result['conclusion']
        })

    return diagnostics


class WalkForwardEvaluator:
    """
    Walk-forward backtesting evaluator.

    Evaluates models across multiple non-overlapping 14-day forecast windows.
    Computes aggregate metrics and statistical tests.
    """

    def __init__(self, data_df: pd.DataFrame, horizon: int = 14,
                 min_train_size: int = 365, step_size: int = 14):
        """
        Initialize walk-forward evaluator.

        Args:
            data_df: Full dataset with date index
            horizon: Forecast horizon (days)
            min_train_size: Minimum training days
            step_size: Days between forecast windows
        """
        self.data_df = data_df.sort_index()
        self.horizon = horizon
        self.min_train_size = min_train_size
        self.step_size = step_size

    def generate_windows(self, n_windows: int = None) -> List[Dict]:
        """
        Generate walk-forward validation windows.

        Args:
            n_windows: Number of windows to generate (None = all possible)

        Returns:
            List of dicts with train_start, train_end, test_start, test_end
        """
        windows = []

        # Start after min training size
        start_idx = self.min_train_size

        while start_idx + self.horizon <= len(self.data_df):
            train_end_idx = start_idx
            test_start_idx = start_idx
            test_end_idx = min(start_idx + self.horizon, len(self.data_df))

            window = {
                'train_start': self.data_df.index[0],
                'train_end': self.data_df.index[train_end_idx - 1],
                'test_start': self.data_df.index[test_start_idx],
                'test_end': self.data_df.index[test_end_idx - 1],
                'train_data': self.data_df.iloc[:train_end_idx],
                'test_data': self.data_df.iloc[test_start_idx:test_end_idx]
            }

            windows.append(window)

            # Move to next window
            start_idx += self.step_size

            if n_windows and len(windows) >= n_windows:
                break

        return windows

    def evaluate_model_walk_forward(self, model_fn: Callable,
                                    model_params: Dict,
                                    windows: List[Dict],
                                    target: str = 'close') -> Dict:
        """
        Evaluate model across multiple walk-forward windows.

        Args:
            model_fn: Model function that takes (train_data, **params) and returns forecast
            model_params: Parameters to pass to model function
            windows: List of validation windows
            target: Target column name

        Returns:
            Dict with aggregated metrics and per-window results
        """
        window_results = []
        all_actuals = []
        all_forecasts = []
        all_errors = []

        for i, window in enumerate(windows):
            try:
                # Train model
                train_data = window['train_data']
                test_data = window['test_data']

                # Get forecast
                result = model_fn(train_data, **model_params)
                forecast_df = result['forecast_df']

                # Match with actuals
                actuals = test_data[target].values[:len(forecast_df)]
                forecasts = forecast_df['forecast'].values[:len(actuals)]

                # Compute metrics for this window
                errors = actuals - forecasts
                abs_errors = np.abs(errors)

                window_metrics = {
                    'window_id': i,
                    'test_start': window['test_start'],
                    'test_end': window['test_end'],
                    'mae': np.mean(abs_errors),
                    'rmse': np.sqrt(np.mean(errors**2)),
                    'mape': np.mean(np.abs(errors / actuals)) * 100,
                    'n_forecasts': len(actuals)
                }

                window_results.append(window_metrics)
                all_actuals.extend(actuals)
                all_forecasts.extend(forecasts)
                all_errors.extend(errors)

            except Exception as e:
                print(f"   ✗ Window {i+1} failed: {str(e)[:50]}")
                continue

        # Aggregate metrics across all windows
        all_actuals = np.array(all_actuals)
        all_forecasts = np.array(all_forecasts)
        all_errors = np.array(all_errors)

        aggregate_metrics = {
            'n_windows': len(window_results),
            'total_forecasts': len(all_actuals),
            'mae_mean': np.mean([w['mae'] for w in window_results]),
            'mae_std': np.std([w['mae'] for w in window_results]),
            'mae_min': np.min([w['mae'] for w in window_results]),
            'mae_max': np.max([w['mae'] for w in window_results]),
            'rmse_mean': np.mean([w['rmse'] for w in window_results]),
            'mape_mean': np.mean([w['mape'] for w in window_results]),
        }

        # Residual diagnostics (on all forecasts)
        diagnostics = residual_diagnostics(all_actuals, all_forecasts)

        return {
            'aggregate_metrics': aggregate_metrics,
            'window_results': window_results,
            'residual_diagnostics': diagnostics,
            'all_actuals': all_actuals,
            'all_forecasts': all_forecasts,
            'all_errors': all_errors
        }

    def compare_models(self, model1_result: Dict, model2_result: Dict,
                      model1_name: str = "Model 1",
                      model2_name: str = "Model 2") -> Dict:
        """
        Compare two models using Diebold-Mariano test.

        Args:
            model1_result: Walk-forward result from model 1
            model2_result: Walk-forward result from model 2
            model1_name: Name of model 1
            model2_name: Name of model 2

        Returns:
            Dict with comparison results
        """
        errors1 = model1_result['all_errors']
        errors2 = model2_result['all_errors']

        # Ensure same length
        min_len = min(len(errors1), len(errors2))
        errors1 = errors1[:min_len]
        errors2 = errors2[:min_len]

        # Diebold-Mariano test
        dm_result = diebold_mariano_test(errors1, errors2, horizon=self.horizon)

        # Summary comparison
        mae1 = model1_result['aggregate_metrics']['mae_mean']
        mae2 = model2_result['aggregate_metrics']['mae_mean']

        comparison = {
            'model1_name': model1_name,
            'model2_name': model2_name,
            'model1_mae': mae1,
            'model2_mae': mae2,
            'mae_improvement': (mae2 - mae1) / mae2 * 100,  # % improvement of model1 over model2
            'diebold_mariano': dm_result
        }

        return comparison


# Example usage
EXAMPLE_USAGE = """
from ground_truth.core.walk_forward_evaluator import WalkForwardEvaluator
from ground_truth.models import xgboost_model

# Initialize evaluator
evaluator = WalkForwardEvaluator(
    data_df=df,
    horizon=14,
    min_train_size=365*2,  # 2 years minimum training
    step_size=14  # Non-overlapping windows
)

# Generate windows
windows = evaluator.generate_windows(n_windows=20)  # 20 windows = ~280 days of evaluation

# Evaluate XGBoost model
xgb_result = evaluator.evaluate_model_walk_forward(
    model_fn=xgboost_model.xgboost_forecast_with_metadata,
    model_params={
        'commodity': 'Coffee',
        'target': 'close',
        'horizon': 14,
        'exog_features': ['temp_c', 'humidity_pct'],
        'lags': [1, 7, 14],
        'windows': [7, 30]
    },
    windows=windows
)

print(f"Aggregate MAE: {xgb_result['aggregate_metrics']['mae_mean']:.2f}")
print(f"MAE Std: {xgb_result['aggregate_metrics']['mae_std']:.2f}")
print(f"White noise: {xgb_result['residual_diagnostics']['is_white_noise']}")

# Compare with random walk
rw_result = evaluator.evaluate_model_walk_forward(
    model_fn=random_walk_model.random_walk_forecast_with_metadata,
    model_params={'commodity': 'Coffee', 'target': 'close', 'horizon': 14},
    windows=windows
)

comparison = evaluator.compare_models(xgb_result, rw_result,
                                     "XGBoost", "RandomWalk")
print(f"Improvement over RW: {comparison['mae_improvement']:.1f}%")
print(f"Statistical significance: {comparison['diebold_mariano']['conclusion']}")
"""

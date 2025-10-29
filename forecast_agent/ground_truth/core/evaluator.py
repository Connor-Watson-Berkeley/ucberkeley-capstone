"""Forecast evaluation metrics and statistical tests.

Calculates performance metrics, statistical significance, and performance regression.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple


def calculate_metrics(actuals: pd.Series, forecasts: pd.Series) -> Dict[str, float]:
    """
    Calculate forecast performance metrics.

    Args:
        actuals: Realized values
        forecasts: Predicted values (same length as actuals)

    Returns:
        Dict with metrics: MAE, RMSE, MAPE, directional_accuracy

    Metrics explained:
        - MAE (Mean Absolute Error): Average absolute difference
        - RMSE (Root Mean Squared Error): Emphasizes large errors
        - MAPE (Mean Absolute Percentage Error): Scale-independent %
        - Directional Accuracy: % of correct up/down predictions
    """
    errors = actuals - forecasts
    abs_errors = np.abs(errors)

    metrics = {
        'mae': float(abs_errors.mean()),
        'rmse': float(np.sqrt((errors ** 2).mean())),
        'mape': float((abs_errors / actuals).mean() * 100),  # As percentage
        'n_samples': len(actuals)
    }

    # Directional accuracy (did we predict up/down correctly?)
    if len(actuals) > 1:
        actual_direction = np.sign(actuals.diff().dropna())
        forecast_direction = np.sign(forecasts.diff().dropna())
        correct_direction = (actual_direction == forecast_direction).sum()
        metrics['directional_accuracy'] = float(correct_direction / len(actual_direction) * 100)

    return metrics


def t_test_comparison(errors_model1: pd.Series, errors_model2: pd.Series,
                      alpha: float = 0.05) -> Dict[str, any]:
    """
    Paired t-test: Is model1 significantly better than model2?

    Args:
        errors_model1: Absolute errors from model 1
        errors_model2: Absolute errors from model 2
        alpha: Significance level (default: 0.05)

    Returns:
        Dict with t_statistic, p_value, is_significant, better_model

    Interpretation:
        - is_significant=True: Models perform differently (reject H0: equal performance)
        - better_model: Which model has lower mean error

    Example:
        result = t_test_comparison(
            abs(actuals - model1_forecasts),
            abs(actuals - model2_forecasts)
        )
        if result['is_significant']:
            print(f"{result['better_model']} wins!")
    """
    # Paired t-test on absolute errors
    t_stat, p_value = stats.ttest_rel(errors_model1, errors_model2)

    # Determine which is better (lower error = better)
    mean1 = errors_model1.mean()
    mean2 = errors_model2.mean()
    better_model = 'model1' if mean1 < mean2 else 'model2'

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'is_significant': (p_value < alpha),
        'better_model': better_model,
        'mean_error_model1': float(mean1),
        'mean_error_model2': float(mean2)
    }


def diebold_mariano_test(errors_model1: pd.Series, errors_model2: pd.Series,
                         h: int = 1, alpha: float = 0.05) -> Dict[str, any]:
    """
    Diebold-Mariano test for forecast accuracy comparison.

    More sophisticated than t-test - accounts for forecast horizon and autocorrelation.

    Args:
        errors_model1: Forecast errors from model 1
        errors_model2: Forecast errors from model 2
        h: Forecast horizon (default: 1)
        alpha: Significance level

    Returns:
        Dict with dm_statistic, p_value, is_significant, better_model

    Note: Preferred over t-test for time series forecasts
    """
    # Loss differential (squared error difference)
    d = errors_model1 ** 2 - errors_model2 ** 2

    # Mean of loss differential
    d_bar = d.mean()

    # Variance of d (with autocorrelation correction)
    gamma = [d.autocorr(lag=i) if i > 0 else d.var() for i in range(h)]
    v_d = (gamma[0] + 2 * sum(gamma[1:])) / len(d)

    # DM test statistic
    dm_stat = d_bar / np.sqrt(v_d)

    # Two-tailed test
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    # Which model is better
    better_model = 'model1' if d_bar < 0 else 'model2'

    return {
        'dm_statistic': float(dm_stat),
        'p_value': float(p_value),
        'is_significant': (p_value < alpha),
        'better_model': better_model,
        'loss_differential': float(d_bar)
    }


def binomial_direction_test(actuals: pd.Series, forecasts: pd.Series,
                             alpha: float = 0.05) -> Dict[str, any]:
    """
    Test if directional accuracy is better than random (50%).

    Args:
        actuals: Realized values
        forecasts: Predicted values
        alpha: Significance level

    Returns:
        Dict with accuracy, p_value, is_significant

    Interpretation:
        - is_significant=True: Model beats random guessing on direction

    Example:
        result = binomial_direction_test(actuals, forecasts)
        if result['is_significant']:
            print(f"Model has {result['accuracy']:.1f}% directional accuracy (beats 50%)")
    """
    # Calculate direction matches
    actual_direction = np.sign(actuals.diff().dropna())
    forecast_direction = np.sign(forecasts.diff().dropna())
    correct = (actual_direction == forecast_direction).sum()
    total = len(actual_direction)

    accuracy = correct / total

    # Binomial test: H0 = accuracy is 50% (random)
    from scipy.stats import binomtest
    p_value = binomtest(correct, total, 0.5, alternative='greater').pvalue

    return {
        'directional_accuracy': float(accuracy * 100),
        'p_value': float(p_value),
        'is_significant': (p_value < alpha),
        'n_correct': int(correct),
        'n_total': int(total)
    }


def performance_regression_test(current_mae: float, historical_mae: pd.Series,
                                 threshold_std: float = 2.0) -> Dict[str, any]:
    """
    Detect performance regression - is current MAE worse than expected?

    Args:
        current_mae: MAE from latest forecast
        historical_mae: Series of past MAE values
        threshold_std: How many std devs above mean triggers alert (default: 2)

    Returns:
        Dict with is_regression, z_score, mean_historical, std_historical

    Use case: Monitor model drift - retrain if performance degrades

    Example:
        result = performance_regression_test(
            current_mae=5.2,
            historical_mae=pd.Series([4.1, 4.3, 4.2, 4.5, 4.0])
        )
        if result['is_regression']:
            print("⚠️ Model performance degraded! Consider retraining.")
    """
    mean_historical = historical_mae.mean()
    std_historical = historical_mae.std()

    # Z-score: how many std devs is current MAE above mean?
    z_score = (current_mae - mean_historical) / std_historical

    is_regression = (z_score > threshold_std)

    return {
        'is_regression': bool(is_regression),
        'z_score': float(z_score),
        'current_mae': float(current_mae),
        'mean_historical_mae': float(mean_historical),
        'std_historical_mae': float(std_historical),
        'threshold_std': threshold_std
    }


def evaluate_forecast(actuals: pd.DataFrame, forecasts: pd.DataFrame,
                      model_name: str) -> Dict[str, any]:
    """
    Comprehensive forecast evaluation - all metrics in one call.

    Args:
        actuals: DataFrame with 'date' and 'close' (realized prices)
        forecasts: DataFrame with 'date' and 'forecast' (predictions)
        model_name: Model identifier

    Returns:
        Dict with all metrics, tests, and metadata

    Example:
        results = evaluate_forecast(actuals_df, forecast_df, 'ARIMA(1,1,1)')
        print(f"MAE: {results['metrics']['mae']:.2f}")
        print(f"Directional accuracy: {results['metrics']['directional_accuracy']:.1f}%")
    """
    # Merge on date
    merged = actuals.merge(forecasts, on='date', how='inner')

    if len(merged) == 0:
        raise ValueError("No matching dates between actuals and forecasts")

    actuals_series = merged['close']
    forecasts_series = merged['forecast']

    # Calculate metrics
    metrics = calculate_metrics(actuals_series, forecasts_series)

    # Directional test
    direction_test = binomial_direction_test(actuals_series, forecasts_series)

    # Package results
    return {
        'model_name': model_name,
        'metrics': metrics,
        'direction_test': direction_test,
        'n_forecasts': len(merged),
        'date_range': {
            'start': str(merged['date'].min()),
            'end': str(merged['date'].max())
        }
    }

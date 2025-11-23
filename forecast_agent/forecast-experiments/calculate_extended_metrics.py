"""
Calculate Extended Metrics for Forecast Evaluation

Includes directional accuracy, hit rates, and other trading-relevant metrics
beyond standard MAPE/RMSE/MAE.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from darts import TimeSeries
from darts.metrics import mape, rmse, mae


def calculate_directional_accuracy(actual: TimeSeries, forecast: TimeSeries) -> float:
    """
    Calculate directional accuracy - % of time forecast correctly predicts
    direction of price movement (up/down/flat).

    Args:
        actual: Actual price series
        forecast: Forecasted price series

    Returns:
        Directional accuracy as percentage (0-100)
    """
    actual_values = actual.values().flatten()
    forecast_values = forecast.values().flatten()

    # Calculate price changes
    actual_direction = np.sign(np.diff(actual_values))
    forecast_direction = np.sign(np.diff(forecast_values))

    # Calculate accuracy
    correct = (actual_direction == forecast_direction).sum()
    total = len(actual_direction)

    return (correct / total) * 100


def calculate_hit_rate(actual: TimeSeries, forecast: TimeSeries, threshold_pct: float = 5.0) -> float:
    """
    Calculate hit rate - % of predictions within threshold_pct of actual value.

    Args:
        actual: Actual price series
        forecast: Forecasted price series
        threshold_pct: Acceptable error threshold (default 5%)

    Returns:
        Hit rate as percentage (0-100)
    """
    actual_values = actual.values().flatten()
    forecast_values = forecast.values().flatten()

    # Calculate percentage errors
    pct_errors = np.abs((forecast_values - actual_values) / actual_values) * 100

    # Count predictions within threshold
    hits = (pct_errors <= threshold_pct).sum()
    total = len(pct_errors)

    return (hits / total) * 100


def calculate_bias(actual: TimeSeries, forecast: TimeSeries) -> float:
    """
    Calculate forecast bias (mean error).
    Positive = over-forecasting, Negative = under-forecasting

    Args:
        actual: Actual price series
        forecast: Forecasted price series

    Returns:
        Mean bias in dollars
    """
    actual_values = actual.values().flatten()
    forecast_values = forecast.values().flatten()

    return np.mean(forecast_values - actual_values)


def calculate_prediction_intervals(forecast: TimeSeries, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate prediction interval bounds (simple approach using std).

    Args:
        forecast: Forecasted price series
        confidence: Confidence level (default 0.95)

    Returns:
        (lower_bound, upper_bound) as percentiles
    """
    forecast_values = forecast.values().flatten()
    mean = np.mean(forecast_values)
    std = np.std(forecast_values)

    # Simple normal approximation
    z_score = 1.96 if confidence == 0.95 else 2.58  # 95% or 99%

    lower = mean - (z_score * std)
    upper = mean + (z_score * std)

    return lower, upper


def calculate_sharpe_ratio(actual: TimeSeries, forecast: TimeSeries, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio based on trading strategy:
    Buy if forecast > current price, Sell if forecast < current price

    Args:
        actual: Actual price series
        forecast: Forecasted price series
        risk_free_rate: Annual risk-free rate (default 0%)

    Returns:
        Sharpe ratio
    """
    actual_values = actual.values().flatten()
    forecast_values = forecast.values().flatten()

    # Calculate returns based on forecast signals
    # Signal: 1 if forecast > actual[t-1], -1 if forecast < actual[t-1]
    signals = np.sign(forecast_values[:-1] - actual_values[:-1])

    # Actual returns
    returns = np.diff(actual_values) / actual_values[:-1]

    # Strategy returns
    strategy_returns = signals * returns

    # Sharpe ratio
    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns)

    if std_return == 0:
        return 0.0

    # Annualize (assuming daily data, 252 trading days)
    sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(252)

    return sharpe


def calculate_all_metrics(actual: TimeSeries, forecast: TimeSeries) -> Dict[str, float]:
    """
    Calculate comprehensive set of forecast evaluation metrics.

    Args:
        actual: Actual price series
        forecast: Forecasted price series

    Returns:
        Dictionary of all metrics
    """
    metrics = {}

    # Standard metrics
    metrics['mape'] = mape(actual, forecast)
    metrics['rmse'] = rmse(actual, forecast)
    metrics['mae'] = mae(actual, forecast)

    # Directional accuracy
    metrics['directional_accuracy'] = calculate_directional_accuracy(actual, forecast)

    # Hit rates at different thresholds
    metrics['hit_rate_5pct'] = calculate_hit_rate(actual, forecast, threshold_pct=5.0)
    metrics['hit_rate_10pct'] = calculate_hit_rate(actual, forecast, threshold_pct=10.0)
    metrics['hit_rate_2pct'] = calculate_hit_rate(actual, forecast, threshold_pct=2.0)

    # Bias
    metrics['bias'] = calculate_bias(actual, forecast)

    # Sharpe ratio (trading performance)
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(actual, forecast)

    # Prediction intervals
    lower, upper = calculate_prediction_intervals(forecast, confidence=0.95)
    metrics['prediction_interval_lower'] = lower
    metrics['prediction_interval_upper'] = upper
    metrics['prediction_interval_width'] = upper - lower

    return metrics


def print_metrics_report(metrics: Dict[str, float], model_name: str = "Model"):
    """Print formatted metrics report."""
    print("=" * 80)
    print(f"{model_name} - Extended Metrics Report")
    print("=" * 80)
    print()

    print("Standard Accuracy Metrics:")
    print(f"  MAPE:  {metrics['mape']:.2f}%")
    print(f"  RMSE:  ${metrics['rmse']:.2f}")
    print(f"  MAE:   ${metrics['mae']:.2f}")
    print()

    print("Directional Accuracy:")
    print(f"  Correct Direction: {metrics['directional_accuracy']:.1f}%")
    print()

    print("Hit Rates (% within threshold):")
    print(f"  Within 2%:  {metrics['hit_rate_2pct']:.1f}%")
    print(f"  Within 5%:  {metrics['hit_rate_5pct']:.1f}%")
    print(f"  Within 10%: {metrics['hit_rate_10pct']:.1f}%")
    print()

    print("Forecast Bias:")
    bias_direction = "over-forecasting" if metrics['bias'] > 0 else "under-forecasting"
    print(f"  Mean Bias: ${metrics['bias']:.2f} ({bias_direction})")
    print()

    print("Trading Performance:")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print()

    print("Prediction Intervals (95% confidence):")
    print(f"  Lower Bound: ${metrics['prediction_interval_lower']:.2f}")
    print(f"  Upper Bound: ${metrics['prediction_interval_upper']:.2f}")
    print(f"  Width: ${metrics['prediction_interval_width']:.2f}")
    print()
    print("=" * 80)


if __name__ == '__main__':
    # Test with dummy data
    print("\nThis is a utility module for calculating extended metrics.")
    print("Import and use calculate_all_metrics() or individual metric functions.")
    print("\nExample usage:")
    print("  from calculate_extended_metrics import calculate_all_metrics")
    print("  metrics = calculate_all_metrics(actual_series, forecast_series)")
    print("  print_metrics_report(metrics, model_name='N-HiTS')")

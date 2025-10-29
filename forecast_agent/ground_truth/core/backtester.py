"""Walk-forward backtesting system for time series models.

Implements rigorous backtesting with multiple forecast windows.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, List, Callable


def walk_forward_backtest(df_pandas: pd.DataFrame, model_fn: Callable,
                          model_params: dict, target: str = 'close',
                          initial_train_days: int = 365 * 3,
                          forecast_horizon: int = 14,
                          step_size: int = 14,
                          n_windows: int = 10) -> Dict:
    """
    Walk-forward backtesting with expanding window.

    Args:
        df_pandas: Full dataset with DatetimeIndex
        model_fn: Model function (e.g., xgboost_forecast_with_metadata)
        model_params: Model parameters
        target: Target column
        initial_train_days: Initial training period (days)
        forecast_horizon: Forecast days ahead
        step_size: Days to move forward each iteration
        n_windows: Number of forecast windows to test

    Returns:
        Dict with:
            - backtest_results: List of forecast windows with actuals
            - performance_summary: Aggregate metrics
            - all_forecasts: Combined DataFrame

    Methodology:
        Window 1: Train on days 1-1095, forecast days 1096-1109
        Window 2: Train on days 1-1109, forecast days 1110-1123
        ...
        Window N: Train on expanding data, forecast next 14 days

    This is more realistic than single test period because:
        - Tests model on multiple market conditions
        - Expanding window mimics production (more data over time)
        - Reveals model stability across different periods
    """
    results = []

    # Start date for backtesting
    start_idx = initial_train_days
    max_idx = len(df_pandas) - forecast_horizon

    if start_idx >= max_idx:
        raise ValueError(f"Not enough data: need {initial_train_days + forecast_horizon} days")

    # Generate backtest windows
    for window_i in range(n_windows):
        test_start_idx = start_idx + (window_i * step_size)

        if test_start_idx + forecast_horizon > len(df_pandas):
            break  # Not enough data left

        # Train data: all data up to test_start
        train_data = df_pandas.iloc[:test_start_idx]
        train_end_date = train_data.index[-1]

        # Test data: next forecast_horizon days
        test_data = df_pandas.iloc[test_start_idx:test_start_idx + forecast_horizon]

        # Generate forecast
        try:
            result = model_fn(
                df_pandas=train_data,
                **model_params
            )

            forecast_df = result['forecast_df']

            # Merge with actuals
            merged = test_data[[target]].reset_index()
            merged.columns = ['date', 'actual']
            merged = merged.merge(forecast_df[['date', 'forecast']], on='date', how='left')

            # Calculate errors
            errors = merged['actual'] - merged['forecast']
            abs_errors = np.abs(errors)

            # Calculate directional accuracy from day 0
            directional_accuracy_from_day0 = 0.0
            if len(merged) > 1:
                day_0_actual = merged['actual'].iloc[0]
                day_0_forecast = merged['forecast'].iloc[0]

                correct_from_day0 = 0
                total_from_day0 = 0

                for i in range(1, len(merged)):
                    actual_higher = merged['actual'].iloc[i] > day_0_actual
                    forecast_higher = merged['forecast'].iloc[i] > day_0_forecast

                    if actual_higher == forecast_higher:
                        correct_from_day0 += 1
                    total_from_day0 += 1

                if total_from_day0 > 0:
                    directional_accuracy_from_day0 = (correct_from_day0 / total_from_day0) * 100

            # Calculate day-to-day directional accuracy
            directional_accuracy = 0.0
            if len(merged) > 1:
                correct_direction = 0
                total_direction = 0

                for i in range(1, len(merged)):
                    actual_direction = merged['actual'].iloc[i] > merged['actual'].iloc[i-1]
                    forecast_direction = merged['forecast'].iloc[i] > merged['forecast'].iloc[i-1]

                    if actual_direction == forecast_direction:
                        correct_direction += 1
                    total_direction += 1

                if total_direction > 0:
                    directional_accuracy = (correct_direction / total_direction) * 100

            results.append({
                'window': window_i + 1,
                'train_start': train_data.index[0],
                'train_end': train_end_date,
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'n_train_days': len(train_data),
                'mae': abs_errors.mean(),
                'rmse': np.sqrt((errors ** 2).mean()),
                'mape': (abs_errors / merged['actual']).mean() * 100,
                'directional_accuracy': directional_accuracy,
                'directional_accuracy_from_day0': directional_accuracy_from_day0,
                'forecast_df': forecast_df,
                'actuals_df': merged,
                'errors': errors
            })

        except Exception as e:
            print(f"Window {window_i + 1} failed: {str(e)[:50]}")
            continue

    # Aggregate performance
    if len(results) == 0:
        return None

    performance_summary = {
        'mean_mae': np.mean([r['mae'] for r in results]),
        'std_mae': np.std([r['mae'] for r in results]),
        'mean_rmse': np.mean([r['rmse'] for r in results]),
        'std_rmse': np.std([r['rmse'] for r in results]),
        'mean_mape': np.mean([r['mape'] for r in results]),
        'std_mape': np.std([r['mape'] for r in results]),
        'mean_directional_accuracy': np.mean([r['directional_accuracy'] for r in results]),
        'std_directional_accuracy': np.std([r['directional_accuracy'] for r in results]),
        'mean_directional_accuracy_from_day0': np.mean([r['directional_accuracy_from_day0'] for r in results]),
        'std_directional_accuracy_from_day0': np.std([r['directional_accuracy_from_day0'] for r in results]),
        'n_windows': len(results),
        'window_mae_range': (min([r['mae'] for r in results]), max([r['mae'] for r in results])),
        'stability_score': 1 / (1 + np.std([r['mae'] for r in results]))  # Higher = more stable
    }

    return {
        'backtest_results': results,
        'performance_summary': performance_summary,
        'model_params': model_params
    }


def compare_backtest_results(backtest_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple models' backtest performance.

    Args:
        backtest_dict: Dict mapping model_name -> backtest results

    Returns:
        DataFrame with comparative metrics
    """
    comparison = []

    for model_name, backtest_result in backtest_dict.items():
        if backtest_result is None:
            continue

        summary = backtest_result['performance_summary']

        comparison.append({
            'model': model_name,
            'mean_mae': summary['mean_mae'],
            'std_mae': summary['std_mae'],
            'mean_rmse': summary['mean_rmse'],
            'std_rmse': summary['std_rmse'],
            'mean_mape': summary['mean_mape'],
            'n_windows': summary['n_windows'],
            'mae_range_min': summary['window_mae_range'][0],
            'mae_range_max': summary['window_mae_range'][1],
            'stability_score': summary['stability_score']
        })

    return pd.DataFrame(comparison).sort_values('mean_mae')

"""Baseline model experiment runner.

Trains all baseline models, generates forecasts, and compares performance.
Can run locally (pandas) or in Databricks (PySpark).
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add ground_truth to path
sys.path.insert(0, os.path.dirname(__file__))

from ground_truth.config.model_registry import (
    BASELINE_MODELS, get_commodity_config, print_model_summary
)
from ground_truth.core import evaluator


def run_experiment_local(commodity: str = 'Coffee', cutoff_date: str = '2023-12-31'):
    """
    Run baseline experiment locally with pandas.

    Args:
        commodity: 'Coffee' or 'Sugar'
        cutoff_date: Training cutoff date for backtesting

    Outputs:
        - Forecast CSVs for each model
        - Performance comparison table
    """
    print("="*60)
    print(f"BASELINE MODEL EXPERIMENT - {commodity}")
    print("="*60)
    print(f"Cutoff date: {cutoff_date}")
    print(f"Forecast horizon: 14 days")
    print()

    # Load data
    print("1. Loading data...")
    data_path = "../data/unified_data_snapshot_all.parquet"
    df = pd.read_parquet(data_path)

    # Filter and aggregate (same as test script)
    df_filtered = df[df['commodity'] == commodity].copy()
    df_agg = df_filtered.groupby(['date', 'commodity']).agg({
        'close': 'first',
        'temp_c': 'mean',
        'humidity_pct': 'mean',
        'precipitation_mm': 'mean'
    }).reset_index()

    df_agg['date'] = pd.to_datetime(df_agg['date'])
    df_agg = df_agg.set_index('date').sort_index()

    # Split train/test
    df_train = df_agg[df_agg.index <= cutoff_date]
    df_test = df_agg[(df_agg.index > cutoff_date) & (df_agg.index <= '2024-01-14')]

    print(f"   Training: {len(df_train)} days ({df_train.index[0]} to {df_train.index[-1]})")
    print(f"   Test: {len(df_test)} days ({df_test.index[0]} to {df_test.index[-1]})")
    print()

    # Train all models
    print("2. Training models...")
    results = {}

    for model_key, config in BASELINE_MODELS.items():
        try:
            print(f"\n   Training: {config['name']}")

            # Get forecast function and params
            forecast_fn = config['function']
            params = config['params'].copy()

            # Add commodity and cutoff_date
            params['commodity'] = commodity
            params['cutoff_date'] = cutoff_date

            # Call model
            result = forecast_fn(df_pandas=df_train, **params)

            # Store forecast
            results[model_key] = {
                'config': config,
                'forecast': result['forecast_df'],
                'metadata': result
            }

            print(f"      ✓ Success")
            print(f"      Forecast: {result['forecast_df']['forecast'].iloc[0]:.2f} to "
                  f"{result['forecast_df']['forecast'].iloc[-1]:.2f}")

        except Exception as e:
            print(f"      ✗ Failed: {e}")
            results[model_key] = None

    # Evaluate against actuals
    print("\n" + "="*60)
    print("3. Evaluating against actuals")
    print("="*60)

    actuals_df = df_test[['close']].reset_index()
    actuals_df.columns = ['date', 'close']

    performance = []

    for model_key, result in results.items():
        if result is None:
            continue

        try:
            forecast_df = result['forecast'].copy()

            # Evaluate
            eval_result = evaluator.evaluate_forecast(
                actuals_df, forecast_df, result['config']['name']
            )

            # Store performance
            metrics = eval_result['metrics']
            direction = eval_result['direction_test']

            performance.append({
                'model': result['config']['name'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'mape': metrics['mape'],
                'directional_accuracy': metrics.get('directional_accuracy', None),
                'direction_significant': direction.get('is_significant', None)
            })

        except Exception as e:
            print(f"\n✗ {result['config']['name']}: Evaluation failed - {e}")

    # Display results
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)

    perf_df = pd.DataFrame(performance)
    perf_df = perf_df.sort_values('mae')  # Sort by MAE (best first)

    print(perf_df.to_string(index=False))
    print()

    # Highlight best model
    best_model = perf_df.iloc[0]
    print(f"\n🏆 Best Model: {best_model['model']}")
    print(f"   MAE: ${best_model['mae']:.2f}")
    print(f"   RMSE: ${best_model['rmse']:.2f}")
    print(f"   MAPE: {best_model['mape']:.2f}%")
    if pd.notna(best_model['directional_accuracy']):
        print(f"   Directional Accuracy: {best_model['directional_accuracy']:.1f}%")

    # Statistical comparison: best vs naive
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)

    if 'naive' in results and results['naive'] is not None:
        naive_forecast = results['naive']['forecast']
        best_key = perf_df.iloc[0]['model']

        # Find best model forecast
        best_forecast = None
        for key, result in results.items():
            if result is not None and result['config']['name'] == best_key:
                best_forecast = result['forecast']
                break

        if best_forecast is not None:
            # Merge with actuals
            merged = actuals_df.merge(naive_forecast[['date', 'forecast']], on='date')
            merged = merged.merge(best_forecast[['date', 'forecast']], on='date', suffixes=('_naive', '_best'))

            naive_errors = (merged['close'] - merged['forecast_naive']).abs()
            best_errors = (merged['close'] - merged['forecast_best']).abs()

            # T-test
            t_result = evaluator.t_test_comparison(best_errors, naive_errors)

            print(f"\nT-test: {best_key} vs Naive")
            print(f"  {best_key} MAE: ${t_result['mean_error_model1']:.2f}")
            print(f"  Naive MAE: ${t_result['mean_error_model2']:.2f}")
            print(f"  p-value: {t_result['p_value']:.4f}")
            print(f"  Significant: {t_result['is_significant']}")

            if t_result['is_significant']:
                print(f"  ✓ {t_result['better_model']} is significantly better")
            else:
                print(f"  ✗ No significant difference")

            # Diebold-Mariano test
            naive_errors_signed = merged['close'] - merged['forecast_naive']
            best_errors_signed = merged['close'] - merged['forecast_best']

            dm_result = evaluator.diebold_mariano_test(best_errors_signed, naive_errors_signed)

            print(f"\nDiebold-Mariano test: {best_key} vs Naive")
            print(f"  DM statistic: {dm_result['dm_statistic']:.4f}")
            print(f"  p-value: {dm_result['p_value']:.4f}")
            print(f"  Significant: {dm_result['is_significant']}")

            if dm_result['is_significant']:
                print(f"  ✓ {dm_result['better_model']} is significantly better")

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save performance comparison
    perf_df.to_csv(f"results/baseline_performance_{commodity}_{timestamp}.csv", index=False)
    print(f"✓ Performance saved to: results/baseline_performance_{commodity}_{timestamp}.csv")

    # Save individual forecasts
    for model_key, result in results.items():
        if result is not None:
            forecast_df = result['forecast']
            forecast_df.to_csv(
                f"results/forecast_{model_key}_{commodity}_{timestamp}.csv",
                index=False
            )

    print(f"✓ Forecasts saved to: results/forecast_*_{commodity}_{timestamp}.csv")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)


def main():
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Print model registry
    print_model_summary()
    print()

    # Run experiment for Coffee
    run_experiment_local(commodity='Coffee', cutoff_date='2023-12-31')

    # Optionally run for Sugar
    # run_experiment_local(commodity='Sugar', cutoff_date='2023-12-31')


if __name__ == "__main__":
    main()

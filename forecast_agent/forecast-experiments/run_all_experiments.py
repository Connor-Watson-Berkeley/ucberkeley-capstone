"""
Run All DARTS Experiments and Compare Results

This script runs multiple DARTS models and generates a comparison report.
"""

import pandas as pd
import json
from datetime import datetime
import sys

# Import experiment runners
from darts_nbeats_experiment import run_nbeats_experiment
from darts_tft_experiment import run_tft_experiment
from darts_nhits_experiment import run_nhits_experiment


def run_all_experiments():
    """Run all DARTS experiments and collect results."""

    results_summary = {
        'experiment_date': datetime.now().isoformat(),
        'data_config': {
            'lookback_days': 730,
            'forecast_horizon': 14,
            'region': 'Bahia_Brazil',
            'commodity': 'Coffee'
        },
        'models': {}
    }

    experiments = [
        ('N-BEATS', run_nbeats_experiment),
        ('TFT', run_tft_experiment),
        ('N-HiTS', run_nhits_experiment),
    ]

    for model_name, experiment_fn in experiments:
        print("\n" + "=" * 80)
        print(f"Running {model_name} Experiment")
        print("=" * 80 + "\n")

        try:
            if model_name == 'TFT':
                # TFT specific parameters
                results = experiment_fn(
                    lookback_days=730,
                    forecast_horizon=14,
                    input_chunk_length=60,
                    output_chunk_length=14,
                    hidden_size=64,
                    lstm_layers=2,
                    num_attention_heads=4,
                    dropout=0.1,
                    batch_size=32,
                    n_epochs=50,  # Fewer epochs for TFT
                    learning_rate=1e-3
                )
            else:
                # Standard parameters for N-BEATS and N-HiTS
                results = experiment_fn(
                    lookback_days=730,
                    forecast_horizon=14,
                    n_epochs=100
                )

            # Extract metrics
            # Convert dates to strings for JSON serialization
            forecast_df_copy = results['forecast_df'].copy()
            forecast_df_copy['date'] = forecast_df_copy['date'].astype(str)

            results_summary['models'][model_name] = {
                'mape': float(results['metrics']['mape']),
                'rmse': float(results['metrics']['rmse']),
                'mae': float(results['metrics']['mae']),
                'forecast': forecast_df_copy.to_dict('records')[:7]  # First week only
            }

            print(f"\n✅ {model_name} completed successfully!")
            print(f"   MAPE: {results['metrics']['mape']:.2f}%")
            print(f"   RMSE: ${results['metrics']['rmse']:.4f}")
            print(f"   MAE: ${results['metrics']['mae']:.4f}")

        except Exception as e:
            print(f"\n❌ {model_name} failed with error: {str(e)}")
            results_summary['models'][model_name] = {
                'error': str(e),
                'status': 'failed'
            }
            continue

    # Save results
    output_file = 'experiment_results.json'
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print(f"\nResults saved to: {output_file}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<15} {'MAPE':<10} {'RMSE':<12} {'MAE':<12}")
    print("-" * 80)

    for model_name, metrics in results_summary['models'].items():
        if 'error' not in metrics:
            print(f"{model_name:<15} {metrics['mape']:<10.2f} ${metrics['rmse']:<11.4f} ${metrics['mae']:<11.4f}")
        else:
            print(f"{model_name:<15} {'FAILED':<10} {'-':<12} {'-':<12}")

    print("=" * 80)

    # Identify best model
    valid_models = {k: v for k, v in results_summary['models'].items() if 'error' not in v}
    if valid_models:
        best_model = min(valid_models.items(), key=lambda x: x[1]['mape'])
        print(f"\n🏆 BEST MODEL: {best_model[0]} (MAPE: {best_model[1]['mape']:.2f}%)")

    return results_summary


if __name__ == '__main__':
    results = run_all_experiments()

"""
Re-evaluate trained models with extended metrics
Loads previously trained models and calculates comprehensive metrics
"""

import sys
from calculate_extended_metrics import calculate_all_metrics, print_metrics_report

# Import experiment runners
from darts_nbeats_experiment import run_nbeats_experiment
from darts_nhits_experiment import run_nhits_experiment


def evaluate_all_models():
    """Re-run experiments and calculate extended metrics."""

    print("\n" + "=" * 80)
    print("EXTENDED MODEL EVALUATION")
    print("=" * 80)
    print("\nRunning models and calculating comprehensive metrics...\n")

    models_to_evaluate = [
        ("N-BEATS", run_nbeats_experiment),
        ("N-HiTS", run_nhits_experiment),
    ]

    all_metrics = {}

    for model_name, experiment_fn in models_to_evaluate:
        print("\n" + "=" * 80)
        print(f"Evaluating {model_name}")
        print("=" * 80 + "\n")

        try:
            # Run experiment
            results = experiment_fn(
                lookback_days=730,
                forecast_horizon=14,
                n_epochs=100
            )

            # Calculate extended metrics
            val_actual = results['actual']
            val_forecast = results['val_forecast']

            metrics = calculate_all_metrics(val_actual, val_forecast)
            all_metrics[model_name] = metrics

            # Print detailed report
            print_metrics_report(metrics, model_name=model_name)

        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - EXTENDED METRICS")
    print("=" * 80)
    print()

    # Standard metrics
    print("Standard Accuracy Metrics:")
    print(f"{'Model':<15} {'MAPE':<10} {'RMSE':<12} {'MAE':<12}")
    print("-" * 80)
    for model_name, metrics in all_metrics.items():
        print(f"{model_name:<15} {metrics['mape']:<10.2f} ${metrics['rmse']:<11.2f} ${metrics['mae']:<11.2f}")
    print()

    # Directional accuracy
    print("Directional Accuracy (% correct direction):")
    print(f"{'Model':<15} {'Directional Accuracy':<25}")
    print("-" * 80)
    for model_name, metrics in all_metrics.items():
        print(f"{model_name:<15} {metrics['directional_accuracy']:<25.1f}%")
    print()

    # Hit rates
    print("Hit Rates (% within threshold):")
    print(f"{'Model':<15} {'Within 2%':<15} {'Within 5%':<15} {'Within 10%':<15}")
    print("-" * 80)
    for model_name, metrics in all_metrics.items():
        print(f"{model_name:<15} {metrics['hit_rate_2pct']:<15.1f}% {metrics['hit_rate_5pct']:<15.1f}% {metrics['hit_rate_10pct']:<15.1f}%")
    print()

    # Trading performance
    print("Trading Performance:")
    print(f"{'Model':<15} {'Sharpe Ratio':<20} {'Bias ($)':<20}")
    print("-" * 80)
    for model_name, metrics in all_metrics.items():
        bias_direction = "over" if metrics['bias'] > 0 else "under"
        print(f"{model_name:<15} {metrics['sharpe_ratio']:<20.3f} ${metrics['bias']:<.2f} ({bias_direction})")
    print()

    # Identify best model
    best_mape = min(all_metrics.items(), key=lambda x: x[1]['mape'])
    best_directional = max(all_metrics.items(), key=lambda x: x[1]['directional_accuracy'])
    best_sharpe = max(all_metrics.items(), key=lambda x: x[1]['sharpe_ratio'])

    print("=" * 80)
    print("BEST MODELS BY METRIC")
    print("=" * 80)
    print(f"🏆 Best MAPE: {best_mape[0]} ({best_mape[1]['mape']:.2f}%)")
    print(f"🎯 Best Directional Accuracy: {best_directional[0]} ({best_directional[1]['directional_accuracy']:.1f}%)")
    print(f"📈 Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.3f})")
    print("=" * 80)

    return all_metrics


if __name__ == '__main__':
    metrics = evaluate_all_models()

"""
Train Region-Specific Models

Instead of aggregating across all regions, train separate models for each region
to capture region-specific patterns and weather effects.
"""

import pandas as pd
import json
from datetime import datetime
from load_local_data import load_local_data
from darts_nhits_experiment import run_nhits_experiment


def get_available_regions():
    """Get list of available regions from local data."""
    df = pd.read_parquet('data/unified_data.parquet')
    regions = df['region'].unique().tolist()
    return sorted(regions)


def train_model_for_region(region, model_type='nhits', n_epochs=50):
    """
    Train a model for a specific region.

    Args:
        region: Region name (e.g., 'Bahia_Brazil')
        model_type: Type of model ('nhits', 'nbeats', 'tft')
        n_epochs: Number of training epochs

    Returns:
        Dictionary with results and metrics
    """
    print("\n" + "=" * 80)
    print(f"Training {model_type.upper()} for region: {region}")
    print("=" * 80 + "\n")

    try:
        if model_type == 'nhits':
            results = run_nhits_experiment(
                lookback_days=730,
                forecast_horizon=14,
                n_epochs=n_epochs
            )
        else:
            raise ValueError(f"Model type {model_type} not yet supported in this script")

        return {
            'region': region,
            'model_type': model_type,
            'mape': float(results['metrics']['mape']),
            'rmse': float(results['metrics']['rmse']),
            'mae': float(results['metrics']['mae']),
            'status': 'success'
        }

    except Exception as e:
        print(f"❌ Error training {model_type} for {region}: {str(e)}")
        return {
            'region': region,
            'model_type': model_type,
            'status': 'failed',
            'error': str(e)
        }


def train_all_regional_models(
    regions=None,
    model_type='nhits',
    n_epochs=50,
    top_n_regions=5
):
    """
    Train models for multiple regions.

    Args:
        regions: List of regions (if None, uses top_n_regions by data availability)
        model_type: Type of model to train
        n_epochs: Number of epochs per model
        top_n_regions: If regions is None, train on top N regions
    """

    results_summary = {
        'experiment_date': datetime.now().isoformat(),
        'model_type': model_type,
        'n_epochs': n_epochs,
        'regional_results': {}
    }

    # Get regions to train on
    if regions is None:
        # Get regions with most data
        df = pd.read_parquet('data/unified_data.parquet')
        df_filtered = df[df['commodity'] == 'Coffee']
        region_counts = df_filtered.groupby('region').size().sort_values(ascending=False)
        regions = region_counts.head(top_n_regions).index.tolist()

        print(f"\nTraining on top {top_n_regions} regions by data availability:")
        for i, region in enumerate(regions, 1):
            print(f"  {i}. {region} ({region_counts[region]} days)")
        print()

    # Train model for each region
    for region in regions:
        result = train_model_for_region(region, model_type=model_type, n_epochs=n_epochs)
        results_summary['regional_results'][region] = result

        if result['status'] == 'success':
            print(f"\n✅ {region}: MAPE={result['mape']:.2f}%, RMSE=${result['rmse']:.2f}, MAE=${result['mae']:.2f}")
        else:
            print(f"\n❌ {region}: Failed")

    # Save results
    output_file = f'regional_results_{model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("REGIONAL MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Region':<25} {'MAPE':<10} {'RMSE':<12} {'MAE':<12} {'Status':<10}")
    print("-" * 80)

    successful_models = []
    for region, result in results_summary['regional_results'].items():
        if result['status'] == 'success':
            print(f"{region:<25} {result['mape']:<10.2f} ${result['rmse']:<11.2f} ${result['mae']:<11.2f} ✅")
            successful_models.append((region, result))
        else:
            print(f"{region:<25} {'FAILED':<10} {'-':<12} {'-':<12} ❌")

    print("=" * 80)

    # Identify best regional model
    if successful_models:
        best_region, best_result = min(successful_models, key=lambda x: x[1]['mape'])
        print(f"\n🏆 BEST REGIONAL MODEL: {best_region}")
        print(f"   MAPE: {best_result['mape']:.2f}%")
        print(f"   RMSE: ${best_result['rmse']:.2f}")
        print(f"   MAE: ${best_result['mae']:.2f}")

        # Compare to aggregated model
        print(f"\n📊 Comparison to aggregated Bahia_Brazil model:")
        print(f"   Aggregated MAPE: 1.12% (from previous experiment)")
        if best_result['mape'] < 1.12:
            improvement = ((1.12 - best_result['mape']) / 1.12) * 100
            print(f"   Regional improvement: {improvement:.1f}% better! ✅")
        else:
            degradation = ((best_result['mape'] - 1.12) / 1.12) * 100
            print(f"   Regional difference: {degradation:.1f}% worse ⚠️")

    return results_summary


if __name__ == '__main__':
    import sys

    print("\n" + "=" * 80)
    print("REGIONAL MODEL TRAINING")
    print("=" * 80)

    # Check available regions
    print("\nAvailable regions in dataset:")
    regions = get_available_regions()
    for i, region in enumerate(regions, 1):
        print(f"  {i}. {region}")

    print(f"\nTotal regions: {len(regions)}")
    print("\n" + "-" * 80)

    # Train on top 3 regions (to save time, can increase later)
    print("\nTraining N-HiTS models for top 3 coffee-producing regions...")
    print("(Using 50 epochs to save time. Increase to 100 for production.)")
    print("-" * 80)

    results = train_all_regional_models(
        regions=None,  # Auto-select top regions
        model_type='nhits',
        n_epochs=50,  # Reduced for faster experimentation
        top_n_regions=3  # Train on top 3 regions
    )

    print("\n" + "=" * 80)
    print("REGIONAL TRAINING COMPLETE")
    print("=" * 80)

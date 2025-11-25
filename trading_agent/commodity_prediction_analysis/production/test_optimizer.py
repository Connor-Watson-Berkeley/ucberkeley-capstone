#!/usr/bin/env python3
"""
Test script for parameter optimizer

Tests:
1. Loading predictions and prices
2. Running Optuna optimization
3. Generating optimized parameters
4. Saving results

Uses minimal configuration for fast testing:
- Single strategy (consensus)
- 5 trials only
- Coffee + synthetic_acc90 (known good data)
"""
import sys
from pathlib import Path

# Add parent directory to path
try:
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir.parent))
except NameError:
    # __file__ not defined in Databricks jobs
    sys.path.insert(0, '/Workspace/Repos/Project_Git/ucberkeley-capstone/trading_agent/commodity_prediction_analysis')

from pyspark.sql import SparkSession
from analysis.optimization.run_parameter_optimization import run_optimization

def test_optimizer(commodity='coffee', model_version='synthetic_acc90'):
    """Test optimizer with minimal configuration."""

    print(f"\n{'='*80}")
    print(f"TESTING PARAMETER OPTIMIZER - {commodity.upper()}")
    print(f"{'='*80}\n")

    # Get Spark session
    try:
        spark = SparkSession.builder.getOrCreate()
        print(f"✓ Got Spark session")
    except Exception as e:
        print(f"❌ ERROR getting Spark session: {e}")
        return False

    # Run optimization with minimal configuration
    print(f"\nRunning optimization with minimal configuration...")
    print(f"  Commodity: {commodity}")
    print(f"  Model: {model_version}")
    print(f"  Strategy: consensus (single strategy for testing)")
    print(f"  Trials: 5 (quick test)")
    print(f"  Objective: efficiency")

    try:
        results = run_optimization(
            commodity=commodity,
            model_version=model_version,
            objective='efficiency',
            n_trials=5,  # Minimal for testing
            strategy_filter=['consensus'],  # Single strategy for speed
            spark=spark
        )

        print(f"\n✅ Optimization completed successfully")
        print(f"\nResults summary:")
        if results and 'best_params' in results:
            for strategy_name, params in results['best_params'].items():
                print(f"  {strategy_name}:")
                for param, value in params.items():
                    print(f"    {param}: {value}")

        return True

    except Exception as e:
        print(f"\n❌ ERROR during optimization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # Test with coffee and synthetic_acc90
    success = test_optimizer('coffee', 'synthetic_acc90')

    if success:
        print(f"\n{'='*80}")
        print("✅ OPTIMIZER TEST COMPLETE")
        print(f"{'='*80}\n")
        sys.exit(0)
    else:
        print(f"\n{'='*80}")
        print("❌ OPTIMIZER TEST FAILED")
        print(f"{'='*80}\n")
        sys.exit(1)

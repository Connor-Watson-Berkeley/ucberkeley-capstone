"""
Parameter Optimization Orchestrator

Modern parameter optimization for trading strategies using Optuna.

Migrated from diagnostics/run_diagnostic_16.py with enhancements:
- Efficiency-aware optimization (optimize for efficiency ratio, not just earnings)
- Integration with theoretical maximum benchmark
- Uses production config and data loaders
- Multi-objective optimization support
- Clean, modular architecture

Usage:
    # Optimize all strategies for efficiency
    python analysis/optimization/run_parameter_optimization.py \\
        --commodity coffee --objective efficiency --trials 200

    # Optimize single strategy for raw earnings
    python analysis/optimization/run_parameter_optimization.py \\
        --commodity coffee --strategy consensus --objective earnings --trials 200

    # Multi-objective optimization
    python analysis/optimization/run_parameter_optimization.py \\
        --commodity coffee --objective multi --trials 500
"""

import sys
import os
from pathlib import Path
import argparse
import pandas as pd
import pickle
from datetime import datetime

# Add parent directories to path
try:
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir.parent.parent))
except NameError:
    # __file__ not defined in Databricks jobs - use hardcoded path
    sys.path.insert(0, '/Workspace/Repos/Project_Git/ucberkeley-capstone/trading_agent/commodity_prediction_analysis')

# Production imports
from production.config import COMMODITY_CONFIGS, VOLUME_PATH
from analysis.theoretical_max import TheoreticalMaxCalculator
from analysis.optimization.optimizer import ParameterOptimizer
from analysis.optimization.search_space import SearchSpaceRegistry


def load_data(spark, commodity, model_version='synthetic_acc90'):
    """
    Load price data and predictions.

    Args:
        spark: SparkSession
        commodity: str (e.g., 'coffee')
        model_version: str (default: 'synthetic_acc90')

    Returns:
        Tuple of (prices_df, prediction_matrices)
    """
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Load price data
    print(f"\n1. Loading price data for {commodity}...")
    market_df = spark.table("commodity.bronze.market").filter(
        f"lower(commodity) = '{commodity}'"
    ).toPandas()

    market_df['date'] = pd.to_datetime(market_df['date']).dt.normalize()
    market_df['price'] = market_df['close']
    prices = market_df[['date', 'price']].sort_values('date').reset_index(drop=True)
    prices = prices[prices['date'] >= '2022-01-01'].reset_index(drop=True)

    print(f"   ✓ Loaded {len(prices)} price points from {prices['date'].min()} to {prices['date'].max()}")

    # Load predictions
    print(f"\n2. Loading predictions for {commodity} - {model_version}...")
    pred_table = f"commodity.trading_agent.predictions_{commodity}"
    pred_df = spark.table(pred_table).filter(
        f"model_version = '{model_version}'"
    ).toPandas()

    # Convert to matrix format
    pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
    prediction_matrices = {}

    for timestamp in pred_df['timestamp'].unique():
        ts_data = pred_df[pred_df['timestamp'] == timestamp]
        matrix = ts_data.pivot_table(
            index='run_id',
            columns='day_ahead',
            values='predicted_price',
            aggfunc='first'
        ).values
        date_key = pd.Timestamp(timestamp).normalize()
        prediction_matrices[date_key] = matrix

    print(f"   ✓ Loaded {len(prediction_matrices)} prediction matrices")

    return prices, prediction_matrices


def calculate_theoretical_max(prices, predictions, config):
    """
    Calculate theoretical maximum earnings.

    Args:
        prices: DataFrame with price data
        predictions: Dict of prediction matrices
        config: Commodity config dict

    Returns:
        float: Theoretical maximum net earnings
    """
    print("\n3. Calculating theoretical maximum...")

    calculator = TheoreticalMaxCalculator(
        prices_df=prices,
        predictions=predictions,
        config={
            'storage_cost_pct_per_day': config['storage_cost_pct_per_day'],
            'transaction_cost_pct': config['transaction_cost_pct']
        }
    )

    optimal_result = calculator.calculate_optimal_policy(
        initial_inventory=config['harvest_volume']
    )

    theoretical_max = optimal_result['total_net_earnings']
    print(f"   ✓ Theoretical maximum: ${theoretical_max:,.2f}")

    return theoretical_max


def get_strategy_classes():
    """
    Import and return all strategy classes.

    Returns:
        List of (strategy_class, strategy_name) tuples
    """
    # Import strategies from production
    from production.strategies import baseline, prediction

    strategies = [
        (baseline.ImmediateSaleStrategy, 'immediate_sale'),
        (baseline.EqualBatchStrategy, 'equal_batch'),
        (baseline.PriceThresholdStrategy, 'price_threshold'),
        (baseline.MovingAverageStrategy, 'moving_average'),
        (prediction.PriceThresholdPredictive, 'price_threshold_predictive'),
        (prediction.MovingAveragePredictive, 'moving_average_predictive'),
        (prediction.ExpectedValueStrategy, 'expected_value'),
        (prediction.ConsensusStrategy, 'consensus'),
        (prediction.RiskAdjustedStrategy, 'risk_adjusted')
    ]

    return strategies


def run_optimization(
    commodity,
    model_version='arima_v1',
    objective='efficiency',
    n_trials=200,
    strategy_filter=None,
    spark=None
):
    """
    Run parameter optimization workflow.

    Args:
        commodity: str (e.g., 'coffee')
        model_version: str (default: 'arima_v1')
        objective: 'earnings', 'efficiency', or 'multi'
        n_trials: Number of Optuna trials per strategy
        strategy_filter: Optional list of strategy names to optimize
        spark: SparkSession

    Returns:
        Dict with optimization results
    """
    start_time = datetime.now()

    print("=" * 80)
    print("PARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Started: {start_time}")
    print(f"Commodity: {commodity}")
    print(f"Model Version: {model_version}")
    print(f"Objective: {objective}")
    print(f"Trials per Strategy: {n_trials}")
    print("=" * 80)

    # Get commodity config
    if commodity not in COMMODITY_CONFIGS:
        raise ValueError(f"Unknown commodity: {commodity}. Available: {list(COMMODITY_CONFIGS.keys())}")

    config = COMMODITY_CONFIGS[commodity]

    # Load data
    prices, predictions = load_data(spark, commodity, model_version)

    # Calculate theoretical max if using efficiency objective
    theoretical_max = None
    if objective in ['efficiency', 'multi']:
        theoretical_max = calculate_theoretical_max(prices, predictions, config)

    # Initialize optimizer
    print("\n" + "=" * 80)
    print("INITIALIZING OPTIMIZER")
    print("=" * 80)

    # Pass full commodity config (required for production BacktestEngine)
    optimizer = ParameterOptimizer(
        prices_df=prices,
        prediction_matrices=predictions,
        config=config,  # Full config with harvest_volume, harvest_windows, etc.
        theoretical_max_earnings=theoretical_max,
        use_production_engine=True  # Use production engine for accurate results
    )

    print("✓ Optimizer initialized")

    # Get strategies to optimize
    all_strategies = get_strategy_classes()

    if strategy_filter:
        strategies = [(cls, name) for cls, name in all_strategies if name in strategy_filter]
        print(f"✓ Filtering to {len(strategies)} strategies: {strategy_filter}")
    else:
        strategies = all_strategies
        print(f"✓ Optimizing all {len(strategies)} strategies")

    # Run optimization
    print("\n" + "=" * 80)
    print("RUNNING OPTIMIZATION")
    print("=" * 80)

    results = optimizer.optimize_all_strategies(
        strategies=strategies,
        n_trials=n_trials,
        objective=objective
    )

    # Prepare parameters for saving
    best_params = {name: params for name, (params, value) in results.items()}

    # Add cost parameters to predictive strategies
    for strategy in ['price_threshold_predictive', 'moving_average_predictive',
                     'expected_value', 'consensus', 'risk_adjusted']:
        if strategy in best_params:
            best_params[strategy]['storage_cost_pct_per_day'] = config['storage_cost_pct_per_day']
            best_params[strategy]['transaction_cost_pct'] = config['transaction_cost_pct']

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    output_dir = f"{VOLUME_PATH}/optimization"
    os.makedirs(output_dir, exist_ok=True)

    # Save parameters pickle
    params_file = f"{output_dir}/optimized_params_{commodity}_{model_version}_{objective}.pkl"
    with open(params_file, 'wb') as f:
        pickle.dump(best_params, f)
    print(f"✓ Saved parameters to: {params_file}")

    # Save full results
    results_data = {
        'execution_time': datetime.now(),
        'commodity': commodity,
        'model_version': model_version,
        'objective': objective,
        'n_trials': n_trials,
        'theoretical_max': theoretical_max,
        'results': results,
        'best_params': best_params
    }

    results_file = f"{output_dir}/optimization_results_{commodity}_{model_version}_{objective}.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"✓ Saved full results to: {results_file}")

    # Save CSV summary
    csv_file = f"{output_dir}/optimization_summary_{commodity}_{model_version}_{objective}.csv"
    summary_rows = []
    for name, (params, value) in results.items():
        summary_rows.append({
            'strategy_name': name,
            'best_value': value,
            'num_params': len(params),
            'objective': objective
        })
    summary_df = pd.DataFrame(summary_rows).sort_values('best_value', ascending=False)
    summary_df.to_csv(csv_file, index=False)
    print(f"✓ Saved summary to: {csv_file}")

    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Completed: {end_time}")
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Strategies optimized: {len(results)}")
    print(f"Total trials: {len(results) * n_trials}")

    if objective == 'efficiency' and theoretical_max:
        print(f"\nTheoretical Maximum: ${theoretical_max:,.2f}")
        print("\nTop 3 by Efficiency:")
        for name, (params, value) in sorted(results.items(), key=lambda x: x[1][1], reverse=True)[:3]:
            efficiency_pct = (value * 100) if objective == 'efficiency' else ((value / theoretical_max) * 100 if theoretical_max > 0 else 0)
            print(f"  {name:35s}: {efficiency_pct:5.1f}%")
    else:
        print("\nTop 3 by Earnings:")
        for name, (params, value) in sorted(results.items(), key=lambda x: x[1][1], reverse=True)[:3]:
            print(f"  {name:35s}: ${value:,.2f}")

    print("=" * 80)

    return {
        'results': results,
        'theoretical_max': theoretical_max,
        'duration_seconds': duration
    }


def main():
    parser = argparse.ArgumentParser(
        description='Optimize trading strategy parameters using Optuna'
    )
    parser.add_argument(
        '--commodity',
        type=str,
        required=True,
        help='Commodity to optimize (e.g., coffee, sugar)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='synthetic_acc90',
        help='Model version (default: synthetic_acc90)'
    )
    parser.add_argument(
        '--objective',
        type=str,
        choices=['earnings', 'efficiency', 'multi'],
        default='efficiency',
        help='Optimization objective (default: efficiency)'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=200,
        help='Number of trials per strategy (default: 200)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        help='Optimize single strategy only (e.g., consensus)'
    )
    parser.add_argument(
        '--strategies',
        type=str,
        help='Comma-separated list of strategies to optimize'
    )

    args = parser.parse_args()

    # Parse strategy filter
    strategy_filter = None
    if args.strategies:
        strategy_filter = [s.strip() for s in args.strategies.split(',')]
    elif args.strategy:
        strategy_filter = [args.strategy]

    # Initialize Spark
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("ParameterOptimization").getOrCreate()
        print("✓ Spark session initialized")
    except Exception as e:
        print(f"Error initializing Spark: {e}")
        return 1

    # Run optimization
    try:
        result = run_optimization(
            commodity=args.commodity,
            model_version=args.model,
            objective=args.objective,
            n_trials=args.trials,
            strategy_filter=strategy_filter,
            spark=spark
        )
        return 0
    except Exception as e:
        print(f"\n✗ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

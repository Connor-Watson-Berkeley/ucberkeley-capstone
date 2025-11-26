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
from pyspark.sql import functions as F

# Add parent directories to path
try:
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir.parent.parent))
except NameError:
    # __file__ not defined in Databricks jobs - use hardcoded path
    sys.path.insert(0, '/Workspace/Repos/Project_Git/ucberkeley-capstone/trading_agent/commodity_prediction_analysis')

# Production imports
from production.config import COMMODITY_CONFIGS, VOLUME_PATH
# NOTE: TheoreticalMaxCalculator removed - using PerfectForesightStrategy instead
# (old DP calculator didn't account for harvest cycles, produced wrong baselines)
from analysis.optimization.optimizer import ParameterOptimizer
from analysis.optimization.search_space import SearchSpaceRegistry


def load_data(spark, commodity, model_version='synthetic_acc90'):
    """
    Load price data and predictions with matched date ranges.

    CRITICAL: Ensures price dates and prediction dates align for valid backtesting.
    Both theoretical max and actual strategies must use the same time period.

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

    # Load ALL price data from unified_data (continuous daily coverage, forward-filled)
    # unified_data grain is (date, commodity, region) but price is same across regions
    # So aggregate by date to get one row per date
    print(f"\n1. Loading price data for {commodity}...")
    all_prices = spark.table("commodity.silver.unified_data").filter(
        f"lower(commodity) = '{commodity}'"
    ).groupBy("date").agg(
        F.first("close").alias("price")  # Price is same across regions
    ).toPandas()

    all_prices['date'] = pd.to_datetime(all_prices['date']).dt.normalize()
    all_prices = all_prices.sort_values('date').reset_index(drop=True)

    print(f"   ✓ Loaded {len(all_prices)} total price points")
    print(f"   Full price range: {all_prices['date'].min()} to {all_prices['date'].max()}")

    # Load ALL predictions for this model (no hardcoded date filter)
    print(f"\n2. Loading predictions for {commodity} - {model_version}...")
    pred_table = f"commodity.trading_agent.predictions_{commodity}"
    pred_df = spark.table(pred_table).filter(
        f"model_version = '{model_version}'"
    ).toPandas()

    # Convert to matrix format
    pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp']).dt.normalize()
    all_prediction_matrices = {}

    for timestamp in pred_df['timestamp'].unique():
        ts_data = pred_df[pred_df['timestamp'] == timestamp]
        matrix = ts_data.pivot_table(
            index='run_id',
            columns='day_ahead',
            values='predicted_price',
            aggfunc='first'
        ).values
        date_key = pd.Timestamp(timestamp).normalize()
        all_prediction_matrices[date_key] = matrix

    print(f"   ✓ Loaded {len(all_prediction_matrices)} total prediction matrices")
    print(f"   Full prediction range: {min(all_prediction_matrices.keys())} to {max(all_prediction_matrices.keys())}")

    # Find overlap between price dates and prediction dates
    print(f"\n3. Finding common date range...")
    all_price_dates = set(all_prices['date'])
    all_pred_dates = set(all_prediction_matrices.keys())
    common_dates = all_price_dates & all_pred_dates

    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between prices and predictions!")

    # Determine analysis period (intersection of both datasets)
    start_date = min(common_dates)
    end_date = max(common_dates)

    print(f"   All price dates: {len(all_price_dates)} ({all_prices['date'].min()} to {all_prices['date'].max()})")
    print(f"   All prediction dates: {len(all_pred_dates)} ({min(all_prediction_matrices.keys())} to {max(all_prediction_matrices.keys())})")
    print(f"   ✓ Common date range: {start_date} to {end_date}")
    print(f"   ✓ Overlap: {len(common_dates)} dates")

    # Filter both datasets to common range
    prices = all_prices[all_prices['date'].isin(common_dates)].reset_index(drop=True)
    prediction_matrices = {date: matrix for date, matrix in all_prediction_matrices.items() if date in common_dates}

    # Validate sufficient coverage using forecast loader standard
    # Standard: 90%+ coverage of PREDICTION period (not all price history) + 730 day minimum
    pred_coverage_pct = len(common_dates) / len(all_pred_dates) * 100
    pred_only = all_pred_dates - all_price_dates
    price_only = all_price_dates - all_pred_dates

    print(f"\n4. Validating coverage...")
    print(f"   Final dataset: {len(prices)} price points, {len(prediction_matrices)} prediction matrices")
    print(f"   Coverage: {pred_coverage_pct:.1f}% of prediction period ({len(common_dates)}/{len(all_pred_dates)} days)")

    if len(pred_only) > 0:
        print(f"   ℹ️  {len(pred_only)} prediction dates excluded (no corresponding prices)")

    if len(price_only) > 0:
        print(f"   ℹ️  {len(price_only)} price dates excluded (no corresponding predictions)")

    # Apply forecast loader standard: 730 day minimum
    if len(common_dates) < 730:
        raise ValueError(f"Insufficient data: only {len(common_dates)} overlapping days (need 730+ for 2 year minimum)")

    # Apply forecast loader standard: 90%+ coverage of prediction period
    if pred_coverage_pct < 90:
        raise ValueError(f"Sparse predictions: only {pred_coverage_pct:.1f}% coverage of prediction period (need 90%+). Check model_version '{model_version}' data availability.")

    return prices, prediction_matrices


def calculate_theoretical_max(prices, predictions, config):
    """
    Calculate theoretical maximum using Linear Programming (Oracle/Clairvoyant Algorithm).

    This implements the "offline optimal solution" from the academic literature on
    commodity trading with perfect foresight (El-Yaniv et al., "One-Way Trading", 2001).

    Uses Linear Programming to find the globally optimal selling policy:
    - Full visibility: Sees ALL future prices (perfect foresight)
    - Optimal decisions: LP solver finds exact optimal sell quantities/timing
    - Accurate dynamics: Models harvest accumulation, storage costs, transaction costs
    - Global optimum: Guaranteed to find the best possible solution

    This provides the TRUE theoretical maximum as a benchmark for strategy evaluation.

    Args:
        prices: DataFrame with price data (columns: date, price)
        predictions: Dict of prediction matrices (not used - we use actual prices)
        config: Commodity config dict (harvest_volume, harvest_windows, costs, etc.)

    Returns:
        float: Theoretical maximum net earnings
    """
    print("\n3. Calculating theoretical maximum...")
    print("   Using Linear Programming (Oracle/Clairvoyant Algorithm)")
    print("   Based on: El-Yaniv et al., One-Way Trading, Algorithmica 2001")

    # Import LP optimizer and BacktestEngine (for harvest schedule generation)
    from production.strategies.lp_optimizer import solve_optimal_liquidation_lp
    from production.core.backtest_engine import BacktestEngine

    # Generate harvest schedule using BacktestEngine (ensures consistency with actual strategies)
    dummy_engine = BacktestEngine(prices, predictions, config)
    harvest_schedule = dummy_engine._generate_harvest_schedule()

    # Solve LP to get globally optimal solution
    result = solve_optimal_liquidation_lp(
        prices_df=prices,
        harvest_schedule=harvest_schedule,
        storage_cost_pct_per_day=config['storage_cost_pct_per_day'],
        transaction_cost_pct=config['transaction_cost_pct']
    )

    theoretical_max = result['max_net_earnings']
    num_trades = len(result['trades'])

    # Count harvest cycles
    harvest_years = harvest_schedule[harvest_schedule['is_harvest_day']]['harvest_year'].unique()
    num_cycles = len(harvest_years)

    print(f"   ✓ Theoretical maximum (LP optimal): ${theoretical_max:,.2f}")
    print(f"   Harvest cycles: {num_cycles}")
    print(f"   Optimal trades: {num_trades}")
    print(f"   Avg per cycle: ${theoretical_max / num_cycles:,.2f}" if num_cycles > 0 else "")

    return theoretical_max


def get_strategy_classes():
    """
    Import and return all strategy classes.

    Returns:
        List of (strategy_class, strategy_name) tuples
    """
    # Import strategies from production
    from production.strategies import baseline, prediction, rolling_horizon_mpc

    strategies = [
        (baseline.ImmediateSaleStrategy, 'immediate_sale'),
        (baseline.EqualBatchStrategy, 'equal_batch'),
        (baseline.PriceThresholdStrategy, 'price_threshold'),
        (baseline.MovingAverageStrategy, 'moving_average'),
        (prediction.PriceThresholdPredictive, 'price_threshold_predictive'),
        (prediction.MovingAveragePredictive, 'moving_average_predictive'),
        (prediction.ExpectedValueStrategy, 'expected_value'),
        (prediction.ConsensusStrategy, 'consensus'),
        (prediction.RiskAdjustedStrategy, 'risk_adjusted'),
        (rolling_horizon_mpc.RollingHorizonMPC, 'rolling_horizon_mpc')
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

    # Get strategies to optimize
    all_strategies = get_strategy_classes()

    if strategy_filter:
        strategies = [(cls, name) for cls, name in all_strategies if name in strategy_filter]
        print(f"✓ Filtering to {len(strategies)} strategies: {strategy_filter}")
    else:
        strategies = all_strategies
        print(f"✓ Optimizing all {len(strategies)} strategies")

    # Separate base and predictive strategies for two-pass matched pair optimization
    base_strategies = []
    predictive_strategies = []
    other_strategies = []

    matched_pairs = {
        'price_threshold': 'price_threshold_predictive',
        'moving_average': 'moving_average_predictive'
    }

    for cls, name in strategies:
        if name in matched_pairs.keys():
            # Base strategies (price_threshold, moving_average)
            base_strategies.append((cls, name))
        elif name in matched_pairs.values():
            # Predictive strategies (price_threshold_predictive, moving_average_predictive)
            predictive_strategies.append((cls, name))
        else:
            # Other strategies (immediate_sale, equal_batch, expected_value, etc.)
            other_strategies.append((cls, name))

    print(f"✓ Base strategies (Pass 1): {[name for _, name in base_strategies]}")
    print(f"✓ Predictive strategies (Pass 2): {[name for _, name in predictive_strategies]}")
    print(f"✓ Other strategies: {[name for _, name in other_strategies]}")

    # ============================================================================
    # PASS 1: Optimize base strategies (price_threshold, moving_average)
    # ============================================================================
    print("\n" + "=" * 80)
    print("PASS 1: OPTIMIZING BASE STRATEGIES (for matched pairs)")
    print("=" * 80)

    pass1_optimizer = ParameterOptimizer(
        prices_df=prices,
        prediction_matrices=predictions,
        config=config,
        theoretical_max_earnings=theoretical_max,
        use_production_engine=True
    )

    pass1_results = {}
    if base_strategies:
        pass1_results = pass1_optimizer.optimize_all_strategies(
            strategies=base_strategies,
            n_trials=n_trials,
            objective=objective
        )
        print(f"\n✓ Pass 1 complete - optimized {len(pass1_results)} base strategies")
    else:
        print("\n⚠️  No base strategies to optimize in Pass 1")

    # ============================================================================
    # PASS 2: Optimize predictive strategies with FIXED base parameters
    # ============================================================================
    print("\n" + "=" * 80)
    print("PASS 2: OPTIMIZING PREDICTIVE STRATEGIES (with fixed base params)")
    print("=" * 80)

    # Extract best base parameters from Pass 1
    fixed_base_params = {}
    for base_name, predictive_name in matched_pairs.items():
        if base_name in pass1_results:
            best_params, _ = pass1_results[base_name]
            fixed_base_params[base_name] = best_params
            print(f"✓ Fixing {base_name} params for {predictive_name} optimization")

    # Create optimizer with fixed base params
    pass2_optimizer = ParameterOptimizer(
        prices_df=prices,
        prediction_matrices=predictions,
        config=config,
        theoretical_max_earnings=theoretical_max,
        use_production_engine=True,
        fixed_base_params=fixed_base_params  # KEY: Fixed base params for matched pairs
    )

    pass2_results = {}
    if predictive_strategies and fixed_base_params:
        pass2_results = pass2_optimizer.optimize_all_strategies(
            strategies=predictive_strategies,
            n_trials=n_trials,
            objective=objective
        )
        print(f"\n✓ Pass 2 complete - optimized {len(pass2_results)} predictive strategies")
    elif predictive_strategies and not fixed_base_params:
        print("\n⚠️  Skipping predictive strategies - no base params from Pass 1")
    else:
        print("\n⚠️  No predictive strategies to optimize in Pass 2")

    # ============================================================================
    # PASS 3: Optimize other strategies (non-matched pairs)
    # ============================================================================
    print("\n" + "=" * 80)
    print("PASS 3: OPTIMIZING OTHER STRATEGIES")
    print("=" * 80)

    pass3_optimizer = ParameterOptimizer(
        prices_df=prices,
        prediction_matrices=predictions,
        config=config,
        theoretical_max_earnings=theoretical_max,
        use_production_engine=True
    )

    pass3_results = {}
    if other_strategies:
        pass3_results = pass3_optimizer.optimize_all_strategies(
            strategies=other_strategies,
            n_trials=n_trials,
            objective=objective
        )
        print(f"\n✓ Pass 3 complete - optimized {len(pass3_results)} other strategies")
    else:
        print("\n⚠️  No other strategies to optimize in Pass 3")

    # Combine all results
    results = {**pass1_results, **pass2_results, **pass3_results}
    print(f"\n✓ Total strategies optimized: {len(results)}")

    # Prepare parameters for saving
    best_params = {name: params for name, (params, value) in results.items()}

    # Add cost parameters to predictive and advanced strategies
    for strategy in ['price_threshold_predictive', 'moving_average_predictive',
                     'expected_value', 'consensus', 'risk_adjusted', 'rolling_horizon_mpc']:
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

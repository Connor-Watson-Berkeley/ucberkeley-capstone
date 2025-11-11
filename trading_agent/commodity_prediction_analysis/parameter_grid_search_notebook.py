# Databricks notebook source
# MAGIC %md
# MAGIC # Grid Search for Optimal Trading Strategy Parameters
# MAGIC
# MAGIC This notebook performs grid search optimization to find the best parameter values
# MAGIC for each trading strategy, maximizing net revenue.
# MAGIC
# MAGIC **Key Features:**
# MAGIC - Tests parameter combinations across reasonable ranges
# MAGIC - Ensures matched pairs share same baseline parameters
# MAGIC - Generates optimal_parameters.json for use in production
# MAGIC - Provides detailed comparison of all tested combinations
# MAGIC
# MAGIC **Usage:**
# MAGIC 1. Set CURRENT_COMMODITY and CURRENT_MODEL below
# MAGIC 2. Choose grid search mode (coarse/fine)
# MAGIC 3. Run all cells
# MAGIC 4. Review results and update main notebook parameters

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Instructions
# MAGIC
# MAGIC **IMPORTANT:** Before running this grid search notebook, you must:
# MAGIC
# MAGIC 1. Open `trading_prediction_analysis_multi_model.py` in Databricks
# MAGIC 2. Run ALL cells in that notebook (this loads all strategy classes and BacktestEngine into memory)
# MAGIC 3. Keep that notebook tab open
# MAGIC 4. Then come back to this grid search notebook and run it
# MAGIC
# MAGIC The `%run` command below will access the variables from the main notebook's execution context.

# COMMAND ----------

# IMPORTANT: For grid search to work, we need strategy classes and BacktestEngine
# These are defined in the main notebook. You have two options:
#
# OPTION 1 (Recommended): Run main notebook first
#   1. Open trading_prediction_analysis_multi_model.py
#   2. Run all cells (or at least cells 1-10 that define classes)
#   3. Keep that tab open
#   4. Come back here and the %run will work
#
# OPTION 2: The code below will define COMMODITY_CONFIGS as fallback
#   But you still need strategy classes from main notebook

# Try to import from main notebook
import_success = False
try:
    %run ./trading_prediction_analysis_multi_model

    # Verify import worked
    if 'COMMODITY_CONFIGS' in dir() and 'ImmediateSaleStrategy' in dir() and 'BacktestEngine' in dir():
        print("✓ Imported main notebook code successfully")
        print("  - Strategy classes available")
        print("  - BacktestEngine available")
        print("  - COMMODITY_CONFIGS available")
        print("  - Database connection available")
        import_success = True
    else:
        print("⚠️  Partial import - some variables missing")

except Exception as e:
    print(f"⚠️  Import warning: {e}")

# Define COMMODITY_CONFIGS as fallback (always define it to prevent errors downstream)
if 'COMMODITY_CONFIGS' not in dir():
    print("\n  → Defining COMMODITY_CONFIGS as fallback...")
    COMMODITY_CONFIGS = {
        'coffee': {
            'commodity': 'coffee',
            'harvest_volume': 50,
            'harvest_windows': [(5, 9)],
            'storage_cost_pct_per_day': 0.025,
            'transaction_cost_pct': 0.25,
            'min_inventory_to_trade': 1.0,
            'max_holding_days': 365
        },
        'sugar': {
            'commodity': 'sugar',
            'harvest_volume': 50,
            'harvest_windows': [(10, 12)],
            'storage_cost_pct_per_day': 0.025,
            'transaction_cost_pct': 0.25,
            'min_inventory_to_trade': 1.0,
            'max_holding_days': 365
        }
    }
    print("  ✓ COMMODITY_CONFIGS defined")

# Check if we have everything we need
if not import_success:
    print("\n" + "=" * 80)
    print("❌ MISSING REQUIRED CLASSES")
    print("=" * 80)
    print("\nStrategy classes and BacktestEngine are required but not loaded.")
    print("\n📋 TO FIX:")
    print("1. Open 'trading_prediction_analysis_multi_model.py' in a new Databricks tab")
    print("2. Run at least cells 1-10 (which define all strategy classes)")
    print("3. Keep that tab open")
    print("4. Come back to this tab and re-run this cell")
    print("\nWithout the strategy classes, grid search cannot run.")
    print("=" * 80)

    # Check if at least some key classes exist
    missing_classes = []
    for cls_name in ['ImmediateSaleStrategy', 'BacktestEngine', 'calculate_metrics']:
        if cls_name not in dir():
            missing_classes.append(cls_name)

    if missing_classes:
        print(f"\nMissing: {', '.join(missing_classes)}")
        print("\nCannot proceed without these classes from main notebook.")
    else:
        print("\n✓ Found required classes - you can proceed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration
CURRENT_COMMODITY = 'coffee'  # 'coffee' or 'sugar'
CURRENT_MODEL = 'sarimax_auto_weather_v1'

# Grid search settings
USE_FINE_GRAIN_GRID = False  # Set True for finer-grained search around optimal values
MAX_COMBINATIONS_PER_STRATEGY = None  # Set to int to sample large grids (e.g., 100)

# Which strategies to optimize (None = all)
STRATEGIES_TO_OPTIMIZE = None  # Or specify: ['price_threshold', 'moving_average', 'consensus']

print(f"Commodity: {CURRENT_COMMODITY}")
print(f"Model: {CURRENT_MODEL}")
print(f"Fine-grain: {USE_FINE_GRAIN_GRID}")
print(f"Max combinations: {MAX_COMBINATIONS_PER_STRATEGY or 'unlimited'}")
print(f"Strategies: {STRATEGIES_TO_OPTIMIZE or 'all'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameter Grid Definitions

# COMMAND ----------

def get_parameter_grids(fine_grain=False):
    """
    Define parameter ranges to test for each strategy.

    Args:
        fine_grain: If True, use finer-grained ranges around optimal values

    Returns:
        dict: Parameter grids for each strategy type
    """
    if fine_grain:
        # Fine-grained search (use after initial coarse search)
        return {
            'immediate_sale': {
                'min_batch_size': [4.0, 5.0, 6.0],
                'sale_frequency_days': [6, 7, 8]
            },
            'equal_batch': {
                'batch_size': [0.22, 0.25, 0.28],
                'frequency_days': [28, 30, 32]
            },
            'price_threshold': {
                'threshold_pct': [0.04, 0.05, 0.06],
                'batch_fraction': [0.23, 0.25, 0.27],
                'max_days_without_sale': [55, 60, 65]
            },
            'moving_average': {
                'ma_period': [28, 30, 32],
                'batch_fraction': [0.23, 0.25, 0.27],
                'max_days_without_sale': [55, 60, 65]
            },
            'consensus': {
                'consensus_threshold': [0.68, 0.70, 0.72],
                'min_return': [0.025, 0.030, 0.035],
                'evaluation_day': [12, 14]
            },
            'expected_value': {
                'min_ev_improvement': [45, 50, 55],
                'baseline_batch': [0.13, 0.15, 0.17],
                'baseline_frequency': [9, 10, 11]
            },
            'risk_adjusted': {
                'min_return': [0.025, 0.030, 0.035],
                'max_uncertainty': [0.30, 0.35, 0.40],
                'consensus_threshold': [0.58, 0.60, 0.62],
                'evaluation_day': [12, 14]
            }
        }
    else:
        # Coarse search (initial broad sweep)
        return {
            'immediate_sale': {
                'min_batch_size': [3.0, 5.0, 7.0, 10.0],
                'sale_frequency_days': [5, 7, 10, 14]
            },
            'equal_batch': {
                'batch_size': [0.15, 0.20, 0.25, 0.30, 0.35],
                'frequency_days': [20, 25, 30, 35, 40]
            },
            'price_threshold': {
                # Matched pair - shares params with PriceThresholdPredictive
                'threshold_pct': [0.02, 0.03, 0.05, 0.07, 0.10],
                'batch_fraction': [0.20, 0.25, 0.30, 0.35],
                'max_days_without_sale': [45, 60, 75, 90]
            },
            'moving_average': {
                # Matched pair - shares params with MovingAveragePredictive
                'ma_period': [20, 25, 30, 35, 40],
                'batch_fraction': [0.20, 0.25, 0.30, 0.35],
                'max_days_without_sale': [45, 60, 75, 90]
            },
            'consensus': {
                'consensus_threshold': [0.60, 0.65, 0.70, 0.75, 0.80],
                'min_return': [0.02, 0.03, 0.04, 0.05],
                'evaluation_day': [10, 12, 14]
            },
            'expected_value': {
                'min_ev_improvement': [30, 40, 50, 60, 75],
                'baseline_batch': [0.10, 0.12, 0.15, 0.18, 0.20],
                'baseline_frequency': [7, 10, 12, 14]
            },
            'risk_adjusted': {
                'min_return': [0.02, 0.03, 0.04, 0.05],
                'max_uncertainty': [0.25, 0.30, 0.35, 0.40],
                'consensus_threshold': [0.55, 0.60, 0.65, 0.70],
                'evaluation_day': [10, 12, 14]
            }
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grid Search Engine

# COMMAND ----------

import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any


class GridSearchOptimizer:
    """
    Optimizes trading strategy parameters via grid search.
    """

    def __init__(self, commodity_config, prices, prediction_matrices, param_grids):
        """
        Initialize optimizer.

        Args:
            commodity_config: Commodity configuration dict
            prices: DataFrame with historical prices
            prediction_matrices: Dict of prediction matrices {date: np.array}
            param_grids: Dict of parameter grids by strategy
        """
        self.commodity_config = commodity_config
        self.prices = prices
        self.prediction_matrices = prediction_matrices
        self.param_grids = param_grids

        # Results storage
        self.all_results = []
        self.optimal_params = {}

    def optimize_all(self, strategy_names=None, max_combinations_per_strategy=None):
        """
        Run grid search for all specified strategies.

        Args:
            strategy_names: List of strategy names, or None for all
            max_combinations_per_strategy: Max combos to test per strategy

        Returns:
            dict: Optimal parameters by strategy
        """
        if strategy_names is None:
            strategy_names = list(self.param_grids.keys())

        print("=" * 80)
        print("GRID SEARCH OPTIMIZATION")
        print("=" * 80)
        print(f"\nCommodity: {self.commodity_config['commodity']}")
        print(f"Strategies to optimize: {len(strategy_names)}")

        # Calculate total combinations
        total_combos = 0
        for name in strategy_names:
            if name in self.param_grids:
                grid = self.param_grids[name]
                n = int(np.prod([len(v) for v in grid.values()]))
                total_combos += n
                print(f"\n{name}:")
                print(f"  Parameters: {list(grid.keys())}")
                print(f"  Combinations: {n:,}")

        print(f"\nTotal combinations: {total_combos:,}")

        # Optimize each strategy
        for name in strategy_names:
            if name not in self.param_grids:
                print(f"\n⚠️  No grid defined for {name}, skipping")
                continue

            print(f"\n{'=' * 80}")
            print(f"OPTIMIZING: {name.upper()}")
            print(f"{'=' * 80}")

            optimal = self._optimize_single_strategy(name, max_combinations_per_strategy)
            self.optimal_params[name] = optimal

            print(f"\n✓ Optimal {name}:")
            print(f"  Net Revenue: ${optimal['net_revenue']:,.2f}")
            print(f"  Parameters:")
            for k, v in optimal['params'].items():
                print(f"    {k}: {v}")

        # Enforce matched pairs
        self._enforce_matched_pairs()

        return self.optimal_params

    def _optimize_single_strategy(self, strategy_name, max_combinations=None):
        """
        Optimize parameters for one strategy.

        Args:
            strategy_name: Name of strategy to optimize
            max_combinations: Max combinations to test

        Returns:
            dict: Optimal parameters and metrics
        """
        grid = self.param_grids[strategy_name]
        param_names = list(grid.keys())
        param_values = list(grid.values())

        # Generate all combinations
        all_combinations = list(itertools.product(*param_values))

        # Sample if needed
        if max_combinations and len(all_combinations) > max_combinations:
            print(f"  Sampling {max_combinations} of {len(all_combinations)} combinations")
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            combinations = [all_combinations[i] for i in indices]
        else:
            combinations = all_combinations

        print(f"  Testing {len(combinations)} combinations...")

        best_result = None
        best_revenue = -np.inf

        for i, param_combo in enumerate(combinations, 1):
            params = dict(zip(param_names, param_combo))

            try:
                # Create strategy with these params
                strategy = self._create_strategy(strategy_name, params)

                # Run backtest
                engine = BacktestEngine(self.prices, self.prediction_matrices,
                                       self.commodity_config)
                results = engine.run(strategy)
                metrics = calculate_metrics(results)

                # Store result
                self.all_results.append({
                    'strategy': strategy_name,
                    'params': params,
                    'net_revenue': metrics['net_earnings'],
                    'total_revenue': metrics['total_revenue'],
                    'total_costs': metrics['total_costs'],
                    'n_trades': metrics['n_trades'],
                    'avg_sale_price': metrics['avg_sale_price']
                })

                # Check if best
                if metrics['net_earnings'] > best_revenue:
                    best_revenue = metrics['net_earnings']
                    best_result = {
                        'params': params,
                        'net_revenue': metrics['net_earnings'],
                        'metrics': metrics
                    }

                # Progress
                if i % 20 == 0 or i == len(combinations):
                    print(f"    {i}/{len(combinations)} | Best: ${best_revenue:,.2f}")

            except Exception as e:
                print(f"    ⚠️  Error with {params}: {e}")
                continue

        return best_result

    def _create_strategy(self, strategy_name, params):
        """
        Create strategy instance with specified parameters.

        Args:
            strategy_name: Strategy name
            params: Parameter dict

        Returns:
            Strategy instance
        """
        # Map strategy names to classes
        if strategy_name == 'immediate_sale':
            return ImmediateSaleStrategy(**params)
        elif strategy_name == 'equal_batch':
            return EqualBatchStrategy(**params)
        elif strategy_name == 'price_threshold':
            return PriceThresholdStrategy(**params)
        elif strategy_name == 'moving_average':
            return MovingAverageStrategy(**params)
        elif strategy_name == 'consensus':
            # Add cost params from commodity config
            full_params = {
                **params,
                'storage_cost_pct_per_day': self.commodity_config['storage_cost_pct_per_day'],
                'transaction_cost_pct': self.commodity_config['transaction_cost_pct']
            }
            return ConsensusStrategy(**full_params)
        elif strategy_name == 'expected_value':
            # Cost params are required
            full_params = {
                'storage_cost_pct_per_day': self.commodity_config['storage_cost_pct_per_day'],
                'transaction_cost_pct': self.commodity_config['transaction_cost_pct'],
                **params
            }
            return ExpectedValueStrategy(**full_params)
        elif strategy_name == 'risk_adjusted':
            # Add cost params
            full_params = {
                **params,
                'storage_cost_pct_per_day': self.commodity_config['storage_cost_pct_per_day'],
                'transaction_cost_pct': self.commodity_config['transaction_cost_pct']
            }
            return RiskAdjustedStrategy(**full_params)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    def _enforce_matched_pairs(self):
        """
        Ensure matched pairs share baseline parameters.

        Matched pairs:
        - price_threshold <-> price_threshold_predictive
        - moving_average <-> moving_average_predictive
        """
        print(f"\n{'=' * 80}")
        print("MATCHED PAIR ENFORCEMENT")
        print(f"{'=' * 80}")

        # Price threshold pair
        if 'price_threshold' in self.optimal_params:
            pt_params = self.optimal_params['price_threshold']['params']
            print(f"\n✓ Price Threshold matched pair parameters:")
            print(f"  threshold_pct: {pt_params['threshold_pct']}")
            print(f"  batch_fraction: {pt_params['batch_fraction']}")
            print(f"  max_days_without_sale: {pt_params['max_days_without_sale']}")
            print(f"\n  → PriceThresholdPredictive will use same values")

        # Moving average pair
        if 'moving_average' in self.optimal_params:
            ma_params = self.optimal_params['moving_average']['params']
            print(f"\n✓ Moving Average matched pair parameters:")
            print(f"  ma_period: {ma_params['ma_period']}")
            print(f"  batch_fraction: {ma_params['batch_fraction']}")
            print(f"  max_days_without_sale: {ma_params['max_days_without_sale']}")
            print(f"\n  → MovingAveragePredictive will use same values")

    def get_results_dataframe(self):
        """
        Get all results as DataFrame.

        Returns:
            DataFrame with all tested combinations
        """
        if not self.all_results:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(self.all_results)

        # Flatten params dict into columns
        params_df = pd.json_normalize(df['params'])

        # Combine
        result_df = pd.concat([
            df.drop('params', axis=1),
            params_df
        ], axis=1)

        return result_df.sort_values('net_revenue', ascending=False)

    def save_optimal_parameters(self, filepath='/dbfs/FileStore/optimal_parameters.json'):
        """
        Save optimal parameters to JSON file.

        Args:
            filepath: Path to save file
        """
        import json
        from datetime import datetime

        output = {
            'generated_at': datetime.now().isoformat(),
            'commodity': self.commodity_config['commodity'],
            'model': CURRENT_MODEL,
            'parameters': {}
        }

        for strategy, result in self.optimal_params.items():
            output['parameters'][strategy] = {
                'params': result['params'],
                'net_revenue': float(result['net_revenue']),
                'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                           for k, v in result['metrics'].items()}
            }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ Saved optimal parameters to: {filepath}")
        return filepath

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Optimization
# MAGIC
# MAGIC **NOTE:** This requires that the main notebook has been run first to define:
# MAGIC - Strategy classes (ImmediateSaleStrategy, etc.)
# MAGIC - BacktestEngine class
# MAGIC - calculate_metrics function
# MAGIC - COMMODITY_CONFIGS
# MAGIC - Databricks connection
# MAGIC - Loaded prices and prediction_matrices

# COMMAND ----------

# Load commodity config
commodity_config = COMMODITY_CONFIGS[CURRENT_COMMODITY]

print("Commodity Configuration:")
print(json.dumps(commodity_config, indent=2))

# COMMAND ----------

# Load data from Unity Catalog (same as main notebook)
print(f"\nLoading data for {CURRENT_COMMODITY.upper()} - {CURRENT_MODEL}...")

# Load prices
real_models = get_available_models(CURRENT_COMMODITY.capitalize(), db_connection)
if len(real_models) > 0:
    from data_access.forecast_loader import load_actuals_from_distributions
    prices = load_actuals_from_distributions(
        commodity=CURRENT_COMMODITY.capitalize(),
        model_version=real_models[0],
        connection=db_connection
    )
    print(f"✓ Loaded {len(prices)} days of price data")
else:
    raise ValueError(f"No models found for {CURRENT_COMMODITY}")

# Load predictions
prediction_matrices, predictions_source = load_prediction_matrices(
    CURRENT_COMMODITY,
    model_version=CURRENT_MODEL,
    connection=db_connection,
    prices=prices
)
print(f"✓ Loaded {len(prediction_matrices)} prediction matrices")
print(f"  Source: {predictions_source}")

# COMMAND ----------

# Initialize grid search
param_grids = get_parameter_grids(fine_grain=USE_FINE_GRAIN_GRID)

optimizer = GridSearchOptimizer(
    commodity_config=commodity_config,
    prices=prices,
    prediction_matrices=prediction_matrices,
    param_grids=param_grids
)

print("✓ Grid search optimizer initialized")

# COMMAND ----------

# Run optimization
optimal_params = optimizer.optimize_all(
    strategy_names=STRATEGIES_TO_OPTIMIZE,
    max_combinations_per_strategy=MAX_COMBINATIONS_PER_STRATEGY
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Analysis

# COMMAND ----------

# Get all results as DataFrame
results_df = optimizer.get_results_dataframe()

print(f"Total combinations tested: {len(results_df)}")
print(f"\nTop 10 results:")
display(results_df.head(10))

# COMMAND ----------

# Show optimal parameters summary
print("=" * 80)
print("OPTIMAL PARAMETERS SUMMARY")
print("=" * 80)

for strategy, result in optimal_params.items():
    print(f"\n{strategy.upper()}")
    print("-" * 40)
    print(f"Net Revenue: ${result['net_revenue']:,.2f}")
    print(f"Total Revenue: ${result['metrics']['total_revenue']:,.2f}")
    print(f"Total Costs: ${result['metrics']['total_costs']:,.2f}")
    print(f"Trades: {result['metrics']['n_trades']}")
    print(f"\nParameters:")
    for param, value in result['params'].items():
        print(f"  {param}: {value}")

# COMMAND ----------

# Save optimal parameters
output_file = optimizer.save_optimal_parameters()
print(f"\n✓ Results saved to: {output_file}")

# Also save full results CSV
results_csv = '/dbfs/FileStore/grid_search_results_all.csv'
results_df.to_csv(results_csv, index=False)
print(f"✓ Full results saved to: {results_csv}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameter Update Guide
# MAGIC
# MAGIC To use the optimal parameters in the main notebook:
# MAGIC
# MAGIC 1. **Update BASELINE_PARAMS** (lines 66-79):
# MAGIC ```python
# MAGIC BASELINE_PARAMS = {
# MAGIC     'equal_batch': {
# MAGIC         'batch_size': <optimal_value>,  # from grid search
# MAGIC         'frequency_days': <optimal_value>
# MAGIC     },
# MAGIC     'price_threshold': {
# MAGIC         'threshold_pct': <optimal_value>
# MAGIC     },
# MAGIC     'moving_average': {
# MAGIC         'ma_period': <optimal_value>
# MAGIC     }
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC 2. **Update PREDICTION_PARAMS** (lines 81-98):
# MAGIC ```python
# MAGIC PREDICTION_PARAMS = {
# MAGIC     'consensus': {
# MAGIC         'consensus_threshold': <optimal_value>,
# MAGIC         'min_return': <optimal_value>,
# MAGIC         'evaluation_day': <optimal_value>
# MAGIC     },
# MAGIC     'expected_value': {
# MAGIC         'min_ev_improvement': <optimal_value>,
# MAGIC         'baseline_batch': <optimal_value>,
# MAGIC         'baseline_frequency': <optimal_value>
# MAGIC     },
# MAGIC     'risk_adjusted': {
# MAGIC         'min_return': <optimal_value>,
# MAGIC         'max_uncertainty': <optimal_value>,
# MAGIC         'consensus_threshold': <optimal_value>,
# MAGIC         'evaluation_day': <optimal_value>
# MAGIC     }
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC 3. **Update strategy instantiation** (lines 3484-3492):
# MAGIC ```python
# MAGIC PriceThresholdPredictive(
# MAGIC     threshold_pct=<optimal_value>,  # same as baseline
# MAGIC     batch_fraction=<optimal_value>,  # same as baseline
# MAGIC     max_days_without_sale=<optimal_value>  # same as baseline
# MAGIC ),
# MAGIC MovingAveragePredictive(
# MAGIC     ma_period=<optimal_value>,  # same as baseline
# MAGIC     batch_fraction=<optimal_value>,  # same as baseline
# MAGIC     max_days_without_sale=<optimal_value>  # same as baseline
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualization: Parameter Impact

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# For each strategy, plot parameter impact on net revenue
for strategy in results_df['strategy'].unique():
    strategy_df = results_df[results_df['strategy'] == strategy]

    # Get parameter columns
    param_cols = [col for col in strategy_df.columns
                  if col not in ['strategy', 'net_revenue', 'total_revenue',
                                'total_costs', 'n_trades', 'avg_sale_price']]

    if len(param_cols) == 0:
        continue

    print(f"\n{'=' * 80}")
    print(f"{strategy.upper()} - Parameter Impact")
    print(f"{'=' * 80}")

    # Create subplots for each parameter
    fig, axes = plt.subplots(1, len(param_cols), figsize=(5*len(param_cols), 4))
    if len(param_cols) == 1:
        axes = [axes]

    for i, param in enumerate(param_cols):
        # Group by parameter value and get mean net revenue
        grouped = strategy_df.groupby(param)['net_revenue'].agg(['mean', 'std', 'count'])

        axes[i].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                        marker='o', capsize=5, capthick=2)
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('Net Revenue ($)')
        axes[i].set_title(f'{param}')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Review optimal parameters** - Check if they make intuitive sense
# MAGIC 2. **Validate performance** - Run full backtest with optimal params
# MAGIC 3. **Compare to baseline** - How much improvement over current params?
# MAGIC 4. **Statistical significance** - Run bootstrap tests
# MAGIC 5. **Update main notebook** - Copy optimal values to BASELINE_PARAMS and PREDICTION_PARAMS
# MAGIC 6. **Re-run multi-model analysis** - See impact across all models
# MAGIC 7. **Consider fine-grained search** - If parameters at grid boundaries

"""
Strategy Parameter Optimizer

Production-ready parameter optimization using Optuna with efficiency-aware objectives.

Key Features:
- Multiple optimization objectives (earnings, efficiency ratio, multi-objective)
- Integration with theoretical maximum benchmark
- Uses production backtest engine
- Full logging and result tracking

Migrated from diagnostics/run_diagnostic_16.py with modern enhancements.
"""

import pandas as pd
import numpy as np
import sys
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime

# Auto-install optuna if needed (for Databricks)
try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("Installing optuna...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "--quiet"])
    import optuna
    from optuna.samplers import TPESampler

from .search_space import SearchSpaceRegistry


class ParameterOptimizer:
    """
    Optimize strategy parameters using Optuna.

    Supports multiple optimization objectives:
    - 'earnings': Maximize raw net earnings (original approach)
    - 'efficiency': Maximize efficiency ratio (Actual / Theoretical Max)
    - 'multi': Multi-objective optimization (Pareto frontier)
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        prediction_matrices: Dict,
        config: Dict,
        backtest_engine_class: Optional[Callable] = None,
        theoretical_max_earnings: Optional[float] = None,
        use_production_engine: bool = True
    ):
        """
        Initialize optimizer.

        Args:
            prices_df: DataFrame with columns ['date', 'price']
            prediction_matrices: Dict mapping date -> prediction matrix (runs × horizons)
            config: Full commodity config dict with:
                - storage_cost_pct_per_day: float
                - transaction_cost_pct: float
                - harvest_volume: float (required for production engine)
                - harvest_windows: list (required for production engine)
                - commodity: str (optional, for production engine)
                - max_holding_days: int (optional)
                - min_inventory_to_trade: float (optional)
            backtest_engine_class: Backtest engine class (overrides use_production_engine if set)
            theoretical_max_earnings: Theoretical maximum earnings for efficiency calculation
            use_production_engine: If True (default), use production BacktestEngine for accuracy
                                  If False, use SimpleBacktestEngine for speed

        Note:
            - Production engine recommended for final optimization (more accurate, harvest-aware)
            - Simple engine acceptable for rapid prototyping/testing
            - If theoretical_max_earnings is None, optimizer will only support 'earnings' objective
        """
        self.prices = prices_df
        self.predictions = prediction_matrices
        self.config = config
        self.theoretical_max = theoretical_max_earnings

        # Determine which engine to use
        if backtest_engine_class is not None:
            # Explicit override
            self.engine_class = backtest_engine_class
        elif use_production_engine:
            # Use production engine for accuracy (default)
            from production.core.backtest_engine import BacktestEngine
            self.engine_class = BacktestEngine
        else:
            # Use simple engine for speed
            self.engine_class = SimpleBacktestEngine

        # Create engine instance
        self.engine = self.engine_class(prices_df, prediction_matrices, config)

        # Search space registry
        self.search_space = SearchSpaceRegistry()

    def optimize_strategy(
        self,
        strategy_class: Callable,
        strategy_name: str,
        n_trials: int = 200,
        objective: str = 'earnings',
        seed: int = 42,
        show_progress: bool = True
    ) -> Tuple[Dict, float, Optional[optuna.Study]]:
        """
        Optimize parameters for a single strategy.

        Args:
            strategy_class: Strategy class to instantiate
            strategy_name: Name of strategy (for search space lookup)
            n_trials: Number of optimization trials (default: 200)
            objective: 'earnings', 'efficiency', or 'multi'
            seed: Random seed for reproducibility
            show_progress: Show progress bar

        Returns:
            Tuple of (best_params, best_value, study)
            - best_params: Dict of optimal parameters
            - best_value: Best objective value achieved
            - study: Optuna study object (for further analysis)

        Raises:
            ValueError: If objective requires theoretical_max but it's not provided
        """
        if objective == 'efficiency' and self.theoretical_max is None:
            raise ValueError(
                "Efficiency optimization requires theoretical_max_earnings. "
                "Pass it to __init__ or use objective='earnings'."
            )

        print(f"\n{'='*80}")
        print(f"OPTIMIZING: {strategy_name}")
        print(f"{'='*80}")
        print(f"Objective: {objective}")
        print(f"Trials: {n_trials}")
        print(f"Started: {datetime.now()}")

        # Create study
        if objective == 'multi':
            # Multi-objective: maximize both earnings and Sharpe ratio
            study = optuna.create_study(
                directions=['maximize', 'maximize'],  # [earnings, sharpe]
                sampler=TPESampler(seed=seed)
            )
        else:
            # Single objective
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=seed)
            )

        # Define objective function
        def objective_function(trial):
            # Get parameter suggestions from search space
            params = self.search_space.get_search_space(trial, strategy_name)

            # Add cost parameters for prediction strategies
            if strategy_name not in ['immediate_sale', 'equal_batch', 'price_threshold', 'moving_average']:
                params['storage_cost_pct_per_day'] = self.config['storage_cost_pct_per_day']
                params['transaction_cost_pct'] = self.config['transaction_cost_pct']

            try:
                # Instantiate strategy with suggested parameters
                strategy = strategy_class(**params)

                # Run backtest
                result = self.engine.run_backtest(strategy)

                # Return objective value(s)
                if objective == 'earnings':
                    return result['net_earnings']

                elif objective == 'efficiency':
                    # Efficiency ratio = Actual / Theoretical Max
                    if self.theoretical_max > 0:
                        efficiency = result['net_earnings'] / self.theoretical_max
                        return efficiency
                    else:
                        # Fallback to earnings if theoretical max is zero
                        return result['net_earnings']

                elif objective == 'multi':
                    # Multi-objective: (earnings, Sharpe ratio)
                    earnings = result['net_earnings']
                    sharpe = result.get('sharpe_ratio', 0.0)
                    return earnings, sharpe

                else:
                    raise ValueError(f"Unknown objective: {objective}")

            except Exception as e:
                print(f"  Trial {trial.number} failed: {e}")
                if objective == 'multi':
                    return -1e9, -1e9  # Return very bad values for multi-objective
                else:
                    return -1e9

        # Run optimization
        study.optimize(
            objective_function,
            n_trials=n_trials,
            show_progress_bar=show_progress
        )

        # Extract best results
        if objective == 'multi':
            # For multi-objective, return the best trial on the Pareto front
            # (We'll return the one with highest earnings among Pareto-optimal)
            pareto_trials = study.best_trials
            if pareto_trials:
                best_trial = max(pareto_trials, key=lambda t: t.values[0])  # Max earnings
                best_params = best_trial.params
                best_value = best_trial.values[0]  # Earnings
            else:
                best_params = {}
                best_value = -1e9
        else:
            best_params = study.best_params
            best_value = study.best_value

        print(f"✓ Optimization complete")
        print(f"  Best value: {best_value:,.2f}")
        print(f"  Trials completed: {len(study.trials)}")
        print(f"Completed: {datetime.now()}")

        return best_params, best_value, study

    def optimize_all_strategies(
        self,
        strategies: List[Tuple[Callable, str]],
        n_trials: int = 200,
        objective: str = 'earnings',
        seed: int = 42
    ) -> Dict[str, Tuple[Dict, float]]:
        """
        Optimize parameters for all provided strategies.

        Args:
            strategies: List of (strategy_class, strategy_name) tuples
            n_trials: Number of trials per strategy
            objective: Optimization objective
            seed: Random seed

        Returns:
            Dict mapping strategy_name -> (best_params, best_value)
        """
        results = {}

        for i, (strategy_class, strategy_name) in enumerate(strategies, 1):
            print(f"\n[{i}/{len(strategies)}] {strategy_name}")

            best_params, best_value, _ = self.optimize_strategy(
                strategy_class=strategy_class,
                strategy_name=strategy_name,
                n_trials=n_trials,
                objective=objective,
                seed=seed
            )

            results[strategy_name] = (best_params, best_value)

        # Print summary
        print("\n" + "="*80)
        print(f"ALL {len(strategies)} STRATEGIES OPTIMIZED")
        print("="*80)
        for name, (params, value) in sorted(results.items(), key=lambda x: x[1][1], reverse=True):
            print(f"{name:35s}: {value:,.2f}")

        return results


class SimpleBacktestEngine:
    """
    Simplified backtest engine for fast parameter optimization.

    This is extracted from diagnostics/run_diagnostic_16.py.
    For production use, consider using production/core/backtest_engine.py instead.
    """

    def __init__(self, prices_df, pred_matrices, config):
        """
        Initialize engine.

        Args:
            prices_df: DataFrame with columns ['date', 'price']
            pred_matrices: Dict mapping date -> prediction matrix
            config: Dict with cost parameters
        """
        self.prices = prices_df
        self.pred = pred_matrices
        self.config = config

    def run_backtest(self, strategy, initial_inventory=50.0):
        """
        Run backtest for a strategy.

        Args:
            strategy: Strategy instance to test
            initial_inventory: Starting inventory in tons

        Returns:
            Dict with:
                - net_earnings: float
                - total_revenue: float
                - transaction_costs: float
                - storage_costs: float
                - num_trades: int
                - final_inventory: float
                - sharpe_ratio: float (if enough trades)
        """
        inventory = initial_inventory
        total_revenue = 0
        trans_costs = 0
        storage_costs = 0
        trades = []
        daily_returns = []

        strategy.reset()
        strategy.set_harvest_start(0)

        for day in range(len(self.prices)):
            date = self.prices.iloc[day]['date']
            price = self.prices.iloc[day]['price']
            hist = self.prices.iloc[:day+1].copy()
            pred = self.pred.get(date)

            # Get strategy decision
            decision = strategy.decide(
                day=day,
                inventory=inventory,
                current_price=price,
                price_history=hist,
                predictions=pred
            )

            # Execute trade if recommended
            if decision['action'] == 'SELL' and decision['amount'] > 0:
                amt = min(decision['amount'], inventory)

                # Calculate revenue and costs
                # Note: price * 20 converts cents/lb to $/ton
                revenue = amt * price * 20
                trans_cost = revenue * self.config['transaction_cost_pct'] / 100

                total_revenue += revenue
                trans_costs += trans_cost
                inventory -= amt

                trades.append({
                    'day': day,
                    'amount': amt,
                    'price': price,
                    'revenue': revenue
                })

                # Track daily return (for Sharpe ratio)
                if len(trades) > 1:
                    prev_price = trades[-2]['price']
                    daily_return = (price - prev_price) / prev_price
                    daily_returns.append(daily_return)

            # Calculate storage costs
            if inventory > 0:
                avg_price = self.prices.iloc[:day+1]['price'].mean()
                storage_cost = inventory * avg_price * 20 * self.config['storage_cost_pct_per_day'] / 100
                storage_costs += storage_cost

        # Calculate net earnings
        net_earnings = total_revenue - trans_costs - storage_costs

        # Calculate Sharpe ratio if enough data
        sharpe_ratio = 0.0
        if len(daily_returns) > 1:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            if std_return > 0:
                sharpe_ratio = (mean_return / std_return) * np.sqrt(252)  # Annualized

        return {
            'net_earnings': net_earnings,
            'total_revenue': total_revenue,
            'transaction_costs': trans_costs,
            'storage_costs': storage_costs,
            'num_trades': len(trades),
            'final_inventory': inventory,
            'sharpe_ratio': sharpe_ratio
        }

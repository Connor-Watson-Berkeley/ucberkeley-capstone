"""
Perfect Foresight Strategy

Establishes the TRUE theoretical maximum for production backtesting.

Uses the production BacktestEngine with perfect knowledge of future prices to:
1. Ensure harvest dynamics match (gradual accumulation, multiple cycles)
2. Respect age constraints (365-day max holding)
3. Apply same cost structure (storage + transaction)
4. Provide apples-to-apples benchmark for statistical validation

This replaces the broken DP-based theoretical max which assumed instant 50-ton
inventory and didn't account for harvest cycles.
"""

import numpy as np
from typing import Dict, Any


class PerfectForesightStrategy:
    """
    Strategy that makes optimal decisions with perfect knowledge of future prices.

    At each decision point, calculates the net value of selling now vs waiting,
    accounting for storage costs, transaction costs, and future price movements.

    This is the theoretical maximum achievable performance given:
    - Perfect predictions (oracle knowledge)
    - Realistic harvest dynamics (gradual accumulation)
    - Realistic constraints (age limits, costs)
    """

    def __init__(
        self,
        storage_cost_pct_per_day: float,
        transaction_cost_pct: float,
        lookback_days: int = 14,
        sell_threshold_pct: float = 0.95
    ):
        """
        Initialize perfect foresight strategy.

        Args:
            storage_cost_pct_per_day: Daily storage cost as percentage (e.g., 0.005 for 0.005%)
            transaction_cost_pct: Transaction cost as percentage (e.g., 0.01 for 0.01%)
            lookback_days: Days ahead to consider (default: 14, matches prediction horizon)
            sell_threshold_pct: Sell if current value is within this % of best future value
                              (default: 0.95 = sell if today is ≥95% of best future)
        """
        self.name = "perfect_foresight"
        self.storage_cost_pct = storage_cost_pct_per_day
        self.transaction_cost_pct = transaction_cost_pct
        self.lookback_days = lookback_days
        self.sell_threshold = sell_threshold_pct

    def reset(self):
        """Reset strategy state (called before each backtest)."""
        pass

    def set_harvest_start(self, start_day: int):
        """Set harvest start day (called by backtest engine)."""
        self.harvest_start = start_day

    def decide(
        self,
        day: int,
        inventory: float,
        current_price: float,
        price_history: Any,
        predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Make sell/hold decision with perfect foresight.

        Args:
            day: Current day index
            inventory: Current inventory (tons)
            current_price: Current price (cents/lb)
            price_history: Historical prices (not used - we have predictions)
            predictions: Prediction matrix (n_runs × n_horizons)
                        Shape: (2000 runs, 14 days ahead)

        Returns:
            Dict with 'action' ('SELL' or 'HOLD'), 'amount', and 'reason'
        """
        if inventory <= 0:
            return {'action': 'HOLD', 'amount': 0, 'reason': 'no_inventory'}

        if predictions is None or len(predictions) == 0:
            # No predictions available - sell immediately
            return {
                'action': 'SELL',
                'amount': inventory,
                'reason': 'no_predictions'
            }

        # Use mean of prediction matrix as "perfect foresight"
        # Shape: (n_runs, n_horizons) -> (n_horizons,)
        future_prices = predictions.mean(axis=0)

        # Limit to lookback window
        future_prices = future_prices[:self.lookback_days]

        # Calculate net value of selling today
        current_price_per_ton = current_price * 20  # Convert cents/lb to $/ton
        revenue_now = inventory * current_price_per_ton
        transaction_cost_now = revenue_now * (self.transaction_cost_pct / 100)
        net_value_now = revenue_now - transaction_cost_now

        # Calculate best future net value
        best_future_value = 0
        best_future_day = 0

        for i, future_price in enumerate(future_prices):
            days_to_wait = i + 1

            # Revenue from selling in the future
            future_price_per_ton = future_price * 20
            future_revenue = inventory * future_price_per_ton
            future_transaction_cost = future_revenue * (self.transaction_cost_pct / 100)

            # Storage costs accumulated while waiting
            # Use average price over waiting period for storage cost calculation
            avg_price_per_ton = (current_price_per_ton + future_price_per_ton) / 2
            cumulative_storage_cost = (
                inventory
                * avg_price_per_ton
                * (self.storage_cost_pct / 100)
                * days_to_wait
            )

            # Net value = future revenue - transaction cost - storage costs
            net_future = future_revenue - future_transaction_cost - cumulative_storage_cost

            if net_future > best_future_value:
                best_future_value = net_future
                best_future_day = days_to_wait

        # Decision: Sell if today's value is ≥ threshold% of best future value
        # Using threshold allows for slight suboptimality to avoid over-waiting
        if net_value_now >= (best_future_value * self.sell_threshold):
            return {
                'action': 'SELL',
                'amount': inventory,
                'reason': f'optimal_timing (now=${net_value_now:,.2f} vs best_future=${best_future_value:,.2f} in {best_future_day}d)'
            }
        else:
            return {
                'action': 'HOLD',
                'amount': 0,
                'reason': f'waiting_for_better_price (expected_gain=${best_future_value - net_value_now:,.2f} in {best_future_day}d)'
            }


class GreedyPerfectForesightStrategy:
    """
    Simplified perfect foresight: sell on the highest price day.

    This is a faster, simpler alternative that just waits for the peak price
    within the prediction horizon. Less sophisticated than PerfectForesightStrategy
    but still establishes a reasonable upper bound.
    """

    def __init__(
        self,
        storage_cost_pct_per_day: float,
        transaction_cost_pct: float,
        lookback_days: int = 14
    ):
        """
        Initialize greedy perfect foresight strategy.

        Args:
            storage_cost_pct_per_day: Daily storage cost percentage
            transaction_cost_pct: Transaction cost percentage
            lookback_days: Days ahead to consider
        """
        self.name = "greedy_perfect_foresight"
        self.storage_cost_pct = storage_cost_pct_per_day
        self.transaction_cost_pct = transaction_cost_pct
        self.lookback_days = lookback_days

    def reset(self):
        """Reset strategy state."""
        pass

    def set_harvest_start(self, start_day: int):
        """Set harvest start day."""
        self.harvest_start = start_day

    def decide(
        self,
        day: int,
        inventory: float,
        current_price: float,
        price_history: Any,
        predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Sell when current price is the peak within prediction horizon.

        Simpler logic: just check if today is the highest price day.
        """
        if inventory <= 0:
            return {'action': 'HOLD', 'amount': 0, 'reason': 'no_inventory'}

        if predictions is None or len(predictions) == 0:
            return {'action': 'SELL', 'amount': inventory, 'reason': 'no_predictions'}

        # Get mean predictions
        future_prices = predictions.mean(axis=0)[:self.lookback_days]

        # Is current price the highest?
        max_future_price = future_prices.max()

        if current_price >= max_future_price:
            return {
                'action': 'SELL',
                'amount': inventory,
                'reason': f'peak_price (current={current_price:.2f} vs max_future={max_future_price:.2f})'
            }
        else:
            return {
                'action': 'HOLD',
                'amount': 0,
                'reason': f'waiting_for_peak (expecting {max_future_price:.2f})'
            }

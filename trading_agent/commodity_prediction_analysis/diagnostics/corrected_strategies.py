"""
Corrected Trading Strategies - Proper Storage Cost Handling

**Purpose:** Prediction strategies that properly account for storage costs in decisions

**Key Fix:**
Net benefit (profit after storage costs) is the PRIMARY decision driver.
Confidence and trend indicators are SECONDARY, used to adjust batch sizing.

**Parameterization:**
All thresholds and decision points are exposed as parameters for grid search optimization.

**Matched Pairs:**
These prediction strategies can be compared to baseline strategies (PriceThreshold, MovingAverage)
that use the same parameters but don't use predictions.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


# =============================================================================
# TECHNICAL INDICATOR CALCULATIONS
# =============================================================================

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return 50.0

    deltas = np.diff(prices[-period-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_prediction_confidence(predictions, horizon_day):
    """Calculate confidence from prediction ensemble using coefficient of variation"""
    if predictions is None or predictions.size == 0:
        return 1.0

    if horizon_day >= predictions.shape[1]:
        horizon_day = predictions.shape[1] - 1

    day_predictions = predictions[:, horizon_day]
    median_pred = np.median(day_predictions)
    std_dev = np.std(day_predictions)

    cv = std_dev / median_pred if median_pred > 0 else 1.0

    return cv


def calculate_adx(price_history, period=14):
    """Calculate Average Directional Index (trend strength)"""
    if len(price_history) < period + 1:
        return 20.0, 0.0, 0.0

    if 'high' in price_history.columns and 'low' in price_history.columns:
        high = price_history['high'].values
        low = price_history['low'].values
    else:
        high = price_history['price'].values
        low = price_history['price'].values

    close = price_history['price'].values

    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(abs(high[1:] - close[:-1]),
                              abs(low[1:] - close[:-1])))

    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                        np.maximum(low[:-1] - low[1:], 0), 0)

    atr = np.mean(tr[-period:])
    if atr > 0:
        plus_di = 100 * np.mean(plus_dm[-period:]) / atr
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr
    else:
        plus_di = 0.0
        minus_di = 0.0

    di_sum = plus_di + minus_di
    if di_sum > 0:
        dx = 100 * abs(plus_di - minus_di) / di_sum
        adx = dx
    else:
        adx = 0.0

    return adx, plus_di, minus_di


# =============================================================================
# BASE STRATEGY CLASS
# =============================================================================

class Strategy(ABC):
    """Base class for all strategies"""

    def __init__(self, name, max_holding_days=365):
        self.name = name
        self.history = []
        self.max_holding_days = max_holding_days
        self.harvest_start_day = None

    @abstractmethod
    def decide(self, day, inventory, current_price, price_history, predictions=None):
        pass

    def set_harvest_start(self, day):
        self.harvest_start_day = day

    def reset(self):
        self.history = []
        self.harvest_start_day = None

    def _days_held(self, day):
        if self.harvest_start_day is None:
            return 0
        return day - self.harvest_start_day

    def _force_liquidation_check(self, day, inventory):
        if self.harvest_start_day is None:
            return None

        days_held = self._days_held(day)
        days_remaining = self.max_holding_days - days_held

        if days_remaining <= 0 and inventory > 0:
            return {'action': 'SELL', 'amount': inventory,
                   'reason': 'max_holding_365d_reached'}
        elif days_remaining <= 30 and inventory > 0:
            sell_fraction = min(1.0, 0.05 * (31 - days_remaining))
            amount = inventory * sell_fraction
            return {'action': 'SELL', 'amount': amount,
                   'reason': f'approaching_365d_deadline_{days_remaining}d_left'}

        return None


# =============================================================================
# CORRECTED: EXPECTED VALUE STRATEGY
# =============================================================================

class ExpectedValueStrategyCorrected(Strategy):
    """
    Expected Value Strategy - Corrected for proper storage cost handling

    Decision Logic:
    1. Calculate net benefit of waiting (includes storage costs)
    2. PRIMARY: If net_benefit > threshold → consider holding
    3. SECONDARY: Use confidence/trend to adjust batch size
    4. If net_benefit < 0 → sell (storage costs exceed gains)

    Parameters (all exposed for grid search):
    - min_net_benefit: Minimum $ benefit required to hold (default: 50)
    - negative_threshold: Below this, sell aggressively (default: -20)
    - high_confidence_cv: CV below this = high confidence (default: 0.05)
    - medium_confidence_cv: CV below this = medium confidence (default: 0.10)
    - strong_trend_adx: ADX above this = strong trend (default: 25)
    - batch_positive_confident: Batch when profitable + confident (default: 0.0)
    - batch_positive_uncertain: Batch when profitable + uncertain (default: 0.05)
    - batch_marginal: Batch when near-zero benefit (default: 0.15)
    - batch_negative_mild: Batch when mildly negative (default: 0.20)
    - batch_negative_strong: Batch when strongly negative (default: 0.30)
    - cooldown_days: Days between trades (default: 7)
    - baseline_batch: Fallback batch size (default: 0.15)
    - baseline_frequency: Days before fallback trade (default: 30)
    """

    def __init__(self,
                 storage_cost_pct_per_day,
                 transaction_cost_pct,
                 min_net_benefit=50,
                 negative_threshold=-20,
                 high_confidence_cv=0.05,
                 medium_confidence_cv=0.10,
                 strong_trend_adx=25,
                 batch_positive_confident=0.0,
                 batch_positive_uncertain=0.05,
                 batch_marginal=0.15,
                 batch_negative_mild=0.20,
                 batch_negative_strong=0.30,
                 cooldown_days=7,
                 baseline_batch=0.15,
                 baseline_frequency=30):
        super().__init__("Expected Value (Corrected)")

        # Cost parameters
        self.storage_cost_pct_per_day = storage_cost_pct_per_day
        self.transaction_cost_pct = transaction_cost_pct

        # Decision thresholds
        self.min_net_benefit = min_net_benefit
        self.negative_threshold = negative_threshold
        self.high_confidence_cv = high_confidence_cv
        self.medium_confidence_cv = medium_confidence_cv
        self.strong_trend_adx = strong_trend_adx

        # Batch sizing parameters
        self.batch_positive_confident = batch_positive_confident
        self.batch_positive_uncertain = batch_positive_uncertain
        self.batch_marginal = batch_marginal
        self.batch_negative_mild = batch_negative_mild
        self.batch_negative_strong = batch_negative_strong

        # Trading frequency
        self.cooldown_days = cooldown_days
        self.baseline_batch = baseline_batch
        self.baseline_frequency = baseline_frequency

        self.last_sale_day = -self.cooldown_days

    def decide(self, day, inventory, current_price, price_history, predictions=None):
        if inventory <= 0:
            return {'action': 'HOLD', 'amount': 0, 'reason': 'no_inventory'}

        forced = self._force_liquidation_check(day, inventory)
        if forced:
            return forced

        days_since_sale = day - self.last_sale_day

        if days_since_sale < self.cooldown_days:
            return {'action': 'HOLD', 'amount': 0,
                   'reason': f'cooldown_{self.cooldown_days - days_since_sale}d'}

        if predictions is None or predictions.size == 0:
            if days_since_sale >= self.baseline_frequency:
                return self._execute_trade(day, inventory, self.baseline_batch,
                                          'no_predictions_fallback')
            return {'action': 'HOLD', 'amount': 0, 'reason': 'no_predictions_waiting'}

        batch_size, reason = self._analyze_expected_value(
            current_price, price_history, predictions
        )

        if batch_size > 0:
            return self._execute_trade(day, inventory, batch_size, reason)
        else:
            return {'action': 'HOLD', 'amount': 0, 'reason': reason}

    def _analyze_expected_value(self, current_price, price_history, predictions):
        """
        PRIMARY: Calculate net benefit (includes storage costs)
        SECONDARY: Use confidence/trend to adjust batch sizing
        """

        # Calculate optimal sale day and net benefit (accounts for storage costs)
        optimal_day, net_benefit = self._find_optimal_sale_day(current_price, predictions)

        # Calculate confidence and trend indicators
        cv_pred = calculate_prediction_confidence(predictions, horizon_day=min(13, predictions.shape[1]-1))
        adx_pred, _, _ = calculate_adx(
            pd.DataFrame({'price': [np.median(predictions[:, h])
                                   for h in range(predictions.shape[1])]})
        )

        # PRIMARY DECISION: Is there net benefit to waiting?
        if net_benefit > self.min_net_benefit:
            # Profitable to wait - adjust batch by confidence
            if cv_pred < self.high_confidence_cv and adx_pred > self.strong_trend_adx:
                # High confidence + strong trend
                batch_size = self.batch_positive_confident
                reason = f'net_benefit_${net_benefit:.0f}_high_conf_hold_to_day{optimal_day}'
            elif cv_pred < self.medium_confidence_cv:
                # Medium confidence
                batch_size = self.batch_positive_uncertain
                reason = f'net_benefit_${net_benefit:.0f}_med_conf_small_hedge_day{optimal_day}'
            else:
                # Lower confidence - larger hedge
                batch_size = self.batch_marginal
                reason = f'net_benefit_${net_benefit:.0f}_low_conf_larger_hedge'

        elif net_benefit > 0:
            # Positive but below threshold - marginal case
            batch_size = self.batch_marginal
            reason = f'marginal_benefit_${net_benefit:.0f}_gradual_liquidation'

        elif net_benefit > self.negative_threshold:
            # Mildly negative - storage costs slightly exceed gains
            batch_size = self.batch_negative_mild
            reason = f'mild_negative_ev_${net_benefit:.0f}_avoid_storage_costs'

        else:
            # Strongly negative - storage costs far exceed gains
            batch_size = self.batch_negative_strong
            reason = f'strong_negative_ev_${net_benefit:.0f}_sell_to_avoid_losses'

        return batch_size, reason

    def _find_optimal_sale_day(self, current_price, predictions):
        """Find the day with maximum expected value after costs"""
        ev_by_day = []
        for h in range(predictions.shape[1]):
            future_price = np.median(predictions[:, h])
            days_to_wait = h + 1

            # Storage cost accumulates linearly
            storage_cost = current_price * (self.storage_cost_pct_per_day / 100) * days_to_wait
            # Transaction cost as % of sale value
            transaction_cost = future_price * (self.transaction_cost_pct / 100)

            # Net expected value
            ev = future_price - storage_cost - transaction_cost
            ev_by_day.append(ev)

        # EV of selling today
        transaction_cost_today = current_price * (self.transaction_cost_pct / 100)
        ev_today = current_price - transaction_cost_today

        # Find optimal day
        optimal_day = np.argmax(ev_by_day)
        net_benefit = ev_by_day[optimal_day] - ev_today

        return optimal_day, net_benefit

    def _execute_trade(self, day, inventory, batch_size, reason):
        amount = inventory * batch_size
        self.last_sale_day = day
        return {'action': 'SELL', 'amount': amount, 'reason': reason}

    def reset(self):
        super().reset()
        self.last_sale_day = -self.cooldown_days


# =============================================================================
# CORRECTED: CONSENSUS STRATEGY
# =============================================================================

class ConsensusStrategyCorrected(Strategy):
    """
    Consensus Strategy - Corrected for proper storage cost handling

    Decision Logic:
    1. Calculate net benefit AND consensus metrics
    2. PRIMARY: Net benefit determines hold vs sell
    3. SECONDARY: Consensus strength adjusts batch sizing

    Parameters (all exposed for grid search):
    - consensus_threshold: % of predictions that must be bullish (default: 0.70)
    - min_return: Minimum required return % (default: 0.03)
    - min_net_benefit: Minimum $ benefit to hold (default: 50)
    - evaluation_day: Which day's predictions to evaluate (default: 14)
    - high_confidence_cv: CV threshold for high confidence (default: 0.05)
    - very_strong_consensus: Threshold for very strong consensus (default: 0.80)
    - moderate_consensus: Threshold for moderate consensus (default: 0.60)
    - batch_strong_consensus: Batch for strong consensus (default: 0.0)
    - batch_moderate: Batch for moderate consensus (default: 0.15)
    - batch_weak: Batch for weak consensus (default: 0.25)
    - batch_bearish: Batch for bearish consensus (default: 0.35)
    - cooldown_days: Days between trades (default: 7)
    """

    def __init__(self,
                 storage_cost_pct_per_day,
                 transaction_cost_pct,
                 consensus_threshold=0.70,
                 min_return=0.03,
                 min_net_benefit=50,
                 evaluation_day=14,
                 high_confidence_cv=0.05,
                 very_strong_consensus=0.80,
                 moderate_consensus=0.60,
                 batch_strong_consensus=0.0,
                 batch_moderate=0.15,
                 batch_weak=0.25,
                 batch_bearish=0.35,
                 cooldown_days=7):
        super().__init__("Consensus (Corrected)")

        # Cost parameters
        self.storage_cost_pct_per_day = storage_cost_pct_per_day
        self.transaction_cost_pct = transaction_cost_pct

        # Decision thresholds
        self.consensus_threshold = consensus_threshold
        self.min_return = min_return
        self.min_net_benefit = min_net_benefit
        self.evaluation_day = evaluation_day
        self.high_confidence_cv = high_confidence_cv
        self.very_strong_consensus = very_strong_consensus
        self.moderate_consensus = moderate_consensus

        # Batch sizing
        self.batch_strong_consensus = batch_strong_consensus
        self.batch_moderate = batch_moderate
        self.batch_weak = batch_weak
        self.batch_bearish = batch_bearish

        self.cooldown_days = cooldown_days
        self.last_sale_day = -self.cooldown_days

    def decide(self, day, inventory, current_price, price_history, predictions=None):
        if inventory <= 0:
            return {'action': 'HOLD', 'amount': 0, 'reason': 'no_inventory'}

        forced = self._force_liquidation_check(day, inventory)
        if forced:
            return forced

        days_since_sale = day - self.last_sale_day

        if days_since_sale < self.cooldown_days:
            return {'action': 'HOLD', 'amount': 0,
                   'reason': f'cooldown_{self.cooldown_days - days_since_sale}d'}

        if predictions is None or predictions.size == 0:
            if days_since_sale >= 30:
                return self._execute_trade(day, inventory, 0.20, 'no_predictions_fallback')
            return {'action': 'HOLD', 'amount': 0, 'reason': 'no_predictions_waiting'}

        batch_size, reason = self._analyze_consensus(
            current_price, price_history, predictions
        )

        if batch_size > 0:
            return self._execute_trade(day, inventory, batch_size, reason)
        else:
            return {'action': 'HOLD', 'amount': 0, 'reason': reason}

    def _analyze_consensus(self, current_price, price_history, predictions):
        """
        PRIMARY: Net benefit determines hold vs sell
        SECONDARY: Consensus strength adjusts batch sizing
        """

        # Calculate confidence
        cv_pred = calculate_prediction_confidence(
            predictions,
            horizon_day=min(self.evaluation_day-1, predictions.shape[1]-1)
        )

        # Get consensus metrics
        eval_day_idx = min(self.evaluation_day, predictions.shape[1]) - 1
        day_preds = predictions[:, eval_day_idx]
        median_pred = np.median(day_preds)
        expected_return = (median_pred - current_price) / current_price
        bullish_pct = np.mean(day_preds > current_price)

        # Calculate net benefit (includes storage costs)
        net_benefit = self._calculate_net_benefit(current_price, predictions)

        # PRIMARY DECISION: Is there net benefit?
        if net_benefit > self.min_net_benefit:
            # Profitable to wait - use consensus to adjust batch
            if (bullish_pct >= self.very_strong_consensus and
                expected_return >= self.min_return and
                cv_pred < self.high_confidence_cv):
                # Very strong consensus + high confidence
                batch_size = self.batch_strong_consensus
                reason = f'very_strong_consensus_{bullish_pct:.0%}_net_benefit_${net_benefit:.0f}_hold'
            elif (bullish_pct >= self.consensus_threshold and
                  expected_return >= self.min_return):
                # Strong consensus
                batch_size = self.batch_strong_consensus
                reason = f'strong_consensus_{bullish_pct:.0%}_net_benefit_${net_benefit:.0f}_hold'
            elif bullish_pct >= self.moderate_consensus:
                # Moderate consensus - small hedge
                batch_size = self.batch_moderate
                reason = f'moderate_consensus_{bullish_pct:.0%}_partial_hold'
            else:
                # Weak consensus despite positive net benefit
                batch_size = self.batch_weak
                reason = f'weak_consensus_{bullish_pct:.0%}_despite_positive_benefit'

        elif net_benefit < 0:
            # Negative net benefit - sell regardless of consensus
            if bullish_pct < 0.40:
                # Bearish + negative benefit
                batch_size = self.batch_bearish
                reason = f'bearish_{bullish_pct:.0%}_negative_benefit_${net_benefit:.0f}'
            else:
                # Mixed signals but costs exceed gains
                batch_size = self.batch_weak
                reason = f'negative_benefit_${net_benefit:.0f}_storage_exceeds_gains'

        else:
            # Marginal benefit
            batch_size = self.batch_moderate
            reason = f'marginal_benefit_${net_benefit:.0f}_consensus_{bullish_pct:.0%}'

        return batch_size, reason

    def _calculate_net_benefit(self, current_price, predictions):
        """Calculate max net benefit across forecast horizon"""
        ev_by_day = []
        for h in range(predictions.shape[1]):
            future_price = np.median(predictions[:, h])
            days_to_wait = h + 1
            storage_cost = current_price * (self.storage_cost_pct_per_day / 100) * days_to_wait
            transaction_cost = future_price * (self.transaction_cost_pct / 100)
            ev = future_price - storage_cost - transaction_cost
            ev_by_day.append(ev)

        transaction_cost_today = current_price * (self.transaction_cost_pct / 100)
        ev_today = current_price - transaction_cost_today

        return max(ev_by_day) - ev_today

    def _execute_trade(self, day, inventory, batch_size, reason):
        amount = inventory * batch_size
        self.last_sale_day = day
        return {'action': 'SELL', 'amount': amount, 'reason': reason}

    def reset(self):
        super().reset()
        self.last_sale_day = -self.cooldown_days


# =============================================================================
# CORRECTED: RISK-ADJUSTED STRATEGY
# =============================================================================

class RiskAdjustedStrategyCorrected(Strategy):
    """
    Risk-Adjusted Strategy - Corrected for proper storage cost handling

    Decision Logic:
    1. Calculate net benefit AND risk metrics
    2. PRIMARY: Net benefit determines hold vs sell
    3. SECONDARY: Risk level adjusts batch sizing

    Parameters (all exposed for grid search):
    - min_return: Minimum required return % (default: 0.03)
    - max_uncertainty_low: CV threshold for low risk (default: 0.05)
    - max_uncertainty_medium: CV threshold for medium risk (default: 0.10)
    - max_uncertainty_high: CV threshold for high risk (default: 0.20)
    - min_net_benefit: Minimum $ benefit to hold (default: 50)
    - strong_trend_adx: ADX threshold for strong trend (default: 25)
    - evaluation_day: Which day to evaluate (default: 14)
    - batch_low_risk: Batch for low risk (default: 0.0)
    - batch_medium_risk: Batch for medium risk (default: 0.10)
    - batch_high_risk: Batch for high risk (default: 0.25)
    - batch_very_high_risk: Batch for very high risk (default: 0.35)
    - cooldown_days: Days between trades (default: 7)
    """

    def __init__(self,
                 storage_cost_pct_per_day,
                 transaction_cost_pct,
                 min_return=0.03,
                 max_uncertainty_low=0.05,
                 max_uncertainty_medium=0.10,
                 max_uncertainty_high=0.20,
                 min_net_benefit=50,
                 strong_trend_adx=25,
                 evaluation_day=14,
                 batch_low_risk=0.0,
                 batch_medium_risk=0.10,
                 batch_high_risk=0.25,
                 batch_very_high_risk=0.35,
                 cooldown_days=7):
        super().__init__("Risk-Adjusted (Corrected)")

        # Cost parameters
        self.storage_cost_pct_per_day = storage_cost_pct_per_day
        self.transaction_cost_pct = transaction_cost_pct

        # Decision thresholds
        self.min_return = min_return
        self.max_uncertainty_low = max_uncertainty_low
        self.max_uncertainty_medium = max_uncertainty_medium
        self.max_uncertainty_high = max_uncertainty_high
        self.min_net_benefit = min_net_benefit
        self.strong_trend_adx = strong_trend_adx
        self.evaluation_day = evaluation_day

        # Batch sizing
        self.batch_low_risk = batch_low_risk
        self.batch_medium_risk = batch_medium_risk
        self.batch_high_risk = batch_high_risk
        self.batch_very_high_risk = batch_very_high_risk

        self.cooldown_days = cooldown_days
        self.last_sale_day = -self.cooldown_days

    def decide(self, day, inventory, current_price, price_history, predictions=None):
        if inventory <= 0:
            return {'action': 'HOLD', 'amount': 0, 'reason': 'no_inventory'}

        forced = self._force_liquidation_check(day, inventory)
        if forced:
            return forced

        days_since_sale = day - self.last_sale_day

        if days_since_sale < self.cooldown_days:
            return {'action': 'HOLD', 'amount': 0,
                   'reason': f'cooldown_{self.cooldown_days - days_since_sale}d'}

        if predictions is None or predictions.size == 0:
            if days_since_sale >= 30:
                return self._execute_trade(day, inventory, 0.18, 'no_predictions_fallback')
            return {'action': 'HOLD', 'amount': 0, 'reason': 'no_predictions_waiting'}

        batch_size, reason = self._analyze_risk_adjusted(
            current_price, price_history, predictions
        )

        if batch_size > 0:
            return self._execute_trade(day, inventory, batch_size, reason)
        else:
            return {'action': 'HOLD', 'amount': 0, 'reason': reason}

    def _analyze_risk_adjusted(self, current_price, price_history, predictions):
        """
        PRIMARY: Net benefit determines hold vs sell
        SECONDARY: Risk level (CV) adjusts batch sizing
        """

        # Calculate risk metrics
        cv_pred = calculate_prediction_confidence(
            predictions,
            horizon_day=min(self.evaluation_day-1, predictions.shape[1]-1)
        )

        adx_pred, _, _ = calculate_adx(
            pd.DataFrame({'price': [np.median(predictions[:, h])
                                   for h in range(predictions.shape[1])]})
        )

        # Get expected return
        eval_day_idx = min(self.evaluation_day, predictions.shape[1]) - 1
        day_preds = predictions[:, eval_day_idx]
        median_pred = np.median(day_preds)
        expected_return = (median_pred - current_price) / current_price

        # Calculate net benefit (includes storage costs)
        net_benefit = self._calculate_net_benefit(current_price, predictions)

        # PRIMARY DECISION: Is there net benefit?
        if net_benefit > self.min_net_benefit and expected_return >= self.min_return:
            # Profitable to wait - use risk to adjust batch
            if cv_pred < self.max_uncertainty_low and adx_pred > self.strong_trend_adx:
                # Very low risk
                batch_size = self.batch_low_risk
                reason = f'very_low_risk_cv{cv_pred:.1%}_net_benefit_${net_benefit:.0f}_hold'
            elif cv_pred < self.max_uncertainty_medium:
                # Low to medium risk
                batch_size = self.batch_low_risk
                reason = f'low_risk_cv{cv_pred:.1%}_net_benefit_${net_benefit:.0f}_hold'
            elif cv_pred < self.max_uncertainty_high:
                # Medium risk - small hedge
                batch_size = self.batch_medium_risk
                reason = f'medium_risk_cv{cv_pred:.1%}_partial_hold'
            else:
                # Higher risk - larger hedge
                batch_size = self.batch_high_risk
                reason = f'high_risk_cv{cv_pred:.1%}_larger_hedge'

        elif net_benefit < 0:
            # Negative net benefit - sell based on risk level
            if cv_pred > self.max_uncertainty_high:
                # High uncertainty + negative benefit
                batch_size = self.batch_very_high_risk
                reason = f'very_high_risk_cv{cv_pred:.1%}_negative_benefit_${net_benefit:.0f}'
            else:
                # Moderate uncertainty but costs exceed gains
                batch_size = self.batch_high_risk
                reason = f'negative_benefit_${net_benefit:.0f}_storage_exceeds_gains'

        else:
            # Marginal benefit or low expected return
            batch_size = self.batch_medium_risk
            reason = f'marginal_benefit_${net_benefit:.0f}_risk_cv{cv_pred:.1%}'

        return batch_size, reason

    def _calculate_net_benefit(self, current_price, predictions):
        """Calculate max net benefit across forecast horizon"""
        ev_by_day = []
        for h in range(predictions.shape[1]):
            future_price = np.median(predictions[:, h])
            days_to_wait = h + 1
            storage_cost = current_price * (self.storage_cost_pct_per_day / 100) * days_to_wait
            transaction_cost = future_price * (self.transaction_cost_pct / 100)
            ev = future_price - storage_cost - transaction_cost
            ev_by_day.append(ev)

        transaction_cost_today = current_price * (self.transaction_cost_pct / 100)
        ev_today = current_price - transaction_cost_today

        return max(ev_by_day) - ev_today

    def _execute_trade(self, day, inventory, batch_size, reason):
        amount = inventory * batch_size
        self.last_sale_day = day
        return {'action': 'SELL', 'amount': amount, 'reason': reason}

    def reset(self):
        super().reset()
        self.last_sale_day = -self.cooldown_days

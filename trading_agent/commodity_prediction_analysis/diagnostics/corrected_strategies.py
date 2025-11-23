"""
Corrected Trading Strategies - Percentage-Based Decision Framework

**Purpose:** Prediction strategies with rational cost-benefit analysis

**Key Design Principles:**
1. NET BENEFIT as PERCENTAGE of current price (scale-invariant)
2. Compare percentage returns to percentage costs (apples-to-apples)
3. Full parameterization (everything exposed for grid search)
4. Reasonable batch sizing (0.0 to ~0.40, rarely sell all inventory)

**The Fix:**
- Calculate net_benefit_pct = (EV_future - EV_today) / current_price
- Compare to min_net_benefit_pct (e.g., 0.5%)
- This is comparable to storage costs (0.025%/day) and transaction costs (0.25%)

**Example:**
- Predictions show +2% price increase over 14 days
- Storage cost: 0.025% × 14 = 0.35%
- Transaction cost: 0.25%
- Net benefit: +2% - 0.6% = +1.4%
- If min_net_benefit_pct = 0.5%: HOLD (rational!)
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

class BaseStrategy(ABC):
    def __init__(self, name):
        self.name = name
        self.harvest_start = None

    @abstractmethod
    def decide(self, day, inventory, current_price, price_history, predictions=None):
        pass

    def reset(self):
        self.harvest_start = None

    def set_harvest_start(self, day):
        self.harvest_start = day

    def _force_liquidation_check(self, day, inventory):
        """Force liquidation approaching day 365"""
        if self.harvest_start is not None:
            days_since_harvest = day - self.harvest_start
            if days_since_harvest >= 365:
                return {'action': 'SELL', 'amount': inventory,
                       'reason': 'forced_liquidation_365d'}
            elif days_since_harvest >= 345:
                days_left = 365 - days_since_harvest
                return {'action': 'SELL', 'amount': inventory * 0.05,
                       'reason': f'approaching_365d_deadline_{days_left}d_left'}
        return None


# =============================================================================
# CORRECTED: EXPECTED VALUE STRATEGY (PERCENTAGE-BASED)
# =============================================================================

class ExpectedValueStrategyCorrected(BaseStrategy):
    """
    Expected value strategy with percentage-based decision framework.

    Decision Logic:
    1. Calculate net_benefit as % of current price
    2. If net_benefit_pct > min_threshold: HOLD (profitable to wait)
    3. If net_benefit_pct < 0: SELL (storage costs exceed gains)
    4. Confidence modulates batch sizing (not primary decision)

    Parameters (all exposed for grid search):
    - Cost parameters: storage_cost_pct_per_day, transaction_cost_pct
    - Decision thresholds: min_net_benefit_pct, negative_threshold_pct
    - Confidence thresholds: high_confidence_cv, medium_confidence_cv
    - Trend threshold: strong_trend_adx
    - Batch sizing: 5 different batch sizes for different scenarios
    - Timing: cooldown_days, baseline_batch, baseline_frequency
    """

    def __init__(self,
                 storage_cost_pct_per_day,
                 transaction_cost_pct,
                 # PERCENTAGE thresholds (scale-invariant)
                 min_net_benefit_pct=0.5,          # Min % return to hold (0.5% = 50 basis points)
                 negative_threshold_pct=-0.3,       # Sell aggressively below this % (-0.3%)
                 # Confidence thresholds (CV = std/mean)
                 high_confidence_cv=0.05,           # CV < 5% = high confidence
                 medium_confidence_cv=0.10,         # CV < 10% = medium confidence
                 # Trend strength (ADX)
                 strong_trend_adx=25,               # ADX > 25 = strong trend
                 # Batch sizing (0.0 to ~0.40, rarely sell all)
                 batch_positive_confident=0.0,      # Hold all when profitable + confident
                 batch_positive_uncertain=0.10,     # Small hedge when profitable + uncertain
                 batch_marginal=0.15,               # Gradual liquidation near zero
                 batch_negative_mild=0.25,          # Sell when mildly negative
                 batch_negative_strong=0.35,        # Sell aggressively when strongly negative
                 # Timing parameters
                 cooldown_days=7,                   # Days between trades
                 baseline_batch=0.15,               # Fallback batch size
                 baseline_frequency=30):            # Days before fallback

        super().__init__("Expected Value (Corrected)")

        # Cost parameters
        self.storage_cost_pct_per_day = storage_cost_pct_per_day
        self.transaction_cost_pct = transaction_cost_pct

        # PERCENTAGE-based decision thresholds
        self.min_net_benefit_pct = min_net_benefit_pct
        self.negative_threshold_pct = negative_threshold_pct

        # Confidence and trend thresholds
        self.high_confidence_cv = high_confidence_cv
        self.medium_confidence_cv = medium_confidence_cv
        self.strong_trend_adx = strong_trend_adx

        # Batch sizing (secondary modulation)
        self.batch_positive_confident = batch_positive_confident
        self.batch_positive_uncertain = batch_positive_uncertain
        self.batch_marginal = batch_marginal
        self.batch_negative_mild = batch_negative_mild
        self.batch_negative_strong = batch_negative_strong

        # Timing
        self.cooldown_days = cooldown_days
        self.baseline_batch = baseline_batch
        self.baseline_frequency = baseline_frequency

        self.last_sale_day = -self.cooldown_days

    def decide(self, day, inventory, current_price, price_history, predictions=None):
        if inventory <= 0:
            return {'action': 'HOLD', 'amount': 0, 'reason': 'no_inventory'}

        # Check forced liquidation
        forced = self._force_liquidation_check(day, inventory)
        if forced:
            return forced

        days_since_sale = day - self.last_sale_day

        # Cooldown period
        if days_since_sale < self.cooldown_days:
            return {'action': 'HOLD', 'amount': 0,
                   'reason': f'cooldown_{self.cooldown_days - days_since_sale}d'}

        # No predictions fallback
        if predictions is None or predictions.size == 0:
            if days_since_sale >= self.baseline_frequency:
                return self._execute_trade(day, inventory, self.baseline_batch,
                                          'no_predictions_fallback')
            return {'action': 'HOLD', 'amount': 0, 'reason': 'no_predictions_waiting'}

        # Main decision logic (PERCENTAGE-BASED)
        batch_size, reason = self._analyze_expected_value_pct(
            current_price, price_history, predictions
        )

        return self._execute_trade(day, inventory, batch_size, reason)

    def _analyze_expected_value_pct(self, current_price, price_history, predictions):
        """
        Calculate expected value and make decision based on PERCENTAGE returns.

        Returns: (batch_size, reason)
        """
        # Find optimal day and net benefit as PERCENTAGE
        optimal_day, net_benefit_pct = self._find_optimal_sale_day_pct(
            current_price, predictions
        )

        # Calculate confidence and trend indicators (secondary)
        cv_pred = calculate_prediction_confidence(
            predictions,
            horizon_day=min(13, predictions.shape[1]-1)
        )
        adx_pred, _, _ = calculate_adx(price_history, period=min(14, len(price_history)-1))

        # PRIMARY DECISION: Is net benefit percentage positive and significant?
        if net_benefit_pct > self.min_net_benefit_pct:
            # Profitable to wait - modulate batch by confidence

            if cv_pred < self.high_confidence_cv and adx_pred > self.strong_trend_adx:
                # High confidence + strong trend: hold all inventory
                batch_size = self.batch_positive_confident
                reason = f'net_benefit_{net_benefit_pct:.2f}%_high_conf_hold_to_day{optimal_day}'

            elif cv_pred < self.medium_confidence_cv:
                # Medium confidence: small hedge
                batch_size = self.batch_positive_uncertain
                reason = f'net_benefit_{net_benefit_pct:.2f}%_med_conf_small_hedge_day{optimal_day}'

            else:
                # Lower confidence: larger hedge
                batch_size = self.batch_marginal
                reason = f'net_benefit_{net_benefit_pct:.2f}%_low_conf_hedge'

        elif net_benefit_pct > 0:
            # Positive but below threshold - marginal case
            batch_size = self.batch_marginal
            reason = f'marginal_benefit_{net_benefit_pct:.2f}%_gradual_liquidation'

        elif net_benefit_pct > self.negative_threshold_pct:
            # Mildly negative - storage costs slightly exceed gains
            batch_size = self.batch_negative_mild
            reason = f'mild_negative_{net_benefit_pct:.2f}%_avoid_storage'

        else:
            # Strongly negative - storage costs far exceed gains
            batch_size = self.batch_negative_strong
            reason = f'strong_negative_{net_benefit_pct:.2f}%_sell_to_cut_losses'

        return batch_size, reason

    def _find_optimal_sale_day_pct(self, current_price, predictions):
        """
        Find optimal sale day and calculate net benefit as PERCENTAGE.

        Returns: (optimal_day, net_benefit_pct)
        where net_benefit_pct is the % return relative to current price
        """
        ev_by_day = []
        for h in range(predictions.shape[1]):
            future_price = np.median(predictions[:, h])
            days_to_wait = h + 1

            # Storage cost accumulates linearly (in cents/lb)
            storage_cost = current_price * (self.storage_cost_pct_per_day / 100) * days_to_wait
            # Transaction cost as % of sale value
            transaction_cost = future_price * (self.transaction_cost_pct / 100)

            # Net expected value per pound
            ev = future_price - storage_cost - transaction_cost
            ev_by_day.append(ev)

        # EV of selling today
        transaction_cost_today = current_price * (self.transaction_cost_pct / 100)
        ev_today = current_price - transaction_cost_today

        # Find optimal day
        optimal_day = np.argmax(ev_by_day)

        # Calculate net benefit as PERCENTAGE of current price
        net_benefit_pct = 100 * (ev_by_day[optimal_day] - ev_today) / current_price

        return optimal_day, net_benefit_pct

    def _execute_trade(self, day, inventory, batch_size, reason):
        amount = inventory * batch_size
        self.last_sale_day = day
        return {'action': 'SELL', 'amount': amount, 'reason': reason}

    def reset(self):
        super().reset()
        self.last_sale_day = -self.cooldown_days


# =============================================================================
# CORRECTED: CONSENSUS STRATEGY (PERCENTAGE-BASED)
# =============================================================================

class ConsensusStrategyCorrected(BaseStrategy):
    """
    Consensus strategy with percentage-based decision framework.

    Decision Logic:
    1. Count % of predictions that are bullish (above min_return)
    2. If consensus >= threshold AND net_benefit_pct > min: HOLD
    3. If bearish consensus: SELL
    4. Batch size modulated by consensus strength

    All thresholds are percentages for scale-invariance.
    """

    def __init__(self,
                 storage_cost_pct_per_day,
                 transaction_cost_pct,
                 # Consensus thresholds
                 consensus_threshold=0.70,           # 70% agreement to act
                 very_strong_consensus=0.85,         # 85% = very strong
                 moderate_consensus=0.60,            # 60% = moderate
                 # Percentage-based decision thresholds
                 min_return=0.03,                    # 3% minimum return
                 min_net_benefit_pct=0.5,            # 0.5% minimum net benefit
                 # Confidence threshold
                 high_confidence_cv=0.05,            # CV < 5% = high confidence
                 # Which day to evaluate
                 evaluation_day=14,
                 # Batch sizing (0.0 to ~0.40)
                 batch_strong_consensus=0.0,         # Hold when very strong consensus
                 batch_moderate=0.15,                # Gradual when moderate
                 batch_weak=0.25,                    # Sell when weak consensus
                 batch_bearish=0.35,                 # Sell aggressively when bearish
                 # Timing
                 cooldown_days=7):

        super().__init__("Consensus (Corrected)")

        # Cost parameters
        self.storage_cost_pct_per_day = storage_cost_pct_per_day
        self.transaction_cost_pct = transaction_cost_pct

        # Consensus thresholds
        self.consensus_threshold = consensus_threshold
        self.very_strong_consensus = very_strong_consensus
        self.moderate_consensus = moderate_consensus

        # PERCENTAGE-based decision thresholds
        self.min_return = min_return
        self.min_net_benefit_pct = min_net_benefit_pct

        # Confidence
        self.high_confidence_cv = high_confidence_cv

        # Evaluation parameters
        self.evaluation_day = evaluation_day

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

        batch_size, reason = self._analyze_consensus_pct(
            current_price, price_history, predictions
        )

        return self._execute_trade(day, inventory, batch_size, reason)

    def _analyze_consensus_pct(self, current_price, price_history, predictions):
        """Calculate consensus and make percentage-based decision"""

        # Evaluate at specific day
        eval_day = min(self.evaluation_day, predictions.shape[1] - 1)
        day_predictions = predictions[:, eval_day]

        # Calculate expected return as percentage
        median_future = np.median(day_predictions)
        expected_return_pct = (median_future - current_price) / current_price

        # Count bullish predictions (those showing sufficient return)
        bullish_count = np.sum(
            (day_predictions - current_price) / current_price > self.min_return
        )
        bullish_pct = bullish_count / len(day_predictions)

        # Calculate confidence
        cv = calculate_prediction_confidence(predictions, eval_day)

        # Calculate net benefit accounting for costs
        days_to_wait = eval_day + 1
        storage_cost_pct = (self.storage_cost_pct_per_day / 100) * days_to_wait
        transaction_cost_pct = self.transaction_cost_pct / 100
        net_benefit_pct = 100 * (expected_return_pct - storage_cost_pct - transaction_cost_pct)

        # Decision based on consensus and net benefit
        if bullish_pct >= self.very_strong_consensus and net_benefit_pct > self.min_net_benefit_pct:
            # Very strong consensus + positive net benefit
            batch_size = self.batch_strong_consensus
            reason = f'very_strong_consensus_{bullish_pct:.0%}_net_{net_benefit_pct:.2f}%_hold'

        elif bullish_pct >= self.consensus_threshold and net_benefit_pct > self.min_net_benefit_pct:
            # Strong consensus + positive net benefit
            if cv < self.high_confidence_cv:
                batch_size = self.batch_strong_consensus
                reason = f'strong_consensus_{bullish_pct:.0%}_high_conf_hold'
            else:
                batch_size = self.batch_moderate
                reason = f'strong_consensus_{bullish_pct:.0%}_med_conf_gradual'

        elif bullish_pct >= self.moderate_consensus:
            # Moderate consensus
            batch_size = self.batch_moderate
            reason = f'moderate_consensus_{bullish_pct:.0%}_gradual'

        elif bullish_pct < (1 - self.consensus_threshold):
            # Bearish consensus (most predictions negative)
            batch_size = self.batch_bearish
            reason = f'bearish_consensus_{bullish_pct:.0%}_sell'

        else:
            # Weak/unclear consensus
            batch_size = self.batch_weak
            reason = f'weak_consensus_{bullish_pct:.0%}_sell'

        return batch_size, reason

    def _execute_trade(self, day, inventory, batch_size, reason):
        amount = inventory * batch_size
        self.last_sale_day = day
        return {'action': 'SELL', 'amount': amount, 'reason': reason}

    def reset(self):
        super().reset()
        self.last_sale_day = -self.cooldown_days


# =============================================================================
# CORRECTED: RISK-ADJUSTED STRATEGY (PERCENTAGE-BASED)
# =============================================================================

class RiskAdjustedStrategyCorrected(BaseStrategy):
    """
    Risk-adjusted strategy with percentage-based framework.

    Decision Logic:
    1. Evaluate expected return as percentage
    2. Measure uncertainty (CV)
    3. If return > threshold AND uncertainty < max AND net_benefit_pct > min: HOLD
    4. Batch size based on risk tier (low/medium/high/very high)

    All thresholds in percentages.
    """

    def __init__(self,
                 storage_cost_pct_per_day,
                 transaction_cost_pct,
                 # Return threshold (percentage)
                 min_return=0.03,                    # 3% minimum return
                 min_net_benefit_pct=0.5,            # 0.5% minimum net benefit
                 # Uncertainty thresholds (CV levels)
                 max_uncertainty_low=0.05,           # CV < 5% = low risk
                 max_uncertainty_medium=0.10,        # CV < 10% = medium risk
                 max_uncertainty_high=0.20,          # CV < 20% = high risk
                 # Trend strength
                 strong_trend_adx=25,
                 # Evaluation day
                 evaluation_day=14,
                 # Batch sizing by risk tier
                 batch_low_risk=0.0,                 # Hold when low risk
                 batch_medium_risk=0.10,             # Small hedge medium risk
                 batch_high_risk=0.25,               # Sell more at high risk
                 batch_very_high_risk=0.35,          # Aggressive at very high risk
                 # Timing
                 cooldown_days=7):

        super().__init__("Risk-Adjusted (Corrected)")

        # Cost parameters
        self.storage_cost_pct_per_day = storage_cost_pct_per_day
        self.transaction_cost_pct = transaction_cost_pct

        # PERCENTAGE-based thresholds
        self.min_return = min_return
        self.min_net_benefit_pct = min_net_benefit_pct

        # Uncertainty (risk) thresholds
        self.max_uncertainty_low = max_uncertainty_low
        self.max_uncertainty_medium = max_uncertainty_medium
        self.max_uncertainty_high = max_uncertainty_high

        # Trend
        self.strong_trend_adx = strong_trend_adx

        # Evaluation
        self.evaluation_day = evaluation_day

        # Batch sizing by risk tier
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
                return self._execute_trade(day, inventory, 0.20, 'no_predictions_fallback')
            return {'action': 'HOLD', 'amount': 0, 'reason': 'no_predictions_waiting'}

        batch_size, reason = self._analyze_risk_adjusted_pct(
            current_price, price_history, predictions
        )

        return self._execute_trade(day, inventory, batch_size, reason)

    def _analyze_risk_adjusted_pct(self, current_price, price_history, predictions):
        """Risk-adjusted decision with percentage-based logic"""

        # Evaluate at specific day
        eval_day = min(self.evaluation_day, predictions.shape[1] - 1)
        day_predictions = predictions[:, eval_day]

        # Calculate expected return as percentage
        median_future = np.median(day_predictions)
        expected_return_pct = (median_future - current_price) / current_price

        # Measure uncertainty (risk)
        cv = calculate_prediction_confidence(predictions, eval_day)

        # Calculate net benefit accounting for costs
        days_to_wait = eval_day + 1
        storage_cost_pct = (self.storage_cost_pct_per_day / 100) * days_to_wait
        transaction_cost_pct = self.transaction_cost_pct / 100
        net_benefit_pct = 100 * (expected_return_pct - storage_cost_pct - transaction_cost_pct)

        # Get trend strength
        adx, _, _ = calculate_adx(price_history, period=min(14, len(price_history)-1))

        # Decision based on risk tier and net benefit
        if expected_return_pct >= self.min_return and net_benefit_pct > self.min_net_benefit_pct:
            # Sufficient expected return and net benefit

            if cv < self.max_uncertainty_low and adx > self.strong_trend_adx:
                # Low risk + strong trend: hold all
                batch_size = self.batch_low_risk
                reason = f'low_risk_cv{cv:.2%}_return{expected_return_pct:.2%}_hold'

            elif cv < self.max_uncertainty_medium:
                # Medium risk: small hedge
                batch_size = self.batch_medium_risk
                reason = f'medium_risk_cv{cv:.2%}_return{expected_return_pct:.2%}_small_hedge'

            elif cv < self.max_uncertainty_high:
                # High risk: larger hedge
                batch_size = self.batch_high_risk
                reason = f'high_risk_cv{cv:.2%}_return{expected_return_pct:.2%}_hedge'

            else:
                # Very high risk: sell aggressively
                batch_size = self.batch_very_high_risk
                reason = f'very_high_risk_cv{cv:.2%}_sell'

        else:
            # Insufficient return or negative net benefit
            if net_benefit_pct < 0:
                batch_size = self.batch_very_high_risk
                reason = f'negative_net_benefit_{net_benefit_pct:.2f}%_sell'
            else:
                batch_size = self.batch_high_risk
                reason = f'insufficient_return_{expected_return_pct:.2%}_sell'

        return batch_size, reason

    def _execute_trade(self, day, inventory, batch_size, reason):
        amount = inventory * batch_size
        self.last_sale_day = day
        return {'action': 'SELL', 'amount': amount, 'reason': reason}

    def reset(self):
        super().reset()
        self.last_sale_day = -self.cooldown_days

"""
Production Strategy Implementations
Extracted from diagnostics/all_strategies_pct.py (improved version)

**Complete Strategy Suite:**
- 4 Baseline strategies
- 5 Prediction strategies
- 2 Perfect foresight strategies (for theoretical maximum calculation)
- Technical indicators
"""

from .base import Strategy
from .baseline import (
    ImmediateSaleStrategy,
    EqualBatchStrategy,
    PriceThresholdStrategy,
    MovingAverageStrategy
)
from .prediction import (
    PriceThresholdPredictive,
    MovingAveragePredictive,
    ExpectedValueStrategy,
    ConsensusStrategy,
    RiskAdjustedStrategy
)
from .perfect_foresight import (
    PerfectForesightStrategy,
    GreedyPerfectForesightStrategy
)
from .indicators import (
    calculate_rsi,
    calculate_adx,
    calculate_prediction_confidence
)

__all__ = [
    # Base
    'Strategy',

    # Baseline strategies (4)
    'ImmediateSaleStrategy',
    'EqualBatchStrategy',
    'PriceThresholdStrategy',
    'MovingAverageStrategy',

    # Prediction strategies (5)
    'PriceThresholdPredictive',
    'MovingAveragePredictive',
    'ExpectedValueStrategy',
    'ConsensusStrategy',
    'RiskAdjustedStrategy',

    # Perfect foresight strategies (2) - for theoretical max
    'PerfectForesightStrategy',
    'GreedyPerfectForesightStrategy',

    # Indicators
    'calculate_rsi',
    'calculate_adx',
    'calculate_prediction_confidence'
]

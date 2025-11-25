"""
Production Strategy Implementations
Extracted from diagnostics/all_strategies_pct.py (improved version)

**Complete Strategy Suite:**
- 4 Baseline strategies
- 5 Prediction strategies
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

    # Indicators
    'calculate_rsi',
    'calculate_adx',
    'calculate_prediction_confidence'
]

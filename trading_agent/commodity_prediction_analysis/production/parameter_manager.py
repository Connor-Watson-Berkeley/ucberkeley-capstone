"""
Parameter Manager

Intelligent parameter management for production backtesting that automatically uses
optimized parameters when available, with graceful fallback to defaults.

Design:
    - Single source of truth for strategy parameters
    - Automatic discovery of optimized parameters from optimizer output
    - Graceful fallback to hardcoded defaults
    - Clear logging of parameter source
    - Support for multiple optimization objectives

Integration:
    production/config.py (defaults)
        ↓
    production/parameter_manager.py (load optimized if available)
        ↓
    production/runners/*.py (use managed parameters)

Usage:
    from production.parameter_manager import ParameterManager

    # Initialize with defaults
    pm = ParameterManager(commodity='coffee', model_version='arima_v1')

    # Get parameters (automatically loads optimized if available)
    baseline_params = pm.get_baseline_params()
    prediction_params = pm.get_prediction_params()

    # Or force specific source
    params = pm.get_params(source='optimized')  # Use optimized only
    params = pm.get_params(source='default')    # Use defaults only
    params = pm.get_params(source='auto')       # Auto (optimized if available)
"""

import os
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, Literal
import warnings

from production.config import (
    BASELINE_PARAMS,
    PREDICTION_PARAMS,
    VOLUME_PATH,
    COMMODITY_CONFIGS
)


class ParameterManager:
    """
    Manages strategy parameters with automatic optimization integration.

    Features:
        - Loads optimized parameters when available
        - Falls back to defaults gracefully
        - Validates parameter compatibility
        - Logs parameter sources for transparency
        - Supports multiple optimization objectives
    """

    def __init__(
        self,
        commodity: str,
        model_version: str = 'arima_v1',
        optimization_objective: str = 'efficiency',
        volume_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize parameter manager.

        Args:
            commodity: Commodity name (e.g., 'coffee', 'sugar')
            model_version: Model version (default: 'arima_v1')
            optimization_objective: Which optimization objective to use
                                   ('efficiency', 'earnings', 'multi')
            volume_path: Override default volume path (for testing)
            verbose: Print parameter loading info
        """
        self.commodity = commodity
        self.model_version = model_version
        self.objective = optimization_objective
        self.volume_path = volume_path or VOLUME_PATH
        self.verbose = verbose

        # Validate commodity
        if commodity not in COMMODITY_CONFIGS:
            raise ValueError(
                f"Unknown commodity: {commodity}. "
                f"Available: {list(COMMODITY_CONFIGS.keys())}"
            )

        # Cache for loaded parameters
        self._optimized_params_cache = None
        self._optimized_params_loaded = False

    def get_optimized_params_path(self, version: str = 'latest', format: str = 'json') -> str:
        """
        Get file path for optimized parameters.

        Args:
            version: 'latest', 'previous', or 'timestamped' (for new JSON format)
            format: 'json' or 'pkl' (for backwards compatibility)

        Returns:
            str: Path to optimized parameters file
        """
        if format == 'json':
            # New JSON format with versioning
            if version == 'latest':
                filename = f"optuna_results_{self.commodity}_{self.model_version}_latest.json"
            elif version == 'previous':
                filename = f"optuna_results_{self.commodity}_{self.model_version}_previous.json"
            else:
                # For timestamped, caller needs to provide full filename
                filename = f"optuna_results_{self.commodity}_{self.model_version}_{version}.json"
        else:
            # Legacy pickle format (for backwards compatibility)
            filename = f"optimized_params_{self.commodity}_{self.model_version}_{self.objective}.pkl"

        return os.path.join(self.volume_path, 'optimization', filename)

    def has_optimized_params(self) -> bool:
        """
        Check if optimized parameters exist for this commodity/model/objective.

        Returns:
            bool: True if optimized params file exists (JSON or pickle)
        """
        # Check JSON first (preferred format)
        json_path = self.get_optimized_params_path(version='latest', format='json')
        if os.path.exists(json_path):
            return True

        # Fall back to legacy pickle format
        pkl_path = self.get_optimized_params_path(format='pkl')
        return os.path.exists(pkl_path)

    def load_optimized_params(self, force_reload: bool = False, version: str = 'latest') -> Optional[Dict[str, Any]]:
        """
        Load optimized parameters from optimizer output (JSON or pickle).

        Args:
            force_reload: Force reload even if cached
            version: 'latest' or 'previous' (for rollback)

        Returns:
            Dict of {strategy_name: params} or None if not available
        """
        # Return cached if available (only for 'latest')
        if version == 'latest' and self._optimized_params_loaded and not force_reload:
            return self._optimized_params_cache

        # Try JSON first (preferred format)
        json_path = self.get_optimized_params_path(version=version, format='json')

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Extract parameters from JSON structure
                # JSON format: {'strategies': {strategy_name: {'parameters': {...}, 'best_value': ...}}}
                params = {}
                for strategy_name, strategy_data in data.get('strategies', {}).items():
                    params[strategy_name] = strategy_data.get('parameters', {})

                if self.verbose:
                    print(f"✓ Loaded optimized parameters from JSON: {json_path}")
                    print(f"  Version: {version}")
                    print(f"  Strategies optimized: {len(params)}")
                    if 'execution_time' in data:
                        print(f"  Optimization date: {data['execution_time']}")
                    if 'theoretical_max' in data and data['theoretical_max']:
                        print(f"  Theoretical max: ${data['theoretical_max']:,.2f}")

                # Cache only 'latest' version
                if version == 'latest':
                    self._optimized_params_cache = params
                    self._optimized_params_loaded = True

                return params

            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to load JSON from {json_path}: {e}")
                # Continue to try pickle format below

        # Fall back to legacy pickle format (only for 'latest')
        if version == 'latest':
            pkl_path = self.get_optimized_params_path(format='pkl')

            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, 'rb') as f:
                        params = pickle.load(f)

                    if self.verbose:
                        print(f"✓ Loaded optimized parameters from pickle: {pkl_path}")
                        print(f"  Strategies optimized: {len(params)}")
                        print(f"  Optimization objective: {self.objective}")

                    self._optimized_params_cache = params
                    self._optimized_params_loaded = True
                    return params

                except Exception as e:
                    warnings.warn(
                        f"Failed to load optimized parameters from {pkl_path}: {e}. "
                        f"Falling back to defaults."
                    )

        # No valid params found
        if self.verbose:
            print(f"ℹ️  No optimized parameters found (version={version})")

        if version == 'latest':
            self._optimized_params_cache = None
            self._optimized_params_loaded = True

        return None

    def get_baseline_params(
        self,
        source: Literal['auto', 'optimized', 'default'] = 'auto'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get baseline strategy parameters.

        Args:
            source: Parameter source
                - 'auto': Use optimized if available, else default
                - 'optimized': Use only optimized (raises error if unavailable)
                - 'default': Always use default hardcoded params

        Returns:
            Dict of {strategy_name: params}
        """
        if source == 'default':
            if self.verbose:
                print("Using default baseline parameters (hardcoded)")
            return BASELINE_PARAMS.copy()

        # Try to load optimized
        optimized = self.load_optimized_params()

        if optimized is None:
            if source == 'optimized':
                raise ValueError(
                    f"Optimized parameters requested but not available for "
                    f"{self.commodity}/{self.model_version}/{self.objective}"
                )
            # Auto mode: fallback to defaults
            if self.verbose:
                print("Using default baseline parameters (optimized not available)")
            return BASELINE_PARAMS.copy()

        # Extract baseline strategy params from optimized
        baseline_names = ['immediate_sale', 'equal_batch', 'price_threshold', 'moving_average']
        baseline_params = {}

        for strategy_name in baseline_names:
            if strategy_name in optimized:
                baseline_params[strategy_name] = optimized[strategy_name]
            else:
                # Fallback to default for this specific strategy
                if strategy_name in BASELINE_PARAMS:
                    baseline_params[strategy_name] = BASELINE_PARAMS[strategy_name]
                    if self.verbose:
                        print(f"  ⚠️  {strategy_name}: Using default (not in optimized)")
                else:
                    baseline_params[strategy_name] = {}

        if self.verbose:
            optimized_count = sum(1 for name in baseline_names if name in optimized)
            print(f"Using baseline parameters: {optimized_count}/{len(baseline_names)} optimized, rest default")

        return baseline_params

    def get_prediction_params(
        self,
        source: Literal['auto', 'optimized', 'default'] = 'auto'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get prediction strategy parameters.

        Args:
            source: Parameter source (same as get_baseline_params)

        Returns:
            Dict of {strategy_name: params}
        """
        if source == 'default':
            if self.verbose:
                print("Using default prediction parameters (hardcoded)")
            return PREDICTION_PARAMS.copy()

        # Try to load optimized
        optimized = self.load_optimized_params()

        if optimized is None:
            if source == 'optimized':
                raise ValueError(
                    f"Optimized parameters requested but not available for "
                    f"{self.commodity}/{self.model_version}/{self.objective}"
                )
            # Auto mode: fallback to defaults
            if self.verbose:
                print("Using default prediction parameters (optimized not available)")
            return PREDICTION_PARAMS.copy()

        # Extract prediction strategy params from optimized
        prediction_names = [
            'price_threshold_predictive',
            'moving_average_predictive',
            'expected_value',
            'consensus',
            'risk_adjusted',
            'rolling_horizon_mpc'
        ]

        prediction_params = {}

        for strategy_name in prediction_names:
            if strategy_name in optimized:
                prediction_params[strategy_name] = optimized[strategy_name]
            else:
                # Fallback to default for this specific strategy
                if strategy_name in PREDICTION_PARAMS:
                    prediction_params[strategy_name] = PREDICTION_PARAMS[strategy_name]
                    if self.verbose:
                        print(f"  ⚠️  {strategy_name}: Using default (not in optimized)")
                else:
                    prediction_params[strategy_name] = {}

        if self.verbose:
            optimized_count = sum(1 for name in prediction_names if name in optimized)
            print(f"Using prediction parameters: {optimized_count}/{len(prediction_names)} optimized, rest default")

        return prediction_params

    def get_all_params(
        self,
        source: Literal['auto', 'optimized', 'default'] = 'auto'
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get all parameters (baseline + prediction).

        Args:
            source: Parameter source (same as get_baseline_params)

        Returns:
            Dict with keys 'baseline' and 'prediction', each containing param dicts
        """
        return {
            'baseline': self.get_baseline_params(source=source),
            'prediction': self.get_prediction_params(source=source)
        }

    def get_parameter_summary(self) -> Dict[str, Any]:
        """
        Get summary of parameter sources and availability.

        Returns:
            Dict with parameter metadata
        """
        has_optimized = self.has_optimized_params()
        optimized = self.load_optimized_params() if has_optimized else None

        summary = {
            'commodity': self.commodity,
            'model_version': self.model_version,
            'optimization_objective': self.objective,
            'has_optimized': has_optimized,
            'optimized_path': self.get_optimized_params_path(),
            'default_baseline_count': len(BASELINE_PARAMS),
            'default_prediction_count': len(PREDICTION_PARAMS),
            'optimized_count': len(optimized) if optimized else 0,
            'strategies_in_optimized': list(optimized.keys()) if optimized else []
        }

        return summary

    def print_summary(self):
        """Print human-readable parameter summary."""
        summary = self.get_parameter_summary()

        print("\n" + "=" * 80)
        print("PARAMETER MANAGER SUMMARY")
        print("=" * 80)
        print(f"Commodity:     {summary['commodity']}")
        print(f"Model Version: {summary['model_version']}")
        print(f"Optimization:  {summary['optimization_objective']}")
        print(f"\nOptimized Parameters: {'✓ Available' if summary['has_optimized'] else '✗ Not Available'}")

        if summary['has_optimized']:
            print(f"  Path: {summary['optimized_path']}")
            print(f"  Strategies: {summary['optimized_count']}")
            print(f"  Names: {', '.join(summary['strategies_in_optimized'])}")

        print(f"\nDefault Parameters:")
        print(f"  Baseline:   {summary['default_baseline_count']} strategies")
        print(f"  Prediction: {summary['default_prediction_count']} strategies")

        print("\nParameter Source (when using source='auto'):")
        if summary['has_optimized']:
            print(f"  ✓ Will use optimized parameters")
        else:
            print(f"  ⚠️  Will use default hardcoded parameters")
        print("=" * 80)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_params_for_backtest(
    commodity: str,
    model_version: str = 'arima_v1',
    optimization_objective: str = 'efficiency',
    source: Literal['auto', 'optimized', 'default'] = 'auto',
    verbose: bool = True
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Convenience function to get parameters for backtest.

    This is the main entry point for production workflows.

    Args:
        commodity: Commodity name
        model_version: Model version
        optimization_objective: Optimization objective used
        source: Parameter source ('auto' recommended)
        verbose: Print loading info

    Returns:
        Dict with 'baseline' and 'prediction' parameter dicts

    Example:
        >>> params = get_params_for_backtest('coffee', 'arima_v1', source='auto')
        >>> baseline_params = params['baseline']
        >>> prediction_params = params['prediction']
    """
    manager = ParameterManager(
        commodity=commodity,
        model_version=model_version,
        optimization_objective=optimization_objective,
        verbose=verbose
    )

    return manager.get_all_params(source=source)


def check_optimized_params_availability(
    commodity: str,
    model_version: str = 'arima_v1',
    optimization_objective: str = 'efficiency'
) -> bool:
    """
    Quick check if optimized parameters are available.

    Args:
        commodity: Commodity name
        model_version: Model version
        optimization_objective: Optimization objective

    Returns:
        bool: True if optimized params exist
    """
    manager = ParameterManager(
        commodity=commodity,
        model_version=model_version,
        optimization_objective=optimization_objective,
        verbose=False
    )

    return manager.has_optimized_params()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ParameterManager',
    'get_params_for_backtest',
    'check_optimized_params_availability'
]

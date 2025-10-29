"""Synthetic forecast generator for trading agent sensitivity analysis.

Creates forecasts with controlled error levels to test trading agent performance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class SyntheticForecastGenerator:
    """
    Generate synthetic forecasts with specified accuracy characteristics.

    Use cases:
    - Test trading agent with different error levels
    - Sensitivity analysis (what MAE is needed for profitability?)
    - Stress testing (how does agent handle bad forecasts?)
    - Directional accuracy testing (precision vs accuracy tradeoff)
    """

    def __init__(self, actuals_df: pd.DataFrame, target: str = 'close'):
        """
        Initialize with actual data.

        Args:
            actuals_df: DataFrame with actual price data
            target: Target column name
        """
        self.actuals_df = actuals_df.copy()
        self.target = target

    def generate_perfect_forecast(self, horizon: int = 14) -> pd.DataFrame:
        """
        Generate perfect forecast (actual values).

        Args:
            horizon: Days to forecast

        Returns:
            DataFrame with perfect forecasts
        """
        forecast_df = self.actuals_df.iloc[:horizon].copy()
        forecast_df['forecast'] = forecast_df[self.target]
        forecast_df['model'] = 'Perfect'
        forecast_df['mae_target'] = 0.0

        return forecast_df[['date', 'forecast', 'model', 'mae_target']]

    def generate_with_mae(self, target_mae: float, horizon: int = 14,
                         directional_accuracy: Optional[float] = None) -> pd.DataFrame:
        """
        Generate forecast with target MAE.

        Args:
            target_mae: Desired mean absolute error ($)
            horizon: Days to forecast
            directional_accuracy: Optional - % of correct directions (0-100)

        Returns:
            DataFrame with synthetic forecasts
        """
        actuals = self.actuals_df[self.target].iloc[:horizon].values
        forecasts = np.zeros(horizon)

        if directional_accuracy is None:
            # Random errors with target MAE
            errors = np.random.normal(0, target_mae / 0.8, size=horizon)
            forecasts = actuals + errors

        else:
            # Control directional accuracy
            dir_acc_fraction = directional_accuracy / 100.0

            for i in range(horizon):
                if i == 0:
                    # First forecast: add noise
                    error = np.random.normal(0, target_mae / 0.8)
                    forecasts[i] = actuals[i] + error
                else:
                    # Check actual direction from day 0
                    actual_direction = actuals[i] > actuals[0]

                    # Decide if we get direction right
                    if np.random.random() < dir_acc_fraction:
                        # Correct direction
                        if actual_direction:
                            # Should be higher - add positive error
                            base = actuals[0] + abs(actuals[i] - actuals[0])
                            error = np.random.normal(0, target_mae)
                        else:
                            # Should be lower - add negative error
                            base = actuals[0] - abs(actuals[i] - actuals[0])
                            error = np.random.normal(0, target_mae)
                    else:
                        # Wrong direction
                        if actual_direction:
                            # Actually higher, but predict lower
                            base = actuals[0] - abs(actuals[i] - actuals[0])
                            error = np.random.normal(0, target_mae)
                        else:
                            # Actually lower, but predict higher
                            base = actuals[0] + abs(actuals[i] - actuals[0])
                            error = np.random.normal(0, target_mae)

                    forecasts[i] = base + error

        # Build DataFrame
        forecast_df = pd.DataFrame({
            'date': self.actuals_df['date'].iloc[:horizon],
            'forecast': forecasts,
            'model': f'Synthetic_MAE{target_mae:.1f}_DIR{directional_accuracy:.0f}' if directional_accuracy else f'Synthetic_MAE{target_mae:.1f}',
            'mae_target': target_mae,
            'dir_acc_target': directional_accuracy if directional_accuracy else None
        })

        return forecast_df

    def generate_sensitivity_suite(self, horizon: int = 14) -> List[pd.DataFrame]:
        """
        Generate a suite of forecasts for sensitivity analysis.

        Creates forecasts with varying MAE and directional accuracy.

        Args:
            horizon: Days to forecast

        Returns:
            List of forecast DataFrames
        """
        suite = []

        # Perfect forecast
        suite.append(self.generate_perfect_forecast(horizon))

        # MAE sensitivity (varying error levels)
        for mae in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
            suite.append(self.generate_with_mae(mae, horizon))

        # Directional accuracy sensitivity (with fixed MAE)
        for dir_acc in [40, 50, 60, 70, 80, 90]:
            suite.append(self.generate_with_mae(target_mae=2.0, horizon=horizon,
                                               directional_accuracy=dir_acc))

        # High directional, high error (good direction, bad magnitude)
        suite.append(self.generate_with_mae(target_mae=5.0, horizon=horizon,
                                           directional_accuracy=80))

        # Low directional, low error (good magnitude, bad direction)
        suite.append(self.generate_with_mae(target_mae=1.0, horizon=horizon,
                                           directional_accuracy=40))

        return suite

    def generate_monte_carlo_forecasts(self, n_simulations: int = 100,
                                      mae_mean: float = 2.0,
                                      mae_std: float = 1.0,
                                      horizon: int = 14) -> List[pd.DataFrame]:
        """
        Generate Monte Carlo simulations of forecasts.

        Args:
            n_simulations: Number of forecast simulations
            mae_mean: Mean MAE for simulations
            mae_std: Std dev of MAE across simulations
            horizon: Days to forecast

        Returns:
            List of simulated forecast DataFrames
        """
        simulations = []

        for i in range(n_simulations):
            # Sample MAE from distribution
            mae = max(0.1, np.random.normal(mae_mean, mae_std))

            # Sample directional accuracy
            dir_acc = np.random.uniform(30, 70)

            forecast_df = self.generate_with_mae(mae, horizon, dir_acc)
            forecast_df['simulation_id'] = i

            simulations.append(forecast_df)

        return simulations

    def save_sensitivity_suite(self, output_dir: str = "synthetic_forecasts"):
        """
        Generate and save full sensitivity analysis suite.

        Args:
            output_dir: Directory to save synthetic forecasts
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        suite = self.generate_sensitivity_suite()

        # Save each forecast
        for forecast_df in suite:
            model_name = forecast_df['model'].iloc[0]
            filename = f"{output_dir}/{model_name}.csv"
            forecast_df.to_csv(filename, index=False)

        # Save actuals
        self.actuals_df.to_csv(f"{output_dir}/actuals.csv", index=False)

        # Create summary
        summary = pd.DataFrame([{
            'model': f['model'].iloc[0],
            'mae_target': f['mae_target'].iloc[0],
            'dir_acc_target': f['dir_acc_target'].iloc[0] if 'dir_acc_target' in f.columns else None,
            'n_forecasts': len(f)
        } for f in suite])

        summary.to_csv(f"{output_dir}/synthetic_suite_summary.csv", index=False)

        print(f"✓ Generated {len(suite)} synthetic forecasts in {output_dir}/")
        print(f"  MAE range: {summary['mae_target'].min():.1f} - {summary['mae_target'].max():.1f}")
        print(f"  Directional accuracy range: {summary['dir_acc_target'].min():.0f}% - {summary['dir_acc_target'].max():.0f}%")

        return summary


def generate_trading_agent_test_data(actuals_df: pd.DataFrame,
                                    output_dir: str = "trading_agent_test_data"):
    """
    Generate comprehensive test data for trading agent evaluation.

    Creates:
    - Baseline forecasts (perfect, random walk, naive)
    - Error sensitivity suite (varying MAE)
    - Directional accuracy suite
    - Monte Carlo simulations
    - Extreme cases (best/worst)

    Args:
        actuals_df: DataFrame with actual price data
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    generator = SyntheticForecastGenerator(actuals_df)

    print("="*80)
    print("  GENERATING TRADING AGENT TEST DATA")
    print("="*80)
    print()

    # 1. Sensitivity suite
    print("1. Generating sensitivity suite...")
    summary = generator.save_sensitivity_suite(f"{output_dir}/sensitivity")
    print()

    # 2. Monte Carlo
    print("2. Generating Monte Carlo simulations (100 runs)...")
    mc_forecasts = generator.generate_monte_carlo_forecasts(n_simulations=100)

    mc_dir = f"{output_dir}/monte_carlo"
    os.makedirs(mc_dir, exist_ok=True)

    for forecast_df in mc_forecasts:
        sim_id = forecast_df['simulation_id'].iloc[0]
        forecast_df.to_csv(f"{mc_dir}/simulation_{sim_id:03d}.csv", index=False)

    print(f"   ✓ Generated 100 Monte Carlo simulations")
    print()

    # 3. Extreme cases
    print("3. Generating extreme cases...")

    extreme_cases = {
        'perfect': generator.generate_perfect_forecast(),
        'terrible': generator.generate_with_mae(20.0),
        'high_error_good_direction': generator.generate_with_mae(10.0, directional_accuracy=90),
        'low_error_bad_direction': generator.generate_with_mae(0.5, directional_accuracy=30),
        'median': generator.generate_with_mae(3.0, directional_accuracy=50)
    }

    extreme_dir = f"{output_dir}/extreme_cases"
    os.makedirs(extreme_dir, exist_ok=True)

    for case_name, forecast_df in extreme_cases.items():
        forecast_df.to_csv(f"{extreme_dir}/{case_name}.csv", index=False)

    print(f"   ✓ Generated {len(extreme_cases)} extreme cases")
    print()

    # Save metadata
    metadata = {
        'generated_date': datetime.now().isoformat(),
        'n_sensitivity_forecasts': len(summary),
        'n_monte_carlo_runs': 100,
        'n_extreme_cases': len(extreme_cases),
        'forecast_horizon': 14,
        'commodity': actuals_df.get('commodity', 'Unknown').iloc[0] if 'commodity' in actuals_df.columns else 'Unknown'
    }

    import json
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("="*80)
    print("  COMPLETE - Trading Agent Test Data Generated")
    print("="*80)
    print(f"  Output: {output_dir}/")
    print(f"  Total forecasts: {len(summary) + 100 + len(extreme_cases)}")
    print()

    return metadata


# Example usage
EXAMPLE_USAGE = """
from ground_truth.testing.synthetic_forecasts import (
    SyntheticForecastGenerator,
    generate_trading_agent_test_data
)

# Load actual data
actuals_df = pd.read_csv("actuals.csv")

# Generate test data for trading agent
metadata = generate_trading_agent_test_data(actuals_df, "trading_agent_test_data")

# Trading agent can now test with:
# 1. Sensitivity suite: How does MAE affect profitability?
# 2. Monte Carlo: Distribution of outcomes
# 3. Extreme cases: Best/worst scenarios

# Example: Test with specific MAE
generator = SyntheticForecastGenerator(actuals_df)
forecast = generator.generate_with_mae(target_mae=2.0, directional_accuracy=60)

# Feed to trading agent for backtesting
trading_agent.backtest(forecast)
"""

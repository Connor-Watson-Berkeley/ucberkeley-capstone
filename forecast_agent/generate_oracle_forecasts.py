#!/usr/bin/env python3
"""
Generate Oracle Forecasts for Trading Sensitivity Analysis

Creates "fake" forecast models with controlled accuracy characteristics to answer:
"How accurate do forecasts need to be for profitable trading?"

Oracle models start with actuals and add:
- Directional noise (flip direction X% of the time)
- MAPE noise (scale to achieve target MAPE)

This lets us test trading strategies against forecasts of known quality.

Usage:
    # Generate reduced suite of oracle models (5 models, 2 years)
    python generate_oracle_forecasts.py --commodity Coffee --start-date 2023-01-01 --end-date 2025-11-17
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict

from databricks import sql


def load_actuals(connection, commodity: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load actual prices from unified_data, aggregated across regions."""
    cursor = connection.cursor()

    # Aggregate across regions to get single price series
    query = f"""
        SELECT date, AVG(close) as close
        FROM commodity.silver.unified_data
        WHERE commodity = '{commodity}'
          AND date >= '{start_date}'
          AND date <= '{end_date}'
        GROUP BY date
        ORDER BY date
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    cursor.close()

    df = pd.DataFrame(rows, columns=columns)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    return df


def generate_ar1_noise(n_steps: int, sigma: float, ar_coef: float = 0.7) -> np.ndarray:
    """
    Generate AR(1) autocorrelated noise.

    ε_t = ρ * ε_{t-1} + η_t, where η_t ~ N(0, σ_η²)

    Args:
        n_steps: Number of time steps
        sigma: Target unconditional standard deviation
        ar_coef: AR(1) coefficient (persistence)

    Returns:
        Array of autocorrelated noise
    """
    # Adjust innovation variance for AR(1) process
    # Var(ε_t) = σ_η² / (1 - ρ²), so σ_η = σ * sqrt(1 - ρ²)
    sigma_innovation = sigma * np.sqrt(1 - ar_coef**2)

    noise = np.zeros(n_steps)
    # Initialize from stationary distribution
    noise[0] = np.random.normal(0, sigma)

    for t in range(1, n_steps):
        noise[t] = ar_coef * noise[t-1] + np.random.normal(0, sigma_innovation)

    return noise


def generate_directional_oracle(
    actuals_df: pd.DataFrame,
    target_directional_accuracy: float,
    horizon: int = 14,
    ar_coef: float = 0.7
) -> List[Dict]:
    """
    Generate oracle forecasts with target directional accuracy.

    Starts with actuals, then flips direction (1-accuracy)% of the time.

    Args:
        actuals_df: DataFrame with actual prices
        target_directional_accuracy: Target accuracy (0-100%)
        horizon: Forecast horizon in days
        ar_coef: AR(1) coefficient for autocorrelated noise

    Returns:
        List of forecast dicts
    """
    forecasts = []

    for i in range(len(actuals_df) - horizon):
        forecast_date = actuals_df.index[i]

        # Get actual future values
        future_actuals = actuals_df.iloc[i+1:i+1+horizon]['close'].values
        current_price = actuals_df.iloc[i]['close']

        # Start with perfect forecast
        forecast_values = future_actuals.copy()

        # Flip direction with probability (1 - target_accuracy)
        flip_probability = (100 - target_directional_accuracy) / 100

        for j in range(len(forecast_values)):
            if np.random.random() < flip_probability:
                # Flip the direction relative to current price
                actual_change = future_actuals[j] - current_price
                flipped_value = current_price - actual_change
                forecast_values[j] = flipped_value

        # Calculate confidence intervals (wider for lower accuracy)
        accuracy_factor = target_directional_accuracy / 100
        std_dev = np.std(future_actuals - current_price) / accuracy_factor

        forecast_row = {
            'forecast_date': forecast_date,
            'forecast_values': forecast_values,
            'std_dev': std_dev
        }
        forecasts.append(forecast_row)

    return forecasts


def generate_mape_oracle(
    actuals_df: pd.DataFrame,
    target_mape: float,
    horizon: int = 14,
    ar_coef: float = 0.7
) -> List[Dict]:
    """
    Generate oracle forecasts with target MAPE using AR(1) noise.

    Adds autocorrelated Gaussian noise scaled to achieve target MAPE.

    Args:
        actuals_df: DataFrame with actual prices
        target_mape: Target MAPE (percentage)
        horizon: Forecast horizon in days
        ar_coef: AR(1) coefficient for autocorrelated noise

    Returns:
        List of forecast dicts
    """
    forecasts = []

    for i in range(len(actuals_df) - horizon):
        forecast_date = actuals_df.index[i]

        # Get actual future values
        future_actuals = actuals_df.iloc[i+1:i+1+horizon]['close'].values

        # Add AR(1) noise to achieve target MAPE
        # MAPE ≈ E[|ε|] / mean(actuals) * 100
        # For Normal: E[|N(0,σ)|] ≈ 0.8 * σ
        # So: σ_target = (target_mape / 100) * mean(actuals) / 0.8
        noise_std = (target_mape / 100) * np.mean(future_actuals) / 0.8

        # Generate AR(1) noise
        noise = generate_ar1_noise(len(future_actuals), noise_std, ar_coef)

        forecast_values = future_actuals + noise

        forecast_row = {
            'forecast_date': forecast_date,
            'forecast_values': forecast_values,
            'std_dev': noise_std
        }
        forecasts.append(forecast_row)

    return forecasts


def write_forecasts_to_db(
    connection,
    forecasts: List[Dict],
    commodity: str,
    model_version: str,
    n_paths: int = 1000
):
    """Write oracle forecasts to distributions and point_forecasts tables."""
    cursor = connection.cursor()

    distributions_rows = []
    point_forecasts_rows = []
    generation_timestamp = datetime.now()
    path_id_base = np.random.randint(1000000, 9999999)

    for forecast in forecasts:
        forecast_date = forecast['forecast_date']
        forecast_values = forecast['forecast_values']
        std_dev = forecast['std_dev']

        # Generate Monte Carlo paths
        for path_idx in range(n_paths):
            path_id = path_id_base + path_idx

            # Add AR(1) noise to create distribution (more realistic)
            ar_coef = 0.7  # Persistence in errors
            path_noise = generate_ar1_noise(len(forecast_values), std_dev, ar_coef)
            path_values = forecast_values + path_noise

            # Create distribution row (14 days, wide format)
            dist_row = {
                'path_id': path_id,
                'commodity': commodity,
                'model_version': model_version,
                'forecast_start_date': forecast_date,
                'data_cutoff_date': forecast_date,  # Oracle uses future data
                'generation_timestamp': generation_timestamp
            }

            for day in range(14):
                if day < len(path_values):
                    dist_row[f'day_{day+1}'] = float(path_values[day])
                else:
                    dist_row[f'day_{day+1}'] = None

            distributions_rows.append(dist_row)

        # Create point forecasts
        for day_idx in range(len(forecast_values)):
            day_ahead = day_idx + 1

            # Confidence intervals
            forecast_mean = float(forecast_values[day_idx])
            vol_scaled = std_dev * np.sqrt(day_idx + 1)
            lower_95 = forecast_mean - 1.96 * vol_scaled
            upper_95 = forecast_mean + 1.96 * vol_scaled

            point_row = {
                'forecast_date': forecast_date,
                'data_cutoff_date': forecast_date,
                'day_ahead': day_ahead,
                'commodity': commodity,
                'model_version': model_version,
                'forecast_mean': forecast_mean,
                'forecast_std': float(std_dev),
                'lower_95': float(lower_95),
                'upper_95': float(upper_95),
                'generation_timestamp': generation_timestamp
            }

            point_forecasts_rows.append(point_row)

        path_id_base += n_paths

    # Write distributions (batch)
    print(f"     Writing {len(distributions_rows):,} distribution rows...")
    chunk_size = 500
    for i in range(0, len(distributions_rows), chunk_size):
        chunk = distributions_rows[i:i+chunk_size]

        value_rows = []
        for row in chunk:
            day_vals = [f"{row[f'day_{i+1}']:.2f}" if row[f'day_{i+1}'] is not None else "NULL" for i in range(14)]
            value_rows.append(
                f"({row['path_id']}, '{row['forecast_start_date']}', '{row['data_cutoff_date']}', "
                f"'{row['generation_timestamp']}', '{row['model_version']}', '{row['commodity']}', "
                f"{', '.join(day_vals)})"
            )

        insert_sql = f"""
        INSERT INTO commodity.forecast.distributions
        (path_id, forecast_start_date, data_cutoff_date, generation_timestamp,
         model_version, commodity, day_1, day_2, day_3, day_4, day_5, day_6, day_7,
         day_8, day_9, day_10, day_11, day_12, day_13, day_14)
        VALUES {', '.join(value_rows)}
        """
        cursor.execute(insert_sql)

    print(f"     ✅ Wrote {len(distributions_rows):,} distributions")

    # Write point forecasts (batch)
    print(f"     Writing {len(point_forecasts_rows):,} point forecast rows...")
    for i in range(0, len(point_forecasts_rows), 1000):
        chunk = point_forecasts_rows[i:i+1000]

        value_rows = []
        for row in chunk:
            value_rows.append(
                f"('{row['forecast_date']}', '{row['data_cutoff_date']}', "
                f"'{row['generation_timestamp']}', {row['day_ahead']}, "
                f"{row['forecast_mean']:.2f}, {row['forecast_std']:.2f}, "
                f"{row['lower_95']:.2f}, {row['upper_95']:.2f}, "
                f"'{row['model_version']}', '{row['commodity']}')"
            )

        insert_sql = f"""
        INSERT INTO commodity.forecast.point_forecasts
        (forecast_date, data_cutoff_date, generation_timestamp, day_ahead,
         forecast_mean, forecast_std, lower_95, upper_95, model_version, commodity)
        VALUES {', '.join(value_rows)}
        """
        cursor.execute(insert_sql)

    print(f"     ✅ Wrote {len(point_forecasts_rows):,} point forecasts")

    cursor.close()


def main():
    parser = argparse.ArgumentParser(description='Generate oracle forecasts for trading sensitivity analysis')
    parser.add_argument('--commodity', type=str, required=True, choices=['Coffee', 'Sugar'])
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Start date YYYY-MM-DD (default: 2023-01-01)')
    parser.add_argument('--end-date', type=str, default='2025-11-17', help='End date YYYY-MM-DD (default: 2025-11-17)')
    parser.add_argument('--n-paths', type=int, default=1000,
                        help='Number of Monte Carlo paths per forecast (default: 1000)')
    parser.add_argument('--ar-coefficient', type=float, default=0.7,
                        help='AR(1) coefficient for autocorrelated noise (default: 0.7)')

    args = parser.parse_args()

    # Reduced oracle suite (5 models)
    directional_accuracies = [70, 85]
    mape_targets = [5, 10, 20]

    # Load credentials
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
    DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")

    if not all([DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH]):
        print("ERROR: Missing Databricks credentials")
        sys.exit(1)

    print("="*80)
    print("ORACLE FORECAST GENERATOR - Trading Sensitivity Analysis")
    print("="*80)
    print(f"Commodity: {args.commodity}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Directional accuracies: {directional_accuracies}")
    print(f"MAPE targets: {mape_targets}")
    print(f"Monte Carlo paths: {args.n_paths}")
    print(f"AR(1) coefficient: {args.ar_coefficient}")
    print("="*80)

    # Connect to Databricks
    print("\n📡 Connecting to Databricks...")
    connection = sql.connect(
        server_hostname=DATABRICKS_HOST.replace('https://', ''),
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )
    print("✅ Connected")

    # Load actuals
    print(f"\n📊 Loading actuals for {args.commodity}...")
    actuals_df = load_actuals(connection, args.commodity, args.start_date, args.end_date)
    print(f"✅ Loaded {len(actuals_df):,} days of data")

    # Generate directional accuracy oracles
    print("\n"+"="*80)
    print("DIRECTIONAL ACCURACY ORACLES")
    print("="*80)

    for accuracy in directional_accuracies:
        model_version = f"oracle_directional_{accuracy}pct"
        print(f"\n🔮 {model_version}")
        print(f"   Target: {accuracy}% directional accuracy")
        print(f"   AR(1) coefficient: {args.ar_coefficient}")

        forecasts = generate_directional_oracle(actuals_df, accuracy, ar_coef=args.ar_coefficient)
        write_forecasts_to_db(connection, forecasts, args.commodity, model_version, args.n_paths)

    # Generate MAPE oracles
    print("\n"+"="*80)
    print("MAPE ORACLES")
    print("="*80)

    for mape in mape_targets:
        model_version = f"oracle_mape_{mape}pct"
        print(f"\n🔮 {model_version}")
        print(f"   Target: {mape}% MAPE")
        print(f"   AR(1) coefficient: {args.ar_coefficient}")

        forecasts = generate_mape_oracle(actuals_df, mape, ar_coef=args.ar_coefficient)
        write_forecasts_to_db(connection, forecasts, args.commodity, model_version, args.n_paths)

    # Summary
    total_models = len(directional_accuracies) + len(mape_targets)
    print("\n"+"="*80)
    print("✅ ORACLE GENERATION COMPLETE")
    print("="*80)
    print(f"Generated {total_models} oracle models:")
    print(f"  - {len(directional_accuracies)} directional accuracy variants")
    print(f"  - {len(mape_targets)} MAPE variants")
    print()
    print("Next steps:")
    print("  1. Query these forecasts in your trading agent")
    print("  2. Measure trading P&L for each oracle model")
    print("  3. Determine minimum forecast quality needed for profitability")
    print("="*80)

    connection.close()


if __name__ == "__main__":
    main()

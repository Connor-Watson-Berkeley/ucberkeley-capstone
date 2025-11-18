#!/usr/bin/env python3
"""
Quick Evaluation Script (Local Development Only - Gitignored)

Spot-check forecast performance without modifying production tables.
Useful for sanity checking during backfill development.

Usage:
    python quick_eval.py --commodity Coffee --model naive --date 2024-01-15
    python quick_eval.py --commodity Coffee --model xgboost --last 5
    python quick_eval.py --commodity Coffee --model naive --start 2024-01-01 --end 2024-03-01
"""

import os
import argparse
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple
import numpy as np
from databricks import sql

# Databricks connection
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")


def connect_databricks():
    """Create Databricks SQL connection."""
    return sql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )


def load_forecast_and_actuals(
    connection,
    commodity: str,
    model_version: str,
    forecast_start_date: date
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load forecast paths, actuals, and trading day flags for a specific forecast window.

    Returns:
        forecast_paths: Array of shape (n_paths, 14)
        actuals: Array of shape (14,)
        is_trading_day: Boolean array of shape (14,)
    """
    cursor = connection.cursor()

    # Load forecast distributions (wide format: day_1, day_2, ..., day_14)
    query_forecast = f"""
    SELECT day_1, day_2, day_3, day_4, day_5, day_6, day_7,
           day_8, day_9, day_10, day_11, day_12, day_13, day_14
    FROM commodity.forecast.distributions
    WHERE commodity = '{commodity}'
      AND model_version = '{model_version}'
      AND forecast_start_date = '{forecast_start_date}'
      AND is_actuals = FALSE
    ORDER BY path_id
    """

    cursor.execute(query_forecast)
    forecast_rows = cursor.fetchall()

    if not forecast_rows:
        raise ValueError(f"No forecast found for {commodity}/{model_version} on {forecast_start_date}")

    # Convert to numpy array (each row is a path)
    forecast_paths = np.array([[row[i] for i in range(14)] for row in forecast_rows])

    # Load actuals (model_version = 'actuals')
    # Note: Using new convention where actuals are stored as model_version='actuals'
    # (Also has is_actuals=TRUE for backwards compatibility)
    query_actuals = f"""
    SELECT day_1, day_2, day_3, day_4, day_5, day_6, day_7,
           day_8, day_9, day_10, day_11, day_12, day_13, day_14
    FROM commodity.forecast.distributions
    WHERE commodity = '{commodity}'
      AND model_version = 'actuals'
      AND forecast_start_date = '{forecast_start_date}'
    LIMIT 1
    """

    cursor.execute(query_actuals)
    actuals_row = cursor.fetchone()

    if actuals_row:
        actuals = np.array([actuals_row[i] if actuals_row[i] is not None else np.nan for i in range(14)])
    else:
        actuals = np.full(14, np.nan)

    # Load trading day flags from unified_data
    query_trading_days = f"""
    SELECT is_trading_day
    FROM commodity.silver.unified_data
    WHERE commodity = '{commodity}'
      AND date >= '{forecast_start_date}'
      AND date < DATE_ADD('{forecast_start_date}', 14)
    ORDER BY date
    LIMIT 14
    """

    cursor.execute(query_trading_days)
    trading_day_rows = cursor.fetchall()

    # Ensure we have 14 values, default to True if missing
    is_trading_day = np.array([
        bool(trading_day_rows[i][0]) if i < len(trading_day_rows) and trading_day_rows[i] and trading_day_rows[i][0] is not None
        else True
        for i in range(14)
    ], dtype=bool)

    cursor.close()

    return forecast_paths, actuals, is_trading_day


def calculate_metrics_at_horizon(
    forecast_paths: np.ndarray,
    actuals: np.ndarray,
    is_trading_day: np.ndarray,
    horizon: int
) -> Dict[str, float]:
    """Calculate metrics for a specific horizon (trading days only)."""
    # Slice to horizon
    forecast_paths_h = forecast_paths[:, :horizon]
    actuals_h = actuals[:horizon]
    is_trading_h = is_trading_day[:horizon]

    # Filter to trading days only
    trading_mask = is_trading_h & ~np.isnan(actuals_h)

    if not trading_mask.any():
        return None

    # Extract only trading days
    actuals_trading = actuals_h[trading_mask]
    forecast_paths_trading = forecast_paths_h[:, trading_mask]

    # Point forecast (median)
    point_forecast = np.median(forecast_paths_trading, axis=0)

    # MAE and RMSE
    errors = point_forecast - actuals_trading
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mape = float(np.mean(np.abs(errors / actuals_trading)) * 100)

    # CRPS (simplified)
    crps_values = []
    for t in range(forecast_paths_trading.shape[1]):
        actual = actuals_trading[t]
        samples = forecast_paths_trading[:, t]
        crps_t = np.mean(np.abs(samples - actual)) - 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
        crps_values.append(crps_t)
    crps = float(np.mean(crps_values))

    # Coverage (80% and 95%)
    p10 = np.percentile(forecast_paths_trading, 10, axis=0)
    p90 = np.percentile(forecast_paths_trading, 90, axis=0)
    p025 = np.percentile(forecast_paths_trading, 2.5, axis=0)
    p975 = np.percentile(forecast_paths_trading, 97.5, axis=0)

    coverage_80 = float(np.mean((actuals_trading >= p10) & (actuals_trading <= p90)))
    coverage_95 = float(np.mean((actuals_trading >= p025) & (actuals_trading <= p975)))

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'crps': crps,
        'coverage_80': coverage_80,
        'coverage_95': coverage_95,
        'num_trading_days': int(trading_mask.sum()),
        'num_total_days': horizon
    }


def evaluate_forecast(commodity: str, model_version: str, forecast_date: date):
    """Evaluate a single forecast and print results."""
    connection = connect_databricks()

    try:
        forecast_paths, actuals, is_trading_day = load_forecast_and_actuals(
            connection, commodity, model_version, forecast_date
        )

        print(f"\n{'='*80}")
        print(f"Forecast: {commodity} / {model_version} / {forecast_date}")
        print(f"{'='*80}")
        print(f"Shape: {forecast_paths.shape[0]} paths × {forecast_paths.shape[1]} days")
        print(f"Trading days in window: {is_trading_day.sum()}/{len(is_trading_day)}")
        print(f"Actuals available: {(~np.isnan(actuals)).sum()}/14")

        # Calculate metrics at 1d, 7d, 14d horizons
        horizons = [1, 7, 14]
        print(f"\n{'Horizon':<10} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'CRPS':<10} {'Cov80':<10} {'Cov95':<10} {'#Days':<10}")
        print("-" * 80)

        for h in horizons:
            metrics = calculate_metrics_at_horizon(forecast_paths, actuals, is_trading_day, h)
            if metrics:
                print(f"{h:>2}d        {metrics['mae']:>8.2f}  {metrics['rmse']:>8.2f}  "
                      f"{metrics['mape']:>8.1f}%  {metrics['crps']:>8.2f}  "
                      f"{metrics['coverage_80']:>8.2%}  {metrics['coverage_95']:>8.2%}  "
                      f"{metrics['num_trading_days']:>2}/{metrics['num_total_days']:>2}")
            else:
                print(f"{h:>2}d        (no data)")

        print()

    finally:
        connection.close()


def get_recent_forecasts(
    commodity: str,
    model_version: str,
    limit: int = 5
) -> List[date]:
    """Get the N most recent forecast dates for a commodity/model."""
    connection = connect_databricks()
    cursor = connection.cursor()

    query = f"""
    SELECT DISTINCT forecast_start_date
    FROM commodity.forecast.distributions
    WHERE commodity = '{commodity}'
      AND model_version = '{model_version}'
      AND is_actuals = FALSE
    ORDER BY forecast_start_date DESC
    LIMIT {limit}
    """

    cursor.execute(query)
    dates = [row[0] for row in cursor.fetchall()]

    cursor.close()
    connection.close()

    return dates


def get_forecasts_in_range(
    commodity: str,
    model_version: str,
    start_date: date,
    end_date: date
) -> List[date]:
    """Get all forecast dates in a date range."""
    connection = connect_databricks()
    cursor = connection.cursor()

    query = f"""
    SELECT DISTINCT forecast_start_date
    FROM commodity.forecast.distributions
    WHERE commodity = '{commodity}'
      AND model_version = '{model_version}'
      AND is_actuals = FALSE
      AND forecast_start_date >= '{start_date}'
      AND forecast_start_date <= '{end_date}'
    ORDER BY forecast_start_date
    """

    cursor.execute(query)
    dates = [row[0] for row in cursor.fetchall()]

    cursor.close()
    connection.close()

    return dates


def main():
    parser = argparse.ArgumentParser(description="Quick forecast evaluation (local development)")
    parser.add_argument("--commodity", required=True, choices=["Coffee", "Sugar"], help="Commodity")
    parser.add_argument("--model", required=True, help="Model version (e.g., 'naive', 'xgboost')")

    # Mutually exclusive date selection
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument("--date", type=str, help="Specific forecast date (YYYY-MM-DD)")
    date_group.add_argument("--last", type=int, help="Evaluate last N forecasts")
    date_group.add_argument("--range", nargs=2, metavar=("START", "END"), help="Date range (YYYY-MM-DD YYYY-MM-DD)")

    args = parser.parse_args()

    # Determine which forecasts to evaluate
    if args.date:
        forecast_dates = [datetime.strptime(args.date, "%Y-%m-%d").date()]
    elif args.last:
        print(f"Finding {args.last} most recent forecasts for {args.commodity}/{args.model}...")
        forecast_dates = get_recent_forecasts(args.commodity, args.model, args.last)
        if not forecast_dates:
            print(f"No forecasts found for {args.commodity}/{args.model}")
            return
    else:  # --range
        start_date = datetime.strptime(args.range[0], "%Y-%m-%d").date()
        end_date = datetime.strptime(args.range[1], "%Y-%m-%d").date()
        print(f"Finding forecasts between {start_date} and {end_date}...")
        forecast_dates = get_forecasts_in_range(args.commodity, args.model, start_date, end_date)
        if not forecast_dates:
            print(f"No forecasts found in range")
            return

    # Evaluate each forecast
    for forecast_date in forecast_dates:
        try:
            evaluate_forecast(args.commodity, args.model, forecast_date)
        except Exception as e:
            print(f"\n❌ Error evaluating {forecast_date}: {e}\n")

    # Summary
    print(f"\n{'='*80}")
    print(f"Evaluated {len(forecast_dates)} forecast(s)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

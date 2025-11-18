"""
Evaluate Historical Forecasts and Populate forecast_metadata Table

Calculates comprehensive performance metrics for all historical forecasts:
- Point forecast metrics: MAE, RMSE, MAPE, directional accuracy
- Probabilistic metrics: CRPS, calibration, coverage, sharpness

Usage:
    # Evaluate all forecasts
    python evaluate_historical_forecasts.py --commodity Coffee

    # Evaluate specific model
    python evaluate_historical_forecasts.py --commodity Coffee --model naive_v1

    # Evaluate specific date range
    python evaluate_historical_forecasts.py --commodity Coffee --start-date 2023-01-01 --end-date 2024-01-01

    # Dry run to see what would be evaluated
    python evaluate_historical_forecasts.py --commodity Coffee --dry-run

    # Run schema migration first (adds probabilistic columns)
    python -c "from databricks import sql; conn = sql.connect(...); cursor = conn.cursor(); cursor.execute(open('sql/add_probabilistic_metrics_to_metadata.sql').read())"

Strategy:
    - Queries commodity.forecast.distributions for all historical forecasts
    - Joins with actuals (path_id = 0) to calculate errors
    - Calculates metrics at multiple horizons (1d, 7d, 14d)
    - Updates commodity.forecast.forecast_metadata with performance metrics
    - Can be run independently or integrated into backfill pipeline
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from databricks import sql
from ground_truth.core.evaluator import calculate_metrics


def calculate_crps(actuals: np.ndarray, forecast_paths: np.ndarray) -> float:
    """
    Calculate Continuous Ranked Probability Score (CRPS).

    CRPS measures the quality of probabilistic forecasts by comparing
    the forecast distribution to the actual outcome.

    Lower CRPS = better forecast

    Args:
        actuals: Array of actual values (shape: [horizon])
        forecast_paths: Array of forecast paths (shape: [n_paths, horizon])

    Returns:
        Average CRPS across the forecast horizon
    """
    n_paths, horizon = forecast_paths.shape

    crps_values = []

    for t in range(horizon):
        if np.isnan(actuals[t]):
            continue

        actual = actuals[t]
        forecast_samples = forecast_paths[:, t]

        # Sort forecast samples
        sorted_samples = np.sort(forecast_samples)

        # CRPS = E[|X - Y|] - 0.5 * E[|X - X'|]
        # where X is forecast, Y is actual, X' is independent copy of X

        # First term: mean absolute error
        term1 = np.mean(np.abs(sorted_samples - actual))

        # Second term: mean pairwise distance (approximated)
        # For efficiency, we use the analytical formula for sorted samples
        n = len(sorted_samples)
        indices = np.arange(1, n + 1)
        term2 = np.sum((2 * indices - 1) * sorted_samples) / (n ** 2) - np.mean(sorted_samples)

        crps = term1 - 0.5 * term2
        crps_values.append(crps)

    return float(np.mean(crps_values)) if crps_values else None


def calculate_calibration(actuals: np.ndarray, forecast_paths: np.ndarray,
                         quantile_levels: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]) -> float:
    """
    Calculate calibration score.

    A well-calibrated forecast has the property that X% of actuals fall
    below the X-th percentile of the forecast distribution.

    Calibration score = mean absolute deviation from ideal calibration

    Args:
        actuals: Array of actual values (shape: [horizon])
        forecast_paths: Array of forecast paths (shape: [n_paths, horizon])
        quantile_levels: List of quantiles to check (0-1)

    Returns:
        Calibration error (lower is better, 0 = perfect calibration)
    """
    n_paths, horizon = forecast_paths.shape

    calibration_errors = []

    for quantile in quantile_levels:
        # Calculate empirical coverage rate
        below_count = 0
        total_count = 0

        for t in range(horizon):
            if np.isnan(actuals[t]):
                continue

            actual = actuals[t]
            forecast_samples = forecast_paths[:, t]

            # Calculate quantile
            forecast_quantile = np.percentile(forecast_samples, quantile * 100)

            # Check if actual is below quantile
            if actual <= forecast_quantile:
                below_count += 1

            total_count += 1

        if total_count > 0:
            empirical_coverage = below_count / total_count
            calibration_error = abs(empirical_coverage - quantile)
            calibration_errors.append(calibration_error)

    return float(np.mean(calibration_errors)) if calibration_errors else None


def calculate_coverage_rate(actuals: np.ndarray, forecast_paths: np.ndarray,
                            confidence_level: float = 0.80) -> float:
    """
    Calculate coverage rate for prediction intervals.

    Coverage rate = fraction of actuals that fall within the prediction interval

    Args:
        actuals: Array of actual values (shape: [horizon])
        forecast_paths: Array of forecast paths (shape: [n_paths, horizon])
        confidence_level: Confidence level (e.g., 0.80 for 80% interval)

    Returns:
        Coverage rate (0-1)
    """
    n_paths, horizon = forecast_paths.shape

    # Calculate prediction interval bounds
    lower_quantile = (1 - confidence_level) / 2
    upper_quantile = 1 - lower_quantile

    in_interval_count = 0
    total_count = 0

    for t in range(horizon):
        if np.isnan(actuals[t]):
            continue

        actual = actuals[t]
        forecast_samples = forecast_paths[:, t]

        lower_bound = np.percentile(forecast_samples, lower_quantile * 100)
        upper_bound = np.percentile(forecast_samples, upper_quantile * 100)

        if lower_bound <= actual <= upper_bound:
            in_interval_count += 1

        total_count += 1

    return float(in_interval_count / total_count) if total_count > 0 else None


def calculate_sharpness(forecast_paths: np.ndarray, confidence_level: float = 0.80) -> float:
    """
    Calculate sharpness of prediction intervals.

    Sharpness = average width of prediction interval
    Lower is better (sharper = more confident)

    Args:
        forecast_paths: Array of forecast paths (shape: [n_paths, horizon])
        confidence_level: Confidence level (e.g., 0.80 for 80% interval)

    Returns:
        Average interval width
    """
    n_paths, horizon = forecast_paths.shape

    lower_quantile = (1 - confidence_level) / 2
    upper_quantile = 1 - lower_quantile

    widths = []

    for t in range(horizon):
        forecast_samples = forecast_paths[:, t]

        lower_bound = np.percentile(forecast_samples, lower_quantile * 100)
        upper_bound = np.percentile(forecast_samples, upper_quantile * 100)

        width = upper_bound - lower_bound
        widths.append(width)

    return float(np.mean(widths))


def get_forecast_windows_to_evaluate(
    connection,
    commodity: str,
    model_version: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> pd.DataFrame:
    """
    Get list of forecast windows that need evaluation.

    Returns DataFrame with columns: [forecast_start_date, model_version, num_paths]
    """
    cursor = connection.cursor()

    # Build query
    where_clauses = [f"commodity = '{commodity}'", "is_actuals = FALSE"]

    if model_version:
        where_clauses.append(f"model_version = '{model_version}'")

    if start_date:
        where_clauses.append(f"forecast_start_date >= '{start_date}'")

    if end_date:
        where_clauses.append(f"forecast_start_date <= '{end_date}'")

    where_clause = " AND ".join(where_clauses)

    query = f"""
    SELECT
        forecast_start_date,
        data_cutoff_date,
        model_version,
        COUNT(DISTINCT path_id) as num_paths
    FROM commodity.forecast.distributions
    WHERE {where_clause}
    GROUP BY forecast_start_date, data_cutoff_date, model_version
    ORDER BY forecast_start_date, model_version
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    df = pd.DataFrame(rows, columns=columns)
    cursor.close()

    return df


def load_forecast_and_actuals(
    connection,
    commodity: str,
    model_version: str,
    forecast_start_date: date
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, date]:
    """
    Load forecast paths, actuals, and trading day flags for a specific forecast window.

    Returns:
        forecast_paths: Array of shape (n_paths, 14)
        actuals: Array of shape (14,)
        is_trading_day: Boolean array of shape (14,) - True for trading days
        data_cutoff_date: Date of last training data
    """
    cursor = connection.cursor()

    # Load forecast paths
    query_forecasts = f"""
    SELECT
        path_id,
        data_cutoff_date,
        day_1, day_2, day_3, day_4, day_5, day_6, day_7,
        day_8, day_9, day_10, day_11, day_12, day_13, day_14
    FROM commodity.forecast.distributions
    WHERE commodity = '{commodity}'
      AND model_version = '{model_version}'
      AND forecast_start_date = '{forecast_start_date}'
      AND is_actuals = FALSE
    ORDER BY path_id
    """

    cursor.execute(query_forecasts)
    forecast_rows = cursor.fetchall()

    # Load actuals (model_version = 'actuals')
    # Note: Using new convention where actuals are stored as model_version='actuals'
    # (Also has is_actuals=TRUE for backwards compatibility)
    query_actuals = f"""
    SELECT
        day_1, day_2, day_3, day_4, day_5, day_6, day_7,
        day_8, day_9, day_10, day_11, day_12, day_13, day_14
    FROM commodity.forecast.distributions
    WHERE commodity = '{commodity}'
      AND model_version = 'actuals'
      AND forecast_start_date = '{forecast_start_date}'
    """

    cursor.execute(query_actuals)
    actuals_row = cursor.fetchall()

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

    cursor.close()

    # Convert to numpy arrays
    if not forecast_rows or not actuals_row:
        return None, None, None, None

    data_cutoff_date = forecast_rows[0][1]

    # Extract day values (skip path_id and data_cutoff_date columns)
    forecast_paths = np.array([[float(val) if val is not None else np.nan
                                for val in row[2:]] for row in forecast_rows])

    actuals = np.array([float(val) if val is not None else np.nan
                       for val in actuals_row[0]])

    # Extract trading day flags (default to True if missing)
    is_trading_day = np.array([row[0] if row else True for row in trading_day_rows])

    # Ensure we have 14 days (pad with True if needed)
    if len(is_trading_day) < 14:
        is_trading_day = np.pad(is_trading_day, (0, 14 - len(is_trading_day)),
                                constant_values=True)

    return forecast_paths, actuals, is_trading_day, data_cutoff_date


def evaluate_forecast_window(
    forecast_paths: np.ndarray,
    actuals: np.ndarray,
    is_trading_day: np.ndarray,
    horizons: List[int] = [1, 7, 14]
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate a single forecast window at multiple horizons.
    **Only evaluates on trading days** to avoid inflated metrics from forward-filled weekend prices.

    Args:
        forecast_paths: Array of shape (n_paths, 14)
        actuals: Array of shape (14,)
        is_trading_day: Boolean array of shape (14,) - True for trading days
        horizons: List of horizons to evaluate (e.g., [1, 7, 14])

    Returns:
        Dict mapping horizon -> metrics dict
    """
    results = {}

    for horizon in horizons:
        # Slice to horizon
        forecast_paths_h = forecast_paths[:, :horizon]
        actuals_h = actuals[:horizon]
        is_trading_h = is_trading_day[:horizon]

        # Filter to trading days only (AND with valid actuals)
        trading_mask = is_trading_h & ~np.isnan(actuals_h)

        if not trading_mask.any():
            continue  # No trading days with valid actuals

        # Extract only trading days
        actuals_trading = actuals_h[trading_mask]
        forecast_paths_trading = forecast_paths_h[:, trading_mask]

        # Calculate point forecast (median of paths) on trading days only
        point_forecast_trading = np.median(forecast_paths_trading, axis=0)

        # Create pandas Series for existing evaluator
        actuals_series = pd.Series(actuals_trading)
        forecast_series = pd.Series(point_forecast_trading)

        # Calculate point metrics using existing evaluator
        point_metrics = calculate_metrics(actuals_series, forecast_series)

        # Calculate probabilistic metrics (on trading days only)
        crps = calculate_crps(actuals_trading, forecast_paths_trading)
        calibration = calculate_calibration(actuals_trading, forecast_paths_trading)
        coverage_80 = calculate_coverage_rate(actuals_trading, forecast_paths_trading, confidence_level=0.80)
        coverage_95 = calculate_coverage_rate(actuals_trading, forecast_paths_trading, confidence_level=0.95)
        sharpness_80 = calculate_sharpness(forecast_paths_trading, confidence_level=0.80)
        sharpness_95 = calculate_sharpness(forecast_paths_trading, confidence_level=0.95)

        # Combine all metrics
        metrics = {
            'mae': point_metrics.get('mae'),
            'rmse': point_metrics.get('rmse'),
            'mape': point_metrics.get('mape'),
            'directional_accuracy': point_metrics.get('directional_accuracy'),
            'directional_accuracy_from_day0': point_metrics.get('directional_accuracy_from_day0'),
            'crps': crps,
            'calibration_score': calibration,
            'coverage_80': coverage_80,
            'coverage_95': coverage_95,
            'sharpness_80': sharpness_80,
            'sharpness_95': sharpness_95,
            'num_days_evaluated': int(trading_mask.sum()),  # Count of trading days evaluated
            'num_total_days': horizon,  # Total days in horizon
            'trading_day_fraction': float(trading_mask.sum() / horizon)  # Fraction that were trading days
        }

        results[horizon] = metrics

    return results


def write_performance_metrics(
    connection,
    commodity: str,
    model_version: str,
    forecast_start_date: date,
    data_cutoff_date: date,
    metrics_by_horizon: Dict[int, Dict[str, float]],
    num_paths: int
):
    """
    Update commodity.forecast.forecast_metadata with performance metrics.

    Uses UPDATE if forecast_id exists, INSERT otherwise.
    """
    cursor = connection.cursor()

    # Generate forecast_id (consistent with backfill scripts)
    forecast_id = f"{model_version}_{commodity}_{forecast_start_date}"
    generation_timestamp = datetime.now()

    # Get metrics for each horizon
    metrics_1d = metrics_by_horizon.get(1, {})
    metrics_7d = metrics_by_horizon.get(7, {})
    metrics_14d = metrics_by_horizon.get(14, {})

    # Check if forecast_id already exists
    cursor.execute(f"""
        SELECT COUNT(*) FROM commodity.forecast.forecast_metadata
        WHERE forecast_id = '{forecast_id}'
    """)
    exists = cursor.fetchone()[0] > 0

    if exists:
        # UPDATE existing row with performance metrics
        update_sql = f"""
        UPDATE commodity.forecast.forecast_metadata
        SET
            mae_1d = {metrics_1d.get('mae') if metrics_1d.get('mae') is not None else 'NULL'},
            mae_7d = {metrics_7d.get('mae') if metrics_7d.get('mae') is not None else 'NULL'},
            mae_14d = {metrics_14d.get('mae') if metrics_14d.get('mae') is not None else 'NULL'},
            rmse_1d = {metrics_1d.get('rmse') if metrics_1d.get('rmse') is not None else 'NULL'},
            rmse_7d = {metrics_7d.get('rmse') if metrics_7d.get('rmse') is not None else 'NULL'},
            rmse_14d = {metrics_14d.get('rmse') if metrics_14d.get('rmse') is not None else 'NULL'},
            mape_1d = {metrics_1d.get('mape') if metrics_1d.get('mape') is not None else 'NULL'},
            mape_7d = {metrics_7d.get('mape') if metrics_7d.get('mape') is not None else 'NULL'},
            mape_14d = {metrics_14d.get('mape') if metrics_14d.get('mape') is not None else 'NULL'},
            crps_1d = {metrics_1d.get('crps') if metrics_1d.get('crps') is not None else 'NULL'},
            crps_7d = {metrics_7d.get('crps') if metrics_7d.get('crps') is not None else 'NULL'},
            crps_14d = {metrics_14d.get('crps') if metrics_14d.get('crps') is not None else 'NULL'},
            calibration_score_1d = {metrics_1d.get('calibration_score') if metrics_1d.get('calibration_score') is not None else 'NULL'},
            calibration_score_7d = {metrics_7d.get('calibration_score') if metrics_7d.get('calibration_score') is not None else 'NULL'},
            calibration_score_14d = {metrics_14d.get('calibration_score') if metrics_14d.get('calibration_score') is not None else 'NULL'},
            coverage_80_1d = {metrics_1d.get('coverage_80') if metrics_1d.get('coverage_80') is not None else 'NULL'},
            coverage_80_7d = {metrics_7d.get('coverage_80') if metrics_7d.get('coverage_80') is not None else 'NULL'},
            coverage_80_14d = {metrics_14d.get('coverage_80') if metrics_14d.get('coverage_80') is not None else 'NULL'},
            coverage_95_1d = {metrics_1d.get('coverage_95') if metrics_1d.get('coverage_95') is not None else 'NULL'},
            coverage_95_7d = {metrics_7d.get('coverage_95') if metrics_7d.get('coverage_95') is not None else 'NULL'},
            coverage_95_14d = {metrics_14d.get('coverage_95') if metrics_14d.get('coverage_95') is not None else 'NULL'},
            sharpness_80_1d = {metrics_1d.get('sharpness_80') if metrics_1d.get('sharpness_80') is not None else 'NULL'},
            sharpness_80_7d = {metrics_7d.get('sharpness_80') if metrics_7d.get('sharpness_80') is not None else 'NULL'},
            sharpness_80_14d = {metrics_14d.get('sharpness_80') if metrics_14d.get('sharpness_80') is not None else 'NULL'},
            sharpness_95_1d = {metrics_1d.get('sharpness_95') if metrics_1d.get('sharpness_95') is not None else 'NULL'},
            sharpness_95_7d = {metrics_7d.get('sharpness_95') if metrics_7d.get('sharpness_95') is not None else 'NULL'},
            sharpness_95_14d = {metrics_14d.get('sharpness_95') if metrics_14d.get('sharpness_95') is not None else 'NULL'},
            actuals_available = {max(metrics_1d.get('num_days_evaluated', 0), metrics_7d.get('num_days_evaluated', 0), metrics_14d.get('num_days_evaluated', 0))}
        WHERE forecast_id = '{forecast_id}'
        """

        cursor.execute(update_sql)

    else:
        # INSERT new row (minimal metadata, focus on metrics)
        insert_sql = f"""
        INSERT INTO commodity.forecast.forecast_metadata (
            forecast_id, forecast_start_date, data_cutoff_date, generation_timestamp,
            model_version, commodity,
            mae_1d, mae_7d, mae_14d,
            rmse_1d, rmse_7d, rmse_14d,
            mape_1d, mape_7d, mape_14d,
            crps_1d, crps_7d, crps_14d,
            calibration_score_1d, calibration_score_7d, calibration_score_14d,
            coverage_80_1d, coverage_80_7d, coverage_80_14d,
            coverage_95_1d, coverage_95_7d, coverage_95_14d,
            sharpness_80_1d, sharpness_80_7d, sharpness_80_14d,
            sharpness_95_1d, sharpness_95_7d, sharpness_95_14d,
            actuals_available, model_success
        ) VALUES (
            '{forecast_id}', '{forecast_start_date}', '{data_cutoff_date}', '{generation_timestamp}',
            '{model_version}', '{commodity}',
            {metrics_1d.get('mae') if metrics_1d.get('mae') is not None else 'NULL'},
            {metrics_7d.get('mae') if metrics_7d.get('mae') is not None else 'NULL'},
            {metrics_14d.get('mae') if metrics_14d.get('mae') is not None else 'NULL'},
            {metrics_1d.get('rmse') if metrics_1d.get('rmse') is not None else 'NULL'},
            {metrics_7d.get('rmse') if metrics_7d.get('rmse') is not None else 'NULL'},
            {metrics_14d.get('rmse') if metrics_14d.get('rmse') is not None else 'NULL'},
            {metrics_1d.get('mape') if metrics_1d.get('mape') is not None else 'NULL'},
            {metrics_7d.get('mape') if metrics_7d.get('mape') is not None else 'NULL'},
            {metrics_14d.get('mape') if metrics_14d.get('mape') is not None else 'NULL'},
            {metrics_1d.get('crps') if metrics_1d.get('crps') is not None else 'NULL'},
            {metrics_7d.get('crps') if metrics_7d.get('crps') is not None else 'NULL'},
            {metrics_14d.get('crps') if metrics_14d.get('crps') is not None else 'NULL'},
            {metrics_1d.get('calibration_score') if metrics_1d.get('calibration_score') is not None else 'NULL'},
            {metrics_7d.get('calibration_score') if metrics_7d.get('calibration_score') is not None else 'NULL'},
            {metrics_14d.get('calibration_score') if metrics_14d.get('calibration_score') is not None else 'NULL'},
            {metrics_1d.get('coverage_80') if metrics_1d.get('coverage_80') is not None else 'NULL'},
            {metrics_7d.get('coverage_80') if metrics_7d.get('coverage_80') is not None else 'NULL'},
            {metrics_14d.get('coverage_80') if metrics_14d.get('coverage_80') is not None else 'NULL'},
            {metrics_1d.get('coverage_95') if metrics_1d.get('coverage_95') is not None else 'NULL'},
            {metrics_7d.get('coverage_95') if metrics_7d.get('coverage_95') is not None else 'NULL'},
            {metrics_14d.get('coverage_95') if metrics_14d.get('coverage_95') is not None else 'NULL'},
            {metrics_1d.get('sharpness_80') if metrics_1d.get('sharpness_80') is not None else 'NULL'},
            {metrics_7d.get('sharpness_80') if metrics_7d.get('sharpness_80') is not None else 'NULL'},
            {metrics_14d.get('sharpness_80') if metrics_14d.get('sharpness_80') is not None else 'NULL'},
            {metrics_1d.get('sharpness_95') if metrics_1d.get('sharpness_95') is not None else 'NULL'},
            {metrics_7d.get('sharpness_95') if metrics_7d.get('sharpness_95') is not None else 'NULL'},
            {metrics_14d.get('sharpness_95') if metrics_14d.get('sharpness_95') is not None else 'NULL'},
            {max(metrics_1d.get('num_days_evaluated', 0), metrics_7d.get('num_days_evaluated', 0), metrics_14d.get('num_days_evaluated', 0))},
            TRUE
        )
        """

        cursor.execute(insert_sql)

    cursor.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate historical forecasts')
    parser.add_argument('--commodity', required=True, choices=['Coffee', 'Sugar'])
    parser.add_argument('--model', type=str, help='Specific model to evaluate (default: all)')
    parser.add_argument('--start-date', type=lambda s: datetime.strptime(s, '%Y-%m-%d').date(),
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=lambda s: datetime.strptime(s, '%Y-%m-%d').date(),
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 7, 14],
                       help='Horizons to evaluate (default: 1 7 14)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be evaluated without executing')

    args = parser.parse_args()

    # Load credentials
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
    DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")

    if not all([DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH]):
        print("ERROR: Missing Databricks credentials")
        sys.exit(1)

    print("=" * 80)
    print("HISTORICAL FORECAST EVALUATION")
    print("=" * 80)
    print(f"Commodity: {args.commodity}")
    print(f"Model: {args.model or 'All models'}")
    print(f"Horizons: {args.horizons}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 80)

    # Connect to Databricks
    connection = sql.connect(
        server_hostname=DATABRICKS_HOST.replace('https://', ''),
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )

    # Get forecast windows to evaluate
    print("\n📊 Finding forecast windows to evaluate...")
    windows_df = get_forecast_windows_to_evaluate(
        connection,
        args.commodity,
        args.model,
        args.start_date,
        args.end_date
    )

    print(f"   Found {len(windows_df)} forecast windows")

    if len(windows_df) == 0:
        print("   No windows to evaluate")
        connection.close()
        return

    # Show summary
    print(f"\n   Date range: {windows_df['forecast_start_date'].min()} to {windows_df['forecast_start_date'].max()}")
    print(f"   Models: {windows_df['model_version'].unique().tolist()}")

    if args.dry_run:
        print("\n🔍 DRY RUN - Would evaluate:")
        print(windows_df.to_string())
        connection.close()
        return

    # Evaluate each window
    print("\n⚙️  Evaluating forecast windows...")

    total_windows = len(windows_df)
    success_count = 0
    error_count = 0

    for idx, row in windows_df.iterrows():
        forecast_start_date = row['forecast_start_date']
        model_version = row['model_version']
        num_paths = row['num_paths']

        print(f"\n[{idx + 1}/{total_windows}] {forecast_start_date} - {model_version}")

        try:
            # Load data
            forecast_paths, actuals, data_cutoff_date = load_forecast_and_actuals(
                connection, args.commodity, model_version, forecast_start_date
            )

            if forecast_paths is None or actuals is None:
                print(f"   ⚠️  No data found, skipping")
                error_count += 1
                continue

            print(f"   Loaded {len(forecast_paths)} paths, cutoff: {data_cutoff_date}")

            # Evaluate
            metrics_by_horizon = evaluate_forecast_window(
                forecast_paths, actuals, args.horizons
            )

            if not metrics_by_horizon:
                print(f"   ⚠️  No valid metrics, skipping")
                error_count += 1
                continue

            # Display metrics
            for horizon, metrics in metrics_by_horizon.items():
                print(f"   {horizon}d: MAE={metrics['mae']:.2f}, CRPS={metrics['crps']:.2f}, "
                      f"Coverage(80%)={metrics['coverage_80']:.2%}")

            # Write to database
            write_performance_metrics(
                connection,
                args.commodity,
                model_version,
                forecast_start_date,
                data_cutoff_date,
                metrics_by_horizon,
                num_paths
            )

            success_count += 1

            # Progress update every 25 windows
            if (idx + 1) % 25 == 0:
                print(f"\n   Progress: {idx + 1}/{total_windows} ({100 * (idx + 1) / total_windows:.1f}%)")
                print(f"   Successful: {success_count}, Errors: {error_count}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            error_count += 1
            continue

    connection.close()

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Successful: {success_count}/{total_windows}")
    print(f"Errors: {error_count}/{total_windows}")
    print("=" * 80)


if __name__ == '__main__':
    main()

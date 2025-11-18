#!/usr/bin/env python3
"""
Backfill Actuals to Distributions Table

Populates model_version='actuals' rows in the distributions table from unified_data.
This creates the ground truth for forecast evaluation.

Hybrid approach:
- Sets model_version='actuals' (new convention)
- Sets is_actuals=TRUE (legacy convention)
- Allows migration without breaking existing code

Usage:
    python backfill_actuals.py --commodity Coffee --start-date 2018-01-01 --end-date 2025-11-17
    python backfill_actuals.py --commodity Sugar --start-date 2018-01-01 --end-date 2025-11-17
"""

import os
import argparse
from datetime import date, timedelta
from typing import List, Tuple
import numpy as np
from databricks import sql
from tqdm import tqdm

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


def get_forecast_dates(
    connection,
    commodity: str,
    start_date: date,
    end_date: date
) -> List[date]:
    """
    Get all trading dates that need actuals backfilled.

    Returns dates where:
    - is_trading_day = TRUE
    - We have actual close prices
    - Date is in the requested range
    """
    cursor = connection.cursor()

    query = f"""
    SELECT DISTINCT date
    FROM commodity.silver.unified_data
    WHERE commodity = '{commodity}'
      AND date >= '{start_date}'
      AND date <= '{end_date}'
      AND is_trading_day = 1
      AND close IS NOT NULL
    ORDER BY date
    """

    cursor.execute(query)
    dates = [row[0] for row in cursor.fetchall()]
    cursor.close()

    return dates


def get_actuals_window(
    connection,
    commodity: str,
    forecast_start_date: date,
    horizon: int = 14
) -> Tuple[np.ndarray, date]:
    """
    Get the next 14 days of actual prices starting from forecast_start_date.

    Returns:
        actuals: Array of shape (14,) with actual close prices
        data_cutoff_date: The day before forecast_start_date
    """
    cursor = connection.cursor()

    # Get the next 14 days of prices (may include weekends/holidays)
    # Note: unified_data may have multiple rows per date (one per region for weather data)
    # But the 'close' price (futures contract) should be the same across all regions
    # So we use DISTINCT ON date to get one row per date
    end_date = forecast_start_date + timedelta(days=horizon - 1)

    query = f"""
    SELECT DISTINCT date, FIRST(close) as close
    FROM commodity.silver.unified_data
    WHERE commodity = '{commodity}'
      AND date >= '{forecast_start_date}'
      AND date <= DATE_ADD('{forecast_start_date}', {horizon + 5})
    GROUP BY date
    ORDER BY date
    LIMIT {horizon}
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    # Fill array (NaN for missing dates)
    actuals = np.full(horizon, np.nan)
    for i, row in enumerate(rows):
        if i < horizon:
            actuals[i] = float(row[1]) if row[1] is not None else np.nan

    # Data cutoff date is the day before forecast starts
    data_cutoff_date = forecast_start_date - timedelta(days=1)

    cursor.close()
    return actuals, data_cutoff_date


def write_actuals_to_distributions(
    connection,
    commodity: str,
    forecast_start_date: date,
    data_cutoff_date: date,
    actuals: np.ndarray
):
    """
    Write actuals to distributions table using hybrid approach.

    Sets:
    - model_version = 'actuals' (new convention)
    - is_actuals = TRUE (legacy convention)
    - path_id = 1 (single "path" of ground truth)
    """
    cursor = connection.cursor()

    # Check if actuals already exist for this date
    cursor.execute(f"""
    SELECT COUNT(*)
    FROM commodity.forecast.distributions
    WHERE commodity = '{commodity}'
      AND forecast_start_date = '{forecast_start_date}'
      AND model_version = 'actuals'
    """)

    if cursor.fetchone()[0] > 0:
        print(f"  ⏭️  Actuals already exist for {forecast_start_date}, skipping...")
        cursor.close()
        return

    # Prepare values for insert
    from datetime import datetime
    generation_timestamp = datetime.now()

    # Build INSERT statement
    insert_query = f"""
    INSERT INTO commodity.forecast.distributions
    (
        path_id, forecast_start_date, data_cutoff_date, generation_timestamp,
        model_version, commodity,
        day_1, day_2, day_3, day_4, day_5, day_6, day_7,
        day_8, day_9, day_10, day_11, day_12, day_13, day_14,
        is_actuals, has_data_leakage
    )
    VALUES
    (
        1,  -- path_id (single ground truth path)
        '{forecast_start_date}',
        '{data_cutoff_date}',
        '{generation_timestamp}',
        'actuals',  -- model_version (new convention)
        '{commodity}',
        {actuals[0] if not np.isnan(actuals[0]) else 'NULL'},
        {actuals[1] if not np.isnan(actuals[1]) else 'NULL'},
        {actuals[2] if not np.isnan(actuals[2]) else 'NULL'},
        {actuals[3] if not np.isnan(actuals[3]) else 'NULL'},
        {actuals[4] if not np.isnan(actuals[4]) else 'NULL'},
        {actuals[5] if not np.isnan(actuals[5]) else 'NULL'},
        {actuals[6] if not np.isnan(actuals[6]) else 'NULL'},
        {actuals[7] if not np.isnan(actuals[7]) else 'NULL'},
        {actuals[8] if not np.isnan(actuals[8]) else 'NULL'},
        {actuals[9] if not np.isnan(actuals[9]) else 'NULL'},
        {actuals[10] if not np.isnan(actuals[10]) else 'NULL'},
        {actuals[11] if not np.isnan(actuals[11]) else 'NULL'},
        {actuals[12] if not np.isnan(actuals[12]) else 'NULL'},
        {actuals[13] if not np.isnan(actuals[13]) else 'NULL'},
        TRUE,  -- is_actuals (legacy convention)
        FALSE  -- has_data_leakage (actuals are never leakage)
    )
    """

    cursor.execute(insert_query)
    cursor.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill actuals to distributions table")
    parser.add_argument("--commodity", required=True, choices=["Coffee", "Sugar"], help="Commodity")
    parser.add_argument("--start-date", required=True, type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without writing")

    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)

    print(f"{'='*80}")
    print(f"Backfilling Actuals: {args.commodity}")
    print(f"{'='*80}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE'}")
    print()

    connection = connect_databricks()

    # Get all trading dates in range
    print("📅 Fetching trading dates...")
    forecast_dates = get_forecast_dates(connection, args.commodity, start_date, end_date)
    print(f"Found {len(forecast_dates)} trading dates\n")

    # Backfill actuals for each date
    for forecast_start_date in tqdm(forecast_dates, desc="Backfilling actuals"):
        try:
            # Get next 14 days of actuals
            actuals, data_cutoff_date = get_actuals_window(
                connection, args.commodity, forecast_start_date
            )

            # Count non-NaN values
            valid_count = (~np.isnan(actuals)).sum()

            if valid_count == 0:
                print(f"  ⚠️  No actuals available for {forecast_start_date}, skipping...")
                continue

            if not args.dry_run:
                # Write to database
                write_actuals_to_distributions(
                    connection, args.commodity, forecast_start_date,
                    data_cutoff_date, actuals
                )
            else:
                print(f"  [DRY RUN] Would write actuals for {forecast_start_date} ({valid_count}/14 days)")

        except Exception as e:
            print(f"  ❌ Error processing {forecast_start_date}: {e}")
            continue

    connection.close()

    print(f"\n{'='*80}")
    print(f"✅ Actuals backfill complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

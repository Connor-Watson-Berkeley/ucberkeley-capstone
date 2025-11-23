"""
Helper module for loading local unified_data

This module provides functions to load data from local parquet file
instead of querying Databricks, saving on serverless costs during experimentation.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def load_local_data(
    commodity='Coffee',
    region=None,
    lookback_days=None,
    start_date=None,
    end_date=None,
    data_file='data/unified_data.parquet'
):
    """
    Load unified_data from local parquet file.

    Args:
        commodity: Commodity name (default: Coffee)
        region: Region name (optional, if None returns all regions aggregated by date)
        lookback_days: Number of days to look back from today (optional)
        start_date: Start date (optional, overrides lookback_days)
        end_date: End date (optional, defaults to today)
        data_file: Path to local parquet file

    Returns:
        DataFrame with price and weather features
    """
    # Load parquet file
    file_path = Path(__file__).parent / data_file
    if not file_path.exists():
        raise FileNotFoundError(
            f"Local data file not found: {file_path}\n"
            f"Run 'python download_data_local.py' to download the data first."
        )

    df = pd.read_parquet(file_path)

    # Filter by commodity
    df = df[df['commodity'] == commodity].copy()

    # Handle date filtering
    if lookback_days and not start_date:
        end_date = datetime.now() if not end_date else pd.to_datetime(end_date)
        start_date = end_date - timedelta(days=lookback_days)

    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df['date'] >= start_date]

    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df['date'] <= end_date]

    # Handle region filtering
    if region:
        df = df[df['region'] == region].copy()
    else:
        # Aggregate across all regions (take average of price and weather)
        agg_dict = {
            'open': 'mean',
            'high': 'mean',
            'low': 'mean',
            'close': 'mean',
            'volume': 'sum',
            'vix': 'first',  # VIX is market-wide, not region-specific
            'temp_max_c': 'mean',
            'temp_min_c': 'mean',
            'temp_mean_c': 'mean',
            'precipitation_mm': 'mean',
            'rain_mm': 'mean',
            'snowfall_cm': 'mean',
            'humidity_mean_pct': 'mean',
            'wind_speed_max_kmh': 'mean',
            'is_trading_day': 'first'
        }
        df = df.groupby('date').agg(agg_dict).reset_index()

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # Handle missing values (forward fill)
    df = df.ffill().bfill()

    return df


def get_available_regions(commodity='Coffee', data_file='data/unified_data.parquet'):
    """
    Get list of available regions for a commodity.

    Args:
        commodity: Commodity name
        data_file: Path to local parquet file

    Returns:
        List of region names
    """
    file_path = Path(__file__).parent / data_file
    if not file_path.exists():
        raise FileNotFoundError(
            f"Local data file not found: {file_path}\n"
            f"Run 'python download_data_local.py' to download the data first."
        )

    df = pd.read_parquet(file_path)
    regions = df[df['commodity'] == commodity]['region'].unique()
    return sorted(regions)


if __name__ == '__main__':
    # Test the function
    print("Available regions for Coffee:")
    regions = get_available_regions('Coffee')
    for region in regions:
        print(f"  - {region}")

    print("\nLoading last 90 days of Coffee data (all regions aggregated)...")
    df = load_local_data(commodity='Coffee', lookback_days=90)
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample data:")
    print(df[['date', 'close', 'temp_mean_c', 'precipitation_mm']].head(10))

    print("\nLoading data for specific region (Bahia_Brazil)...")
    df_region = load_local_data(commodity='Coffee', region='Bahia_Brazil', lookback_days=90)
    print(f"Loaded {len(df_region)} rows")
    print(f"Date range: {df_region['date'].min()} to {df_region['date'].max()}")

"""
Download unified_data from Databricks to local parquet file

This script downloads the unified_data table once and saves it locally
to avoid Databricks serverless charges during experimentation.
"""

import pandas as pd
from databricks import sql
import os
from pathlib import Path

def download_unified_data(output_file='data/unified_data.parquet', commodity='Coffee'):
    """
    Download unified_data from Databricks and save locally.

    Args:
        output_file: Path to save the data (relative to forecast-experiments/)
        commodity: Commodity to download (default: Coffee)
    """
    print("=" * 80)
    print("Downloading Unified Data from Databricks")
    print("=" * 80)
    print(f"Commodity: {commodity}")
    print()

    # Create data directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Connect to Databricks
    print("Connecting to Databricks...")
    connection = sql.connect(
        server_hostname=os.getenv('DATABRICKS_HOST'),
        http_path=os.getenv('DATABRICKS_HTTP_PATH'),
        access_token=os.getenv('DATABRICKS_TOKEN')
    )

    cursor = connection.cursor()

    # Query all data for the commodity
    query = f"""
    SELECT
        date,
        is_trading_day,
        commodity,
        region,
        open,
        high,
        low,
        close,
        volume,
        vix,
        temp_max_c,
        temp_min_c,
        temp_mean_c,
        precipitation_mm,
        rain_mm,
        snowfall_cm,
        humidity_mean_pct,
        wind_speed_max_kmh
    FROM commodity.silver.unified_data
    WHERE commodity = '{commodity}'
    ORDER BY date, region
    """

    print("Querying data...")
    cursor.execute(query)
    result = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    print(f"Retrieved {len(result)} rows")

    # Create DataFrame
    df = pd.DataFrame(result, columns=columns)
    cursor.close()
    connection.close()

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Save to parquet
    print(f"Saving to {output_file}...")
    df.to_parquet(output_file, index=False)

    # Print summary
    print()
    print("=" * 80)
    print("Download Summary")
    print("=" * 80)
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Regions: {df['region'].nunique()}")
    print(f"Unique regions: {', '.join(sorted(df['region'].unique()))}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Saved to: {output_file}")
    print("=" * 80)
    print()
    print("✓ Data download complete!")
    print()
    print("Note: This file is gitignored and won't be committed to the repository.")

    return df


if __name__ == '__main__':
    # Download Coffee data
    df = download_unified_data(
        output_file='data/unified_data.parquet',
        commodity='Coffee'
    )

    # Show sample
    print("\nSample data:")
    print(df.head(10))

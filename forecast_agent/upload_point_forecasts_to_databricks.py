"""Upload point_forecasts parquet to Databricks.

Uploads local point_forecasts.parquet file to commodity.forecast.point_forecasts table.

Usage:
    python upload_point_forecasts_to_databricks.py
"""

import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from ground_truth.storage.databricks_writer import DatabricksForecastWriter


def upload_point_forecasts_to_databricks(parquet_path: str = "production_forecasts/point_forecasts.parquet"):
    """
    Upload local point_forecasts parquet file to Databricks.

    Args:
        parquet_path: Path to local point_forecasts.parquet file
    """

    print("="*80)
    print("UPLOADING POINT FORECASTS TO DATABRICKS")
    print("="*80)

    # Load local parquet
    print(f"\n[1/3] Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Models: {df['model_version'].nunique()}")
    print(f"  Commodities: {df['commodity'].unique()}")

    # Initialize Databricks writer
    print("\n[2/3] Connecting to Databricks...")
    writer = DatabricksForecastWriter()

    # Upload point forecasts
    print("\n[3/3] Uploading point forecasts...")
    writer.write_point_forecasts(df, mode="append")

    # Verify
    writer.verify_tables()

    print("\n" + "="*80)
    print("✅ UPLOAD COMPLETE")
    print("="*80)
    print(f"\nPoint forecasts available at: commodity.forecast.point_forecasts")
    print(f"Total rows uploaded: {len(df):,}")


if __name__ == "__main__":
    # Upload point forecasts from local parquet
    upload_point_forecasts_to_databricks()

"""Validate forecast pipeline end-to-end after table migration.

This script tests:
1. Data loading from commodity.forecast tables
2. Forecast generation locally
3. Upload to Databricks
4. Query validation from Databricks

Usage:
    python validate_forecast_pipeline.py
"""

import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ground_truth.storage.databricks_writer import DatabricksForecastWriter
from ground_truth.storage.production_writer import ProductionForecastWriter
from ground_truth.config.model_registry import BASELINE_MODELS
from datetime import datetime, timedelta


def validate_pipeline():
    """Run end-to-end validation of forecast pipeline."""

    print("="*80)
    print("FORECAST PIPELINE VALIDATION")
    print("="*80)

    # Test 1: Check Databricks connection and tables exist
    print("\n[1/5] Testing Databricks connection...")
    try:
        writer = DatabricksForecastWriter()
        with writer._get_connection() as conn:
            cursor = conn.cursor()

            # Check distributions table
            cursor.execute(f"""
                SELECT COUNT(*) FROM {writer.catalog}.{writer.schema}.distributions
            """)
            dist_count = cursor.fetchone()[0]
            print(f"  ✓ commodity.forecast.distributions: {dist_count:,} rows")

            # Check point_forecasts table
            cursor.execute(f"""
                SELECT COUNT(*) FROM {writer.catalog}.{writer.schema}.point_forecasts
            """)
            point_count = cursor.fetchone()[0]
            print(f"  ✓ commodity.forecast.point_forecasts: {point_count:,} rows")

            cursor.close()
    except Exception as e:
        print(f"  ✗ Databricks connection failed: {e}")
        return False

    # Test 2: Load data from unified_data
    print("\n[2/5] Loading Coffee data...")
    try:
        data_path = "../data/unified_data_snapshot_all.parquet"
        if not os.path.exists(data_path):
            print(f"  ✗ Data file not found: {data_path}")
            return False

        df = pd.read_parquet(data_path)
        df_coffee = df[df['commodity'] == 'Coffee'].copy()
        df_agg = df_coffee.groupby('date').agg({'close': 'first'}).reset_index()
        df_agg['date'] = pd.to_datetime(df_agg['date'])
        df_agg = df_agg.set_index('date').sort_index()

        print(f"  ✓ Loaded {len(df_agg):,} days of Coffee data")
        print(f"  ✓ Date range: {df_agg.index.min().date()} to {df_agg.index.max().date()}")
    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")
        return False

    # Test 3: Generate a test forecast
    print("\n[3/5] Generating test forecast (Random Walk model)...")
    try:
        model_config = BASELINE_MODELS['random_walk']
        result = model_config['function'](
            df_pandas=df_agg.tail(100),  # Only use last 100 days for quick test
            commodity='Coffee'
        )

        if not result.get('success', True):
            print(f"  ✗ Model failed to converge")
            return False

        forecast_df = result['forecast_df']
        print(f"  ✓ Generated 14-day forecast")
        print(f"  ✓ Forecast range: ${forecast_df['forecast'].iloc[0]:.2f} to ${forecast_df['forecast'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"  ✗ Forecast generation failed: {e}")
        return False

    # Test 4: Write to local parquet (via ProductionForecastWriter)
    print("\n[4/5] Writing to local parquet...")
    try:
        local_writer = ProductionForecastWriter("production_forecasts")

        # Create test distribution
        import numpy as np
        test_paths = np.random.randn(10, 14) * 5 + forecast_df['forecast'].values

        forecast_start_date = df_agg.index[-1] + timedelta(days=1)
        data_cutoff_date = df_agg.index[-1]

        local_writer.write_distributions(
            forecast_start_date=forecast_start_date,
            data_cutoff_date=data_cutoff_date,
            model_version='random_walk_v1_test',
            commodity='Coffee',
            sample_paths=test_paths,
            generation_timestamp=datetime.now(),
            n_paths=10
        )

        print(f"  ✓ Written to: {local_writer.distributions_path}")
        print(f"  ✓ Total local rows: {len(local_writer.distributions):,}")
    except Exception as e:
        print(f"  ✗ Local write failed: {e}")
        return False

    # Test 5: Upload to Databricks and verify
    print("\n[5/5] Uploading test data to Databricks...")
    try:
        # Read the test data
        test_df = local_writer.distributions[local_writer.distributions['model_version'] == 'random_walk_v1_test']

        print(f"  Test data: {len(test_df)} rows")

        # Upload to Databricks
        writer.write_distributions(test_df, mode="append")

        # Verify it's there
        with writer._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT COUNT(*)
                FROM {writer.catalog}.{writer.schema}.distributions
                WHERE model_version = 'random_walk_v1_test'
            """)
            uploaded_count = cursor.fetchone()[0]
            cursor.close()

        if uploaded_count != len(test_df):
            print(f"  ✗ Row count mismatch: uploaded {uploaded_count}, expected {len(test_df)}")
            return False

        print(f"  ✓ Uploaded {uploaded_count} test rows to Databricks")

        # Clean up test data
        with writer._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                DELETE FROM {writer.catalog}.{writer.schema}.distributions
                WHERE model_version = 'random_walk_v1_test'
            """)
            cursor.close()

        print(f"  ✓ Cleaned up test data")

    except Exception as e:
        print(f"  ✗ Upload/verification failed: {e}")
        return False

    # Summary
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80)
    print("\nValidation Results:")
    print("  ✓ Databricks connection working")
    print("  ✓ Tables accessible (commodity.forecast.distributions, point_forecasts)")
    print("  ✓ Data loading working")
    print("  ✓ Forecast generation working")
    print("  ✓ Local parquet write working")
    print("  ✓ Databricks upload working")
    print("  ✓ Query validation working")
    print("\n🎉 Forecast pipeline is ready for production!")

    return True


if __name__ == "__main__":
    success = validate_pipeline()
    sys.exit(0 if success else 1)

"""
Quick test script for feature engineering functions.

Tests with local Parquet data to ensure functions work before Databricks deployment.
"""

import sys
import os

# Add ground_truth to path
sys.path.insert(0, os.path.dirname(__file__))

from databricks.connect import DatabricksSession
from ground_truth.features import aggregators, covariate_projection, transformers

def main():
    print("="*60)
    print("Feature Engineering Test Suite")
    print("="*60)

    # Initialize Databricks Connect session
    print("\n1. Initializing Databricks Connect session...")
    spark = DatabricksSession.builder.getOrCreate()

    # Load local data
    print("\n2. Loading local data...")
    data_path = "../data/unified_data_snapshot_all.parquet"
    df = spark.read.parquet(data_path)

    print(f"   Loaded {df.count()} rows")
    print(f"   Columns: {df.columns[:10]}...")  # Show first 10 columns

    # Test 1: Aggregate regions (mean)
    print("\n" + "="*60)
    print("TEST 1: Aggregate Regions (Mean)")
    print("="*60)

    features = ['close', 'temp_c', 'humidity_pct', 'precipitation_mm']

    try:
        df_agg = aggregators.aggregate_regions_mean(
            df_spark=df,
            commodity='Coffee',
            features=features,
            cutoff_date='2024-01-01'
        )

        print(f"✓ Success! Shape: {df_agg.count()} rows")
        print("\nSample output:")
        df_agg.select('date', 'commodity', 'close', 'temp_c').show(5)

    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 2: Aggregate regions (weighted)
    print("\n" + "="*60)
    print("TEST 2: Aggregate Regions (Weighted)")
    print("="*60)

    production_weights = {
        'Colombia': 0.3,
        'Brazil': 0.4,
        'Vietnam': 0.3
    }

    try:
        df_weighted = aggregators.aggregate_regions_weighted(
            df_spark=df,
            commodity='Coffee',
            features=features,
            cutoff_date='2024-01-01',
            production_weights=production_weights
        )

        print(f"✓ Success! Shape: {df_weighted.count()} rows")
        print("\nSample output:")
        df_weighted.select('date', 'commodity', 'close', 'temp_c').show(5)

    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 3: Pivot regions
    print("\n" + "="*60)
    print("TEST 3: Pivot Regions as Features")
    print("="*60)

    try:
        df_pivot = aggregators.pivot_regions_as_features(
            df_spark=df,
            commodity='Coffee',
            features=['close', 'temp_c'],
            cutoff_date='2024-01-01'
        )

        print(f"✓ Success! Shape: {df_pivot.count()} rows, {len(df_pivot.columns)} columns")
        print(f"\nColumn sample: {df_pivot.columns[:10]}...")

    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 4: Covariate Projection (pandas-based)
    print("\n" + "="*60)
    print("TEST 4: Covariate Projection")
    print("="*60)

    # Convert aggregated data to pandas for projection testing
    df_pandas = df_agg.toPandas()
    df_pandas['date'] = pd.to_datetime(df_pandas['date'])
    df_pandas = df_pandas.set_index('date').sort_index()

    # Test 4a: Persist last value
    print("\n4a. Persist Last Value:")
    try:
        import pandas as pd
        projected = covariate_projection.persist_last_value(
            df_pandas=df_pandas,
            features=['temp_c', 'humidity_pct'],
            horizon=14
        )

        print(f"✓ Success! Projected {len(projected)} days")
        print("\nProjected values:")
        print(projected.head())

    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 4b: Seasonal average
    print("\n4b. Seasonal Average:")
    try:
        projected_seasonal = covariate_projection.seasonal_average(
            df_pandas=df_pandas,
            features=['temp_c', 'humidity_pct'],
            horizon=14,
            lookback_years=3
        )

        print(f"✓ Success! Projected {len(projected_seasonal)} days")
        print("\nProjected values:")
        print(projected_seasonal.head())

    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 4c: None needed
    print("\n4c. None Needed (for ARIMA):")
    try:
        projected_none = covariate_projection.none_needed(
            df_pandas=df_pandas,
            features=[],
            horizon=14
        )

        print(f"✓ Success! Returns: {projected_none}")

    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 5: Transformers (lags, diffs, rolling)
    print("\n" + "="*60)
    print("TEST 5: Feature Transformers")
    print("="*60)

    # Test 5a: Add lags
    print("\n5a. Add Lags:")
    try:
        df_lagged = transformers.add_lags(
            df_spark=df_agg,
            features=['close'],
            lags=[1, 7],
            cutoff_date='2024-01-01'
        )

        print(f"✓ Success! Columns: {df_lagged.columns}")
        print("\nSample with lags:")
        df_lagged.select('date', 'close', 'close_lag_1', 'close_lag_7').show(10)

    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 5b: Add rolling stats
    print("\n5b. Add Rolling Stats:")
    try:
        df_rolling = transformers.add_rolling_stats(
            df_spark=df_agg,
            features=['close'],
            windows=[7, 30],
            cutoff_date='2024-01-01'
        )

        print(f"✓ Success!")
        print("\nSample with rolling stats:")
        df_rolling.select('date', 'close', 'close_rolling_mean_7', 'close_rolling_std_7').show(10)

    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 5c: Add differences
    print("\n5c. Add Differences:")
    try:
        df_diff = transformers.add_differences(
            df_spark=df_agg,
            features=['close'],
            periods=[1, 7],
            cutoff_date='2024-01-01'
        )

        print(f"✓ Success!")
        print("\nSample with differences:")
        df_diff.select('date', 'close', 'close_diff_1', 'close_diff_7').show(10)

    except Exception as e:
        print(f"✗ Failed: {e}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("""
All feature engineering functions tested!

Next steps:
1. Fix any failed tests above
2. Build baseline forecasting models
3. Test end-to-end forecasting pipeline

Ready to build models when tests pass.
    """)

    spark.stop()

if __name__ == "__main__":
    main()

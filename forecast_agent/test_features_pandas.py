"""
Quick test script for feature engineering - pandas version.

Tests covariate projection and validates function signatures.
PySpark functions will be tested in Databricks.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add ground_truth to path
sys.path.insert(0, os.path.dirname(__file__))

from ground_truth.features import covariate_projection

def main():
    print("="*60)
    print("Feature Engineering Test Suite (Pandas)")
    print("="*60)

    # Load local data
    print("\n1. Loading local data...")
    data_path = "../data/unified_data_snapshot_all.parquet"
    df = pd.read_parquet(data_path)

    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"   Commodities: {df['commodity'].unique()}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

    # Filter and prepare data
    print("\n2. Preparing test data (Coffee, aggregated)...")
    df_coffee = df[df['commodity'] == 'Coffee'].copy()

    # Simple aggregation (average across regions)
    df_agg = df_coffee.groupby(['date', 'commodity']).agg({
        'close': 'first',
        'temp_c': 'mean',
        'humidity_pct': 'mean',
        'precipitation_mm': 'mean'
    }).reset_index()

    # Simulate what we'd get from PySpark aggregation
    df_agg.columns = ['date', 'commodity', 'close', 'temp_c', 'humidity_pct', 'precipitation_mm']
    df_agg['date'] = pd.to_datetime(df_agg['date'])
    df_agg = df_agg.set_index('date').sort_index()

    # Keep only through 2023 for testing
    df_agg = df_agg[df_agg.index <= '2023-12-31']

    print(f"   Aggregated to {len(df_agg)} days")
    print(f"   Columns: {list(df_agg.columns)}")
    print(f"\n   Sample data:")
    print(df_agg.head())

    # Test 1: None needed
    print("\n" + "="*60)
    print("TEST 1: None Needed (for ARIMA)")
    print("="*60)

    try:
        result = covariate_projection.none_needed(
            df_pandas=df_agg,
            features=[],
            horizon=14
        )

        print(f"✓ Success! Returns: {result}")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Persist last value
    print("\n" + "="*60)
    print("TEST 2: Persist Last Value")
    print("="*60)

    try:
        projected = covariate_projection.persist_last_value(
            df_pandas=df_agg,
            features=['temp_c', 'humidity_pct', 'precipitation_mm'],
            horizon=14
        )

        print(f"✓ Success! Projected {len(projected)} days")
        print(f"\nProjected index: {projected.index[0]} to {projected.index[-1]}")
        print(f"\nLast training value:")
        print(df_agg[['temp_c', 'humidity_pct', 'precipitation_mm']].iloc[-1])
        print(f"\nFirst projected value (should match):")
        print(projected[['temp_c', 'humidity_pct', 'precipitation_mm']].iloc[0])
        print(f"\nAll projected values (should be constant):")
        print(projected.head(7))

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Seasonal average
    print("\n" + "="*60)
    print("TEST 3: Seasonal Average")
    print("="*60)

    try:
        projected_seasonal = covariate_projection.seasonal_average(
            df_pandas=df_agg,
            features=['temp_c', 'humidity_pct'],
            horizon=14,
            lookback_years=3
        )

        print(f"✓ Success! Projected {len(projected_seasonal)} days")
        print(f"\nProjected values:")
        print(projected_seasonal.head(7))

        # Compare to persist (should be different)
        if projected is not None and projected_seasonal is not None:
            temp_diff = abs(projected['temp_c'].iloc[0] - projected_seasonal['temp_c'].iloc[0])
            print(f"\nDifference from persist method: {temp_diff:.2f}°C")
            print("(Should be non-zero if seasonal pattern exists)")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Linear trend
    print("\n" + "="*60)
    print("TEST 4: Linear Trend")
    print("="*60)

    try:
        projected_trend = covariate_projection.linear_trend(
            df_pandas=df_agg,
            features=['temp_c'],
            horizon=14,
            lookback_days=30
        )

        print(f"✓ Success! Projected {len(projected_trend)} days")
        print(f"\nLast 5 training values:")
        print(df_agg['temp_c'].iloc[-5:])
        print(f"\nProjected trend:")
        print(projected_trend['temp_c'].head(7))

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Weather API (should raise NotImplementedError)
    print("\n" + "="*60)
    print("TEST 5: Weather API (should fail gracefully)")
    print("="*60)

    try:
        projected_api = covariate_projection.weather_forecast_api(
            df_pandas=df_agg,
            features=['temp_c'],
            horizon=14
        )

        print(f"✗ Unexpected success! Should raise NotImplementedError")

    except NotImplementedError as e:
        print(f"✓ Correctly raises NotImplementedError")
        print(f"   Message: {str(e)[:80]}...")

    except Exception as e:
        print(f"✗ Wrong exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

    # Test 6: Get projection function helper
    print("\n" + "="*60)
    print("TEST 6: Get Projection Function Helper")
    print("="*60)

    try:
        fn = covariate_projection.get_projection_function('persist')
        print(f"✓ Success! Got function: {fn.__name__}")

        # Try invalid method
        try:
            bad_fn = covariate_projection.get_projection_function('invalid_method')
            print(f"✗ Should have raised ValueError for invalid method")
        except ValueError as e:
            print(f"✓ Correctly raises ValueError for invalid method")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("""
Pandas-based tests complete!

What we tested:
✓ Covariate projection functions (none, persist, seasonal, trend, API stub)
✓ Function signatures and return types
✓ Error handling

What's NOT tested (requires Databricks):
- PySpark aggregators (aggregate_regions_mean, weighted, pivot)
- PySpark transformers (lags, diffs, rolling stats)

Next steps:
1. Review test output above
2. If all passed: Build baseline models
3. Test full pipeline in Databricks

Ready to code baseline models!
    """)

if __name__ == "__main__":
    main()

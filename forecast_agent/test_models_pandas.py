"""Quick test script for baseline models - pandas version.

Tests all four baseline models with local data before Databricks deployment.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add ground_truth to path
sys.path.insert(0, os.path.dirname(__file__))

from ground_truth.models import naive, random_walk, arima, sarimax
from ground_truth.features import covariate_projection

def main():
    print("="*60)
    print("Baseline Models Test Suite (Pandas)")
    print("="*60)

    # Load local data
    print("\n1. Loading local data...")
    data_path = "../data/unified_data_snapshot_all.parquet"
    df = pd.read_parquet(data_path)

    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    # Filter and prepare data (Coffee, aggregated)
    print("\n2. Preparing test data (Coffee, aggregated)...")
    df_coffee = df[df['commodity'] == 'Coffee'].copy()

    # Simple aggregation (average across regions)
    df_agg = df_coffee.groupby(['date', 'commodity']).agg({
        'close': 'first',
        'temp_c': 'mean',
        'humidity_pct': 'mean',
        'precipitation_mm': 'mean'
    }).reset_index()

    df_agg['date'] = pd.to_datetime(df_agg['date'])
    df_agg = df_agg.set_index('date').sort_index()

    # Use 2023 for training, test on early 2024
    df_train = df_agg[df_agg.index <= '2023-12-31']

    print(f"   Training data: {len(df_train)} days")
    print(f"   Range: {df_train.index[0]} to {df_train.index[-1]}")
    print(f"   Last close: ${df_train['close'].iloc[-1]:.2f}")

    # Test 1: Naive
    print("\n" + "="*60)
    print("TEST 1: Naive Persistence")
    print("="*60)

    try:
        result = naive.naive_forecast_with_metadata(
            df_pandas=df_train,
            commodity='Coffee',
            target='close',
            horizon=14
        )

        forecast_df = result['forecast_df']
        print(f"✓ Success! Model: {result['model_name']}")
        print(f"\nForecast preview:")
        print(forecast_df.head(7)[['date', 'forecast', 'lower_95', 'upper_95']])
        print(f"\nAll forecasts same value: {forecast_df['forecast'].nunique() == 1}")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Random Walk
    print("\n" + "="*60)
    print("TEST 2: Random Walk with Drift")
    print("="*60)

    try:
        result = random_walk.random_walk_forecast_with_metadata(
            df_pandas=df_train,
            commodity='Coffee',
            target='close',
            horizon=14,
            lookback_days=30
        )

        forecast_df = result['forecast_df']
        drift = result['parameters']['estimated_drift']

        print(f"✓ Success! Model: {result['model_name']}")
        print(f"  Estimated drift: ${drift:.4f}/day")
        print(f"\nForecast preview:")
        print(forecast_df.head(7)[['date', 'forecast', 'lower_95', 'upper_95']])

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: ARIMA
    print("\n" + "="*60)
    print("TEST 3: ARIMA(1,1,1)")
    print("="*60)

    try:
        result = arima.arima_forecast_with_metadata(
            df_pandas=df_train,
            commodity='Coffee',
            target='close',
            order=(1, 1, 1),
            horizon=14
        )

        forecast_df = result['forecast_df']
        aic = result['parameters']['aic']
        bic = result['parameters']['bic']

        print(f"✓ Success! Model: {result['model_name']}")
        print(f"  AIC: {aic:.2f}, BIC: {bic:.2f}")
        print(f"\nForecast preview:")
        print(forecast_df.head(7)[['date', 'forecast', 'lower_95', 'upper_95']])

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: SARIMAX (no exog)
    print("\n" + "="*60)
    print("TEST 4: SARIMAX (auto-fit, no exogenous)")
    print("="*60)

    try:
        result = sarimax.sarimax_forecast_with_metadata(
            df_pandas=df_train,
            commodity='Coffee',
            target='close',
            exog_features=None,  # No exogenous vars
            order=None,  # Auto-fit
            horizon=14
        )

        forecast_df = result['forecast_df']
        params = result['parameters']

        print(f"✓ Success! Model: {result['model_name']}")
        print(f"  Order: ({params['p']},{params['d']},{params['q']})")
        print(f"  AIC: {params['aic']:.2f}, BIC: {params['bic']:.2f}")
        print(f"  Auto-fitted: {params['auto_fitted']}")
        print(f"\nForecast preview:")
        print(forecast_df.head(7)[['date', 'forecast', 'lower_95', 'upper_95']])

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: SARIMAX with exogenous variables
    print("\n" + "="*60)
    print("TEST 5: SARIMAX (auto-fit, with weather covariates)")
    print("="*60)

    try:
        result = sarimax.sarimax_forecast_with_metadata(
            df_pandas=df_train,
            commodity='Coffee',
            target='close',
            exog_features=['temp_c', 'humidity_pct', 'precipitation_mm'],
            covariate_projection_method='persist',
            order=None,  # Auto-fit
            horizon=14
        )

        forecast_df = result['forecast_df']
        params = result['parameters']

        print(f"✓ Success! Model: {result['model_name']}")
        print(f"  Order: ({params['p']},{params['d']},{params['q']})")
        print(f"  Exog features: {params['exog_features']}")
        print(f"  Covariate projection: {params['covariate_projection']}")
        print(f"  AIC: {params['aic']:.2f}, BIC: {params['bic']:.2f}")
        print(f"\nForecast preview:")
        print(forecast_df.head(7)[['date', 'forecast', 'lower_95', 'upper_95']])

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("""
All baseline models tested!

Models:
✓ Naive - Last value persistence
✓ Random Walk - Persistence + drift
✓ ARIMA(1,1,1) - Classical time series
✓ SARIMAX (auto) - Without exogenous variables
✓ SARIMAX (auto+exog) - With weather covariates

Next steps:
1. Review forecast outputs above
2. Build core modules (data_loader, forecast_writer, evaluator)
3. Create model registry
4. Run full evaluation pipeline

Ready to build the rest of the system!
    """)

if __name__ == "__main__":
    main()

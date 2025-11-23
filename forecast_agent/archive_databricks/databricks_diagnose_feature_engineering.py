# Databricks notebook source
"""
Diagnostic Script: Test Feature Engineering Integration
Minimal test to isolate the failure point
"""

# COMMAND ----------

# MAGIC %pip install databricks-sql-connector pmdarima

# COMMAND ----------

# Restart Python to load newly installed packages
dbutils.library.restartPython()

# COMMAND ----------

print("=" * 80)
print("DIAGNOSTIC: Feature Engineering Integration Test")
print("=" * 80)

# COMMAND ----------

# Test 1: Basic imports
print("\n[TEST 1] Basic imports...")
try:
    import sys
    import os
    import pandas as pd
    import numpy as np
    print("✓ Basic imports successful")
except Exception as e:
    print(f"✗ Basic imports failed: {e}")
    raise

# COMMAND ----------

# Test 2: Python path setup
print("\n[TEST 2] Python path setup...")
try:
    forecast_agent_path = '/Workspace/Repos/Project_Git/ucberkeley-capstone/forecast_agent'
    if forecast_agent_path not in sys.path:
        sys.path.insert(0, forecast_agent_path)
    os.chdir(forecast_agent_path)
    print(f"✓ Path added: {forecast_agent_path}")
    print(f"✓ Current directory: {os.getcwd()}")
except Exception as e:
    print(f"✗ Path setup failed: {e}")
    raise

# COMMAND ----------

# Test 3: sklearn availability
print("\n[TEST 3] sklearn availability...")
try:
    import sklearn
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    print(f"✓ sklearn version: {sklearn.__version__}")
except Exception as e:
    print(f"✗ sklearn import failed: {e}")
    raise

# COMMAND ----------

# Test 4: Import feature engineering module
print("\n[TEST 4] Import feature engineering module...")
try:
    from ground_truth.features.data_preparation import prepare_data_for_model
    print("✓ prepare_data_for_model imported successfully")
except Exception as e:
    print(f"✗ Feature engineering import failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    raise

# COMMAND ----------

# Test 5: Import model registry
print("\n[TEST 5] Import model registry...")
try:
    from ground_truth.config.model_registry import BASELINE_MODELS
    print(f"✓ Model registry imported, {len(BASELINE_MODELS)} models available")
    print(f"   Models: {list(BASELINE_MODELS.keys())}")
except Exception as e:
    print(f"✗ Model registry import failed: {e}")
    import traceback
    traceback.print_exc()
    raise

# COMMAND ----------

# Test 6: Load sample data
print("\n[TEST 6] Load sample data from unified_data...")
try:
    query = """
        SELECT *
        FROM commodity.silver.unified_data
        WHERE commodity = 'Coffee'
        AND date >= '2024-01-01'
        AND date <= '2024-01-10'
        ORDER BY date
    """
    spark_df = spark.sql(query)
    pandas_df = spark_df.toPandas()
    pandas_df['date'] = pd.to_datetime(pandas_df['date'])

    print(f"✓ Loaded {len(pandas_df)} rows")
    print(f"   Columns: {list(pandas_df.columns)}")
    print(f"   Unique dates: {pandas_df['date'].nunique()}")
    print(f"   Unique regions: {pandas_df['region'].nunique() if 'region' in pandas_df.columns else 'N/A'}")
    print(f"\n   Sample data shape: {pandas_df.shape}")
    print(f"   First few rows:")
    print(pandas_df.head(3))
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    raise

# COMMAND ----------

# Test 7: Test feature engineering with aggregate strategy
print("\n[TEST 7] Test feature engineering (aggregate strategy)...")
try:
    prepared_df = prepare_data_for_model(
        raw_data=pandas_df,
        commodity='Coffee',
        region_strategy='aggregate',
        gdelt_strategy='aggregate',
        feature_columns=['close', 'temp_mean_c', 'humidity_mean_pct', 'precipitation_mm']
    )

    print(f"✓ Feature engineering successful!")
    print(f"   Input shape: {pandas_df.shape}")
    print(f"   Output shape: {prepared_df.shape}")
    print(f"   Output columns: {list(prepared_df.columns)}")
    print(f"   Output index: {prepared_df.index.name}")
    print(f"\n   Prepared data sample:")
    print(prepared_df.head(3))
except Exception as e:
    print(f"✗ Feature engineering failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    raise

# COMMAND ----------

# Test 8: Test with model config
print("\n[TEST 8] Test with model config (sarimax_auto_weather)...")
try:
    model_config = BASELINE_MODELS['sarimax_auto_weather']
    params = model_config.get('params', {})

    print(f"   Model params: {params}")

    region_strategy = params.get('region_strategy', 'aggregate')
    gdelt_strategy = params.get('gdelt_strategy', None)
    gdelt_themes = params.get('gdelt_themes', None)
    feature_columns = params.get('exog_features', None)

    if feature_columns:
        target = params.get('target', 'close')
        feature_columns_with_target = [target] + feature_columns
    else:
        feature_columns_with_target = None

    print(f"   Region strategy: {region_strategy}")
    print(f"   GDELT strategy: {gdelt_strategy}")
    print(f"   Features: {feature_columns_with_target}")

    prepared_df = prepare_data_for_model(
        raw_data=pandas_df,
        commodity='Coffee',
        region_strategy=region_strategy,
        gdelt_strategy=gdelt_strategy,
        gdelt_themes=gdelt_themes,
        feature_columns=feature_columns_with_target
    )

    print(f"✓ Model config-based feature engineering successful!")
    print(f"   Output shape: {prepared_df.shape}")
    print(f"   Output columns: {list(prepared_df.columns)}")
except Exception as e:
    print(f"✗ Model config feature engineering failed: {e}")
    import traceback
    traceback.print_exc()
    raise

# COMMAND ----------

print("\n" + "=" * 80)
print("✅ ALL DIAGNOSTIC TESTS PASSED!")
print("=" * 80)
print("\nFeature engineering integration is working correctly.")
print("The failure must be in the training loop itself.")

# Databricks notebook source
"""
Minimal test notebook to verify basic execution
"""

# COMMAND ----------

print("Test 1: Basic execution works")
print(f"Python version: {__import__('sys').version}")

# COMMAND ----------

# Test 2: Import sys.path
import sys
print("Test 2: sys.path")
for i, path in enumerate(sys.path[:5]):
    print(f"  {i}: {path}")

# COMMAND ----------

# Test 3: Try to add workspace path
forecast_agent_path = '/Workspace/Repos/Project_Git/ucberkeley-capstone/forecast_agent'
if forecast_agent_path not in sys.path:
    sys.path.insert(0, forecast_agent_path)
print(f"Test 3: Added {forecast_agent_path} to sys.path")

# COMMAND ----------

# Test 4: List files in that directory
import os
print("Test 4: Files in forecast_agent directory")
try:
    files = os.listdir(forecast_agent_path)
    print(f"Found {len(files)} files/directories")
    for f in sorted(files)[:10]:
        print(f"  - {f}")
except Exception as e:
    print(f"Error listing files: {e}")

# COMMAND ----------

# Test 5: Try simple import
print("Test 5: Attempting import")
try:
    import backfill_rolling_window_spark
    print("✓ Import successful!")
    print(f"  Module: {backfill_rolling_window_spark}")
    print(f"  Has backfill function: {hasattr(backfill_rolling_window_spark, 'backfill_all_models_spark')}")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()

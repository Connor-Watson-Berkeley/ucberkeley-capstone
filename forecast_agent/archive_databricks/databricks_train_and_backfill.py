# Databricks notebook source
"""
Databricks Notebook: Train + Spark Backfill
Runs the Spark-parallelized forecast backfill for massive speedup.

This notebook runs the backfill using direct execution without imports.
"""

# COMMAND ----------

# Set up paths for module imports
import sys
import os

# Databricks workspace path
forecast_agent_path = '/Workspace/Repos/Project_Git/ucberkeley-capstone/forecast_agent'

# Add to Python path
if forecast_agent_path not in sys.path:
    sys.path.insert(0, forecast_agent_path)

print(f"✓ Added {forecast_agent_path} to Python path")

# COMMAND ----------

# Import the backfill function with retry logic (handles serverless race condition)
import time

retries = 5
backfill_all_models_spark = None

for attempt in range(retries):
    try:
        from backfill_rolling_window_spark import backfill_all_models_spark
        print(f"✓ Successfully imported backfill_all_models_spark (attempt {attempt + 1}/{retries})")
        break
    except (ImportError, ModuleNotFoundError) as e:
        if attempt < retries - 1:
            print(f"Import attempt {attempt + 1} failed: {e}")
            print(f"Retrying in 3 seconds... (serverless cluster may still be initializing)")
            time.sleep(3)
        else:
            print(f"Failed to import after {retries} attempts")
            raise

if backfill_all_models_spark is None:
    raise ImportError("Failed to import backfill_all_models_spark after all retries")

# COMMAND ----------

# Run the Spark backfill for Coffee
# This will use pretrained models and parallelize across the cluster
print("="*80)
print("Starting Spark-Parallelized Backfill for Coffee")
print("="*80)

backfill_all_models_spark(
    commodities=['Coffee'],
    models=['naive', 'xgboost', 'sarimax_auto_weather'],
    train_frequency='semiannually',
    start_date='2018-01-01',
    end_date='2025-11-17',
    num_partitions=50  # Adjust based on cluster size
)

print("="*80)
print("Backfill Complete!")
print("="*80)

# Databricks notebook source
"""
Databricks Notebook: Train + Spark Backfill
Runs the Spark-parallelized forecast backfill for massive speedup.

This notebook uses %run to load the backfill_rolling_window_spark module.
"""

# COMMAND ----------

# MAGIC %run ./backfill_rolling_window_spark

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

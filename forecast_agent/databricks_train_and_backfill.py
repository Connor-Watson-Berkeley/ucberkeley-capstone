# Databricks notebook source
"""
Databricks Notebook: Train + Spark Backfill
Runs the Spark-parallelized forecast backfill for massive speedup.

This notebook wraps the backfill_rolling_window_spark.py script to run it as a Databricks job.
"""

# COMMAND ----------

# Import the Spark backfill function
import sys
import os

# In Databricks notebooks, use the notebook's directory
# This works whether running locally or in Databricks
try:
    # Get the notebook's directory from dbutils
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    forecast_agent_dir = os.path.dirname(notebook_path.replace("/Workspace", ""))
except:
    # Fallback for local execution or if dbutils isn't available
    forecast_agent_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()

# Ensure we can import from forecast_agent directory
if forecast_agent_dir not in sys.path:
    sys.path.insert(0, forecast_agent_dir)

# Import the backfill function
from backfill_rolling_window_spark import backfill_all_models_spark

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

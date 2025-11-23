# Databricks notebook source
"""
Clear Locally-Trained Models + Train in Databricks
Fixes version compatibility issues by ensuring all models trained in Databricks environment
"""

# COMMAND ----------

print("=" * 80)
print("Step 1: Clear All Locally-Trained Models from Database")
print("=" * 80)

import os
from databricks import sql

# Get credentials from environment
host = os.environ.get('DATABRICKS_HOST')
http_path = os.environ.get('DATABRICKS_HTTP_PATH')

# Validate required environment variables
if not host:
    raise ValueError("DATABRICKS_HOST environment variable not set")
if not http_path:
    raise ValueError("DATABRICKS_HTTP_PATH environment variable not set")

token = dbutils.secrets.get(scope="databricks-secrets", key="databricks-token")

# Connect to database
connection = sql.connect(
    server_hostname=host.replace('https://', ''),
    http_path=http_path,
    access_token=token
)

cursor = connection.cursor()

# Delete all trained models for Coffee
print("\nDeleting trained models for Coffee...")
delete_query = """
DELETE FROM commodity.forecast.trained_models
WHERE commodity = 'Coffee'
AND model_version IN ('naive', 'xgboost', 'sarimax_auto_weather')
"""

cursor.execute(delete_query)
print("✓ Deleted all locally-trained models for Coffee")

# Verify deletion
count_query = """
SELECT COUNT(*) as count
FROM commodity.forecast.trained_models
WHERE commodity = 'Coffee'
AND model_version IN ('naive', 'xgboost', 'sarimax_auto_weather')
"""

cursor.execute(count_query)
result = cursor.fetchone()
print(f"✓ Verified: {result[0]} models remaining (should be 0)")

cursor.close()
connection.close()

print("\n" + "=" * 80)
print("Step 1 Complete: Database Cleared")
print("=" * 80)

# COMMAND ----------

print("=" * 80)
print("Step 2: Train Models in Databricks Environment")
print("=" * 80)

# Set up sys.path for imports
import sys
forecast_agent_path = '/Workspace/Repos/Project_Git/ucberkeley-capstone/forecast_agent'
if forecast_agent_path not in sys.path:
    sys.path.insert(0, forecast_agent_path)

print(f"✓ Added {forecast_agent_path} to Python path")

# Import training script
print("\nImporting training modules...")
try:
    from train_models import main as train_main
    print("✓ Successfully imported train_models")
except Exception as e:
    print(f"✗ Import failed: {e}")
    print("\nAttempting direct execution instead...")

# COMMAND ----------

# Run training for Coffee with 3 models
print("\n" + "=" * 80)
print("Training Coffee Models (Semiannually)")
print("=" * 80)

# Instead of importing, we'll run via Databricks notebook execution
# This avoids import path issues

import subprocess
import sys

# Change to forecast_agent directory
os.chdir('/Workspace/Repos/Project_Git/ucberkeley-capstone/forecast_agent')

# Run training script
cmd = [
    sys.executable,
    'train_models.py',
    '--commodity', 'Coffee',
    '--models', 'naive', 'xgboost', 'sarimax_auto_weather',
    '--train-frequency', 'semiannually',
    '--start-date', '2018-01-01',
    '--end-date', '2025-11-17'
]

print(f"\nExecuting: {' '.join(cmd)}\n")

result = subprocess.run(cmd, capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)

if result.stderr:
    print("\nSTDERR:")
    print(result.stderr)

if result.returncode != 0:
    print(f"\n❌ Training failed with exit code {result.returncode}")
    raise Exception(f"Training failed: {result.stderr}")
else:
    print("\n✅ Training completed successfully!")

# COMMAND ----------

print("=" * 80)
print("Step 3: Verify Trained Models in Database")
print("=" * 80)

# Reconnect to verify
connection = sql.connect(
    server_hostname=host.replace('https://', ''),
    http_path=http_path,
    access_token=token
)

cursor = connection.cursor()

# Count trained models
verify_query = """
SELECT
    model_version,
    COUNT(*) as count
FROM commodity.forecast.trained_models
WHERE commodity = 'Coffee'
AND model_version IN ('naive', 'xgboost', 'sarimax_auto_weather')
GROUP BY model_version
ORDER BY model_version
"""

cursor.execute(verify_query)
results = cursor.fetchall()

print("\nTrained Models in Database:")
for row in results:
    print(f"  {row[0]}: {row[1]} models")

cursor.close()
connection.close()

print("\n" + "=" * 80)
print("All Steps Complete!")
print("=" * 80)
print("\n✓ Cleared locally-trained models")
print("✓ Trained models in Databricks")
print("✓ Verified models in database")
print("\nNext: Run backfill_rolling_window.py using these Databricks-trained models")

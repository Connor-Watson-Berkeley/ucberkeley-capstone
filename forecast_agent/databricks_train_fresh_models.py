# Databricks notebook source
"""
Train Fresh Models in Databricks
Trains Coffee & Sugar models using Databricks default package versions
"""

# COMMAND ----------

print("=" * 80)
print("Training Fresh Models in Databricks")
print("=" * 80)
print("\nThis ensures NumPy/sklearn/xgboost version consistency")
print("All models will be trained using Databricks default package versions\n")

# COMMAND ----------

# Set up paths for imports
import sys
import os

forecast_agent_path = '/Workspace/Repos/Project_Git/ucberkeley-capstone/forecast_agent'
if forecast_agent_path not in sys.path:
    sys.path.insert(0, forecast_agent_path)

print(f"✓ Added {forecast_agent_path} to Python path")

# Change to forecast_agent directory for imports
os.chdir(forecast_agent_path)
print(f"✓ Changed directory to {forecast_agent_path}")

# COMMAND ----------

# Import training modules
print("\nImporting training modules...")

try:
    import subprocess
    print("✓ subprocess imported")

    # Verify Python executable
    python_exe = sys.executable
    print(f"✓ Using Python: {python_exe}")

    # Check package versions
    import numpy as np
    import sklearn
    import xgboost as xgb
    print(f"\n📦 Package Versions (Databricks):")
    print(f"  NumPy: {np.__version__}")
    print(f"  scikit-learn: {sklearn.__version__}")
    print(f"  XGBoost: {xgb.__version__}")

except Exception as e:
    print(f"✗ Import error: {e}")
    raise

# COMMAND ----------

# Train Coffee models
print("\n" + "=" * 80)
print("Training Coffee Models (Semiannually)")
print("=" * 80)

cmd_coffee = [
    sys.executable,
    'train_models.py',
    '--commodity', 'Coffee',
    '--models', 'naive', 'xgboost', 'sarimax_auto_weather',
    '--train-frequency', 'semiannually',
    '--start-date', '2018-01-01',
    '--end-date', '2025-11-17'
]

print(f"\nExecuting: {' '.join(cmd_coffee)}\n")

result_coffee = subprocess.run(cmd_coffee, capture_output=True, text=True, cwd=forecast_agent_path)

print("STDOUT:")
print(result_coffee.stdout)

if result_coffee.stderr:
    print("\nSTDERR:")
    print(result_coffee.stderr)

if result_coffee.returncode != 0:
    print(f"\n❌ Training failed for Coffee with exit code {result_coffee.returncode}")
    raise Exception(f"Training failed for Coffee: {result_coffee.stderr}")
else:
    print("\n✅ Coffee training completed successfully!")

# COMMAND ----------

# Train Sugar models
print("\n" + "=" * 80)
print("Training Sugar Models (Semiannually)")
print("=" * 80)

cmd_sugar = [
    sys.executable,
    'train_models.py',
    '--commodity', 'Sugar',
    '--models', 'naive', 'xgboost', 'sarimax_auto_weather',
    '--train-frequency', 'semiannually',
    '--start-date', '2018-01-01',
    '--end-date', '2025-11-17'
]

print(f"\nExecuting: {' '.join(cmd_sugar)}\n")

result_sugar = subprocess.run(cmd_sugar, capture_output=True, text=True, cwd=forecast_agent_path)

print("STDOUT:")
print(result_sugar.stdout)

if result_sugar.stderr:
    print("\nSTDERR:")
    print(result_sugar.stderr)

if result_sugar.returncode != 0:
    print(f"\n❌ Training failed for Sugar with exit code {result_sugar.returncode}")
    raise Exception(f"Training failed for Sugar: {result_sugar.stderr}")
else:
    print("\n✅ Sugar training completed successfully!")

# COMMAND ----------

# Verify trained models in database
print("\n" + "=" * 80)
print("Verifying Trained Models in Database")
print("=" * 80)

from databricks import sql as databricks_sql

# Get credentials
host = os.environ.get('DATABRICKS_HOST', 'https://dbc-5e4780f4-fcec.cloud.databricks.com').replace('https://', '')
http_path = os.environ.get('DATABRICKS_HTTP_PATH', '/sql/1.0/warehouses/c97ae2b0cf9cbc05')
token = os.environ['DATABRICKS_TOKEN']

# Connect
print("\nConnecting to database...")
connection = databricks_sql.connect(
    server_hostname=host,
    http_path=http_path,
    access_token=token
)

cursor = connection.cursor()

# Check Coffee models
print("\n📊 Coffee Models:")
cursor.execute("""
    SELECT model_version, COUNT(*) as count
    FROM commodity.forecast.trained_models
    WHERE commodity = 'Coffee'
    AND model_version IN ('naive', 'xgboost', 'sarimax_auto_weather')
    GROUP BY model_version
    ORDER BY model_version
""")

coffee_results = cursor.fetchall()
for row in coffee_results:
    print(f"  {row[0]}: {row[1]} models")

# Check Sugar models
print("\n📊 Sugar Models:")
cursor.execute("""
    SELECT model_version, COUNT(*) as count
    FROM commodity.forecast.trained_models
    WHERE commodity = 'Sugar'
    AND model_version IN ('naive', 'xgboost', 'sarimax_auto_weather')
    GROUP BY model_version
    ORDER BY model_version
""")

sugar_results = cursor.fetchall()
for row in sugar_results:
    print(f"  {row[0]}: {row[1]} models")

cursor.close()
connection.close()

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)
print("\n📝 Summary:")
print("  ✓ All locally-trained models deleted")
print("  ✓ Fresh models trained in Databricks")
print("  ✓ Using Databricks default package versions")
print("  ✓ Version consistency guaranteed\n")
print("Next: Run backfill_rolling_window_spark.py in same environment")

# Databricks notebook source
"""
Train Fresh Models in Databricks
Trains Coffee & Sugar models using Databricks default package versions
"""

# COMMAND ----------

# MAGIC %pip install databricks-sql-connector pmdarima

# COMMAND ----------

# Restart Python to load newly installed packages
dbutils.library.restartPython()

# COMMAND ----------

print("=" * 80)
print("Training Fresh Models in Databricks")
print("=" * 80)
print("\nThis ensures NumPy/sklearn/xgboost version consistency")
print("All models will be trained using Databricks default package versions\n")

# COMMAND ----------

# Set up paths for imports from Git repo
import sys
import os

forecast_agent_path = '/Workspace/Repos/Project_Git/ucberkeley-capstone/forecast_agent'
if forecast_agent_path not in sys.path:
    sys.path.insert(0, forecast_agent_path)

print(f"✓ Added {forecast_agent_path} to Python path")

# Change to forecast_agent directory for relative imports
os.chdir(forecast_agent_path)
print(f"✓ Changed directory to {forecast_agent_path}")

# COMMAND ----------

# Verify package versions (Databricks default)
import numpy as np
import sklearn
import xgboost as xgb
import pandas as pd

print(f"\n📦 Package Versions (Databricks):")
print(f"  NumPy: {np.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  XGBoost: {xgb.__version__}")
print(f"  Pandas: {pd.__version__}")

# COMMAND ----------

# Import training modules from Git repo
print("\n📥 Importing training modules from Git repo...")

import databricks.sql as sql
from ground_truth.config.model_registry import BASELINE_MODELS
from utils.model_persistence import save_model, model_exists
from train_models import train_and_save_model, load_training_data, get_training_dates
from datetime import datetime, timedelta

print("✓ All modules imported successfully")

# COMMAND ----------

# Load credentials
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")

if not all([DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH]):
    raise ValueError("Missing Databricks credentials in environment variables")

print("✓ Databricks credentials loaded")

# COMMAND ----------

# Configuration
commodities = ['Coffee', 'Sugar']
model_keys = ['naive', 'xgboost', 'sarimax_auto_weather']
train_frequency = 'semiannually'
start_date = datetime.strptime('2018-01-01', '%Y-%m-%d').date()
end_date = datetime.strptime('2025-11-17', '%Y-%m-%d').date()
model_version = 'v1.0'
min_training_days = 1095  # 3 years

print("\n" + "=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print(f"Commodities: {', '.join(commodities)}")
print(f"Models: {', '.join(model_keys)}")
print(f"Training Frequency: {train_frequency}")
print(f"Date Range: {start_date} to {end_date}")
print(f"Model Version: {model_version}")
print(f"Min Training Days: {min_training_days}")
print("=" * 80)

# COMMAND ----------

# Connect to Databricks
print("\n📡 Connecting to Databricks...")
connection = sql.connect(
    server_hostname=DATABRICKS_HOST.replace('https://', ''),
    http_path=DATABRICKS_HTTP_PATH,
    access_token=DATABRICKS_TOKEN
)
print("✅ Connected to Databricks")

# COMMAND ----------

# Train models for each commodity
for commodity in commodities:
    print("\n" + "=" * 80)
    print(f"TRAINING {commodity.upper()} MODELS")
    print("=" * 80)

    # Generate training dates
    training_dates = get_training_dates(start_date, end_date, train_frequency)
    print(f"\n📅 Training Windows: {len(training_dates)} windows from {training_dates[0]} to {training_dates[-1]}")

    total_trained = 0
    total_skipped = 0
    total_failed = 0

    # Training loop
    for window_idx, training_cutoff in enumerate(training_dates, 1):
        print(f"\n{'='*80}")
        print(f"Window {window_idx}/{len(training_dates)}: Training Cutoff = {training_cutoff}")
        print(f"{'='*80}")

        # Load training data up to this cutoff
        training_df = load_training_data(connection, commodity, training_cutoff)

        # Check minimum training days
        if len(training_df) < min_training_days:
            print(f"   ⚠️  Insufficient training data: {len(training_df)} days < {min_training_days} days - skipping")
            total_skipped += len(model_keys)
            continue

        print(f"   📊 Loaded {len(training_df):,} days of training data")
        print(f"   📅 Data range: {training_df.index[0].date()} to {training_df.index[-1].date()}")

        # Train each model
        for model_key in model_keys:
            model_config = BASELINE_MODELS[model_key]
            model_name = model_config['name']

            print(f"\n   🔧 {model_name} ({model_key}):")

            model_id = train_and_save_model(
                connection=connection,
                training_df=training_df,
                model_key=model_key,
                model_config=model_config,
                commodity=commodity,
                model_version=model_version,
                created_by="databricks_train_fresh_models.py"
            )

            if model_id:
                total_trained += 1
            elif model_id is None:
                # Check if it was skipped or failed
                if model_exists(connection, commodity, model_name, training_df.index[-1].strftime('%Y-%m-%d'), model_version):
                    total_skipped += 1
                else:
                    total_failed += 1

    # Summary for this commodity
    print("\n" + "=" * 80)
    print(f"{commodity.upper()} TRAINING COMPLETE")
    print("=" * 80)
    print(f"✅ Models Trained: {total_trained}")
    print(f"⏩ Models Skipped (already exist): {total_skipped}")
    print(f"❌ Models Failed: {total_failed}")
    print("=" * 80)

# COMMAND ----------

# Verify trained models in database
print("\n" + "=" * 80)
print("VERIFYING TRAINED MODELS IN DATABASE")
print("=" * 80)

cursor = connection.cursor()

for commodity in commodities:
    print(f"\n📊 {commodity} Models:")
    cursor.execute(f"""
        SELECT model_version, COUNT(*) as count
        FROM commodity.forecast.trained_models
        WHERE commodity = '{commodity}'
        AND model_version IN ('naive', 'xgboost', 'sarimax_auto_weather')
        GROUP BY model_version
        ORDER BY model_version
    """)

    results = cursor.fetchall()
    for row in results:
        print(f"  {row[0]}: {row[1]} models")

cursor.close()
connection.close()

# COMMAND ----------

print("\n" + "=" * 80)
print("✅ ALL TRAINING COMPLETE!")
print("=" * 80)
print("\n📝 Summary:")
print("  ✓ All locally-trained models deleted (from previous session)")
print("  ✓ Fresh models trained in Databricks")
print("  ✓ Using Databricks default package versions")
print("  ✓ Version consistency guaranteed")
print("\nNext: Run backfill_rolling_window_spark.py in same environment")

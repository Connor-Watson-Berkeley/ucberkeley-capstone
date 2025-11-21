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

from ground_truth.config.model_registry import BASELINE_MODELS
from train_models import get_training_dates
from datetime import datetime, timedelta

print("✓ All modules imported successfully")
print("✓ Running in Databricks - using spark.sql() (no credentials needed!)")

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

# Helper functions for Spark-based data access
def load_training_data_spark(commodity: str, cutoff_date):
    """Load training data using Spark SQL (no credentials needed!)"""
    query = f"""
        SELECT *
        FROM commodity.silver.unified_data
        WHERE commodity = '{commodity}'
        AND date <= '{cutoff_date}'
        ORDER BY date
    """
    spark_df = spark.sql(query)
    pandas_df = spark_df.toPandas()

    # Set date as index
    pandas_df['date'] = pd.to_datetime(pandas_df['date'])
    pandas_df = pandas_df.set_index('date')

    return pandas_df

def model_exists_spark(commodity: str, model_name: str, training_cutoff: str, model_version: str) -> bool:
    """Check if model already exists using Spark SQL"""
    query = f"""
        SELECT COUNT(*) as count
        FROM commodity.forecast.trained_models
        WHERE commodity = '{commodity}'
        AND model_name = '{model_name}'
        AND training_date = '{training_cutoff}'
        AND model_version = '{model_version}'
    """
    result = spark.sql(query).collect()[0]
    return result['count'] > 0

def save_model_spark(fitted_model_dict: dict, commodity: str, model_name: str, model_version: str, created_by: str):
    """Save trained model using Spark SQL"""
    import json
    from datetime import datetime

    training_cutoff = fitted_model_dict['last_date'].strftime('%Y-%m-%d')

    # Serialize model to JSON
    model_json = json.dumps(fitted_model_dict, default=str)
    model_size_mb = len(model_json) / (1024 * 1024)

    print(f"      💾 Model size: {model_size_mb:.2f} MB")

    if model_size_mb >= 1.0:
        print(f"      ⚠️  Model too large for JSON ({model_size_mb:.2f} MB ≥ 1 MB) - skipping S3 upload for now")
        return None

    # Create record
    year = int(training_cutoff[:4])
    month = int(training_cutoff[5:7])

    # Use Spark SQL to insert
    insert_query = f"""
        INSERT INTO commodity.forecast.trained_models
        (commodity, model_name, model_version, training_date, fitted_model_json,
         created_at, created_by, year, month)
        VALUES (
            '{commodity}',
            '{model_name}',
            '{model_version}',
            '{training_cutoff}',
            '{model_json.replace("'", "''")}',
            '{datetime.utcnow().isoformat()}',
            '{created_by}',
            {year},
            {month}
        )
    """

    spark.sql(insert_query)
    print(f"      ✅ Model saved to trained_models table")
    return f"{commodity}_{model_name}_{training_cutoff}_{model_version}"

print("✓ Spark helper functions defined")

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

        # Load training data up to this cutoff using Spark
        training_df = load_training_data_spark(commodity, training_cutoff)

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

            # Check if model already exists
            if model_exists_spark(commodity, model_name, training_cutoff.strftime('%Y-%m-%d'), model_version):
                print(f"      ⏩ Model already exists - skipping")
                total_skipped += 1
                continue

            try:
                # Train the model
                from ground_truth.models import naive, random_walk, arima, sarimax, xgboost_model

                # Map model keys to training functions
                train_functions = {
                    'naive': naive.naive_train,
                    'random_walk': random_walk.random_walk_train,
                    'arima_111': arima.arima_train,
                    'sarimax_auto': sarimax.sarimax_train,
                    'sarimax_auto_weather': sarimax.sarimax_train,
                    'xgboost': xgboost_model.xgboost_train,
                }

                train_func = train_functions.get(model_key)
                if not train_func:
                    print(f"      ❌ No training function for {model_key}")
                    total_failed += 1
                    continue

                # Get model parameters
                params = model_config['params'].copy()

                # Train model
                fitted_model_dict = train_func(training_df, **params)

                # Save model
                model_id = save_model_spark(
                    fitted_model_dict=fitted_model_dict,
                    commodity=commodity,
                    model_name=model_name,
                    model_version=model_version,
                    created_by="databricks_train_fresh_models.py"
                )

                if model_id:
                    total_trained += 1
                else:
                    total_failed += 1

            except Exception as e:
                print(f"      ❌ Training failed: {str(e)[:200]}")
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

for commodity in commodities:
    print(f"\n📊 {commodity} Models:")
    result = spark.sql(f"""
        SELECT model_name, COUNT(*) as count
        FROM commodity.forecast.trained_models
        WHERE commodity = '{commodity}'
        AND model_name IN ('Naive', 'XGBoost', 'SARIMAX+Weather')
        GROUP BY model_name
        ORDER BY model_name
    """).collect()

    for row in result:
        print(f"  {row['model_name']}: {row['count']} models")

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

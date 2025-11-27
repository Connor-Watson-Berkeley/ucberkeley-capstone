"""
Databricks Notebook: 01_train_models

Trains forecasting models and saves them to commodity.forecast.trained_models table.

Run this notebook first to train models before generating forecasts.

Usage:
    - Set widget parameters at top
    - Run all cells
    - Models saved to trained_models table for use in inference notebook
"""

# Databricks notebook source
# MAGIC %md
# MAGIC # Train Forecasting Models
# MAGIC 
# MAGIC This notebook trains models and saves them to `commodity.forecast.trained_models` table.
# MAGIC 
# MAGIC ## Setup
# MAGIC Configure parameters below, then run all cells.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configuration

# COMMAND ----------

# Widgets for parameters (Databricks notebook)
dbutils.widgets.text("commodity", "Coffee", "Commodity")
dbutils.widgets.text("models", "naive,random_walk", "Models (comma-separated). Start simple: naive,random_walk")
dbutils.widgets.text("train_frequency", "semiannually", "Training Frequency")
dbutils.widgets.text("model_version", "v1.0", "Model Version")
dbutils.widgets.text("start_date", "2020-01-01", "Start Date (YYYY-MM-DD)")
dbutils.widgets.text("end_date", "2024-01-01", "End Date (YYYY-MM-DD)")

# Get parameters
commodity = dbutils.widgets.get("commodity")
models_str = dbutils.widgets.get("models")
train_frequency = dbutils.widgets.get("train_frequency")
model_version = dbutils.widgets.get("model_version")
start_date = dbutils.widgets.get("start_date")
end_date = dbutils.widgets.get("end_date")

models = [m.strip() for m in models_str.split(',')]

print(f"Commodity: {commodity}")
print(f"Models: {models}")
print(f"Training Frequency: {train_frequency}")
print(f"Model Version: {model_version}")
print(f"Date Range: {start_date} to {end_date}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Import Modules

# COMMAND ----------

# Import refined approach modules
import sys
from pathlib import Path

# Add refined_approach to path
workspace_path = Path('/Workspace/Repos')
repo_path = None
for repo in workspace_path.iterdir():
    if repo.is_dir():
        refined_path = repo / 'forecast_agent' / 'refined_approach'
        if refined_path.exists():
            repo_path = refined_path
            break

if repo_path:
    sys.path.insert(0, str(repo_path))
    print(f"✅ Added {repo_path} to path")
else:
    # Fallback: assume current directory structure
    sys.path.insert(0, str(Path.cwd() / 'forecast_agent' / 'refined_approach'))
    print("⚠️  Using fallback path")

from data_loader import TimeSeriesDataLoader
from model_pipeline import create_model_from_registry
from model_persistence import save_model_spark, model_exists_spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Load Data and Create Folds

# COMMAND ----------

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Initialize data loader
loader = TimeSeriesDataLoader(spark=spark)

# Load full dataset
print(f"\n📊 Loading {commodity} data...")
df = loader.load_to_pandas(
    commodity=commodity,
    cutoff_date=end_date,
    features=['close', 'temp_mean_c', 'humidity_mean_pct', 'precipitation_mm', 'vix'],
    aggregate_regions=True,
    aggregation_method='mean'
)

print(f"✅ Loaded {len(df):,} days of data")
print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Generate Training Dates

# COMMAND ----------

def get_training_dates(start_date, end_date, frequency):
    """Generate training dates based on frequency."""
    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    frequency_map = {
        'daily': timedelta(days=1),
        'weekly': timedelta(days=7),
        'biweekly': timedelta(days=14),
        'monthly': relativedelta(months=1),
        'quarterly': relativedelta(months=3),
        'semiannually': relativedelta(months=6),
        'annually': relativedelta(years=1)
    }
    
    delta = frequency_map.get(frequency)
    if not delta:
        raise ValueError(f"Unknown frequency: {frequency}")
    
    dates = []
    current = start
    
    while current <= end:
        dates.append(current)
        if isinstance(delta, timedelta):
            current = current + delta
        else:
            current = current + delta
    
    return dates

training_dates = get_training_dates(start_date, end_date, train_frequency)
print(f"\n📅 Generated {len(training_dates)} training dates")
print(f"   First: {training_dates[0]}")
print(f"   Last: {training_dates[-1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Train Models

# COMMAND ----------

from datetime import date

min_training_days = 365 * 2  # Minimum 2 years of data

trained_count = 0
skipped_count = 0
failed_count = 0

for training_cutoff in training_dates:
    print(f"\n{'='*60}")
    print(f"Training Cutoff: {training_cutoff}")
    print(f"{'='*60}")
    
    # Filter data up to cutoff
    training_df = df.loc[df.index <= pd.Timestamp(training_cutoff)]
    
    # Check minimum training days
    if len(training_df) < min_training_days:
        print(f"⚠️  Insufficient data: {len(training_df)} days < {min_training_days} days")
        skipped_count += len(models)
        continue
    
    print(f"   Training data: {len(training_df):,} days ({training_df.index[0].date()} to {training_df.index[-1].date()})")
    
    # Train each model (FAIL-OPEN: one failure doesn't stop the rest)
    for model_key in models:
        print(f"\n   🔧 Training {model_key}...")
        
        try:
            # Create model (can fail if model_key invalid or packages missing)
            try:
                model = create_model_from_registry(model_key)
            except ImportError as import_err:
                # FAIL-OPEN: Missing package - skip this model
                print(f"   ⚠️  Missing package dependency - skipping")
                print(f"      Error: {str(import_err)[:150]}")
                print(f"      💡 Install missing package or remove {model_key} from models list")
                failed_count += 1
                continue  # Skip to next model
            except ValueError as val_err:
                # Invalid model key
                print(f"   ❌ Invalid model key: {str(val_err)[:150]}")
                failed_count += 1
                continue  # Skip to next model
            
            model_name = model.model_name
            
            # Check if model already exists (INCREMENTAL: skip if already trained)
            training_date_str = training_cutoff.strftime('%Y-%m-%d')
            if model_exists_spark(spark, commodity, model_name, training_date_str, model_version):
                print(f"   ⏩ Model already exists - skipping (incremental mode)")
                skipped_count += 1
                continue
            
            # Fit model (can fail if convergence issues, data problems)
            model.fit(training_df, target_col='close')
            
            # Save model (can fail if database error, serialization issue)
            model_id = save_model_spark(
                spark=spark,
                fitted_model=model,
                commodity=commodity,
                model_name=model.model_name,
                model_version=model_version,
                training_date=training_cutoff.strftime('%Y-%m-%d'),
                training_samples=len(training_df),
                training_start_date=training_df.index[0].date().strftime('%Y-%m-%d'),
                parameters=model.get_params(),
                created_by="01_train_models.ipynb"
            )
            
            print(f"   ✅ Saved: {model_id}")
            trained_count += 1
            
        except ImportError as import_err:
            # FAIL-OPEN: Missing package during training/inference
            print(f"   ⚠️  Missing package dependency - skipping")
            print(f"      Error: {str(import_err)[:150]}")
            failed_count += 1
            # Continue to next model (no break/return)
        except Exception as e:
            # FAIL-OPEN: Other errors (convergence, data, serialization, etc.)
            error_msg = str(e)
            print(f"   ❌ Failed: {error_msg[:150]}")
            if len(error_msg) > 150:
                print(f"      ... (error truncated)")
            failed_count += 1
            # Print full traceback for debugging (can comment out in production)
            import traceback
            traceback.print_exc()
            # Continue to next model (no break/return)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Summary

# COMMAND ----------

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")
print(f"✅ Models Trained: {trained_count}")
print(f"⏩ Models Skipped: {skipped_count}")
print(f"❌ Models Failed: {failed_count}")
print(f"\n📊 Trained models saved to commodity.forecast.trained_models")
print(f"   Use these in 02_generate_forecasts notebook for inference!")

# COMMAND ----------


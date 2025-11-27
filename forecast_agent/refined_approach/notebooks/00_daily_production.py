"""
Databricks Notebook: 00_daily_production

Daily production workflow: Check if retraining needed, generate today's forecast.

This notebook is designed to run daily (via Databricks Jobs) and:
1. Check if models need retraining based on cadence
2. Train new models if needed
3. Generate forecasts for today using most recent trained models
4. Write to distributions table

Usage:
    - Set up as Databricks Job scheduled to run daily
    - Automatically handles retraining cadence
    - Only generates forecast for today
"""

# Databricks notebook source
# MAGIC %md
# MAGIC # Daily Production Forecast Workflow
# MAGIC 
# MAGIC Runs daily to:
# MAGIC 1. Check if models need retraining (based on cadence)
# MAGIC 2. Train new models if needed
# MAGIC 3. Generate today's forecast using most recent trained models
# MAGIC 4. Populate distributions table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configuration

# COMMAND ----------

# Widgets for parameters
dbutils.widgets.text("commodity", "Coffee", "Commodity")
dbutils.widgets.text("models", "naive,random_walk,xgboost", "Models (comma-separated)")
dbutils.widgets.text("train_frequency", "semiannually", "Training Frequency")
dbutils.widgets.text("model_version", "v1.0", "Model Version")
dbutils.widgets.text("forecast_date", "", "Forecast Date (YYYY-MM-DD, empty = today)")

# Get parameters
commodity = dbutils.widgets.get("commodity")
models_str = dbutils.widgets.get("models")
train_frequency = dbutils.widgets.get("train_frequency")
model_version = dbutils.widgets.get("model_version")
forecast_date_str = dbutils.widgets.get("forecast_date")

models = [m.strip() for m in models_str.split(',')]

# Use today if forecast_date not specified
from datetime import date, datetime
if forecast_date_str:
    forecast_date = datetime.strptime(forecast_date_str, '%Y-%m-%d').date()
else:
    forecast_date = date.today()

print(f"Commodity: {commodity}")
print(f"Models: {models}")
print(f"Training Frequency: {train_frequency}")
print(f"Model Version: {model_version}")
print(f"Forecast Date: {forecast_date}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Import Modules

# COMMAND ----------

# Import refined approach modules
import sys
from pathlib import Path
import pandas as pd

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
    sys.path.insert(0, str(Path.cwd() / 'forecast_agent' / 'refined_approach'))
    print("⚠️  Using fallback path")

from data_loader import TimeSeriesDataLoader
from model_pipeline import create_model_from_registry
from model_persistence import save_model_spark, model_exists_spark, load_model_spark
from distributions_writer import DistributionsWriter, get_existing_forecast_dates
from daily_production import should_retrain_today, get_most_recent_trained_model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Check Training Status and Retrain if Needed

# COMMAND ----------

from datetime import timedelta

print(f"\n{'='*60}")
print("PHASE 1: Training Check")
print(f"{'='*60}")

# Get data up to forecast_date (yesterday, since we're forecasting today)
data_cutoff = forecast_date - timedelta(days=1)

# Load data for training
loader = TimeSeriesDataLoader(spark=spark)
print(f"\n📊 Loading {commodity} data up to {data_cutoff}...")
df = loader.load_to_pandas(
    commodity=commodity,
    cutoff_date=data_cutoff.strftime('%Y-%m-%d'),
    features=['close', 'temp_mean_c', 'humidity_mean_pct', 'precipitation_mm', 'vix'],
    aggregate_regions=True,
    aggregation_method='mean'
)

print(f"✅ Loaded {len(df):,} days of data")

# Check each model to see if retraining is needed
models_trained = 0
models_skipped = 0

for model_key in models:
    try:
        model = create_model_from_registry(model_key)
        model_name = model.model_name
        
        # Check if retraining is needed
        needs_training = should_retrain_today(
            spark=spark,
            commodity=commodity,
            model_name=model_name,
            train_frequency=train_frequency,
            model_version=model_version,
            today=forecast_date
        )
        
        if not needs_training:
            print(f"\n   ⏩ {model_name}: No retraining needed (within cadence)")
            models_skipped += 1
            continue
        
        # Check if already trained for this date
        if model_exists_spark(spark, commodity, model_name, forecast_date.strftime('%Y-%m-%d'), model_version):
            print(f"\n   ⏩ {model_name}: Already trained for {forecast_date}")
            models_skipped += 1
            continue
        
        print(f"\n   🔧 {model_name}: Retraining needed (cadence check passed)")
        
        # Train model
        try:
            model.fit(df, target_col='close')
            
            # Save model
            model_id = save_model_spark(
                spark=spark,
                fitted_model=model,
                commodity=commodity,
                model_name=model_name,
                model_version=model_version,
                training_date=forecast_date.strftime('%Y-%m-%d'),
                training_samples=len(df),
                training_start_date=df.index[0].date().strftime('%Y-%m-%d'),
                parameters=model.get_params(),
                created_by="00_daily_production.ipynb"
            )
            
            print(f"   ✅ Trained and saved: {model_id}")
            models_trained += 1
            
        except Exception as e:
            print(f"   ❌ Training failed: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"\n   ❌ Error checking {model_key}: {str(e)[:100]}")

print(f"\n📊 Training Summary:")
print(f"   ✅ Trained: {models_trained}")
print(f"   ⏩ Skipped: {models_skipped}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Generate Today's Forecasts

# COMMAND ----------

print(f"\n{'='*60}")
print("PHASE 2: Forecast Generation")
print(f"{'='*60}")

# Check if forecast already exists
existing_dates = get_existing_forecast_dates(spark, commodity, model_version)

if forecast_date in existing_dates:
    print(f"\n⏩ Forecast for {forecast_date} already exists - skipping")
else:
    print(f"\n📊 Generating forecast for {forecast_date}...")
    
    # Load data up to forecast_date (includes today's data if available)
    forecast_data_df = loader.load_to_pandas(
        commodity=commodity,
        cutoff_date=forecast_date.strftime('%Y-%m-%d'),
        features=['close', 'temp_mean_c', 'humidity_mean_pct', 'precipitation_mm', 'vix'],
        aggregate_regions=True,
        aggregation_method='mean'
    )
    
    forecasts_generated = 0
    forecasts_failed = 0
    
    writer = DistributionsWriter(spark=spark)
    all_forecasts = []
    
    for model_key in models:
        try:
            model = create_model_from_registry(model_key)
            model_name = model.model_name
            
            print(f"\n   🔧 {model_name}...")
            
            # Get most recent trained model (training_date <= forecast_date)
            model_info = get_most_recent_trained_model(
                spark=spark,
                commodity=commodity,
                model_name=model_name,
                forecast_date=forecast_date,
                model_version=model_version
            )
            
            if not model_info:
                print(f"   ⚠️  No trained model found (need training_date <= {forecast_date})")
                forecasts_failed += 1
                continue
            
            print(f"   📦 Using model trained on {model_info['training_date']}")
            
            # Load model
            loaded_data = load_model_spark(
                spark=spark,
                commodity=commodity,
                model_name=model_name,
                training_date=model_info['training_date'].strftime('%Y-%m-%d'),
                model_version=model_version
            )
            
            if not loaded_data:
                print(f"   ⚠️  Could not load model")
                forecasts_failed += 1
                continue
            
            # Reconstruct model (simplified - actual implementation depends on model type)
            fitted_model_dict = loaded_data['fitted_model']
            
            # For ModelPipeline, we can reconstruct it
            if isinstance(fitted_model_dict, dict):
                # Reconstruct model from saved state
                if 'model_type' in fitted_model_dict and fitted_model_dict['model_type'] in ['naive', 'randomwalk']:
                    # Simple models can be reconstructed
                    model = create_model_from_registry(model_key)
                    if hasattr(model, 'last_value') and 'last_value' in fitted_model_dict:
                        model.last_value = fitted_model_dict['last_value']
                    if hasattr(model, 'last_date') and 'last_date' in fitted_model_dict:
                        model.last_date = pd.Timestamp(fitted_model_dict['last_date'])
                    model.is_fitted = True
            
            # Generate forecast
            forecast_df = model.predict(horizon=14)
            
            # Generate Monte Carlo paths
            from distributions_writer import DistributionsWriter
            paths = writer._generate_paths_from_mean(
                forecast_df['forecast'].values,
                std=2.5,  # TODO: Get from model
                n_paths=2000
            )
            
            forecast_data = {
                'forecast_start_date': forecast_date,
                'data_cutoff_date': model_info['training_date'],
                'paths': paths,
                'mean_forecast': forecast_df['forecast'].values,
                'forecast_std': 2.5,  # TODO: Get from model
                'model_version': f"{model_key}_{model_version}",  # Use model_key as version identifier
                'commodity': commodity
            }
            
            all_forecasts.append(forecast_data)
            forecasts_generated += 1
            print(f"   ✅ Generated forecast")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:100]}")
            forecasts_failed += 1
            import traceback
            traceback.print_exc()
    
    # Write all forecasts to distributions table
    if all_forecasts:
        print(f"\n💾 Writing {len(all_forecasts)} forecasts to distributions table...")
        
        for forecast_data in all_forecasts:
            # Get data_cutoff_date from model_info
            model_info = get_most_recent_trained_model(
                spark=spark,
                commodity=commodity,
                model_name=forecast_data['model_version'].split('_')[0],  # Extract model name
                forecast_date=forecast_date,
                model_version=model_version
            )
            
            if model_info:
                writer.write_distributions(
                    forecasts=[forecast_data],
                    commodity=commodity,
                    model_version=forecast_data['model_version'],
                    data_cutoff_date=model_info['training_date']
                )
        
        print(f"✅ Wrote forecasts to commodity.forecast.distributions")
    
    print(f"\n📊 Forecast Summary:")
    print(f"   ✅ Generated: {forecasts_generated}")
    print(f"   ❌ Failed: {forecasts_failed}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Final Summary

# COMMAND ----------

print(f"\n{'='*60}")
print("DAILY PRODUCTION COMPLETE")
print(f"{'='*60}")
print(f"Date: {forecast_date}")
print(f"Commodity: {commodity}")
print(f"\nTraining:")
print(f"   ✅ Trained: {models_trained}")
print(f"   ⏩ Skipped: {models_skipped}")
print(f"\nForecasts:")
if forecast_date not in existing_dates:
    print(f"   ✅ Generated: {forecasts_generated}")
    print(f"   ❌ Failed: {forecasts_failed}")
else:
    print(f"   ⏩ Already exists - skipped")
print(f"\n{'='*60}")

# COMMAND ----------


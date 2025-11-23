# Databricks notebook source
# MAGIC %md
# MAGIC # Forecast Agent - Databricks Quickstart
# MAGIC
# MAGIC Production-ready commodity price forecasting on Databricks.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Installs dependencies
# MAGIC 2. Loads data from commodity.silver.unified_data
# MAGIC 3. Runs production forecast (SARIMAX+Weather)
# MAGIC 4. Writes to production tables
# MAGIC 5. Exports for trading agent
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Unity Catalog table: `commodity.silver.unified_data`
# MAGIC - Cluster with 8GB+ memory

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# Install required packages
%pip install statsmodels pmdarima xgboost prophet neuralprophet statsforecast

# COMMAND ----------

# Restart Python kernel
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Setup Imports and Path

# COMMAND ----------

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add forecast_agent to path
sys.path.insert(0, '/Workspace/Repos/<YOUR_USERNAME>/ucberkeley-capstone/forecast_agent')

# COMMAND ----------

# Verify imports
from ground_truth.storage.production_writer import ProductionForecastWriter
from ground_truth.config.model_registry import BASELINE_MODELS
from ground_truth.models.sarimax import sarimax_forecast

print(f"Found {len(BASELINE_MODELS)} models in registry")
print("Imports successful!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Data

# COMMAND ----------

# Load unified data from commodity.silver schema
df_spark = spark.table("commodity.silver.unified_data")

# Filter for Coffee
df_coffee_spark = df_spark.filter(df_spark.commodity == 'Coffee')

# Convert to Pandas (required for time series models)
df_coffee = df_coffee_spark.toPandas()
df_coffee['date'] = pd.to_datetime(df_coffee['date'])
df_coffee = df_coffee.set_index('date').sort_index()

print(f"Loaded {len(df_coffee):,} rows for Coffee")
print(f"Date range: {df_coffee.index.min().date()} to {df_coffee.index.max().date()}")
print(f"Columns: {list(df_coffee.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Initialize Production Writer

# COMMAND ----------

# Initialize writer (outputs to DBFS)
writer = ProductionForecastWriter("/dbfs/production_forecasts")

print("Production writer initialized")
print(f"Output directory: /dbfs/production_forecasts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run Production Forecast

# COMMAND ----------

# Configuration
commodity = 'Coffee'
model_version = 'sarimax_weather_v1'
horizon = 14
exog_features = ['temp_c', 'humidity_pct', 'precipitation_mm']

# Training metadata
data_cutoff_date = df_coffee.index[-1]
generation_timestamp = datetime.now()

print(f"Training model: {model_version}")
print(f"Data cutoff: {data_cutoff_date.date()}")
print(f"Horizon: {horizon} days")
print(f"Features: {exog_features}")
print()

# Run forecast
result = sarimax_forecast(
    df_pandas=df_coffee,
    commodity=commodity,
    target='close',
    horizon=horizon,
    exog_features=exog_features
)

print(f"Forecast generated:")
print(result['forecast_df'].head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write Point Forecasts

# COMMAND ----------

# Write point forecasts to production table
rows_written = writer.write_point_forecasts(
    forecast_df=result['forecast_df'],
    model_version=model_version,
    commodity=commodity,
    data_cutoff_date=data_cutoff_date,
    generation_timestamp=generation_timestamp
)

print(f"Wrote {rows_written} point forecasts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Generate and Write Distributions

# COMMAND ----------

# Generate Monte Carlo sample paths for risk analysis
forecast_values = result['forecast_df']['forecast'].values
n_paths = 2000

# Simple normal distribution (can be enhanced with bootstrap/simulation)
sample_paths = np.random.normal(
    loc=forecast_values,
    scale=2.5,  # Residual std from walk-forward evaluation
    size=(n_paths, horizon)
)

print(f"Generated {n_paths} Monte Carlo paths")
print(f"Sample path shape: {sample_paths.shape}")

# COMMAND ----------

# Write distributions to production table
rows_written = writer.write_distributions(
    forecast_start_date=result['forecast_df']['date'].iloc[0],
    data_cutoff_date=data_cutoff_date,
    model_version=model_version,
    commodity=commodity,
    sample_paths=sample_paths,
    generation_timestamp=generation_timestamp,
    n_paths=n_paths
)

print(f"Wrote {rows_written} distribution paths")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Export for Trading Agent

# COMMAND ----------

# Export forecast in JSON format for trading agent
trading_forecast = writer.export_for_trading_agent(
    commodity=commodity,
    model_version=model_version,
    output_path='/dbfs/trading_agent_forecast.json'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. View Results

# COMMAND ----------

# Load and display point forecasts
point_forecasts = spark.read.parquet("/dbfs/production_forecasts/point_forecasts.parquet")
display(point_forecasts.orderBy("forecast_date"))

# COMMAND ----------

# Load and display distributions (first 10 paths)
distributions = spark.read.parquet("/dbfs/production_forecasts/distributions.parquet")
display(distributions.filter("path_id <= 10").orderBy("path_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Optional: Write to Unity Catalog

# COMMAND ----------

# Write to commodity.silver schema (append mode)
point_forecasts.write.mode("append").saveAsTable("commodity.forecast.point_forecasts")
distributions.write.mode("append").saveAsTable("commodity.forecast.distributions")

print("Written to Unity Catalog:")
print("  - commodity.forecast.point_forecasts")
print("  - commodity.forecast.distributions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production Deployment Complete!
# MAGIC
# MAGIC **Outputs:**
# MAGIC - Point forecasts: `/dbfs/production_forecasts/point_forecasts.parquet`
# MAGIC - Distributions: `/dbfs/production_forecasts/distributions.parquet`
# MAGIC - Trading agent JSON: `/dbfs/trading_agent_forecast.json`
# MAGIC
# MAGIC **Next Steps:**
# MAGIC 1. Schedule this notebook to run weekly (Databricks Workflows)
# MAGIC 2. Set up monitoring/alerts
# MAGIC 3. Integrate with trading agent
# MAGIC 4. Backfill historical forecasts with actuals
# MAGIC
# MAGIC **Weekly Retraining Schedule:**
# MAGIC - Frequency: Monday at midnight
# MAGIC - Cron: `0 0 * * 1`
# MAGIC - Cluster: 8GB+ memory recommended

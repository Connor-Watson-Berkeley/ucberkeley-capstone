# Databricks notebook source
# MAGIC %md
# MAGIC # Spark-Parallelized Forecast Backfill
# MAGIC
# MAGIC **Purpose**: Run historical forecast backfill using Spark for massive parallelization
# MAGIC
# MAGIC **Performance**:
# MAGIC - Sequential: ~40-120 hours for 143,750 forecasts
# MAGIC - Spark (100 cores): ~1-3 hours
# MAGIC
# MAGIC **Architecture**:
# MAGIC 1. Generate all (commodity, model, date) forecast tasks
# MAGIC 2. Load and broadcast pretrained models to all workers
# MAGIC 3. Distribute tasks across cluster using `mapPartitions`
# MAGIC 4. Each worker: loads data → runs inference → generates Monte Carlo paths
# MAGIC 5. Write results directly to Delta tables
# MAGIC
# MAGIC **Cluster Recommendations**:
# MAGIC - Runtime: 13.3 LTS ML (includes Spark, Pandas, NumPy)
# MAGIC - Workers: 10-20 workers (Standard_DS3_v2 or similar)
# MAGIC - Driver: Standard_DS3_v2
# MAGIC - Autoscaling: Enabled
# MAGIC
# MAGIC **⚠️ Important**: Uses pretrained models from `commodity.forecast.trained_models` table. Ensure models are trained first using `train_models.py`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# Install dependencies
%pip install statsmodels==0.14.0 pmdarima==2.0.4 xgboost==2.0.3 prophet==1.1.5 -q

# COMMAND ----------

import sys
from pathlib import Path

# Update this path to your Databricks Repos location
REPO_PATH = "/Workspace/Repos/<YOUR_USERNAME>/ucberkeley-capstone/forecast_agent"

# Add to Python path
sys.path.insert(0, REPO_PATH)

print(f"✓ Added {REPO_PATH} to Python path")

# COMMAND ----------

# Import Spark backfill function
from backfill_rolling_window_spark import backfill_all_models_spark

print("✓ Loaded Spark backfill function")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration Options

# COMMAND ----------

# CONFIGURATION
commodities = ['Coffee', 'Sugar']
models = ['naive', 'sarimax_auto_weather', 'xgboost']  # Add more as needed
train_frequency = 'semiannually'  # Must match how models were trained
start_date = '2018-01-01'
end_date = '2025-11-17'
num_partitions = 200  # Adjust based on cluster size (rule of thumb: 2-3x number of cores)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Options
# MAGIC
# MAGIC Available models (see `ground_truth/config/model_registry.py`):
# MAGIC - `naive` - Last observed value
# MAGIC - `random_walk` - Random walk with drift
# MAGIC - `arima` - ARIMA(5,1,0)
# MAGIC - `sarimax_auto` - Auto SARIMAX
# MAGIC - `sarimax_auto_weather` - Auto SARIMAX with weather features
# MAGIC - `sarimax_auto_weather_vix` - Auto SARIMAX with weather + VIX
# MAGIC - `xgboost` - XGBoost with technical indicators
# MAGIC - `prophet_v1` - Facebook Prophet
# MAGIC - And 17 more...
# MAGIC
# MAGIC Set `models = None` to backfill ALL 25 models.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run Backfill

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preview: How many forecasts will be generated?

# COMMAND ----------

from datetime import datetime
import pandas as pd

start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
num_dates = len(pd.date_range(start_dt, end_dt, freq='D'))
num_models = len(models) if models else 25
num_commodities = len(commodities)

total_forecasts = num_dates * num_models * num_commodities
total_paths = total_forecasts * 2000  # 2000 Monte Carlo paths per forecast

print(f"📊 Backfill Scope:")
print(f"  Commodities: {num_commodities}")
print(f"  Models: {num_models}")
print(f"  Date Range: {start_date} to {end_date} ({num_dates} days)")
print(f"  Total Forecasts: {total_forecasts:,}")
print(f"  Total Paths (2000 per forecast): {total_paths:,}")
print()
print(f"⏱️  Estimated Runtime:")
print(f"  Sequential: {total_forecasts / 60:.0f} - {total_forecasts / 20:.0f} hours")
print(f"  Spark ({num_partitions} partitions): 1-3 hours")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute Backfill
# MAGIC
# MAGIC **⚠️ This will take 1-3 hours depending on cluster size**

# COMMAND ----------

backfill_all_models_spark(
    commodities=commodities,
    models=models,
    train_frequency=train_frequency,
    start_date=start_date,
    end_date=end_date,
    num_partitions=num_partitions,
    batch_size=10000
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verify Results

# COMMAND ----------

# Check what was written to distributions table
result = spark.sql("""
SELECT
    commodity,
    model_version,
    COUNT(DISTINCT forecast_start_date) as num_dates,
    COUNT(*) / 2000 as num_forecasts,  -- Each forecast has 2000 paths
    MIN(forecast_start_date) as earliest_date,
    MAX(forecast_start_date) as latest_date
FROM commodity.forecast.distributions
WHERE is_actuals = FALSE
GROUP BY commodity, model_version
ORDER BY commodity, num_forecasts DESC
""")

display(result)

# COMMAND ----------

# Check specific model coverage
model_to_check = 'sarimax_auto_weather'

result = spark.sql(f"""
SELECT
    DATE_TRUNC('month', forecast_start_date) as month,
    COUNT(DISTINCT forecast_start_date) as num_dates,
    COUNT(*) / 2000 as num_forecasts
FROM commodity.forecast.distributions
WHERE commodity = 'Coffee'
  AND model_version = '{model_to_check}'
  AND is_actuals = FALSE
GROUP BY DATE_TRUNC('month', forecast_start_date)
ORDER BY month
""")

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Next Steps
# MAGIC
# MAGIC After backfill completes:
# MAGIC
# MAGIC 1. **Backfill Actuals** (if not done):
# MAGIC    ```python
# MAGIC    %run ./backfill_actuals
# MAGIC    backfill_actuals(commodity='Coffee', start_date='2018-01-01', end_date='2025-11-17')
# MAGIC    backfill_actuals(commodity='Sugar', start_date='2018-01-01', end_date='2025-11-17')
# MAGIC    ```
# MAGIC
# MAGIC 2. **Evaluate Forecasts**:
# MAGIC    ```bash
# MAGIC    python evaluate_historical_forecasts.py --commodity Coffee --models sarimax_auto_weather
# MAGIC    ```
# MAGIC
# MAGIC 3. **Check Coverage**:
# MAGIC    ```bash
# MAGIC    python check_backfill_coverage.py
# MAGIC    ```
# MAGIC
# MAGIC 4. **Generate Dashboards**:
# MAGIC    ```bash
# MAGIC    python experiments/generate_model_comparison_dashboard.py
# MAGIC    ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix: Troubleshooting

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Pretrained Models Availability
# MAGIC
# MAGIC The Spark backfill requires pretrained models to exist in the `commodity.forecast.trained_models` table.

# COMMAND ----------

# Check what pretrained models exist
result = spark.sql("""
SELECT
    commodity,
    model_version,
    training_date,
    train_frequency,
    model_type,
    storage_type
FROM commodity.forecast.trained_models
ORDER BY commodity, model_version, training_date DESC
""")

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### If Models Missing: Train Them First
# MAGIC
# MAGIC ```bash
# MAGIC # On local machine or Databricks
# MAGIC python train_models.py \
# MAGIC     --commodity Coffee \
# MAGIC     --models sarimax_auto_weather xgboost \
# MAGIC     --train-frequency semiannually \
# MAGIC     --start-date 2018-01-01 \
# MAGIC     --end-date 2025-11-17
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Monitor Spark Job Progress
# MAGIC
# MAGIC While backfill is running:
# MAGIC 1. Click **Spark UI** in cluster details
# MAGIC 2. Look for the `mapPartitions` stage
# MAGIC 3. Check progress: Tasks completed / Total tasks
# MAGIC 4. Monitor memory usage and adjust cluster if needed

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performance Tuning
# MAGIC
# MAGIC If backfill is slow:
# MAGIC
# MAGIC **Increase Parallelism**:
# MAGIC - Set `num_partitions = 400` (or higher)
# MAGIC - Add more workers to cluster
# MAGIC
# MAGIC **Optimize Memory**:
# MAGIC - Use larger worker types (Standard_DS4_v2)
# MAGIC - Enable autoscaling
# MAGIC
# MAGIC **Reduce Scope** (for testing):
# MAGIC - Limit date range: `start_date='2024-01-01'`
# MAGIC - Limit models: `models=['naive', 'xgboost']`
# MAGIC - Single commodity: `commodities=['Coffee']`

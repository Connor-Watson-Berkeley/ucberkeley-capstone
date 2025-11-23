# Databricks notebook source
# MAGIC %md
# MAGIC # DARTS Temporal Fusion Transformer (TFT) Experiment
# MAGIC
# MAGIC This notebook trains a TFT model on coffee price data with weather covariates using the DARTS library.
# MAGIC
# MAGIC **Model Features:**
# MAGIC - Temporal Fusion Transformer with attention mechanisms
# MAGIC - Multi-horizon forecasting with probabilistic outputs
# MAGIC - Incorporates weather covariates for improved accuracy
# MAGIC - Built-in interpretability for feature importance

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install DARTS
# MAGIC
# MAGIC Install DARTS with all dependencies

# COMMAND ----------

# MAGIC %pip install darts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae
from darts.utils.likelihood_models import QuantileRegression

import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data from Databricks

# COMMAND ----------

# Configuration
COMMODITY = 'Coffee'
LOOKBACK_DAYS = 365
FORECAST_HORIZON = 14

# Load data using Spark SQL
query = f"""
SELECT
    date,
    price,
    temperature_2m_mean,
    precipitation_sum,
    wind_speed_10m_max,
    relative_humidity_2m_mean,
    soil_moisture_0_to_7cm_mean,
    shortwave_radiation_sum,
    et0_fao_evapotranspiration_sum
FROM commodity.silver.unified_data
WHERE commodity = '{COMMODITY}'
    AND date >= DATE_SUB(CURRENT_DATE(), {LOOKBACK_DAYS})
ORDER BY date
"""

print(f"Loading {COMMODITY} data for last {LOOKBACK_DAYS} days...")
df = spark.sql(query).toPandas()

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Handle missing values
df = df.fillna(method='ffill').fillna(method='bfill')

print(f"✓ Loaded {len(df)} rows")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print(f"  Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare DARTS TimeSeries

# COMMAND ----------

# Define weather covariates
weather_covariates = [
    'temperature_2m_mean',
    'precipitation_sum',
    'wind_speed_10m_max',
    'relative_humidity_2m_mean',
    'soil_moisture_0_to_7cm_mean',
    'shortwave_radiation_sum',
    'et0_fao_evapotranspiration_sum'
]

# Set datetime index
df_indexed = df.set_index('date')

# Create target series
print("Creating DARTS TimeSeries...")
target_series = TimeSeries.from_dataframe(
    df_indexed,
    value_cols='price',
    freq='D'
)

# Create covariate series
covariate_series = TimeSeries.from_dataframe(
    df_indexed,
    value_cols=weather_covariates,
    freq='D'
)

print(f"✓ Target series length: {len(target_series)}")
print(f"✓ Covariate series length: {len(covariate_series)}")
print(f"✓ Number of covariates: {len(weather_covariates)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Validation Split and Scaling

# COMMAND ----------

# Split into train/validation
train_size = int(len(target_series) * 0.8)
train_target = target_series[:train_size]
val_target = target_series[train_size:]

train_covariates = covariate_series[:train_size]
val_covariates = covariate_series[train_size:]

print(f"Train size: {len(train_target)} days")
print(f"Validation size: {len(val_target)} days")

# Scale data
print("\nScaling data...")
target_scaler = Scaler()
train_target_scaled = target_scaler.fit_transform(train_target)
val_target_scaled = target_scaler.transform(val_target)

covariate_scaler = Scaler()
train_covariates_scaled = covariate_scaler.fit_transform(train_covariates)
val_covariates_scaled = covariate_scaler.transform(val_covariates)

print("✓ Scaling completed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure and Train TFT Model

# COMMAND ----------

# Model hyperparameters
INPUT_CHUNK_LENGTH = 60
OUTPUT_CHUNK_LENGTH = 14
HIDDEN_SIZE = 64
LSTM_LAYERS = 2
NUM_ATTENTION_HEADS = 4
DROPOUT = 0.1
BATCH_SIZE = 32
N_EPOCHS = 50
LEARNING_RATE = 1e-3

print("=" * 80)
print("TFT Model Configuration")
print("=" * 80)
print(f"Input sequence length: {INPUT_CHUNK_LENGTH} days")
print(f"Output sequence length: {OUTPUT_CHUNK_LENGTH} days")
print(f"Hidden layer size: {HIDDEN_SIZE}")
print(f"LSTM layers: {LSTM_LAYERS}")
print(f"Attention heads: {NUM_ATTENTION_HEADS}")
print(f"Dropout: {DROPOUT}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {N_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print("=" * 80)

# Initialize TFT model
model = TFTModel(
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=OUTPUT_CHUNK_LENGTH,
    hidden_size=HIDDEN_SIZE,
    lstm_layers=LSTM_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    dropout=DROPOUT,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    optimizer_kwargs={'lr': LEARNING_RATE},
    likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),  # Probabilistic forecasting
    random_state=42,
    force_reset=True,
    save_checkpoints=True,
    pl_trainer_kwargs={
        "accelerator": "auto",
        "callbacks": [],
    }
)

print("\n✓ Model initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model

# COMMAND ----------

print("Training TFT model...")
print(f"This may take several minutes with {N_EPOCHS} epochs...")

model.fit(
    series=train_target_scaled,
    past_covariates=train_covariates_scaled,
    val_series=val_target_scaled,
    val_past_covariates=val_covariates_scaled,
    verbose=True
)

print("\n✓ Training completed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Forecasts and Evaluate

# COMMAND ----------

# Make predictions on validation set
print("Generating forecasts on validation set...")
val_forecast_scaled = model.predict(
    n=len(val_target),
    series=train_target_scaled,
    past_covariates=train_covariates_scaled,
    num_samples=100  # For probabilistic forecasting
)

# Inverse transform predictions
val_forecast = target_scaler.inverse_transform(val_forecast_scaled)

# Calculate metrics
mape_score = mape(val_target, val_forecast)
rmse_score = rmse(val_target, val_forecast)
mae_score = mae(val_target, val_forecast)

print("=" * 80)
print("Validation Metrics")
print("=" * 80)
print(f"MAPE: {mape_score:.2f}%")
print(f"RMSE: ${rmse_score:.4f}")
print(f"MAE: ${mae_score:.4f}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Future Forecast

# COMMAND ----------

# Generate forecast for next 14 days
print(f"Generating {FORECAST_HORIZON}-day forecast...")
forecast_scaled = model.predict(
    n=FORECAST_HORIZON,
    series=train_target_scaled,
    past_covariates=train_covariates_scaled,
    num_samples=100  # For probabilistic forecasting
)

# Inverse transform predictions
forecast = target_scaler.inverse_transform(forecast_scaled)

# Extract quantiles
forecast_median = forecast.quantile(0.5)
forecast_lower = forecast.quantile(0.1)
forecast_upper = forecast.quantile(0.9)

# Create forecast DataFrame
forecast_df = pd.DataFrame({
    'date': forecast.time_index,
    'forecast_median': forecast_median.values().flatten(),
    'forecast_lower_10': forecast_lower.values().flatten(),
    'forecast_upper_90': forecast_upper.values().flatten()
})

print("\n" + "=" * 80)
print(f"{COMMODITY} Price Forecast (Next {FORECAST_HORIZON} Days)")
print("=" * 80)
print(forecast_df.to_string(index=False))
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Results

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Validation Performance
val_dates = val_target.time_index
val_actual = val_target.values().flatten()
val_pred_median = val_forecast.quantile(0.5).values().flatten()
val_pred_lower = val_forecast.quantile(0.1).values().flatten()
val_pred_upper = val_forecast.quantile(0.9).values().flatten()

ax1.plot(val_dates, val_actual, label='Actual', color='black', linewidth=2)
ax1.plot(val_dates, val_pred_median, label='Forecast (Median)', color='blue', linewidth=2)
ax1.fill_between(val_dates, val_pred_lower, val_pred_upper,
                  alpha=0.3, color='blue', label='10th-90th Percentile')
ax1.set_title(f'{COMMODITY} Price - Validation Set Performance', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Price (USD)', fontsize=12)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Future Forecast
forecast_dates = forecast_df['date']
ax2.plot(forecast_dates, forecast_df['forecast_median'],
         label='Forecast (Median)', color='green', linewidth=2, marker='o')
ax2.fill_between(forecast_dates, forecast_df['forecast_lower_10'],
                  forecast_df['forecast_upper_90'],
                  alpha=0.3, color='green', label='10th-90th Percentile')
ax2.set_title(f'{COMMODITY} Price - {FORECAST_HORIZON}-Day Forecast', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Price (USD)', fontsize=12)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print(f"\n✓ Visualization complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 80)
print("DARTS TFT Experiment Summary")
print("=" * 80)
print(f"Commodity: {COMMODITY}")
print(f"Training samples: {len(train_target)}")
print(f"Validation samples: {len(val_target)}")
print(f"\nModel Performance (Validation):")
print(f"  MAPE: {mape_score:.2f}%")
print(f"  RMSE: ${rmse_score:.4f}")
print(f"  MAE: ${mae_score:.4f}")
print(f"\nForecast Summary:")
print(f"  Horizon: {FORECAST_HORIZON} days")
print(f"  Median forecast: ${forecast_df['forecast_median'].mean():.2f} (avg)")
print(f"  Range: ${forecast_df['forecast_median'].min():.2f} - ${forecast_df['forecast_median'].max():.2f}")
print("=" * 80)
print("\n✓ Experiment completed successfully!")

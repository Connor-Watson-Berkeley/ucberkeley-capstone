# Databricks notebook source
"""
LSTM Coffee Price Forecasting Experiment
Designed for Databricks ML cluster with TensorFlow
"""

# COMMAND ----------

# MAGIC %pip install tensorflow scikit-learn

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import timedelta

print("Packages imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Data from unified_data

# COMMAND ----------

print("="*80)
print("LSTM Coffee Price Forecasting Experiment")
print("="*80)
print()

# Load data from unified_data (NOT bronze tables!)
commodity = 'Coffee'
train_end = '2023-12-31'
test_end = '2024-01-14'

query = f"""
SELECT
    date,
    close,
    temp_mean_c,
    humidity_mean_pct,
    precipitation_mm
FROM commodity.silver.unified_data
WHERE commodity = '{commodity}'
AND date <= '{test_end}'
ORDER BY date
"""

df_spark = spark.sql(query)
df = df_spark.toPandas()
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df.sort_index()

print(f"Loaded {len(df)} days of data")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Features: {list(df.columns)}")
print()
print("First 5 rows:")
print(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Split Train/Test

# COMMAND ----------

train_end_dt = pd.to_datetime(train_end)
df_train = df[df.index <= train_end_dt]
df_test = df[(df.index > train_end_dt) & (df.index <= test_end)]

print(f"Training: {len(df_train)} days ({df_train.index[0]} to {df_train.index[-1]})")
print(f"Test: {len(df_test)} days ({df_test.index[0]} to {df_test.index[-1]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Normalize Features

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit on training data only
feature_cols = ['close', 'temp_mean_c', 'humidity_mean_pct', 'precipitation_mm']
df_train_scaled = df_train.copy()
df_train_scaled[feature_cols] = scaler_X.fit_transform(df_train[feature_cols])

# Scale target separately for inverse transform later
scaler_y.fit(df_train[['close']])

print("Feature scaling complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create Sequences for LSTM

# COMMAND ----------

def create_sequences(data, lookback=60, horizon=14):
    """
    Create input sequences and targets for LSTM training.

    Args:
        data: DataFrame with features
        lookback: Number of days to look back (sequence length)
        horizon: Number of days to forecast ahead

    Returns:
        X: Input sequences (samples, lookback, features)
        y: Target values (samples, horizon)
    """
    X, y = [], []

    feature_data = data[feature_cols].values
    target_data = data['close'].values

    for i in range(lookback, len(data) - horizon + 1):
        X.append(feature_data[i-lookback:i])
        y.append(target_data[i:i+horizon])

    return np.array(X), np.array(y)


lookback = 60
horizon = 14

X_train, y_train = create_sequences(df_train_scaled, lookback=lookback, horizon=horizon)

print(f"X_train shape: {X_train.shape} (samples, lookback, features)")
print(f"y_train shape: {y_train.shape} (samples, horizon)")
print(f"Training samples: {len(X_train)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Build and Train LSTM Model

# COMMAND ----------

from tensorflow import keras
from tensorflow.keras import layers

n_features = X_train.shape[2]

model = keras.Sequential([
    layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(lookback, n_features)),
    layers.Dropout(0.2),
    layers.LSTM(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(horizon)  # Output: 14-day forecast
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print(model.summary())

# COMMAND ----------

# Train the model
print("Training LSTM model...")
print()

epochs = 50
batch_size = 32

history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    verbose=1
)

print()
print("Training complete!")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Generate Forecast

# COMMAND ----------

# Get last 60 days from training data for forecast
forecast_input = df_train.iloc[-lookback:][feature_cols].copy()
forecast_input_scaled = scaler_X.transform(forecast_input)
X_forecast = forecast_input_scaled.reshape(1, lookback, n_features)

# Predict
y_pred_scaled = model.predict(X_forecast, verbose=0)[0]

# Inverse transform to get actual prices
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Create forecast dates
forecast_start = df_train.index[-1] + timedelta(days=1)
forecast_dates = pd.date_range(start=forecast_start, periods=horizon, freq='D')

forecast_df = pd.DataFrame({
    'date': forecast_dates,
    'forecast': y_pred
})

print("14-Day Coffee Price Forecast:")
print(forecast_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Evaluate Forecast Accuracy

# COMMAND ----------

if len(df_test) > 0:
    # Compare forecast vs actual
    actual_values = df_test['close'].iloc[:horizon].values
    forecast_values = y_pred[:len(actual_values)]

    mae = np.mean(np.abs(actual_values - forecast_values))
    rmse = np.sqrt(np.mean((actual_values - forecast_values)**2))
    mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100

    print("="*80)
    print("FORECAST ACCURACY METRICS")
    print("="*80)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print()

    # Show predictions vs actuals
    comparison_df = pd.DataFrame({
        'date': forecast_dates[:len(actual_values)],
        'forecast': forecast_values,
        'actual': actual_values,
        'error': forecast_values - actual_values,
        'abs_error': np.abs(forecast_values - actual_values),
        'pct_error': ((forecast_values - actual_values) / actual_values) * 100
    })

    print("Forecast vs Actual (all 14 days):")
    print(comparison_df)
    print()
    print("="*80)
    print("LSTM EXPERIMENT COMPLETE!")
    print("="*80)
else:
    print("No test data available for evaluation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Results (Optional)

# COMMAND ----------

import matplotlib.pyplot as plt

if len(df_test) > 0:
    plt.figure(figsize=(12, 6))

    # Plot last 30 days of training data
    plt.plot(df_train.index[-30:], df_train['close'][-30:],
             label='Training Data (last 30 days)', marker='o', color='blue')

    # Plot forecast
    plt.plot(forecast_dates[:len(actual_values)], forecast_values,
             label='LSTM Forecast', marker='s', color='green', linestyle='--')

    # Plot actual test values
    plt.plot(forecast_dates[:len(actual_values)], actual_values,
             label='Actual Values', marker='x', color='red')

    plt.xlabel('Date')
    plt.ylabel('Coffee Price (close)')
    plt.title('LSTM Coffee Price Forecast vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    display(plt.gcf())
    plt.close()

# COMMAND ----------

# Print summary statistics
print("="*80)
print("SUMMARY")
print("="*80)
print(f"Model: LSTM (64→32 units, 2 layers)")
print(f"Lookback: {lookback} days")
print(f"Forecast horizon: {horizon} days")
print(f"Training samples: {len(X_train)}")
print(f"Training period: {df_train.index[0]} to {df_train.index[-1]}")
print(f"Features: close, temp_mean_c, humidity_mean_pct, precipitation_mm")
print()
if 'mae' in locals():
    print(f"Test MAE: {mae:.4f} ({mape:.2f}% MAPE)")
print()
print("Next steps:")
print("1. Experiment with different architectures (more layers, units)")
print("2. Try different lookback periods (30, 90, 120 days)")
print("3. Add more features (VIX, forex, sentiment)")
print("4. Implement walk-forward validation")
print("5. Compare against baseline models (naive, XGBoost, SARIMAX)")
print("="*80)

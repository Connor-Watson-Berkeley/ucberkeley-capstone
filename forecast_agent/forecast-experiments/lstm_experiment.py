"""LSTM Coffee Price Forecasting Experiment.

Simple experiment to test LSTM model performance for 14-day coffee price forecasting.
Designed to run in Databricks with TensorFlow/Keras.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sequences(data, lookback=60, horizon=14):
    """
    Create input sequences and targets for LSTM training.

    Args:
        data: DataFrame with features (close, temp_mean_c, humidity_mean_pct, precipitation_mm)
        lookback: Number of days to look back (sequence length)
        horizon: Number of days to forecast ahead

    Returns:
        X: Input sequences (samples, lookback, features)
        y: Target values (samples, horizon)
    """
    X, y = [], []

    # Features: close, temp_mean_c, humidity_mean_pct, precipitation_mm
    feature_cols = ['close', 'temp_mean_c', 'humidity_mean_pct', 'precipitation_mm']
    feature_data = data[feature_cols].values
    target_data = data['close'].values

    for i in range(lookback, len(data) - horizon + 1):
        X.append(feature_data[i-lookback:i])
        y.append(target_data[i:i+horizon])

    return np.array(X), np.array(y)


def build_lstm_model(lookback, n_features, horizon):
    """
    Build LSTM model architecture.

    Args:
        lookback: Sequence length (days)
        n_features: Number of input features
        horizon: Forecast horizon (days)

    Returns:
        Compiled Keras model
    """
    from tensorflow import keras
    from tensorflow.keras import layers

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

    return model


def run_lstm_experiment(commodity='Coffee', train_end='2023-12-31', test_end='2024-01-14',
                        lookback=60, horizon=14, epochs=50, batch_size=32):
    """
    Run LSTM forecasting experiment.

    Args:
        commodity: 'Coffee' or 'Sugar'
        train_end: End date for training data
        test_end: End date for test forecast
        lookback: Sequence length (days)
        horizon: Forecast horizon (days)
        epochs: Training epochs
        batch_size: Training batch size

    Returns:
        Dictionary with model, forecast, and metrics
    """
    print("="*80)
    print(f"LSTM Coffee Price Forecasting Experiment")
    print("="*80)
    print(f"Commodity: {commodity}")
    print(f"Training end: {train_end}")
    print(f"Test period: {train_end} to {test_end}")
    print(f"Lookback: {lookback} days")
    print(f"Horizon: {horizon} days")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print()

    # Step 1: Load data from unified_data
    print("Step 1: Loading data from commodity.silver.unified_data...")
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    # Query unified_data (NOT bronze tables!)
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

    print(f"  Loaded {len(df)} days of data")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Features: {list(df.columns)}")
    print()

    # Step 2: Split train/test
    print("Step 2: Splitting train/test...")
    train_end_dt = pd.to_datetime(train_end)
    df_train = df[df.index <= train_end_dt]
    df_test = df[(df.index > train_end_dt) & (df.index <= test_end)]

    print(f"  Training: {len(df_train)} days ({df_train.index[0]} to {df_train.index[-1]})")
    print(f"  Test: {len(df_test)} days ({df_test.index[0]} to {df_test.index[-1]})")
    print()

    # Step 3: Normalize features
    print("Step 3: Normalizing features...")
    from sklearn.preprocessing import StandardScaler

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fit on training data only
    feature_cols = ['close', 'temp_mean_c', 'humidity_mean_pct', 'precipitation_mm']
    df_train_scaled = df_train.copy()
    df_train_scaled[feature_cols] = scaler_X.fit_transform(df_train[feature_cols])

    # Scale target separately for inverse transform later
    scaler_y.fit(df_train[['close']])

    print(f"  Feature scaling fitted on training data")
    print()

    # Step 4: Create sequences
    print("Step 4: Creating input sequences...")
    X_train, y_train = create_sequences(df_train_scaled, lookback=lookback, horizon=horizon)

    print(f"  X_train shape: {X_train.shape} (samples, lookback, features)")
    print(f"  y_train shape: {y_train.shape} (samples, horizon)")
    print(f"  Training samples: {len(X_train)}")
    print()

    # Step 5: Build and train model
    print("Step 5: Building and training LSTM model...")
    n_features = X_train.shape[2]

    model = build_lstm_model(lookback=lookback, n_features=n_features, horizon=horizon)

    print(model.summary())
    print()

    print("Training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )

    print()
    print(f"  Training complete!")
    print(f"  Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print()

    # Step 6: Generate forecast
    print("Step 6: Generating forecast...")

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

    print(forecast_df)
    print()

    # Step 7: Evaluate if test data available
    if len(df_test) > 0:
        print("Step 7: Evaluating forecast accuracy...")

        # Compare forecast vs actual
        actual_values = df_test['close'].iloc[:horizon].values
        forecast_values = y_pred[:len(actual_values)]

        mae = np.mean(np.abs(actual_values - forecast_values))
        rmse = np.sqrt(np.mean((actual_values - forecast_values)**2))
        mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100

        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print()

        # Show first few predictions vs actuals
        comparison_df = pd.DataFrame({
            'date': forecast_dates[:len(actual_values)],
            'forecast': forecast_values,
            'actual': actual_values,
            'error': forecast_values - actual_values
        })

        print("Forecast vs Actual (first 5 days):")
        print(comparison_df.head())
        print()

        return {
            'model': model,
            'forecast': forecast_df,
            'metrics': {'mae': mae, 'rmse': rmse, 'mape': mape},
            'comparison': comparison_df,
            'history': history
        }
    else:
        return {
            'model': model,
            'forecast': forecast_df,
            'history': history
        }


if __name__ == '__main__':
    # Run experiment
    results = run_lstm_experiment(
        commodity='Coffee',
        train_end='2023-12-31',
        test_end='2024-01-14',
        lookback=60,
        horizon=14,
        epochs=50,
        batch_size=32
    )

    print("="*80)
    print("LSTM Experiment Complete!")
    print("="*80)

    if 'metrics' in results:
        print(f"MAE: {results['metrics']['mae']:.4f}")
        print(f"RMSE: {results['metrics']['rmse']:.4f}")
        print(f"MAPE: {results['metrics']['mape']:.2f}%")

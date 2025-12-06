"""
Quick TFT debug test to identify the JSON serialization issue.
"""

import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from darts import TimeSeries
from darts.models import TFTModel

# Load small dataset
print("Loading data...")
df = pd.read_parquet('data/unified_data_with_forex.parquet')
df_bahia = df[df['region'] == 'Bahia_Brazil'].sort_values('date').reset_index(drop=True)

# Use only first 500 days for quick test
df_bahia = df_bahia.head(500)

# Prepare minimal features
weather_features = ['temp_mean_c', 'precipitation_mm']
df_bahia[weather_features] = df_bahia[weather_features].fillna(method='ffill').fillna(0)

# Create TimeSeries
target = TimeSeries.from_dataframe(
    df_bahia,
    time_col='date',
    value_cols='close',
    freq='D',
    fill_missing_dates=True
)

covariates = TimeSeries.from_dataframe(
    df_bahia,
    time_col='date',
    value_cols=weather_features,
    freq='D',
    fill_missing_dates=True
)

train_size = int(len(target) * 0.8)
train_target = target[:train_size]

print(f"Data ready: {len(target)} days, train: {train_size}")

# Try creating TFT model
print("\nCreating TFT model...")
try:
    model = TFTModel(
        input_chunk_length=30,
        output_chunk_length=1,
        hidden_size=32,
        lstm_layers=1,
        num_attention_heads=2,
        n_epochs=5,  # Just 5 epochs for testing
        batch_size=16,
        add_relative_index=True,  # ← FIX: Auto-generate future covariates from time index
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "enable_progress_bar": True,
            "enable_model_summary": False
        },
        random_state=42,
        force_reset=True,
        save_checkpoints=False
    )
    print("✓ Model created successfully")

    # Try training
    print("\nTraining model...")
    model.fit(
        series=train_target,
        past_covariates=covariates[:train_size]
    )
    print("✓ Training completed successfully")

    # Try prediction
    print("\nMaking prediction...")
    forecast = model.predict(
        n=1,
        series=train_target,
        past_covariates=covariates[:train_size]
    )
    print("✓ Prediction successful")
    print(f"\nForecast value: {forecast.values()[0][0]:.2f}")

    print("\n" + "="*50)
    print("SUCCESS! TFT works - no JSON issues found")
    print("="*50)

except Exception as e:
    print(f"\n{'='*50}")
    print(f"ERROR: {type(e).__name__}")
    print(f"{'='*50}")
    print(f"{e}")
    import traceback
    traceback.print_exc()

#!/usr/bin/env python3
"""
SIMPLE 5-STEP TRAINING WORKFLOW
================================

This script implements the core training workflow with maximum clarity:

STEP 1: Read model configs
STEP 2: Load training data
STEP 3: Fit models
STEP 4: Save fitted models
STEP 5: Use fitted models for inference (placeholder)

NO Spark complexity - just simple SQL queries and model training.
"""

# ============================================================================
# INSTALL GROUND_TRUTH PACKAGE (Databricks only)
# ============================================================================

try:
    # In Databricks, install ground_truth package from DBFS
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                          "/dbfs/FileStore/packages/ground_truth-0.1.0-py3-none-any.whl",
                          "--quiet"])
    print("✓ Installed ground_truth package from DBFS")
except Exception as e:
    print(f"Note: Could not install from DBFS (may already be installed or running locally): {e}")

# ============================================================================
# IMPORTS
# ============================================================================

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

# Model imports
from ground_truth.models.naive import naive_forecast_with_metadata
from ground_truth.models.xgboost_model import xgboost_forecast_with_metadata
from ground_truth.models.sarimax import sarimax_forecast_with_metadata

# NOTE: In Databricks notebooks, we use the built-in 'spark' session instead of databricks.sql connector
# NOTE: Weather and sentiment features already in unified_data table - no feature engineering needed

# ============================================================================
# STEP 1: READ MODEL CONFIGS
# ============================================================================

print("="*80)
print("STEP 1: Read Model Configs")
print("="*80)

# Model configurations (from ground_truth/config/model_registry.py)
BASELINE_MODELS = {
    'naive': {
        'name': 'naive',
        'function': naive_forecast_with_metadata,
        'params': {
            'target': 'close',
            'horizon': 14
        }
    },
    'xgboost': {
        'name': 'xgboost',
        'function': xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_mean_c', 'humidity_mean_pct', 'precipitation_mm', 'vix'],
            'horizon': 14
        }
    },
    'sarimax_auto_weather': {
        'name': 'sarimax_auto_weather',
        'function': sarimax_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_mean_c', 'humidity_mean_pct', 'precipitation_mm'],
            'horizon': 14,
            'auto_arima': True
        }
    }
}

# Training configuration
COMMODITIES = ['Coffee', 'Sugar']
MODEL_KEYS = ['naive', 'xgboost', 'sarimax_auto_weather']
TRAIN_FREQUENCY = 'semiannually'  # Train every 6 months
START_DATE = '2018-01-01'
END_DATE = '2025-11-17'
MODEL_VERSION = 'v1.0'

print(f"\nModels to train: {MODEL_KEYS}")
print(f"Commodities: {COMMODITIES}")
print(f"Training frequency: {TRAIN_FREQUENCY}")
print(f"Date range: {START_DATE} to {END_DATE}")

# ============================================================================
# STEP 2: LOAD TRAINING DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Load Training Data")
print("="*80)

def load_training_data(commodity: str, cutoff_date: str, lookback_days: int = 1460):
    """
    Load training data from unified_data table using Spark SQL

    Args:
        commodity: Commodity name (e.g., 'Coffee')
        cutoff_date: Training cutoff date (YYYY-MM-DD)
        lookback_days: Days of history to load (default 1460 = 4 years)

    Returns:
        pd.DataFrame with training data
    """
    print(f"\nLoading data for {commodity} up to {cutoff_date}...")

    # Calculate lookback date
    cutoff_dt = pd.to_datetime(cutoff_date)
    lookback_dt = cutoff_dt - timedelta(days=lookback_days)
    lookback_str = lookback_dt.strftime('%Y-%m-%d')

    # Simple SQL query
    query = f"""
        SELECT
            date,
            commodity,
            region,
            close,
            temp_mean_c,
            humidity_mean_pct,
            precipitation_mm,
            vix
        FROM commodity.silver.unified_data
        WHERE commodity = '{commodity}'
          AND date >= '{lookback_str}'
          AND date <= '{cutoff_date}'
        ORDER BY date
    """

    # Execute query using Spark SQL (built-in 'spark' session available in Databricks)
    df = spark.sql(query).toPandas()

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"  Loaded {len(df)} rows ({df['date'].min()} to {df['date'].max()})")

    return df

def prepare_features(df: pd.DataFrame, model_config: dict):
    """
    Prepare features for model training

    Args:
        df: Raw training data
        model_config: Model configuration dict

    Returns:
        pd.DataFrame with engineered features
    """
    # Set date as index
    df = df.set_index('date')

    # Extract exog_features if specified
    exog_features = model_config['params'].get('exog_features', [])

    if not exog_features:
        # Naive model - no features needed
        return df[['close']]

    # Select target + exog features
    required_cols = ['close'] + exog_features
    return df[required_cols]

# ============================================================================
# STEP 3: FIT MODELS
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Fit Models")
print("="*80)

def generate_training_dates(start_date: str, end_date: str, frequency: str = 'semiannually'):
    """Generate training window dates based on frequency"""
    dates = []
    current = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    if frequency == 'semiannually':
        months_increment = 6
    elif frequency == 'monthly':
        months_increment = 1
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        # Add months
        month = current.month + months_increment
        year = current.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        current = current.replace(year=year, month=month)

    return dates

def train_model(commodity: str, model_key: str, cutoff_date: str):
    """
    Train a single model

    Args:
        commodity: Commodity name
        model_key: Model key (e.g., 'naive', 'xgboost')
        cutoff_date: Training cutoff date

    Returns:
        dict with fitted model and metadata
    """
    model_config = BASELINE_MODELS[model_key]
    model_name = model_config['name']

    print(f"\n  Training {model_name} for {commodity} (cutoff: {cutoff_date})...")

    # Load training data
    raw_df = load_training_data(commodity, cutoff_date)

    # Prepare features
    training_df = prepare_features(raw_df, model_config)

    # Train model
    model_function = model_config['function']
    params = model_config['params']

    result = model_function(training_df, commodity, fitted_model=None, **params)

    fitted_model = result['fitted_model']

    print(f"    ✓ Model trained successfully")

    return {
        'commodity': commodity,
        'model_name': model_name,
        'training_window_end': cutoff_date,
        'fitted_model': fitted_model,
        'model_version': MODEL_VERSION
    }

# ============================================================================
# STEP 4: SAVE FITTED MODELS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Save Fitted Models")
print("="*80)

def save_fitted_model(model_data: dict):
    """
    Save fitted model to trained_models table using Spark SQL

    Args:
        model_data: Dict with model data and fitted model
    """
    commodity = model_data['commodity']
    model_name = model_data['model_name']
    training_window_end = model_data['training_window_end']
    fitted_model = model_data['fitted_model']
    model_version = model_data['model_version']

    print(f"\n  Saving {model_name} for {commodity} (cutoff: {training_window_end})...")

    # Serialize fitted model to JSON
    # Extract model type
    model_type = fitted_model.get('model_type', 'unknown')

    # Simple JSON serialization
    fitted_model_json = json.dumps({
        'model_type': model_type,
        'fitted_model': str(fitted_model.get('fitted_model', '')),
        'last_date': str(fitted_model.get('last_date', '')),
        'target': fitted_model.get('target', 'close')
    })

    # Escape single quotes in JSON for SQL
    fitted_model_json_escaped = fitted_model_json.replace("'", "''")

    # Calculate year/month for partitioning
    training_dt = pd.to_datetime(training_window_end)
    year = training_dt.year
    month = training_dt.month

    # Insert into trained_models table using Spark SQL
    insert_sql = f"""
        INSERT INTO commodity.forecast.trained_models
        (commodity, model_name, model_version, training_cutoff_date,
         year, month, fitted_model_json, created_at)
        VALUES (
            '{commodity}',
            '{model_name}',
            '{model_version}',
            '{training_window_end}',
            {year},
            {month},
            '{fitted_model_json_escaped}',
            CURRENT_TIMESTAMP()
        )
    """

    # Execute using Spark SQL (built-in 'spark' session available in Databricks)
    spark.sql(insert_sql)

    print(f"    ✓ Model saved to trained_models table")

# ============================================================================
# STEP 5: USE FITTED MODELS FOR INFERENCE (Placeholder)
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Use Fitted Models for Inference")
print("="*80)
print("\nThis step will be implemented in the backfill workflow.")
print("For now, we focus on training and persisting models.")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training workflow"""
    print("\n" + "="*80)
    print("STARTING TRAINING WORKFLOW")
    print("="*80)

    # Generate training dates
    training_dates = generate_training_dates(START_DATE, END_DATE, TRAIN_FREQUENCY)
    print(f"\nTraining windows: {len(training_dates)}")
    print(f"First: {training_dates[0]}, Last: {training_dates[-1]}")

    total_models = len(COMMODITIES) * len(MODEL_KEYS) * len(training_dates)
    print(f"\nTotal models to train: {total_models}")
    print(f"  {len(COMMODITIES)} commodities × {len(MODEL_KEYS)} models × {len(training_dates)} windows")

    # Train all models
    trained_count = 0

    for commodity in COMMODITIES:
        print(f"\n{'='*80}")
        print(f"COMMODITY: {commodity}")
        print(f"{'='*80}")

        for cutoff_date in training_dates:
            print(f"\nTraining Window: {cutoff_date}")

            for model_key in MODEL_KEYS:
                try:
                    # Train model
                    model_data = train_model(commodity, model_key, cutoff_date)

                    # Save model
                    save_fitted_model(model_data)

                    trained_count += 1
                    print(f"\n  Progress: {trained_count}/{total_models} models trained")

                except Exception as e:
                    print(f"\n  ❌ Error training {model_key}: {str(e)}")
                    continue

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nSuccessfully trained {trained_count}/{total_models} models")
    print(f"\nModels saved to: commodity.forecast.trained_models")
    print(f"Next step: Run backfill_rolling_window.py for inference")

if __name__ == '__main__':
    main()

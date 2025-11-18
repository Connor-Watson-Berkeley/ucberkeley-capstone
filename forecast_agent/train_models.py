"""
Train Forecasting Models and Persist to Database

Phase 1 of train-once/inference-many architecture.
Trains N models on training windows and saves fitted models to commodity.forecast.trained_models.

Usage:
    # Train all models semiannually for Coffee
    python train_models.py --commodity Coffee --models naive random_walk arima_111 sarimax_auto_weather xgboost --train-frequency semiannually

    # Train specific model monthly
    python train_models.py --commodity Coffee --models naive --train-frequency monthly

    # Train with custom date range
    python train_models.py --commodity Coffee --models xgboost --train-frequency quarterly --start-date 2020-01-01 --end-date 2024-01-01
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from databricks import sql
from ground_truth.config.model_registry import BASELINE_MODELS
from utils.model_persistence import save_model, model_exists
from ground_truth.models import naive, random_walk, arima, sarimax, xgboost_model, prophet_model


# Mapping from model registry keys to training functions
MODEL_TRAIN_FUNCTIONS = {
    'naive': naive.naive_train,
    'random_walk': random_walk.random_walk_train,
    'arima_111': arima.arima_train,
    'sarimax_auto': sarimax.sarimax_train,
    'sarimax_auto_weather': sarimax.sarimax_train,
    'sarimax_auto_weather_seasonal': sarimax.sarimax_train,
    'xgboost': xgboost_model.xgboost_train,
    'prophet': prophet_model.prophet_train,
    # Add more models as they are refactored
}


def get_training_dates(
    start_date: date,
    end_date: date,
    frequency: str
) -> List[date]:
    """
    Generate training dates based on frequency.

    Args:
        start_date: First possible training date
        end_date: Last possible training date
        frequency: 'daily', 'weekly', 'monthly', 'quarterly', 'semiannually', 'annually'

    Returns:
        List of dates to train models on
    """
    training_dates = []
    current = start_date

    frequency_map = {
        'daily': timedelta(days=1),
        'weekly': timedelta(days=7),
        'biweekly': timedelta(days=14),
        'monthly': relativedelta(months=1),
        'quarterly': relativedelta(months=3),
        'semiannually': relativedelta(months=6),
        'annually': relativedelta(years=1)
    }

    if frequency not in frequency_map:
        raise ValueError(f"Unknown frequency: {frequency}. Choose from {list(frequency_map.keys())}")

    delta = frequency_map[frequency]

    while current <= end_date:
        training_dates.append(current)

        if isinstance(delta, timedelta):
            current = current + delta
        else:  # relativedelta
            current = current + delta

    return training_dates


def load_training_data(connection, commodity: str, cutoff_date: date) -> pd.DataFrame:
    """Load all data up to cutoff_date for training from unified_data table."""
    cursor = connection.cursor()

    # Load from commodity.silver.unified_data (includes weather, fx rates, etc.)
    query = f"""
        SELECT
            date,
            close,
            open,
            high,
            low,
            volume,
            temp_mean_c,
            humidity_mean_pct,
            precipitation_mm,
            vix,
            cop_usd
        FROM commodity.silver.unified_data
        WHERE commodity = '{commodity}'
          AND date <= '{cutoff_date}'
        ORDER BY date
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    df = pd.DataFrame(rows, columns=columns)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    cursor.close()
    return df


def train_and_save_model(
    connection,
    training_df: pd.DataFrame,
    model_key: str,
    model_config: Dict,
    commodity: str,
    model_version: str = "v1.0",
    created_by: str = "train_models.py"
) -> Optional[str]:
    """
    Train a single model and save to database.

    Args:
        connection: Databricks SQL connection
        training_df: Training data (pandas DataFrame with DatetimeIndex)
        model_key: Model key from registry (e.g., 'naive', 'sarimax_auto_weather')
        model_config: Model configuration from registry
        commodity: 'Coffee' or 'Sugar'
        model_version: Version string (e.g., 'v1.0')
        created_by: User/process name

    Returns:
        model_id if successful, None if failed
    """
    model_name = model_config['name']
    model_params = model_config['params'].copy()
    train_fn = MODEL_TRAIN_FUNCTIONS.get(model_key)

    if train_fn is None:
        print(f"     ⚠️  Model '{model_key}' does not have train/predict separation yet - skipping")
        return None

    # Extract training metadata
    training_date = training_df.index[-1].strftime('%Y-%m-%d')
    training_start_date = training_df.index[0].strftime('%Y-%m-%d')
    training_samples = len(training_df)

    # Check if model already exists
    if model_exists(connection, commodity, model_name, training_date, model_version):
        print(f"     ⏩ Model already exists - skipping")
        return None

    try:
        # Train model
        print(f"     Training {model_name} on {training_samples} days...")

        # Extract target and features from params
        target = model_params.get('target', 'close')

        # Call the appropriate training function based on model type
        if model_key == 'naive':
            fitted_model_dict = train_fn(training_df, target=target)
        elif model_key == 'random_walk':
            lookback_days = model_params.get('lookback_days', 30)
            fitted_model_dict = train_fn(training_df, target=target, lookback_days=lookback_days)
        elif model_key == 'arima_111':
            order = model_params.get('order', (1, 1, 1))
            fitted_model_dict = train_fn(training_df, target=target, order=order)
        elif model_key in ['sarimax_auto', 'sarimax_auto_weather', 'sarimax_auto_weather_seasonal']:
            exog_features = model_params.get('exog_features', None)
            order = model_params.get('order', None)
            seasonal_order = model_params.get('seasonal_order', (0, 0, 0, 0))
            covariate_projection_method = model_params.get('covariate_projection_method', 'persist')
            fitted_model_dict = train_fn(
                training_df,
                target=target,
                exog_features=exog_features,
                order=order,
                seasonal_order=seasonal_order,
                covariate_projection_method=covariate_projection_method
            )
        elif model_key == 'prophet':
            exog_features = model_params.get('exog_features', None)
            weekly_seasonality = model_params.get('weekly_seasonality', True)
            yearly_seasonality = model_params.get('yearly_seasonality', True)
            fitted_model_dict = train_fn(
                training_df,
                target=target,
                exog_features=exog_features,
                weekly_seasonality=weekly_seasonality,
                yearly_seasonality=yearly_seasonality
            )
        elif model_key == 'xgboost':
            exog_features = model_params.get('exog_features', None)
            lags = model_params.get('lags', [1, 7, 14])
            windows = model_params.get('windows', [7, 30])
            fitted_model_dict = train_fn(
                training_df,
                target=target,
                exog_features=exog_features,
                lags=lags,
                windows=windows
            )
        else:
            print(f"     ❌ Unknown model type: {model_key}")
            return None

        # Save fitted model to database
        print(f"     Saving model to database...")
        model_id = save_model(
            connection=connection,
            fitted_model=fitted_model_dict,
            commodity=commodity,
            model_name=model_name,
            model_version=model_version,
            training_date=training_date,
            training_samples=training_samples,
            training_start_date=training_start_date,
            parameters=model_params,
            created_by=created_by,
            is_active=True
        )

        print(f"     ✅ Saved as {model_id}")
        return model_id

    except Exception as e:
        print(f"     ❌ Error training {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Train forecasting models and persist to database')
    parser.add_argument('--commodity', type=str, required=True, choices=['Coffee', 'Sugar'],
                        help='Commodity to train models for')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='Model keys from registry (e.g., naive random_walk arima_111 sarimax_auto_weather xgboost)')
    parser.add_argument('--train-frequency', type=str, required=True,
                        choices=['daily', 'weekly', 'biweekly', 'monthly', 'quarterly', 'semiannually', 'annually'],
                        help='How often to train models')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date for training windows (YYYY-MM-DD). Default: 3 years before most recent data')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for training windows (YYYY-MM-DD). Default: most recent date in data')
    parser.add_argument('--model-version', type=str, default='v1.0',
                        help='Model version tag (default: v1.0)')
    parser.add_argument('--min-training-days', type=int, default=1095,
                        help='Minimum days of training data required (default: 1095 = 3 years)')

    args = parser.parse_args()

    # Load credentials from environment
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
    DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")

    if not all([DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH]):
        print("ERROR: Missing Databricks credentials in environment variables")
        print("Set DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH")
        sys.exit(1)

    # Validate models
    for model_key in args.models:
        if model_key not in BASELINE_MODELS:
            print(f"ERROR: Unknown model '{model_key}'")
            print(f"Available models: {list(BASELINE_MODELS.keys())}")
            sys.exit(1)

    print("=" * 80)
    print("TRAIN MODELS - Phase 1 of Train-Once/Inference-Many")
    print("=" * 80)
    print(f"Commodity: {args.commodity}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Training Frequency: {args.train_frequency}")
    print(f"Model Version: {args.model_version}")
    print(f"Min Training Days: {args.min_training_days}")
    print("=" * 80)

    # Connect to Databricks
    print("\n📡 Connecting to Databricks...")
    connection = sql.connect(
        server_hostname=DATABRICKS_HOST.replace('https://', ''),
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )
    print("✅ Connected")

    # Load full dataset to determine date range
    print(f"\n📊 Loading {args.commodity} data from unified_data...")
    cursor = connection.cursor()
    cursor.execute(f"""
        SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as row_count
        FROM commodity.silver.unified_data
        WHERE commodity = '{args.commodity}'
    """)
    min_date, max_date, row_count = cursor.fetchone()
    cursor.close()

    print(f"   Data range: {min_date} to {max_date} ({row_count:,} days)")

    # Determine training window range
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    else:
        # Default: Start 3 years before most recent data + min_training_days
        data_end = datetime.strptime(str(max_date), '%Y-%m-%d').date()
        start_date = data_end - timedelta(days=1095) + timedelta(days=args.min_training_days)

    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    else:
        end_date = datetime.strptime(str(max_date), '%Y-%m-%d').date()

    # Generate training dates
    training_dates = get_training_dates(start_date, end_date, args.train_frequency)
    print(f"\n📅 Training Windows: {len(training_dates)} windows from {training_dates[0]} to {training_dates[-1]}")

    # Training loop
    total_trained = 0
    total_skipped = 0
    total_failed = 0

    for window_idx, training_cutoff in enumerate(training_dates, 1):
        print(f"\n{'='*80}")
        print(f"Window {window_idx}/{len(training_dates)}: Training Cutoff = {training_cutoff}")
        print(f"{'='*80}")

        # Load training data up to this cutoff
        training_df = load_training_data(connection, args.commodity, training_cutoff)

        # Check minimum training days
        if len(training_df) < args.min_training_days:
            print(f"   ⚠️  Insufficient training data: {len(training_df)} days < {args.min_training_days} days - skipping")
            total_skipped += len(args.models)
            continue

        print(f"   📊 Loaded {len(training_df):,} days of training data")
        print(f"   📅 Data range: {training_df.index[0].date()} to {training_df.index[-1].date()}")

        # Train each model
        for model_key in args.models:
            model_config = BASELINE_MODELS[model_key]
            model_name = model_config['name']

            print(f"\n   🔧 {model_name} ({model_key}):")

            model_id = train_and_save_model(
                connection=connection,
                training_df=training_df,
                model_key=model_key,
                model_config=model_config,
                commodity=args.commodity,
                model_version=args.model_version,
                created_by="train_models.py"
            )

            if model_id:
                total_trained += 1
            elif model_id is None and MODEL_TRAIN_FUNCTIONS.get(model_key) is not None:
                total_failed += 1
            else:
                total_skipped += 1

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"✅ Models Trained: {total_trained}")
    print(f"⏩ Models Skipped (already exist): {total_skipped}")
    print(f"❌ Models Failed: {total_failed}")
    print("=" * 80)

    if total_trained > 0:
        print(f"\n📊 Trained models are now available in commodity.forecast.trained_models")
        print(f"   Use these in backfill_rolling_window.py for fast inference!")

    connection.close()


if __name__ == "__main__":
    main()

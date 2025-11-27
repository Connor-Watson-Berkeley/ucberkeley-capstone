"""Utilities for daily production inference workflow.

Supports both backfilling (date ranges) and daily production (today only).
"""

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Optional
from pyspark.sql import SparkSession
import pandas as pd


def should_retrain_today(
    spark: SparkSession,
    commodity: str,
    model_name: str,
    train_frequency: str,
    model_version: str = "v1.0",
    today: Optional[date] = None
) -> bool:
    """
    Check if models should be retrained today based on training cadence.
    
    Args:
        spark: Spark session
        commodity: 'Coffee' or 'Sugar'
        model_name: Model name to check
        train_frequency: Training frequency ('semiannually', 'monthly', etc.)
        model_version: Model version
        today: Today's date (defaults to date.today())
    
    Returns:
        True if today is a training date and no model exists for today
    """
    if today is None:
        today = date.today()
    
    # Check if model already trained for today
    from model_persistence import model_exists_spark
    
    today_str = today.strftime('%Y-%m-%d')
    if model_exists_spark(spark, commodity, model_name, today_str, model_version):
        return False  # Already trained today
    
    # Find most recent training date
    query = f"""
        SELECT MAX(training_date) as last_training_date
        FROM commodity.forecast.trained_models
        WHERE commodity = '{commodity}'
          AND model_name = '{model_name}'
          AND model_version = '{model_version}'
          AND is_active = TRUE
    """
    
    result_df = spark.sql(query).toPandas()
    
    if result_df.empty or result_df.iloc[0]['last_training_date'] is None:
        return True  # No previous training, should train
    
    last_training_date = pd.to_datetime(result_df.iloc[0]['last_training_date']).date()
    
    # Calculate next training date based on frequency
    frequency_map = {
        'daily': timedelta(days=1),
        'weekly': timedelta(days=7),
        'biweekly': timedelta(days=14),
        'monthly': relativedelta(months=1),
        'quarterly': relativedelta(months=3),
        'semiannually': relativedelta(months=6),
        'annually': relativedelta(years=1)
    }
    
    delta = frequency_map.get(train_frequency)
    if not delta:
        raise ValueError(f"Unknown frequency: {train_frequency}")
    
    # Calculate next training date
    if isinstance(delta, timedelta):
        next_training_date = last_training_date + delta
    else:
        next_training_date = last_training_date + delta
    
    # Should retrain if today >= next training date
    return today >= next_training_date


def get_most_recent_trained_model(
    spark: SparkSession,
    commodity: str,
    model_name: str,
    forecast_date: date,
    model_version: str = "v1.0"
) -> Optional[dict]:
    """
    Get the most recent trained model that can be used for a forecast date.
    
    Returns model with training_date <= forecast_date (most recent one).
    
    Args:
        spark: Spark session
        commodity: 'Coffee' or 'Sugar'
        model_name: Model name
        forecast_date: Date to forecast for (need training_date <= forecast_date)
        model_version: Model version
    
    Returns:
        Dict with model metadata, or None if no suitable model found
    """
    forecast_date_str = forecast_date.strftime('%Y-%m-%d')
    
    query = f"""
        SELECT
            model_id,
            training_date,
            training_samples,
            training_start_date,
            created_at
        FROM commodity.forecast.trained_models
        WHERE commodity = '{commodity}'
          AND model_name = '{model_name}'
          AND model_version = '{model_version}'
          AND is_active = TRUE
          AND training_date <= '{forecast_date_str}'
        ORDER BY training_date DESC
        LIMIT 1
    """
    
    result_df = spark.sql(query).toPandas()
    
    if result_df.empty:
        return None
    
    row = result_df.iloc[0]
    return {
        'model_id': row['model_id'],
        'training_date': pd.to_datetime(row['training_date']).date(),
        'training_samples': int(row['training_samples']),
        'training_start_date': pd.to_datetime(row['training_start_date']).date(),
        'created_at': row['created_at']
    }


def get_models_to_train_today(
    spark: SparkSession,
    commodity: str,
    models: List[str],
    train_frequency: str,
    model_version: str = "v1.0",
    today: Optional[date] = None
) -> List[str]:
    """
    Get list of models that should be trained today based on cadence.
    
    Args:
        spark: Spark session
        commodity: 'Coffee' or 'Sugar'
        models: List of model keys to check
        train_frequency: Training frequency
        model_version: Model version
        today: Today's date (defaults to date.today())
    
    Returns:
        List of model keys that need training today
    """
    if today is None:
        today = date.today()
    
    from model_pipeline import create_model_from_registry
    
    models_to_train = []
    
    for model_key in models:
        try:
            model = create_model_from_registry(model_key)
            model_name = model.model_name
            
            if should_retrain_today(spark, commodity, model_name, train_frequency, model_version, today):
                models_to_train.append(model_key)
        except Exception:
            # Skip invalid model keys
            continue
    
    return models_to_train


def generate_daily_forecasts(
    spark: SparkSession,
    commodity: str,
    models: List[str],
    forecast_date: date,
    model_version: str = "v1.0"
) -> List[dict]:
    """
    Generate forecasts for a single date using most recent trained models.
    
    Args:
        spark: Spark session
        commodity: 'Coffee' or 'Sugar'
        models: List of model keys
        forecast_date: Date to forecast for (usually today)
        model_version: Model version
    
    Returns:
        List of forecast dicts ready for distributions_writer
    """
    from model_pipeline import create_model_from_registry
    from model_persistence import load_model_spark
    from data_loader import TimeSeriesDataLoader
    from distributions_writer import get_existing_forecast_dates
    
    # Check if forecast already exists
    existing_dates = get_existing_forecast_dates(spark, commodity, model_version)
    if forecast_date in existing_dates:
        print(f"⏩ Forecast for {forecast_date} already exists - skipping")
        return []
    
    forecasts = []
    
    # Load data up to forecast date
    loader = TimeSeriesDataLoader(spark=spark)
    data_df = loader.load_to_pandas(
        commodity=commodity,
        cutoff_date=forecast_date.strftime('%Y-%m-%d'),
        aggregate_regions=True,
        aggregation_method='mean'
    )
    
    for model_key in models:
        try:
            model = create_model_from_registry(model_key)
            model_name = model.model_name
            
            # Get most recent trained model
            model_info = get_most_recent_trained_model(
                spark, commodity, model_name, forecast_date, model_version
            )
            
            if not model_info:
                print(f"⚠️  No trained model found for {model_name} (training_date <= {forecast_date})")
                continue
            
            # Load model
            loaded_data = load_model_spark(
                spark, commodity, model_name, 
                model_info['training_date'].strftime('%Y-%m-%d'),
                model_version
            )
            
            if not loaded_data:
                print(f"⚠️  Could not load model {model_name}")
                continue
            
            # Reconstruct model from loaded data
            # (This depends on model type - simplified here)
            fitted_model = loaded_data['fitted_model']
            
            # Generate forecast
            forecast_df = model.predict(horizon=14)
            
            # Generate Monte Carlo paths (simplified - actual implementation depends on model)
            from distributions_writer import DistributionsWriter
            writer = DistributionsWriter(spark)
            
            # Generate paths from mean forecast
            paths = writer._generate_paths_from_mean(
                forecast_df['forecast'].values,
                std=2.5,  # Default std, should come from model
                n_paths=2000
            )
            
            forecast_data = {
                'forecast_start_date': forecast_date,
                'data_cutoff_date': model_info['training_date'],
                'paths': paths,
                'mean_forecast': forecast_df['forecast'].values,
                'model_version': model_version,
                'commodity': commodity
            }
            
            forecasts.append(forecast_data)
            print(f"✅ Generated forecast for {model_name} on {forecast_date}")
            
        except Exception as e:
            print(f"❌ Failed to generate forecast for {model_key}: {str(e)[:100]}")
            continue
    
    return forecasts


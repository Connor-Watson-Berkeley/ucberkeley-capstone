"""
Model Persistence Utilities

Functions to save and load trained forecasting models to/from Databricks.

Storage Strategy:
- Small models (< 1MB): Store as JSON in fitted_model_json column
- Large models (>= 1MB): Serialize to S3, store path in fitted_model_s3_path column
"""

import os
import json
import pickle
import hashlib
from datetime import datetime
from typing import Dict, Optional

# Conditional import: databricks.sql only works for remote connections
# In Databricks notebooks, we use Spark SQL instead
try:
    from databricks import sql
    HAS_DATABRICKS_SQL = True
except ImportError:
    HAS_DATABRICKS_SQL = False
    # Running in Databricks notebook - will use Spark SQL

import boto3


# Globals for Databricks detection
_IS_DATABRICKS = None
_SPARK = None


def is_databricks():
    """Detect if running in Databricks environment."""
    global _IS_DATABRICKS, _SPARK
    if _IS_DATABRICKS is None:
        try:
            from pyspark.sql import SparkSession
            _SPARK = SparkSession.builder.getOrCreate()
            _IS_DATABRICKS = True
        except:
            _IS_DATABRICKS = False
    return _IS_DATABRICKS


def generate_model_id(commodity: str, model_name: str, training_date: str, version: str = "v1.0") -> str:
    """Generate unique model ID."""
    return f"{commodity}_{model_name}_{training_date}_{version}"


def serialize_model(fitted_model: any, model_name: str) -> tuple:
    """
    Serialize model to JSON or pickle.

    Returns:
        (serialized_data, format, size_bytes)
        format: 'json' or 'pickle'
    """
    # Try JSON first (for simple models like naive, random_walk)
    if model_name.lower() in ['naive', 'randomwalk', 'random_walk']:
        try:
            serialized = json.dumps(fitted_model)
            return serialized, 'json', len(serialized.encode('utf-8'))
        except (TypeError, ValueError):
            pass  # Fall back to pickle

    # Use pickle for complex models (XGBoost, Prophet, etc.)
    serialized = pickle.dumps(fitted_model)
    return serialized, 'pickle', len(serialized)


def deserialize_model(data: any, format: str):
    """Deserialize model from JSON or pickle."""
    if format == 'json':
        if isinstance(data, str):
            return json.loads(data)
        return data
    elif format == 'pickle':
        if isinstance(data, bytes):
            return pickle.loads(data)
        return pickle.loads(data.encode('utf-8'))
    else:
        raise ValueError(f"Unknown format: {format}")


def upload_to_s3(data: bytes, s3_path: str) -> bool:
    """Upload serialized model to S3."""
    # Parse S3 path: s3://bucket/key
    if not s3_path.startswith('s3://'):
        raise ValueError(f"Invalid S3 path: {s3_path}")

    parts = s3_path[5:].split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''

    s3_client = boto3.client('s3')
    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=data)
        return True
    except Exception as e:
        print(f"❌ Error uploading to S3: {e}")
        return False


def download_from_s3(s3_path: str) -> Optional[bytes]:
    """Download serialized model from S3."""
    if not s3_path.startswith('s3://'):
        raise ValueError(f"Invalid S3 path: {s3_path}")

    parts = s3_path[5:].split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''

    s3_client = boto3.client('s3')
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()
    except Exception as e:
        print(f"❌ Error downloading from S3: {e}")
        return None


def save_model(
    connection,
    fitted_model: any,
    commodity: str,
    model_name: str,
    model_version: str,
    training_date: str,
    training_samples: int,
    training_start_date: str,
    parameters: Dict,
    created_by: str = "backfill_script",
    is_active: bool = True,
    s3_bucket: str = "groundtruth-capstone"
) -> str:
    """
    Save trained model to Databricks.

    Args:
        connection: Databricks SQL connection
        fitted_model: The fitted model object
        commodity: 'Coffee' or 'Sugar'
        model_name: 'Naive', 'XGBoost', etc.
        model_version: Version string (e.g., 'v1.0', 'backfill_2024')
        training_date: Cutoff date for training (YYYY-MM-DD)
        training_samples: Number of training days
        training_start_date: First date in training data (YYYY-MM-DD)
        parameters: Dict of model parameters
        created_by: User/process name
        is_active: Whether model is active for inference
        s3_bucket: S3 bucket for large models

    Returns:
        model_id of saved model
    """
    cursor = connection.cursor()

    # Generate model ID
    model_id = generate_model_id(commodity, model_name, training_date, model_version)

    # Serialize model
    serialized, format_type, size_bytes = serialize_model(fitted_model, model_name)

    # Determine storage strategy (1MB threshold)
    SIZE_THRESHOLD = 1 * 1024 * 1024  # 1MB

    if size_bytes < SIZE_THRESHOLD:
        # Store inline as JSON
        fitted_model_json = serialized if format_type == 'json' else None
        fitted_model_s3_path = None

        # If pickle, convert to base64 for storage
        if format_type == 'pickle':
            import base64
            fitted_model_json = base64.b64encode(serialized).decode('utf-8')
    else:
        # Store in S3
        s3_key = f"models/{commodity}/{model_name}/{training_date}/{model_id}.pkl"
        s3_path = f"s3://{s3_bucket}/{s3_key}"

        if upload_to_s3(serialized, s3_path):
            fitted_model_json = None
            fitted_model_s3_path = s3_path
        else:
            raise Exception("Failed to upload model to S3")

    # Extract year/month for partitioning
    year = int(training_date.split('-')[0])
    month = int(training_date.split('-')[1])

    # Insert into table
    insert_sql = """
    INSERT INTO commodity.forecast.trained_models (
        model_id, commodity, model_name, model_version,
        training_date, training_samples, training_start_date,
        parameters, fitted_model_json, fitted_model_s3_path,
        model_size_bytes, created_at, created_by, is_active,
        year, month
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    cursor.execute(insert_sql, (
        model_id,
        commodity,
        model_name,
        model_version,
        training_date,
        training_samples,
        training_start_date,
        json.dumps(parameters),
        fitted_model_json,
        fitted_model_s3_path,
        size_bytes,
        datetime.now(),
        created_by,
        is_active,
        year,
        month
    ))

    cursor.close()
    return model_id


def load_model(
    connection,
    commodity: str,
    model_name: str,
    training_date: str,
    model_version: str = "v1.0"
) -> Optional[Dict]:
    """
    Load trained model from Databricks.

    Args:
        connection: Databricks SQL connection (None if running in Databricks)
        commodity: 'Coffee' or 'Sugar'
        model_name: 'Naive', 'XGBoost', etc.
        training_date: Training cutoff date (YYYY-MM-DD)
        model_version: Version string

    Returns:
        Dict with 'fitted_model', 'parameters', 'metadata'
        Returns None if model not found
    """
    model_id = generate_model_id(commodity, model_name, training_date, model_version)

    # Query model
    query_sql = f"""
    SELECT
        fitted_model_json, fitted_model_s3_path, parameters,
        training_samples, training_start_date, model_size_bytes,
        created_at, is_active
    FROM commodity.forecast.trained_models
    WHERE model_id = '{model_id}'
    """

    if is_databricks():
        # Running in Databricks - use Spark SQL
        import pandas as pd
        result_df = _SPARK.sql(query_sql).toPandas()
        if result_df.empty:
            return None
        row = result_df.iloc[0]
        fitted_model_json = row['fitted_model_json']
        fitted_model_s3_path = row['fitted_model_s3_path']
        parameters_json = row['parameters']
        training_samples = row['training_samples']
        training_start_date = row['training_start_date']
        model_size_bytes = row['model_size_bytes']
        created_at = row['created_at']
        is_active = row['is_active']
    else:
        # Running locally - use databricks.sql connection
        cursor = connection.cursor()
        cursor.execute(query_sql.replace("'{model_id}'", "?"), (model_id,))
        row = cursor.fetchone()
        cursor.close()

        if not row:
            return None

        (fitted_model_json, fitted_model_s3_path, parameters_json,
         training_samples, training_start_date, model_size_bytes,
         created_at, is_active) = row

    # Deserialize model
    if fitted_model_json:
        # Model stored inline
        try:
            # Try JSON first
            fitted_model = json.loads(fitted_model_json)
        except (json.JSONDecodeError, ValueError):
            # Try base64-encoded pickle
            import base64
            fitted_model = pickle.loads(base64.b64decode(fitted_model_json))
    elif fitted_model_s3_path:
        # Model stored in S3
        serialized = download_from_s3(fitted_model_s3_path)
        if serialized is None:
            return None
        fitted_model = pickle.loads(serialized)
    else:
        raise ValueError(f"Model {model_id} has no serialized data")

    return {
        'fitted_model': fitted_model,
        'parameters': json.loads(parameters_json),
        'metadata': {
            'model_id': model_id,
            'commodity': commodity,
            'model_name': model_name,
            'training_date': training_date,
            'training_samples': training_samples,
            'training_start_date': training_start_date,
            'model_size_bytes': model_size_bytes,
            'created_at': created_at,
            'is_active': is_active
        }
    }


def model_exists(
    connection,
    commodity: str,
    model_name: str,
    training_date: str,
    model_version: str = "v1.0"
) -> bool:
    """Check if model exists in database."""
    cursor = connection.cursor()

    model_id = generate_model_id(commodity, model_name, training_date, model_version)

    cursor.execute(
        "SELECT COUNT(*) FROM commodity.forecast.trained_models WHERE model_id = ?",
        (model_id,)
    )
    count = cursor.fetchone()[0]
    cursor.close()

    return count > 0

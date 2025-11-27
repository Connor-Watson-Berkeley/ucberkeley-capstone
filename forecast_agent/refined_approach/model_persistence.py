"""Model persistence for Databricks (Spark-based).

Simplified version that works in Databricks notebooks using Spark SQL.
"""

import json
import pickle
import base64
from datetime import datetime
from typing import Dict, Optional
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit


def serialize_model(fitted_model: any, model_name: str) -> tuple:
    """
    Serialize model to JSON or pickle.
    
    Returns:
        (serialized_data, format_type, size_bytes)
    """
    # Try JSON first for simple models
    if model_name.lower() in ['naive', 'randomwalk', 'random_walk']:
        try:
            # Extract serializable data from ModelPipeline
            model_data = {
                'model_type': model_name,
                'params': fitted_model.get_params() if hasattr(fitted_model, 'get_params') else {},
            }
            
            # Add model-specific data
            if hasattr(fitted_model, 'last_value'):
                model_data['last_value'] = float(fitted_model.last_value) if fitted_model.last_value is not None else None
            if hasattr(fitted_model, 'last_date'):
                model_data['last_date'] = str(fitted_model.last_date) if fitted_model.last_date else None
            if hasattr(fitted_model, 'drift'):
                model_data['drift'] = float(fitted_model.drift) if fitted_model.drift is not None else None
            
            serialized = json.dumps(model_data)
            return serialized, 'json', len(serialized.encode('utf-8'))
        except (TypeError, ValueError) as e:
            print(f"JSON serialization failed: {e}, falling back to pickle")
    
    # Use pickle for complex models
    serialized = pickle.dumps(fitted_model)
    return serialized, 'pickle', len(serialized)


def save_model_spark(
    spark: SparkSession,
    fitted_model: any,
    commodity: str,
    model_name: str,
    model_version: str,
    training_date: str,
    training_samples: int,
    training_start_date: str,
    parameters: Dict,
    created_by: str = "notebook",
    is_active: bool = True,
    s3_bucket: str = "groundtruth-capstone"
) -> str:
    """
    Save trained model using Spark SQL (for Databricks notebooks).
    
    Args:
        spark: Spark session
        fitted_model: Fitted model object (ModelPipeline or similar)
        commodity: 'Coffee' or 'Sugar'
        model_name: Model name (e.g., 'Naive', 'XGBoost')
        model_version: Version string (e.g., 'v1.0')
        training_date: Cutoff date (YYYY-MM-DD)
        training_samples: Number of training days
        training_start_date: First date (YYYY-MM-DD)
        parameters: Model parameters dict
        created_by: User/process name
        is_active: Whether model is active
        s3_bucket: S3 bucket for large models
    
    Returns:
        model_id string
    """
    # Generate model ID
    model_id = f"{commodity}_{model_name}_{training_date}_{model_version}"
    
    # Serialize model
    serialized, format_type, size_bytes = serialize_model(fitted_model, model_name)
    
    SIZE_THRESHOLD = 1 * 1024 * 1024  # 1MB
    
    # Determine storage strategy
    if size_bytes < SIZE_THRESHOLD:
        # Store inline as JSON/base64
        if format_type == 'json':
            fitted_model_json = serialized
            fitted_model_s3_path = None
        else:
            # Base64 encode pickle for JSON column
            fitted_model_json = base64.b64encode(serialized).decode('utf-8')
            fitted_model_s3_path = None
    else:
        # Store in S3 (for large models)
        # TODO: Implement S3 upload if needed
        # For now, use base64 for everything
        fitted_model_json = base64.b64encode(serialized).decode('utf-8')
        fitted_model_s3_path = None
        print(f"⚠️  Large model ({size_bytes} bytes) - consider S3 storage")
    
    # Extract year/month
    year = int(training_date.split('-')[0])
    month = int(training_date.split('-')[1])
    
    # Prepare insert using Spark SQL
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, DateType, TimestampType
    
    # Create row data
    row_data = [(
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
    )]
    
    # Create DataFrame
    schema = StructType([
        StructField("model_id", StringType(), False),
        StructField("commodity", StringType(), False),
        StructField("model_name", StringType(), False),
        StructField("model_version", StringType(), False),
        StructField("training_date", StringType(), False),
        StructField("training_samples", IntegerType(), False),
        StructField("training_start_date", StringType(), False),
        StructField("parameters", StringType(), False),
        StructField("fitted_model_json", StringType(), True),
        StructField("fitted_model_s3_path", StringType(), True),
        StructField("model_size_bytes", IntegerType(), False),
        StructField("created_at", TimestampType(), False),
        StructField("created_by", StringType(), False),
        StructField("is_active", BooleanType(), False),
        StructField("year", IntegerType(), False),
        StructField("month", IntegerType(), False)
    ])
    
    df = spark.createDataFrame(row_data, schema)
    
    # Cast training_date columns to DateType
    from pyspark.sql.functions import to_date
    df = df.withColumn("training_date", to_date(col("training_date"))) \
           .withColumn("training_start_date", to_date(col("training_start_date")))
    
    # Write to table (append mode)
    df.write.mode("append").saveAsTable("commodity.forecast.trained_models")
    
    print(f"✅ Saved model: {model_id}")
    return model_id


def model_exists_spark(
    spark: SparkSession,
    commodity: str,
    model_name: str,
    training_date: str,
    model_version: str = "v1.0"
) -> bool:
    """
    Check if model already exists in trained_models table (Spark-based).
    
    Args:
        spark: Spark session
        commodity: 'Coffee' or 'Sugar'
        model_name: Model name
        training_date: Training cutoff date (YYYY-MM-DD)
        model_version: Version string
    
    Returns:
        True if model exists, False otherwise
    """
    model_id = f"{commodity}_{model_name}_{training_date}_{model_version}"
    
    query = f"""
        SELECT COUNT(*) as count
        FROM commodity.forecast.trained_models
        WHERE model_id = '{model_id}'
    """
    
    result_df = spark.sql(query).toPandas()
    return result_df.iloc[0]['count'] > 0


def load_model_spark(
    spark: SparkSession,
    commodity: str,
    model_name: str,
    training_date: str,
    model_version: str = "v1.0"
) -> Optional[Dict]:
    """
    Load trained model from table using Spark SQL.
    
    Args:
        spark: Spark session
        commodity: 'Coffee' or 'Sugar'
        model_name: Model name
        training_date: Training cutoff date (YYYY-MM-DD)
        model_version: Version string
    
    Returns:
        Dict with 'fitted_model', 'parameters', 'metadata'
        Returns None if not found
    """
    model_id = f"{commodity}_{model_name}_{training_date}_{model_version}"
    
    # Query table
    query = f"""
        SELECT
            fitted_model_json,
            fitted_model_s3_path,
            parameters,
            training_samples,
            training_start_date,
            model_size_bytes,
            created_at,
            is_active
        FROM commodity.forecast.trained_models
        WHERE model_id = '{model_id}'
    """
    
    result_df = spark.sql(query).toPandas()
    
    if result_df.empty:
        return None
    
    row = result_df.iloc[0]
    
    # Deserialize model
    if row['fitted_model_json']:
        try:
            # Try JSON first
            model_data = json.loads(row['fitted_model_json'])
            fitted_model = model_data  # Store as dict for now
            format_type = 'json'
        except (json.JSONDecodeError, ValueError):
            # Try base64 pickle
            fitted_model_bytes = base64.b64decode(row['fitted_model_json'])
            fitted_model = pickle.loads(fitted_model_bytes)
            format_type = 'pickle'
    elif row['fitted_model_s3_path']:
        # TODO: Implement S3 download
        raise NotImplementedError("S3 model loading not yet implemented")
    else:
        raise ValueError(f"Model {model_id} has no serialized data")
    
    return {
        'fitted_model': fitted_model,
        'parameters': json.loads(row['parameters']),
        'metadata': {
            'model_id': model_id,
            'commodity': commodity,
            'model_name': model_name,
            'training_date': training_date,
            'training_samples': int(row['training_samples']),
            'training_start_date': str(row['training_start_date']),
            'model_size_bytes': int(row['model_size_bytes']),
            'created_at': row['created_at'],
            'is_active': row['is_active']
        }
    }


"""
Create Trained Models Table for Persistent Model Storage

This table stores fitted forecasting models to enable:
1. Train-once/inference-many pattern in backfills
2. Model versioning and reproducibility
3. Reuse of trained models across different forecast dates

Storage Strategy:
- Small models (naive, random_walk): Store as JSON in fitted_model_json column
- Large models (XGBoost, TFT, Prophet): Store in S3, reference path in table
"""

import os
from databricks import sql
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")

if not all([DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH]):
    print("ERROR: Missing environment variables!")
    print("Create forecast_agent/infrastructure/.env with credentials")
    exit(1)


def create_trained_models_table():
    """Create commodity.forecast.trained_models table"""
    print("=" * 80)
    print("Creating commodity.forecast.trained_models Table")
    print("=" * 80)

    connection = sql.connect(
        server_hostname=DATABRICKS_HOST.replace("https://", ""),
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )
    cursor = connection.cursor()

    # Create table
    print("\nCreating trained_models table...")
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS commodity.forecast.trained_models (
        -- Model Identification (Primary Key)
        model_id STRING COMMENT 'Unique model identifier: {commodity}_{model_name}_{training_date}_{version}',

        -- Model Metadata
        commodity STRING COMMENT 'Commodity: Coffee or Sugar',
        model_name STRING COMMENT 'Model type: Naive, XGBoost, Prophet, etc.',
        model_version STRING COMMENT 'Model version (e.g., v1.0, backfill_2024)',

        -- Training Information
        training_date DATE COMMENT 'Cutoff date for training data (last date in training set)',
        training_samples INT COMMENT 'Number of training samples (days)',
        training_start_date DATE COMMENT 'First date in training data',

        -- Model Parameters
        parameters STRING COMMENT 'JSON string of model parameters (lags, windows, horizon, etc.)',

        -- Model Storage (two options: inline JSON or S3 path)
        fitted_model_json STRING COMMENT 'Small models stored as JSON (naive, random_walk)',
        fitted_model_s3_path STRING COMMENT 'S3 path for large models (XGBoost, Prophet, TFT)',
        model_size_bytes BIGINT COMMENT 'Size of serialized model in bytes',

        -- Model Metrics (optional - for monitoring model drift)
        training_loss DOUBLE COMMENT 'Training loss/error metric',
        validation_loss DOUBLE COMMENT 'Validation loss/error metric (if available)',

        -- Audit Fields
        created_at TIMESTAMP COMMENT 'When model was trained and stored',
        created_by STRING COMMENT 'User/process that created the model (e.g., backfill_script)',
        is_active BOOLEAN COMMENT 'Whether this model is active for inference',

        -- Partition Columns (for efficient querying)
        year INT COMMENT 'Year of training_date (partition)',
        month INT COMMENT 'Month of training_date (partition)'
    )
    USING DELTA
    PARTITIONED BY (commodity, model_name, year, month)
    LOCATION 's3://groundtruth-capstone/delta/forecast/trained_models/'
    COMMENT 'Trained forecasting models for persistent storage and reuse'
    """

    try:
        cursor.execute(create_table_sql)
        print("✅ Table created successfully")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("⚠️  Table already exists")
        else:
            print(f"❌ Error creating table: {e}")
            cursor.close()
            connection.close()
            return False

    # Create indexes for common queries
    print("\nOptimizing table...")
    try:
        # Z-ORDER by common query columns
        cursor.execute("""
            OPTIMIZE commodity.forecast.trained_models
            ZORDER BY (commodity, model_name, training_date, is_active)
        """)
        print("✅ Table optimized with Z-ORDER")
    except Exception as e:
        print(f"⚠️  Optimization skipped: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print("✅ commodity.forecast.trained_models table created")
    print("\n📝 Table Schema:")
    print("   - Primary Key: model_id (composite of commodity, model_name, training_date)")
    print("   - Storage: JSON (small models) or S3 path (large models)")
    print("   - Partitions: commodity, model_name, year, month")
    print("\n📝 Usage:")
    print("   1. Train model and save: model_persistence.save_model()")
    print("   2. Load model for inference: model_persistence.load_model()")
    print("   3. Query active models: SELECT * WHERE is_active = TRUE")
    print("=" * 80)

    cursor.close()
    connection.close()
    return True


if __name__ == "__main__":
    create_trained_models_table()

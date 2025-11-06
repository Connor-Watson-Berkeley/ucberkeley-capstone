"""Write forecast distributions to Databricks catalog.

This module extends ProductionForecastWriter to support writing directly
to Databricks Unity Catalog instead of local parquet files.

Destination tables:
- commodity.forecast.point_forecasts
- commodity.forecast.distributions
- commodity.forecast.forecast_metadata
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from typing import Optional
from databricks import sql
from databricks.sdk import WorkspaceClient


class DatabricksForecastWriter:
    """
    Write forecast distributions to Databricks Unity Catalog.

    Target schema: commodity.forecast

    Tables:
    - commodity.forecast.point_forecasts
    - commodity.forecast.distributions
    - commodity.forecast.forecast_metadata
    """

    def __init__(self,
                 server_hostname: Optional[str] = None,
                 http_path: Optional[str] = None,
                 token: Optional[str] = None):
        """
        Initialize Databricks writer.

        Args:
            server_hostname: Databricks workspace URL (e.g., dbc-abc123.cloud.databricks.com)
            http_path: SQL warehouse HTTP path
            token: Personal access token

        If not provided, reads from:
        1. Environment variables (DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN)
        2. Config file at ../../../infra/.databrickscfg
        """

        self.server_hostname = server_hostname or os.environ.get("DATABRICKS_HOST", "").replace("https://", "")
        self.http_path = http_path or os.environ.get("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/3cede8561503a13c")
        self.token = token or os.environ.get("DATABRICKS_TOKEN")

        # Try reading from config file if not found
        if not self.token or not self.server_hostname:
            config_path = os.path.join(os.path.dirname(__file__), "../../../infra/.databrickscfg")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    for line in f:
                        if line.startswith('host'):
                            self.server_hostname = line.split('=')[1].strip().replace('https://', '')
                        elif line.startswith('token'):
                            self.token = line.split('=')[1].strip()

        if not all([self.server_hostname, self.http_path, self.token]):
            raise ValueError(
                "Missing Databricks credentials. Provide via arguments or environment variables:\n"
                "  DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN"
            )

        self.catalog = "commodity"
        self.schema = "forecast"

    def _get_connection(self):
        """Get Databricks SQL connection."""
        return sql.connect(
            server_hostname=self.server_hostname,
            http_path=self.http_path,
            access_token=self.token
        )

    def setup_schema(self):
        """Create forecast schema and tables if they don't exist."""

        print("Setting up commodity.forecast schema...")

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Use existing commodity catalog (created by research_agent)
            print("  Using commodity catalog...")
            cursor.execute("USE CATALOG commodity")

            # Create forecast schema (if not exists)
            print("  Ensuring forecast schema exists...")
            cursor.execute(f"""
                CREATE SCHEMA IF NOT EXISTS {self.catalog}.{self.schema}
                COMMENT 'Forecast schema: Model outputs, distributions, and metadata'
            """)

            # Drop existing table if it doesn't match contract
            print("  Dropping existing distributions table (if exists)...")
            cursor.execute(f"DROP TABLE IF EXISTS {self.catalog}.{self.schema}.distributions")

            # Create distributions table with is_actuals and has_data_leakage
            print("  Creating distributions table...")
            cursor.execute(f"""
                CREATE TABLE {self.catalog}.{self.schema}.distributions (
                    path_id INT COMMENT 'Sample path ID (1-2000, 0=actuals)',
                    forecast_start_date DATE COMMENT 'First day of forecast',
                    data_cutoff_date DATE COMMENT 'Last training date',
                    generation_timestamp TIMESTAMP COMMENT 'When generated',
                    model_version STRING COMMENT 'Model identifier',
                    commodity STRING COMMENT 'Coffee or Sugar',
                    day_1 FLOAT,
                    day_2 FLOAT,
                    day_3 FLOAT,
                    day_4 FLOAT,
                    day_5 FLOAT,
                    day_6 FLOAT,
                    day_7 FLOAT,
                    day_8 FLOAT,
                    day_9 FLOAT,
                    day_10 FLOAT,
                    day_11 FLOAT,
                    day_12 FLOAT,
                    day_13 FLOAT,
                    day_14 FLOAT,
                    is_actuals BOOLEAN COMMENT 'TRUE for path_id=0 (actuals row), FALSE for forecast paths',
                    has_data_leakage BOOLEAN COMMENT 'TRUE if forecast_start_date <= data_cutoff_date (should be FALSE in production)'
                )
                COMMENT 'Monte Carlo paths for risk analysis (VaR, CVaR)'
                PARTITIONED BY (model_version, commodity)
            """)

            # Drop existing point_forecasts table if it exists
            print("  Dropping existing point_forecasts table (if exists)...")
            cursor.execute(f"DROP TABLE IF EXISTS {self.catalog}.{self.schema}.point_forecasts")

            # Create point forecasts table
            print("  Creating point_forecasts table...")
            cursor.execute(f"""
                CREATE TABLE {self.catalog}.{self.schema}.point_forecasts (
                    forecast_date DATE COMMENT 'Date being forecasted',
                    data_cutoff_date DATE COMMENT 'Last date in training data',
                    generation_timestamp TIMESTAMP COMMENT 'When forecast was generated',
                    day_ahead INT COMMENT 'Days ahead from data_cutoff_date',
                    forecast_mean DECIMAL(10,2) COMMENT 'Point forecast',
                    forecast_std DECIMAL(10,2) COMMENT 'Forecast standard deviation',
                    lower_95 DECIMAL(10,2) COMMENT '95% prediction interval lower bound',
                    upper_95 DECIMAL(10,2) COMMENT '95% prediction interval upper bound',
                    model_version STRING COMMENT 'Model identifier',
                    commodity STRING COMMENT 'Coffee or Sugar',
                    model_success BOOLEAN COMMENT 'Did model converge successfully?',
                    actual_close DECIMAL(10,2) COMMENT 'Realized price (NULL if future)',
                    has_data_leakage BOOLEAN COMMENT 'TRUE if forecast_date <= data_cutoff_date (should be FALSE in production)'
                )
                COMMENT 'Point forecasts with prediction intervals and actuals'
                PARTITIONED BY (commodity, forecast_date)
            """)

            # Drop existing forecast_actuals table if it exists
            print("  Dropping existing forecast_actuals table (if exists)...")
            cursor.execute(f"DROP TABLE IF EXISTS {self.catalog}.{self.schema}.forecast_actuals")

            # Create forecast actuals table
            print("  Creating forecast_actuals table...")
            cursor.execute(f"""
                CREATE TABLE {self.catalog}.{self.schema}.forecast_actuals (
                    forecast_date DATE COMMENT 'Date of realized price',
                    commodity STRING COMMENT 'Coffee or Sugar',
                    actual_close DECIMAL(10,2) COMMENT 'Realized closing price'
                )
                COMMENT 'Realized prices for backtesting'
                PARTITIONED BY (commodity)
            """)

            # Drop existing forecast_metadata table if it exists
            print("  Dropping existing forecast_metadata table (if exists)...")
            cursor.execute(f"DROP TABLE IF EXISTS {self.catalog}.{self.schema}.forecast_metadata")

            # Create forecast metadata table
            print("  Creating forecast_metadata table...")
            cursor.execute(f"""
                CREATE TABLE {self.catalog}.{self.schema}.forecast_metadata (
                    forecast_id STRING COMMENT 'Unique identifier: model_version_commodity_forecast_date',
                    forecast_start_date DATE COMMENT 'First day of forecast horizon',
                    data_cutoff_date DATE COMMENT 'Last training date',
                    generation_timestamp TIMESTAMP COMMENT 'When forecast was generated',
                    model_version STRING COMMENT 'Model identifier',
                    commodity STRING COMMENT 'Coffee or Sugar',

                    -- Performance Metrics
                    mae_1d DECIMAL(10,4) COMMENT '1-day ahead MAE',
                    mae_7d DECIMAL(10,4) COMMENT '7-day ahead MAE',
                    mae_14d DECIMAL(10,4) COMMENT '14-day ahead MAE',
                    rmse_1d DECIMAL(10,4) COMMENT '1-day ahead RMSE',
                    rmse_7d DECIMAL(10,4) COMMENT '7-day ahead RMSE',
                    rmse_14d DECIMAL(10,4) COMMENT '14-day ahead RMSE',
                    mape_1d DECIMAL(10,4) COMMENT '1-day ahead MAPE (%)',
                    mape_7d DECIMAL(10,4) COMMENT '7-day ahead MAPE (%)',
                    mape_14d DECIMAL(10,4) COMMENT '14-day ahead MAPE (%)',

                    -- Timing Metrics
                    training_time_seconds DECIMAL(10,2) COMMENT 'Time to train model',
                    inference_time_seconds DECIMAL(10,2) COMMENT 'Time to generate 14-day forecast',
                    total_time_seconds DECIMAL(10,2) COMMENT 'Total time (training + inference)',

                    -- Infrastructure
                    hardware_type STRING COMMENT 'CPU type or "local" if run locally',
                    num_cores INT COMMENT 'Number of CPU cores used',
                    memory_gb INT COMMENT 'RAM allocated (GB)',
                    cluster_id STRING COMMENT 'Databricks cluster ID (NULL if local)',

                    -- Data Quality
                    training_days INT COMMENT 'Number of days in training set',
                    actuals_available INT COMMENT 'Number of actuals available (0-14)',
                    has_data_leakage BOOLEAN COMMENT 'TRUE if forecast_start_date <= data_cutoff_date',
                    model_success BOOLEAN COMMENT 'Did model train successfully?',

                    -- Additional Metadata
                    notes STRING COMMENT 'Any additional notes or warnings'
                )
                COMMENT 'Forecast performance metrics and timing data'
                PARTITIONED BY (model_version, commodity)
            """)

            cursor.close()

        print(f"✓ Schema setup complete: {self.catalog}.{self.schema}")
        print(f"  Tables created: distributions, point_forecasts, forecast_actuals, forecast_metadata")

    def write_distributions(self, df: pd.DataFrame, mode: str = "append"):
        """
        Write distributions DataFrame to Databricks.

        Args:
            df: Distributions DataFrame (from ProductionForecastWriter)
            mode: 'append' or 'overwrite'
        """

        if len(df) == 0:
            print("⚠ Warning: Empty DataFrame, nothing to write")
            return

        table_name = f"{self.catalog}.{self.schema}.distributions"
        print(f"\nWriting {len(df):,} rows to {table_name}...")

        # Convert datetime columns to proper types
        df = df.copy()
        df['forecast_start_date'] = pd.to_datetime(df['forecast_start_date']).dt.date
        df['data_cutoff_date'] = pd.to_datetime(df['data_cutoff_date']).dt.date
        df['generation_timestamp'] = pd.to_datetime(df['generation_timestamp'])

        # Ensure column order matches table schema
        expected_cols = [
            'path_id', 'forecast_start_date', 'data_cutoff_date',
            'generation_timestamp', 'model_version', 'commodity',
            'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7',
            'day_8', 'day_9', 'day_10', 'day_11', 'day_12', 'day_13', 'day_14',
            'is_actuals', 'has_data_leakage'
        ]

        df = df[expected_cols]

        # Write using Spark (via Databricks SDK)
        from pyspark.sql import SparkSession

        try:
            spark = SparkSession.builder.getOrCreate()
            spark_df = spark.createDataFrame(df)

            spark_df.write \
                .format("delta") \
                .mode(mode) \
                .saveAsTable(table_name)

            print(f"✓ Wrote {len(df):,} rows to {table_name}")

            # Verify
            count = spark.sql(f"SELECT COUNT(*) FROM {table_name}").collect()[0][0]
            print(f"  Total rows in table: {count:,}")

        except Exception as e:
            # Fallback: use SQL INSERT (slower but works without Spark)
            print(f"  Spark write failed, using SQL INSERT: {e}")
            self._write_via_sql_insert(df, table_name)

    def _write_via_sql_insert(self, df: pd.DataFrame, table_name: str):
        """Fallback method: write via SQL INSERT statements (batch of 1000)."""

        print(f"  Writing {len(df):,} rows via SQL INSERT (may take a while)...")

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build VALUES clause
            batch_size = 1000
            total_batches = (len(df) + batch_size - 1) // batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx]

                values_list = []
                for _, row in batch_df.iterrows():
                    # Format row as SQL values
                    values = []
                    for col in df.columns:
                        val = row[col]
                        if pd.isna(val):
                            values.append("NULL")
                        elif isinstance(val, (int, float, np.integer, np.floating)):
                            values.append(str(val))
                        elif isinstance(val, bool):
                            values.append("TRUE" if val else "FALSE")
                        else:
                            # String - escape single quotes
                            escaped_val = str(val).replace("'", "''")
                            values.append(f"'{escaped_val}'")

                    values_list.append(f"({', '.join(values)})")

                # Execute INSERT
                cols = ', '.join(df.columns)
                values_str = ',\n'.join(values_list)

                sql = f"""
                    INSERT INTO {table_name} ({cols})
                    VALUES {values_str}
                """

                cursor.execute(sql)

                print(f"    Batch {batch_idx + 1}/{total_batches}: {len(batch_df)} rows")

            cursor.close()

        print(f"✓ Inserted {len(df):,} rows via SQL")

    def write_point_forecasts(self, df: pd.DataFrame, mode: str = "append"):
        """
        Write point forecasts DataFrame to Databricks.

        Args:
            df: Point forecasts DataFrame (from ProductionForecastWriter)
            mode: 'append' or 'overwrite'
        """

        if len(df) == 0:
            print("⚠ Warning: Empty DataFrame, nothing to write")
            return

        table_name = f"{self.catalog}.{self.schema}.point_forecasts"
        print(f"\nWriting {len(df):,} rows to {table_name}...")

        # Convert datetime columns to proper types
        df = df.copy()
        df['forecast_date'] = pd.to_datetime(df['forecast_date']).dt.date
        df['data_cutoff_date'] = pd.to_datetime(df['data_cutoff_date']).dt.date
        df['generation_timestamp'] = pd.to_datetime(df['generation_timestamp'])

        # Ensure column order matches table schema
        expected_cols = [
            'forecast_date', 'data_cutoff_date', 'generation_timestamp',
            'day_ahead', 'forecast_mean', 'forecast_std',
            'lower_95', 'upper_95', 'model_version', 'commodity', 'model_success',
            'actual_close', 'has_data_leakage'
        ]

        df = df[expected_cols]

        # Write using Spark (via Databricks SDK)
        from pyspark.sql import SparkSession

        try:
            spark = SparkSession.builder.getOrCreate()
            spark_df = spark.createDataFrame(df)

            spark_df.write \
                .format("delta") \
                .mode(mode) \
                .saveAsTable(table_name)

            print(f"✓ Wrote {len(df):,} rows to {table_name}")

            # Verify
            count = spark.sql(f"SELECT COUNT(*) FROM {table_name}").collect()[0][0]
            print(f"  Total rows in table: {count:,}")

        except Exception as e:
            # Fallback: use SQL INSERT (slower but works without Spark)
            print(f"  Spark write failed, using SQL INSERT: {e}")
            self._write_via_sql_insert(df, table_name)

    def verify_tables(self):
        """Verify tables exist and show row counts."""

        print(f"\nVerifying {self.catalog}.{self.schema} tables...")

        with self._get_connection() as conn:
            cursor = conn.cursor()

            for table in ['distributions', 'point_forecasts', 'forecast_actuals']:
                full_name = f"{self.catalog}.{self.schema}.{table}"

                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {full_name}")
                    count = cursor.fetchone()[0]
                    print(f"  ✓ {table}: {count:,} rows")
                except Exception as e:
                    print(f"  ✗ {table}: {e}")

            cursor.close()


def upload_distributions_to_databricks(parquet_path: str = "production_forecasts/distributions.parquet"):
    """
    Upload local distributions parquet file to Databricks.

    Args:
        parquet_path: Path to local distributions.parquet file
    """

    print("="*80)
    print("UPLOADING DISTRIBUTIONS TO DATABRICKS")
    print("="*80)

    # Load local parquet
    print(f"\n[1/3] Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Models: {df['model_version'].nunique()}")
    print(f"  Commodities: {df['commodity'].unique()}")

    # Initialize Databricks writer
    print("\n[2/3] Connecting to Databricks...")
    writer = DatabricksForecastWriter()

    # Create schema and tables
    writer.setup_schema()

    # Upload distributions
    print("\n[3/3] Uploading distributions...")
    writer.write_distributions(df, mode="append")

    # Verify
    writer.verify_tables()

    print("\n" + "="*80)
    print("✅ UPLOAD COMPLETE")
    print("="*80)
    print(f"\nDistributions available at: commodity.forecast.distributions")
    print(f"Total rows uploaded: {len(df):,}")


if __name__ == "__main__":
    # Upload distributions from local parquet
    upload_distributions_to_databricks()

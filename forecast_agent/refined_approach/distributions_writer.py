"""Writer for distributions table with data leakage prevention.

Ensures only data leakage-free forecasts are written to distributions table.
"""

from typing import List, Dict
from datetime import date, datetime
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.functions import col, to_date, lit
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, TimestampType, FloatType, BooleanType, DateType


def get_existing_forecast_dates(
    spark: SparkSession,
    commodity: str,
    model_version: str
) -> set:
    """
    Get set of dates that already have forecasts in distributions table.
    
    Useful for incremental/resume execution - skip forecasts that already exist.
    
    Args:
        spark: Spark session
        commodity: 'Coffee' or 'Sugar'
        model_version: Model version string
    
    Returns:
        Set of forecast_start_date values (as date objects)
    """
    query = f"""
        SELECT DISTINCT forecast_start_date
        FROM commodity.forecast.distributions
        WHERE commodity = '{commodity}'
          AND model_version = '{model_version}'
          AND is_actuals = FALSE
    """
    
    result_df = spark.sql(query).toPandas()
    
    if result_df.empty:
        return set()
    
    # Convert to set of date objects
    existing_dates = set(pd.to_datetime(result_df['forecast_start_date']).dt.date)
    return existing_dates


class DistributionsWriter:
    """Writes forecasts to distributions table, ensuring no data leakage."""
    
    def __init__(self, spark: SparkSession):
        """
        Initialize writer.
        
        Args:
            spark: Spark session
        """
        self.spark = spark
    
    def write_distributions(self,
                           forecasts: List[Dict],
                           commodity: str,
                           model_version: str,
                           data_cutoff_date: date) -> None:
        """
        Write forecast distributions to table, filtering out data leakage.
        
        Args:
            forecasts: List of forecast dicts with:
                - forecast_start_date: date
                - paths: List of dicts with path_id and values (list of 14 floats)
                - mean_forecast: array of 14 floats (optional, for point forecasts)
            commodity: 'Coffee' or 'Sugar'
            model_version: Model identifier string
            data_cutoff_date: Training cutoff date (last date in training data)
        
        Requirements:
        - Only writes forecasts where forecast_start_date > data_cutoff_date
        - Sets has_data_leakage=FALSE for all rows
        - Generates 2000 paths (path_id 1-2000)
        """
        if not forecasts:
            return
        
        # Filter out data leakage: only forecast_start_date > data_cutoff_date
        valid_forecasts = [
            f for f in forecasts
            if f['forecast_start_date'] > data_cutoff_date
        ]
        
        if not valid_forecasts:
            print(f"⚠️  No data leakage-free forecasts (all have forecast_start_date <= {data_cutoff_date})")
            return
        
        print(f"✅ Writing {len(valid_forecasts)} data leakage-free forecasts (filtered {len(forecasts) - len(valid_forecasts)} with leakage)")
        
        # Build rows for distributions table
        rows = []
        generation_timestamp = datetime.now()
        
        for forecast in valid_forecasts:
            forecast_start_date = forecast['forecast_start_date']
            paths = forecast.get('paths', [])
            
            # Ensure we have paths (generate if needed from mean_forecast)
            if not paths and 'mean_forecast' in forecast:
                paths = self._generate_paths_from_mean(forecast['mean_forecast'], forecast.get('forecast_std', 2.5))
            
            # Write each path
            for path in paths:
                path_id = path.get('path_id', 0)
                values = path.get('values', [])
                
                # Ensure 14 days
                if len(values) < 14:
                    values.extend([None] * (14 - len(values)))
                elif len(values) > 14:
                    values = values[:14]
                
                row = {
                    'path_id': path_id,
                    'forecast_start_date': forecast_start_date,
                    'data_cutoff_date': data_cutoff_date,
                    'generation_timestamp': generation_timestamp,
                    'model_version': model_version,
                    'commodity': commodity,
                    'day_1': float(values[0]) if values[0] is not None else None,
                    'day_2': float(values[1]) if len(values) > 1 and values[1] is not None else None,
                    'day_3': float(values[2]) if len(values) > 2 and values[2] is not None else None,
                    'day_4': float(values[3]) if len(values) > 3 and values[3] is not None else None,
                    'day_5': float(values[4]) if len(values) > 4 and values[4] is not None else None,
                    'day_6': float(values[5]) if len(values) > 5 and values[5] is not None else None,
                    'day_7': float(values[6]) if len(values) > 6 and values[6] is not None else None,
                    'day_8': float(values[7]) if len(values) > 7 and values[7] is not None else None,
                    'day_9': float(values[8]) if len(values) > 8 and values[8] is not None else None,
                    'day_10': float(values[9]) if len(values) > 9 and values[9] is not None else None,
                    'day_11': float(values[10]) if len(values) > 10 and values[10] is not None else None,
                    'day_12': float(values[11]) if len(values) > 11 and values[11] is not None else None,
                    'day_13': float(values[12]) if len(values) > 12 and values[12] is not None else None,
                    'day_14': float(values[13]) if len(values) > 13 and values[13] is not None else None,
                    'is_actuals': False,  # Forecasts only
                    'has_data_leakage': False  # Always False - we filtered leakage above
                }
                rows.append(row)
        
        if not rows:
            return
        
        # Create DataFrame
        df = self.spark.createDataFrame(rows)
        
        # Cast types to match table schema
        df = df.withColumn("path_id", col("path_id").cast(IntegerType())) \
               .withColumn("forecast_start_date", to_date(col("forecast_start_date"))) \
               .withColumn("data_cutoff_date", to_date(col("data_cutoff_date"))) \
               .withColumn("generation_timestamp", col("generation_timestamp").cast(TimestampType())) \
               .withColumn("model_version", col("model_version").cast(StringType())) \
               .withColumn("commodity", col("commodity").cast(StringType())) \
               .withColumn("is_actuals", col("is_actuals").cast(BooleanType())) \
               .withColumn("has_data_leakage", col("has_data_leakage").cast(BooleanType()))
        
        # Cast day columns to FloatType
        for i in range(1, 15):
            day_col = f"day_{i}"
            df = df.withColumn(day_col, col(day_col).cast(FloatType()))
        
        # Write to table (append mode)
        df.write.mode("append").saveAsTable("commodity.forecast.distributions")
        
        print(f"✅ Wrote {len(rows)} distribution rows to commodity.forecast.distributions")
    
    def _generate_paths_from_mean(self, mean_forecast: np.ndarray, std: float, n_paths: int = 2000) -> List[Dict]:
        """
        Generate Monte Carlo paths from mean forecast.
        
        Args:
            mean_forecast: Array of 14 mean values
            std: Standard deviation for noise
            n_paths: Number of paths to generate
        
        Returns:
            List of path dicts
        """
        paths = []
        for path_id in range(1, n_paths + 1):
            # Generate noise
            noise = np.random.normal(0, std, len(mean_forecast))
            path_values = (mean_forecast + noise).tolist()
            
            paths.append({
                'path_id': path_id,
                'values': path_values
            })
        
        return paths


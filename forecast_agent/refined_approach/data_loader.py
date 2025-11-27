"""Data loader for time-series cross-validation folds.

Inspired by DS261 FlightDelayDataLoader - clean and simple.
"""

from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql import functions as F


class TimeSeriesDataLoader:
    """Loads time-series data and creates temporal folds for cross-validation.
    
    Handles both Spark and pandas DataFrames for flexibility.
    """
    
    def __init__(self, 
                 spark: Optional[SparkSession] = None,
                 table_name: str = "commodity.silver.unified_data"):
        """
        Initialize data loader.
        
        Args:
            spark: SparkSession (optional - only needed for Spark mode)
            table_name: Delta table name for commodity data
        """
        self.spark = spark
        self.table_name = table_name
        self.folds = []
        
    def load_from_spark(self,
                       commodity: str,
                       cutoff_date: Optional[str] = None) -> SparkDataFrame:
        """
        Load data from Databricks Delta table.
        
        Args:
            commodity: 'Coffee' or 'Sugar'
            cutoff_date: Optional cutoff date for backtesting (YYYY-MM-DD)
            
        Returns:
            Spark DataFrame with date column and target/features
        """
        if not self.spark:
            raise ValueError("Spark session required for load_from_spark()")
            
        df = self.spark.table(self.table_name)
        df = df.filter(f"commodity = '{commodity}'")
        
        if cutoff_date:
            df = df.filter(f"date <= '{cutoff_date}'")
            
        return df.orderBy("date")
    
    def load_to_pandas(self,
                      commodity: str,
                      cutoff_date: Optional[str] = None,
                      features: Optional[List[str]] = None,
                      aggregate_regions: bool = True,
                      aggregation_method: str = 'mean') -> pd.DataFrame:
        """
        Load data and convert to pandas DataFrame.
        
        Handles unified_data grain: (date, commodity, region) by aggregating regions.
        
        Args:
            commodity: 'Coffee' or 'Sugar'
            cutoff_date: Optional cutoff date for backtesting
            features: Optional list of feature columns to select
            aggregate_regions: If True, aggregate across regions. If False, keep region grain.
            aggregation_method: 'mean', 'first', or 'weighted' (for region aggregation)
            
        Returns:
            Pandas DataFrame with DatetimeIndex (aggregated to date level)
        """
        if not self.spark:
            raise ValueError("Spark session required for load_to_pandas()")
            
        df_spark = self.load_from_spark(commodity, cutoff_date)
        
        # Select features if specified
        if features:
            # Always include date and commodity for aggregation
            cols = ["date", "commodity"]
            if "region" not in features and not aggregate_regions:
                cols.append("region")
            for f in features:
                if f not in cols:
                    cols.append(f)
            df_spark = df_spark.select(cols)
        
        # Aggregate across regions (unified_data has region grain)
        if aggregate_regions:
            # Market data is same across regions, so we can use first() or mean()
            # Weather data varies by region, so we aggregate
            from pyspark.sql import functions as F
            
            # Group by date and commodity
            if aggregation_method == 'mean':
                # Average weather features across regions
                agg_exprs = [F.first(col).alias(col) if col in ['date', 'commodity', 'close', 'open', 'high', 'low', 'volume', 'vix'] 
                            else F.avg(col).alias(col) 
                            for col in df_spark.columns if col != 'region']
            elif aggregation_method == 'first':
                # Use first region's values (market data is identical anyway)
                agg_exprs = [F.first(col).alias(col) for col in df_spark.columns if col != 'region']
            elif aggregation_method == 'weighted':
                # TODO: Implement weighted aggregation by production
                raise NotImplementedError("Weighted aggregation not yet implemented")
            else:
                raise ValueError(f"Unknown aggregation_method: {aggregation_method}")
            
            df_spark = df_spark.groupBy("date", "commodity").agg(*agg_exprs)
        
        # Convert to pandas
        df = df_spark.toPandas()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Drop commodity column if only one commodity
        if 'commodity' in df.columns:
            unique_commodities = df['commodity'].unique()
            if len(unique_commodities) == 1:
                df = df.drop('commodity', axis=1)
        
        return df
    
    def create_temporal_folds(self,
                             df: pd.DataFrame,
                             date_col: str = 'date',
                             n_folds: int = 3,
                             test_fold: bool = True,
                             min_train_size: int = 365,
                             step_size: int = 14,
                             horizon: int = 14) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create sliding-window temporal folds for cross-validation.
        
        Similar to DS261 split.py approach - simple and clean.
        
        Args:
            df: Pandas DataFrame with datetime index
            date_col: Name of date column (if not index)
            n_folds: Number of CV folds
            test_fold: Whether to create final test fold
            min_train_size: Minimum training days
            step_size: Days between forecast windows
            horizon: Forecast horizon (days)
            
        Returns:
            List of (train_df, val_df) tuples for CV folds,
            plus (train_df, test_df) if test_fold=True
        """
        # Ensure datetime index
        if date_col in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(date_col)
        df = df.sort_index()
        
        # Get date range
        start_date = df.index[0]
        end_date = df.index[-1]
        total_days = (end_date - start_date).days
        
        # Calculate period size
        n_periods = n_folds + 1 if test_fold else n_folds
        period_days = (total_days - min_train_size - horizon) // n_periods
        
        if period_days <= 0:
            raise ValueError(f"Not enough data: {total_days} days < min_train_size ({min_train_size}) + horizon ({horizon})")
        
        folds = []
        
        # CV folds (sliding window)
        for f in range(n_folds):
            # Training period: from start to period f
            train_start = start_date
            train_end = start_date + timedelta(days=min_train_size + f * period_days)
            
            # Validation period: period f+1
            val_start = train_end
            val_end = min(val_start + timedelta(days=period_days), end_date)
            
            # Extract dataframes
            train_df = df.loc[train_start:train_end].copy()
            val_df = df.loc[val_start:val_end].copy()
            
            if len(train_df) >= min_train_size and len(val_df) >= horizon:
                folds.append((train_df, val_df))
        
        # Test fold (all training data up to final period, test on final period)
        if test_fold:
            test_start = start_date + timedelta(days=min_train_size + n_folds * period_days)
            test_end = end_date
            
            train_df = df.loc[start_date:test_start].copy()
            test_df = df.loc[test_start:test_end].copy()
            
            if len(train_df) >= min_train_size and len(test_df) >= horizon:
                folds.append((train_df, test_df))
        
        return folds
    
    def create_walk_forward_folds(self,
                                 df: pd.DataFrame,
                                 min_train_size: int = 365,
                                 step_size: int = 14,
                                 horizon: int = 14,
                                 max_folds: Optional[int] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward validation folds (expanding window).
        
        More realistic for production - each fold uses all available data up to test period.
        
        Args:
            df: Pandas DataFrame with datetime index
            min_train_size: Minimum training days
            step_size: Days between forecast windows
            horizon: Forecast horizon (days)
            max_folds: Maximum number of folds to generate (None = all possible)
            
        Returns:
            List of (train_df, test_df) tuples
        """
        # Ensure datetime index and sorted
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        df = df.sort_index()
        
        folds = []
        start_idx = min_train_size
        
        while start_idx + horizon <= len(df):
            train_df = df.iloc[:start_idx].copy()
            test_df = df.iloc[start_idx:start_idx + horizon].copy()
            
            folds.append((train_df, test_df))
            
            start_idx += step_size
            
            if max_folds and len(folds) >= max_folds:
                break
        
        return folds


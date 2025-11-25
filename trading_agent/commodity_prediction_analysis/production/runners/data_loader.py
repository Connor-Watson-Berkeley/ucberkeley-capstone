"""
Data Loading Module
Handles loading prices and prediction matrices for backtesting
"""

import pandas as pd
import pickle
from typing import Dict, Tuple, Optional, Any


class DataLoader:
    """Loads price data and prediction matrices for commodity-model pairs"""

    def __init__(self, spark=None):
        """
        Initialize data loader

        Args:
            spark: Spark session (required for loading from Delta tables)
        """
        self.spark = spark

    def load_commodity_data(
        self,
        commodity: str,
        model_version: str,
        data_paths: Dict[str, str]
    ) -> Tuple[pd.DataFrame, Dict[Any, Any]]:
        """
        Load prices and prediction matrices for a commodity-model pair

        Args:
            commodity: Commodity name (e.g., 'coffee', 'sugar')
            model_version: Model version identifier
            data_paths: Dictionary of data paths from config

        Returns:
            Tuple of (prices_df, prediction_matrices_dict)

        Raises:
            ValueError: If required data cannot be loaded
        """
        print(f"\nLoading data for {commodity.upper()} - {model_version}...")

        # Load prices
        prices = self._load_prices(commodity, data_paths)
        print(f"  ✓ Loaded {len(prices)} days of prices")

        # Load prediction matrices
        prediction_matrices = self._load_prediction_matrices(model_version, data_paths)
        print(f"  ✓ Loaded {len(prediction_matrices)} prediction matrices")

        # Validate data
        self._validate_data(prices, prediction_matrices, model_version)

        return prices, prediction_matrices

    def _load_prices(self, commodity: str, data_paths: Dict[str, str]) -> pd.DataFrame:
        """
        Load price data from Delta table

        Args:
            commodity: Commodity name to filter by
            data_paths: Dictionary containing 'prices_source' key

        Returns:
            DataFrame with columns ['date', 'price']
        """
        if self.spark is None:
            raise ValueError("Spark session required to load prices from Delta table")

        # Load from unified_data and filter by commodity
        # Note: unified_data has 'close' column, rename to 'price' for consistency
        prices = self.spark.table(data_paths['prices_source']) \
            .filter(f"commodity = '{commodity.title()}'") \
            .select('date', 'close') \
            .toPandas()

        # Rename 'close' to 'price' for consistency with backtesting code
        prices = prices.rename(columns={'close': 'price'})

        # CRITICAL: Normalize dates to midnight for dictionary lookup compatibility
        prices['date'] = pd.to_datetime(prices['date']).dt.normalize()

        return prices

    def _load_prediction_matrices(
        self,
        model_version: str,
        data_paths: Dict[str, str]
    ) -> Dict[Any, Any]:
        """
        Load prediction matrices from pickle file

        Args:
            model_version: Model version to determine source type
            data_paths: Dictionary with matrix file paths

        Returns:
            Dictionary mapping {timestamp: numpy_array(n_paths, n_horizons)}
        """
        # Determine source type and select appropriate path
        if model_version.startswith('synthetic_'):
            matrix_path = data_paths['prediction_matrices']
            source_type = "SYNTHETIC"
        else:
            matrix_path = data_paths['prediction_matrices_real']
            source_type = "REAL"

        try:
            with open(matrix_path, 'rb') as f:
                prediction_matrices = pickle.load(f)

            # Inspect structure
            if len(prediction_matrices) > 0:
                sample_matrix = list(prediction_matrices.values())[0]
                print(f"  ✓ Source: {source_type}")
                print(f"  ✓ Matrix structure: {sample_matrix.shape[0]} runs × {sample_matrix.shape[1]} horizons")

            return prediction_matrices

        except FileNotFoundError:
            raise ValueError(
                f"Prediction matrices not found at {matrix_path}. "
                f"Run data preparation first (notebook 01 or 02)."
            )
        except Exception as e:
            raise ValueError(f"Error loading prediction matrices: {e}")

    def _validate_data(
        self,
        prices: pd.DataFrame,
        prediction_matrices: Dict[Any, Any],
        model_version: str
    ) -> None:
        """
        Validate loaded data quality and alignment

        Args:
            prices: Price DataFrame
            prediction_matrices: Prediction matrix dictionary
            model_version: Model version for error messages
        """
        # Check prices structure
        required_cols = ['date', 'price']
        missing_cols = set(required_cols) - set(prices.columns)
        if missing_cols:
            raise ValueError(f"Prices missing required columns: {missing_cols}")

        # Check for null prices
        if prices['price'].isna().any():
            null_count = prices['price'].isna().sum()
            raise ValueError(f"Found {null_count} null prices in data")

        # Check prediction matrices structure
        if len(prediction_matrices) == 0:
            raise ValueError(f"No prediction matrices found for {model_version}")

        # Date alignment check
        pred_keys = set(prediction_matrices.keys())
        price_dates = set(prices['date'].tolist())
        overlap = pred_keys.intersection(price_dates)

        match_rate = len(overlap) / len(pred_keys) if len(pred_keys) > 0 else 0

        if match_rate < 0.5:
            raise ValueError(
                f"Poor date alignment between prices and predictions. "
                f"Match rate: {match_rate*100:.1f}%. "
                f"Expected at least 50% overlap."
            )

        print(f"  ✓ Validation passed - {len(overlap)} matching dates ({match_rate*100:.1f}% coverage)")

    def get_data_summary(
        self,
        prices: pd.DataFrame,
        prediction_matrices: Dict[Any, Any]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for loaded data

        Args:
            prices: Price DataFrame
            prediction_matrices: Prediction matrix dictionary

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'n_price_days': len(prices),
            'price_date_range': (prices['date'].min(), prices['date'].max()),
            'price_range': (prices['price'].min(), prices['price'].max()),
            'n_prediction_dates': len(prediction_matrices),
            'avg_price': prices['price'].mean(),
            'std_price': prices['price'].std()
        }

        if len(prediction_matrices) > 0:
            sample_matrix = list(prediction_matrices.values())[0]
            summary['prediction_runs'] = sample_matrix.shape[0]
            summary['prediction_horizons'] = sample_matrix.shape[1]

        return summary

    def discover_model_versions(
        self,
        commodity: str,
        forecast_table: str = "commodity.forecast.distributions"
    ) -> Tuple[list, list]:
        """
        Discover all available model versions for a commodity

        Args:
            commodity: Commodity name
            forecast_table: Unity Catalog table with forecasts

        Returns:
            Tuple of (synthetic_versions, real_versions)
        """
        if self.spark is None:
            raise ValueError("Spark session required to discover model versions")

        print(f"\nDiscovering model versions for {commodity}...")

        # Check synthetic predictions (from generated tables)
        synthetic_versions = []
        try:
            output_schema = "commodity.trading_agent"
            pred_table = f"{output_schema}.predictions_{commodity.lower()}"
            synthetic_df = self.spark.table(pred_table).select("model_version").distinct()
            synthetic_versions = [row.model_version for row in synthetic_df.collect()]
            if synthetic_versions:
                print(f"  Synthetic models: {synthetic_versions}")
        except Exception:
            pass  # Table may not exist yet

        # Check real predictions (from forecast table)
        real_versions = []
        try:
            real_df = self.spark.table(forecast_table) \
                .filter(f"commodity = '{commodity.title()}' AND is_actuals = false") \
                .select("model_version") \
                .distinct() \
                .orderBy("model_version")
            real_versions = [row.model_version for row in real_df.collect()]
            if real_versions:
                print(f"  Real models: {real_versions}")
        except Exception as e:
            print(f"  Warning: Could not check real predictions: {e}")

        # Combine
        all_versions = list(set(synthetic_versions + real_versions))

        if len(all_versions) == 0:
            print(f"  ⚠️  No model versions found for {commodity}")
        else:
            print(f"  ✓ Found {len(all_versions)} total model versions")

        return synthetic_versions, real_versions

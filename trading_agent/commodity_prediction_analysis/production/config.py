"""
Production Configuration
Central configuration for production trading agent system
Uses updated costs from diagnostic research
"""

import pandas as pd
import os

# =============================================================================
# COMMODITY CONFIGURATIONS
# =============================================================================

COMMODITY_CONFIGS = {
    'coffee': {
        'commodity': 'coffee',
        'harvest_volume': 50,  # tons per year
        'harvest_windows': [(5, 9)],  # May-September
        'storage_cost_pct_per_day': 0.005,   # 0.005% per day (updated from research)
        'transaction_cost_pct': 0.01,        # 0.01% per transaction (updated from research)
        'min_inventory_to_trade': 1.0,
        'max_holding_days': 365
    },
    'sugar': {
        'commodity': 'sugar',
        'harvest_volume': 50,
        'harvest_windows': [(10, 12)],  # October-December
        'storage_cost_pct_per_day': 0.005,
        'transaction_cost_pct': 0.01,
        'min_inventory_to_trade': 1.0,
        'max_holding_days': 365
    }
}

# =============================================================================
# STRATEGY PARAMETERS
# =============================================================================

# Baseline strategy parameters (no predictions)
BASELINE_PARAMS = {
    'immediate_sale': {},  # No parameters

    'equal_batches': {
        'n_batches': 4,  # Sell in 4 equal batches
        'days_between': 30  # 30 days between batches
    },

    'price_threshold': {
        'ma_window': 30,  # 30-day moving average
        'threshold_pct': 0.02  # Sell when price > MA + 2%
    },

    'moving_average': {
        'ma_window': 30  # 30-day moving average for crossover
    }
}

# Prediction-based strategy parameters
PREDICTION_PARAMS = {
    'consensus': {
        'bullish_threshold': 0.70,  # 70% of paths must be bullish
        'lookback_days': 7  # Look at 7-day ahead predictions
    },

    'expected_value': {
        'discount_rate': 0.0001,  # Daily discount rate
        'min_ev_improvement': 0.01  # Require 1% EV improvement to hold
    },

    'risk_adjusted': {
        'risk_aversion': 0.5,  # Balance between return and uncertainty
        'confidence_threshold': 0.80  # 80% confidence level
    },

    'price_threshold_predictive': {
        'ma_window': 30,
        'threshold_pct': 0.02,
        'prediction_weight': 0.5  # Weight on prediction signal
    },

    'moving_average_predictive': {
        'ma_window': 30,
        'prediction_weight': 0.5  # Weight on prediction signal
    }
}

# =============================================================================
# DATA PATHS AND TABLES
# =============================================================================

# Unity Catalog paths
OUTPUT_SCHEMA = "commodity.trading_agent"
MARKET_TABLE = "commodity.silver.unified_data"
FORECAST_TABLE = "commodity.forecast.distributions"

# Volume paths for file storage
VOLUME_PATH = "/Volumes/commodity/trading_agent/files"

# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

ANALYSIS_CONFIG = {
    'backtest_start_date': '2022-01-01',  # Start of backtest period
    'synthetic_accuracies': [0.6, 0.7, 0.8, 0.9, 1.0],  # Synthetic accuracy levels
    'n_monte_carlo_runs': 2000,  # Number of Monte Carlo paths
    'forecast_horizon': 14,  # Days ahead to forecast
    'validation_metrics': ['mape', 'rmse', 'mae']  # Metrics for validation
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_data_paths(commodity, model_version):
    """
    Get data paths for a specific commodity and model version

    Args:
        commodity: Commodity name (e.g., 'coffee')
        model_version: Model version (e.g., 'arima_v1', 'synthetic_acc90')

    Returns:
        dict: Dictionary of data paths
            - prices_prepared: Delta table with prepared prices
            - prediction_matrices: Synthetic prediction matrix file
            - prediction_matrices_real: Real prediction matrix file
            - results_detailed: Output file for detailed results
            - charts_dir: Directory for output charts
    """
    return {
        # Input paths
        'prices_prepared': f"{OUTPUT_SCHEMA}.prices_{commodity.lower()}",
        'prediction_matrices': f"{VOLUME_PATH}/prediction_matrices_{commodity.lower()}_synthetic_acc90.pkl",
        'prediction_matrices_real': f"{VOLUME_PATH}/prediction_matrices_{commodity.lower()}_{model_version}_real.pkl",

        # Output paths
        'results_detailed': f"{VOLUME_PATH}/results_detailed_{commodity.lower()}_{model_version}.pkl",
        'charts_dir': f"{VOLUME_PATH}/charts/{commodity.lower()}/{model_version}/"
    }


def get_model_versions(commodity, spark_session=None):
    """
    Discover available model versions for a commodity from forecast table

    Args:
        commodity: Commodity name (e.g., 'coffee')
        spark_session: SparkSession (optional, will create if not provided)

    Returns:
        list: List of model versions available
    """
    if spark_session is None:
        from pyspark.sql import SparkSession
        spark_session = SparkSession.builder.getOrCreate()

    query = f"""
    SELECT DISTINCT model_version
    FROM {FORECAST_TABLE}
    WHERE commodity = '{commodity.capitalize()}'
    AND is_actuals = FALSE
    ORDER BY model_version
    """

    result_df = spark_session.sql(query).toPandas()

    return result_df['model_version'].tolist()


def load_forecast_data(commodity, model_version, spark_session=None):
    """
    Load forecast prediction data from table for a specific commodity and model

    Args:
        commodity: Commodity name (e.g., 'coffee')
        model_version: Model version (e.g., 'arima_v1')
        spark_session: SparkSession (optional, will create if not provided)

    Returns:
        pd.DataFrame: Wide-format DataFrame with columns:
            - forecast_start_date
            - run_id
            - day_1, day_2, ..., day_14 (predicted prices)
    """
    if spark_session is None:
        from pyspark.sql import SparkSession
        spark_session = SparkSession.builder.getOrCreate()

    # Query to get predictions in wide format
    query = f"""
    SELECT
        forecast_start_date,
        run_id,
        MAX(CASE WHEN day_ahead = 1 THEN predicted_price END) as day_1,
        MAX(CASE WHEN day_ahead = 2 THEN predicted_price END) as day_2,
        MAX(CASE WHEN day_ahead = 3 THEN predicted_price END) as day_3,
        MAX(CASE WHEN day_ahead = 4 THEN predicted_price END) as day_4,
        MAX(CASE WHEN day_ahead = 5 THEN predicted_price END) as day_5,
        MAX(CASE WHEN day_ahead = 6 THEN predicted_price END) as day_6,
        MAX(CASE WHEN day_ahead = 7 THEN predicted_price END) as day_7,
        MAX(CASE WHEN day_ahead = 8 THEN predicted_price END) as day_8,
        MAX(CASE WHEN day_ahead = 9 THEN predicted_price END) as day_9,
        MAX(CASE WHEN day_ahead = 10 THEN predicted_price END) as day_10,
        MAX(CASE WHEN day_ahead = 11 THEN predicted_price END) as day_11,
        MAX(CASE WHEN day_ahead = 12 THEN predicted_price END) as day_12,
        MAX(CASE WHEN day_ahead = 13 THEN predicted_price END) as day_13,
        MAX(CASE WHEN day_ahead = 14 THEN predicted_price END) as day_14
    FROM {FORECAST_TABLE}
    WHERE commodity = '{commodity.capitalize()}'
        AND model_version = '{model_version}'
        AND is_actuals = FALSE
    GROUP BY forecast_start_date, run_id
    ORDER BY forecast_start_date, run_id
    """

    spark_df = spark_session.sql(query)
    predictions_wide = spark_df.toPandas()

    return predictions_wide


def load_price_data(commodity, start_date=None, spark_session=None):
    """
    Load historical price data for a commodity

    Args:
        commodity: Commodity name (e.g., 'coffee')
        start_date: Start date for prices (YYYY-MM-DD), None for all data
        spark_session: SparkSession (optional, will create if not provided)

    Returns:
        pd.DataFrame: DataFrame with columns 'date' and 'price'
    """
    if spark_session is None:
        from pyspark.sql import SparkSession
        spark_session = SparkSession.builder.getOrCreate()

    where_clause = f"WHERE commodity = '{commodity}'"
    if start_date:
        where_clause += f" AND date >= '{start_date}'"

    query = f"""
    SELECT date, price
    FROM {MARKET_TABLE}
    {where_clause}
    ORDER BY date
    """

    spark_df = spark_session.sql(query)
    prices_df = spark_df.toPandas()
    prices_df['date'] = pd.to_datetime(prices_df['date'])

    return prices_df


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'COMMODITY_CONFIGS',
    'BASELINE_PARAMS',
    'PREDICTION_PARAMS',
    'OUTPUT_SCHEMA',
    'MARKET_TABLE',
    'FORECAST_TABLE',
    'VOLUME_PATH',
    'ANALYSIS_CONFIG',
    'get_data_paths',
    'get_model_versions',
    'load_forecast_data',
    'load_price_data'
]

"""Pytest configuration and shared fixtures.

Provides SparkSession and test data for all unit tests.
"""

import pytest
from pyspark.sql import SparkSession
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture(scope="session")
def spark():
    """Create a local Spark session for testing."""
    spark = (SparkSession.builder
             .master("local[1]")
             .appName("forecast_agent_tests")
             .config("spark.sql.shuffle.partitions", "1")
             .getOrCreate())

    yield spark

    spark.stop()


@pytest.fixture
def sample_unified_data(spark):
    """Create sample unified data for testing.

    Returns Spark DataFrame with structure matching commodity.silver.unified_data:
    - 2 commodities (Coffee, Sugar)
    - 2 regions each
    - 10 days of data
    - All required columns
    """
    # Generate 10 days of data
    dates = [datetime(2024, 1, i) for i in range(1, 11)]

    data = []
    for date in dates:
        day_num = date.day

        # Coffee - Colombia
        data.append((
            date, 'Coffee', 'Colombia',
            185.0 + day_num * 0.5,  # close
            25.0 + day_num * 0.1,    # temp_c
            75.0 + day_num * 0.5,    # humidity_pct
            2.0 + day_num * 0.1,     # precipitation_mm
            15.0,                     # vix
            3800.0,                   # cop_usd
        ))

        # Coffee - Vietnam
        data.append((
            date, 'Coffee', 'Vietnam',
            185.0 + day_num * 0.5,  # close (same price)
            30.0 + day_num * 0.1,    # temp_c (warmer)
            80.0 + day_num * 0.5,    # humidity_pct (more humid)
            1.5 + day_num * 0.1,     # precipitation_mm
            15.0,                     # vix
            3800.0,                   # cop_usd
        ))

        # Sugar - Brazil
        data.append((
            date, 'Sugar', 'Brazil',
            22.0 + day_num * 0.05,   # close
            28.0 + day_num * 0.1,    # temp_c
            70.0 + day_num * 0.5,    # humidity_pct
            3.0 + day_num * 0.1,     # precipitation_mm
            15.0,                     # vix
            5.0,                      # brl_usd
        ))

        # Sugar - India
        data.append((
            date, 'Sugar', 'India',
            22.0 + day_num * 0.05,   # close (same price)
            32.0 + day_num * 0.1,    # temp_c (hotter)
            65.0 + day_num * 0.5,    # humidity_pct (less humid)
            2.0 + day_num * 0.1,     # precipitation_mm
            15.0,                     # vix
            83.0,                     # inr_usd
        ))

    schema = ['date', 'commodity', 'region', 'close', 'temp_c', 'humidity_pct',
              'precipitation_mm', 'vix', 'fx_rate']

    df = spark.createDataFrame(data, schema)

    return df


@pytest.fixture
def sample_pandas_data():
    """Create sample pandas DataFrame for testing model functions."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    df = pd.DataFrame({
        'date': dates,
        'close': 185.0 + pd.Series(range(100)) * 0.5,
        'temp_c': 25.0 + pd.Series(range(100)) * 0.1,
        'humidity_pct': 75.0 + pd.Series(range(100)) * 0.2,
        'precipitation_mm': 2.0 + pd.Series(range(100)) * 0.05
    })

    df = df.set_index('date')

    return df

"""GDELT sentiment integration for commodity forecasting.

Fetches and processes GDELT event data for Coffee and Sugar commodities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


def create_mock_gdelt_sentiment(start_date: str, end_date: str,
                                 commodity: str = 'Coffee') -> pd.DataFrame:
    """
    Create mock GDELT sentiment data for prototyping.

    In production, this would query GDELT BigQuery tables.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        commodity: 'Coffee' or 'Sugar'

    Returns:
        DataFrame with columns: date, commodity, avg_tone, event_count,
                               positive_events, negative_events, sentiment_score
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate realistic sentiment patterns
    np.random.seed(42 if commodity == 'Coffee' else 43)

    # Base sentiment with trend
    trend = np.linspace(-1, 1, len(date_range))

    # Seasonal component (yearly cycle)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365)

    # Random noise
    noise = np.random.normal(0, 1.5, len(date_range))

    # Combine components
    avg_tone = trend + seasonal + noise
    avg_tone = np.clip(avg_tone, -10, 10)  # GDELT tone ranges -10 to +10

    # Event counts (higher for Coffee, varies by day)
    base_events = 150 if commodity == 'Coffee' else 100
    event_count = np.random.poisson(base_events, len(date_range))

    # Positive/negative event split based on tone
    positive_pct = (avg_tone + 10) / 20  # Normalize to 0-1
    positive_events = (event_count * positive_pct).astype(int)
    negative_events = event_count - positive_events

    # Normalized sentiment score (-1 to +1)
    sentiment_score = avg_tone / 10

    df = pd.DataFrame({
        'date': date_range,
        'commodity': commodity,
        'avg_tone': avg_tone,
        'event_count': event_count,
        'positive_events': positive_events,
        'negative_events': negative_events,
        'sentiment_score': sentiment_score,
        'sentiment_volatility': pd.Series(sentiment_score).rolling(7).std().fillna(0).values,
        'sentiment_momentum': pd.Series(sentiment_score).diff(7).fillna(0).values
    })

    return df


def fetch_gdelt_sentiment_sql(commodity: str, start_date: str, end_date: str) -> str:
    """
    Generate SQL query for GDELT BigQuery.

    Example query structure for production deployment.

    Args:
        commodity: 'Coffee' or 'Sugar'
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD

    Returns:
        SQL query string
    """
    # Keyword mapping for GDELT search
    keywords = {
        'Coffee': ['coffee', 'arabica', 'robusta', 'coffee bean', 'coffee price',
                  'coffee export', 'coffee production', 'brazil coffee', 'colombia coffee'],
        'Sugar': ['sugar', 'cane sugar', 'sugar price', 'sugar export',
                 'sugar production', 'sugar market', 'brazil sugar', 'india sugar']
    }

    keyword_list = "', '".join(keywords.get(commodity, []))

    sql_query = f"""
    WITH gdelt_events AS (
        SELECT
            DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(SQLDATE AS STRING))) as event_date,
            AVG(CAST(AvgTone AS FLOAT64)) as avg_tone,
            COUNT(*) as event_count,
            SUM(CASE WHEN CAST(AvgTone AS FLOAT64) > 0 THEN 1 ELSE 0 END) as positive_events,
            SUM(CASE WHEN CAST(AvgTone AS FLOAT64) < 0 THEN 1 ELSE 0 END) as negative_events,
            AVG(CAST(GoldsteinScale AS FLOAT64)) as avg_goldstein
        FROM
            `gdelt-bq.gdeltv2.events`
        WHERE
            DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(SQLDATE AS STRING)))
                BETWEEN '{start_date}' AND '{end_date}'
            AND (
                LOWER(SOURCEURL) LIKE ANY (UNNEST(['{keyword_list}']))
                OR LOWER(Actor1Name) LIKE ANY (UNNEST(['{keyword_list}']))
                OR LOWER(Actor2Name) LIKE ANY (UNNEST(['{keyword_list}']))
            )
        GROUP BY
            event_date
    )
    SELECT
        event_date as date,
        '{commodity}' as commodity,
        avg_tone,
        event_count,
        positive_events,
        negative_events,
        avg_tone / 10.0 as sentiment_score,
        avg_goldstein as conflict_score
    FROM
        gdelt_events
    ORDER BY
        event_date
    """

    return sql_query


def process_gdelt_features(gdelt_df: pd.DataFrame,
                           rolling_windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
    """
    Process GDELT sentiment data into forecast features.

    Args:
        gdelt_df: Raw GDELT data with sentiment_score column
        rolling_windows: Windows for rolling statistics

    Returns:
        DataFrame with engineered sentiment features
    """
    df = gdelt_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Rolling sentiment statistics
    for window in rolling_windows:
        df[f'sentiment_ma_{window}'] = df['sentiment_score'].rolling(window).mean()
        df[f'sentiment_std_{window}'] = df['sentiment_score'].rolling(window).std()
        df[f'sentiment_min_{window}'] = df['sentiment_score'].rolling(window).min()
        df[f'sentiment_max_{window}'] = df['sentiment_score'].rolling(window).max()

    # Sentiment momentum (rate of change)
    df['sentiment_momentum_1d'] = df['sentiment_score'].diff(1)
    df['sentiment_momentum_7d'] = df['sentiment_score'].diff(7)

    # Sentiment acceleration (second derivative)
    df['sentiment_acceleration'] = df['sentiment_momentum_1d'].diff(1)

    # Event volume features
    df['event_volume_trend'] = df['event_count'].rolling(14).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0,
        raw=False
    )

    # Positive/negative ratio
    df['positive_ratio'] = df['positive_events'] / (df['event_count'] + 1)
    df['negative_ratio'] = df['negative_events'] / (df['event_count'] + 1)

    # Sentiment regime (categorical: very negative, negative, neutral, positive, very positive)
    df['sentiment_regime'] = pd.cut(
        df['sentiment_score'],
        bins=[-np.inf, -0.3, -0.1, 0.1, 0.3, np.inf],
        labels=['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
    )

    # Encode regime as dummies
    regime_dummies = pd.get_dummies(df['sentiment_regime'], prefix='regime')
    df = pd.concat([df, regime_dummies], axis=1)

    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')

    return df


def merge_gdelt_with_price_data(price_df: pd.DataFrame,
                                gdelt_df: pd.DataFrame,
                                commodity: str) -> pd.DataFrame:
    """
    Merge GDELT sentiment data with price data.

    Args:
        price_df: DataFrame with price data (must have 'date' and 'commodity')
        gdelt_df: Processed GDELT sentiment data
        commodity: Commodity to merge for

    Returns:
        Merged DataFrame with both price and sentiment features
    """
    # Filter for specific commodity
    price_commodity = price_df[price_df['commodity'] == commodity].copy()
    gdelt_commodity = gdelt_df[gdelt_df['commodity'] == commodity].copy()

    # Ensure date columns are datetime
    price_commodity['date'] = pd.to_datetime(price_commodity['date'])
    gdelt_commodity['date'] = pd.to_datetime(gdelt_commodity['date'])

    # Merge on date
    merged = pd.merge(
        price_commodity,
        gdelt_commodity,
        on=['date', 'commodity'],
        how='left'
    )

    # Forward fill missing sentiment data (weekends, holidays)
    sentiment_cols = [col for col in merged.columns if 'sentiment' in col or
                     'event' in col or 'positive' in col or 'negative' in col or
                     'regime' in col]

    merged[sentiment_cols] = merged[sentiment_cols].fillna(method='ffill')

    return merged


def get_sentiment_feature_names() -> List[str]:
    """
    Get list of all sentiment feature names.

    Useful for model feature selection.
    """
    features = ['sentiment_score', 'avg_tone', 'event_count',
               'positive_events', 'negative_events',
               'sentiment_momentum_1d', 'sentiment_momentum_7d',
               'sentiment_acceleration', 'event_volume_trend',
               'positive_ratio', 'negative_ratio']

    # Rolling features
    for window in [7, 14, 30]:
        features.extend([
            f'sentiment_ma_{window}',
            f'sentiment_std_{window}',
            f'sentiment_min_{window}',
            f'sentiment_max_{window}'
        ])

    # Regime dummies
    features.extend([
        'regime_very_negative', 'regime_negative', 'regime_neutral',
        'regime_positive', 'regime_very_positive'
    ])

    return features


# Example usage documentation
EXAMPLE_USAGE = """
# Example: Fetch and process GDELT sentiment

from ground_truth.features.gdelt_sentiment import (
    create_mock_gdelt_sentiment,
    process_gdelt_features,
    merge_gdelt_with_price_data
)

# 1. Create mock GDELT data (or fetch from BigQuery in production)
gdelt_raw = create_mock_gdelt_sentiment(
    start_date='2015-01-01',
    end_date='2024-12-31',
    commodity='Coffee'
)

# 2. Process into features
gdelt_processed = process_gdelt_features(gdelt_raw)

# 3. Merge with price data
merged_data = merge_gdelt_with_price_data(
    price_df=df_prices,
    gdelt_df=gdelt_processed,
    commodity='Coffee'
)

# 4. Use in forecast models
from ground_truth.models.xgboost_model import xgboost_forecast_with_metadata

result = xgboost_forecast_with_metadata(
    df_pandas=merged_data,
    commodity='Coffee',
    target='close',
    exog_features=['temp_c', 'sentiment_score', 'sentiment_ma_7', 'event_count'],
    horizon=14
)
"""

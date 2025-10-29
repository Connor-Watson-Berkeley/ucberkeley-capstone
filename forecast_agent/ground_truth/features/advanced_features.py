"""Advanced feature engineering for time series forecasting.

Technical indicators, Fourier features, and interaction terms.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def add_technical_indicators(df: pd.DataFrame, target: str = 'close',
                             periods: List[int] = [14, 30, 60]) -> pd.DataFrame:
    """
    Add technical trading indicators.

    Features:
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands
        - Price momentum
    """
    df_out = df.copy()

    for period in periods:
        # RSI (Relative Strength Index)
        delta = df[target].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df_out[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Price momentum
        df_out[f'momentum_{period}'] = df[target].pct_change(periods=period)

        # Volatility (std of returns)
        df_out[f'volatility_{period}'] = df[target].pct_change().rolling(window=period).std()

    # MACD (12, 26, 9 standard)
    exp1 = df[target].ewm(span=12, adjust=False).mean()
    exp2 = df[target].ewm(span=26, adjust=False).mean()
    df_out['macd'] = exp1 - exp2
    df_out['macd_signal'] = df_out['macd'].ewm(span=9, adjust=False).mean()
    df_out['macd_hist'] = df_out['macd'] - df_out['macd_signal']

    # Bollinger Bands (20-day standard)
    rolling_mean = df[target].rolling(window=20).mean()
    rolling_std = df[target].rolling(window=20).std()
    df_out['bb_upper'] = rolling_mean + (2 * rolling_std)
    df_out['bb_lower'] = rolling_mean - (2 * rolling_std)
    df_out['bb_width'] = df_out['bb_upper'] - df_out['bb_lower']
    df_out['bb_position'] = (df[target] - df_out['bb_lower']) / df_out['bb_width']

    return df_out


def add_fourier_features(df: pd.DataFrame, n_harmonics: int = 3,
                         period: int = 365) -> pd.DataFrame:
    """
    Add Fourier features for seasonality.

    Args:
        df: DataFrame with DatetimeIndex
        n_harmonics: Number of harmonics (sin/cos pairs)
        period: Seasonality period in days (365 for yearly)
    """
    df_out = df.copy()

    # Day of year (0-364)
    day_of_year = df.index.dayofyear

    for i in range(1, n_harmonics + 1):
        df_out[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * day_of_year / period)
        df_out[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * day_of_year / period)

    return df_out


def add_interaction_features(df: pd.DataFrame,
                            feature_pairs: List[tuple] = None) -> pd.DataFrame:
    """
    Add interaction terms between features.

    Args:
        df: DataFrame with features
        feature_pairs: List of (feature1, feature2) tuples to interact
    """
    df_out = df.copy()

    if feature_pairs is None:
        # Default interactions: weather features
        if 'temp_c' in df.columns and 'humidity_pct' in df.columns:
            df_out['temp_humidity'] = df['temp_c'] * df['humidity_pct']

        if 'temp_c' in df.columns and 'precipitation_mm' in df.columns:
            df_out['temp_precip'] = df['temp_c'] * df['precipitation_mm']
    else:
        for f1, f2 in feature_pairs:
            if f1 in df.columns and f2 in df.columns:
                df_out[f'{f1}_x_{f2}'] = df[f1] * df[f2]

    return df_out


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical encodings of time features.

    Features:
        - Day of week (sin/cos)
        - Day of month (sin/cos)
        - Day of year (sin/cos)
        - Week of year (sin/cos)
    """
    df_out = df.copy()

    # Day of week (0-6)
    df_out['dayofweek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df_out['dayofweek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # Day of month (1-31)
    df_out['dayofmonth_sin'] = np.sin(2 * np.pi * df.index.day / 31)
    df_out['dayofmonth_cos'] = np.cos(2 * np.pi * df.index.day / 31)

    # Day of year (1-365)
    df_out['dayofyear_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df_out['dayofyear_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)

    # Week of year (1-52)
    df_out['weekofyear_sin'] = np.sin(2 * np.pi * df.index.isocalendar().week / 52)
    df_out['weekofyear_cos'] = np.cos(2 * np.pi * df.index.isocalendar().week / 52)

    return df_out


def add_price_patterns(df: pd.DataFrame, target: str = 'close',
                       windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    Add price pattern features.

    Features:
        - Distance from MA
        - Price channels (high/low ranges)
        - Trend strength
    """
    df_out = df.copy()

    for window in windows:
        # Distance from moving average
        ma = df[target].rolling(window=window).mean()
        df_out[f'dist_from_ma_{window}'] = (df[target] - ma) / ma

        # Price channel position
        rolling_max = df[target].rolling(window=window).max()
        rolling_min = df[target].rolling(window=window).min()
        price_range = rolling_max - rolling_min
        df_out[f'channel_position_{window}'] = (df[target] - rolling_min) / (price_range + 1e-8)

        # Trend strength (slope of linear regression)
        def trend_slope(series):
            if len(series) < 2:
                return np.nan
            x = np.arange(len(series))
            return np.polyfit(x, series.values, 1)[0]

        df_out[f'trend_slope_{window}'] = df[target].rolling(window=window).apply(trend_slope, raw=False)

    return df_out


def create_advanced_features(df: pd.DataFrame, target: str = 'close',
                            include_technical: bool = True,
                            include_fourier: bool = True,
                            include_cyclical: bool = True,
                            include_patterns: bool = True,
                            include_interactions: bool = True) -> pd.DataFrame:
    """
    Create comprehensive advanced feature set.

    Args:
        df: DataFrame with DatetimeIndex
        target: Target column
        include_*: Flags for feature groups

    Returns:
        DataFrame with all advanced features
    """
    df_out = df.copy()

    if include_technical:
        df_out = add_technical_indicators(df_out, target)

    if include_fourier:
        df_out = add_fourier_features(df_out, n_harmonics=3)

    if include_cyclical:
        df_out = add_cyclical_time_features(df_out)

    if include_patterns:
        df_out = add_price_patterns(df_out, target)

    if include_interactions:
        df_out = add_interaction_features(df_out)

    # Fill NaN values created by rolling windows
    df_out = df_out.fillna(method='bfill').fillna(method='ffill')

    return df_out


def get_advanced_feature_names(target: str = 'close') -> List[str]:
    """
    Get list of all advanced feature names that will be created.

    Useful for feature selection in models.
    """
    features = []

    # Technical indicators
    for period in [14, 30, 60]:
        features.extend([
            f'rsi_{period}',
            f'momentum_{period}',
            f'volatility_{period}'
        ])

    features.extend(['macd', 'macd_signal', 'macd_hist',
                    'bb_upper', 'bb_lower', 'bb_width', 'bb_position'])

    # Fourier features
    for i in range(1, 4):
        features.extend([f'fourier_sin_{i}', f'fourier_cos_{i}'])

    # Cyclical time features
    features.extend([
        'dayofweek_sin', 'dayofweek_cos',
        'dayofmonth_sin', 'dayofmonth_cos',
        'dayofyear_sin', 'dayofyear_cos',
        'weekofyear_sin', 'weekofyear_cos'
    ])

    # Price patterns
    for window in [5, 10, 20]:
        features.extend([
            f'dist_from_ma_{window}',
            f'channel_position_{window}',
            f'trend_slope_{window}'
        ])

    # Interactions (if weather data present)
    features.extend(['temp_humidity', 'temp_precip'])

    return features

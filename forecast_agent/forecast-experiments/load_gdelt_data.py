"""
Load and Merge GDELT Sentiment Data

Downloads GDELT sentiment data and merges with price/weather data.
Creates enhanced dataset with sentiment features as additional covariates.
"""

import os
import pandas as pd
from databricks import sql


def download_gdelt_data(commodity='Coffee', output_file='data/gdelt_coffee.parquet'):
    """
    Download GDELT sentiment data from Databricks.

    Args:
        commodity: Commodity to filter on (default: 'Coffee')
        output_file: Local parquet file path
    """
    print(f"Downloading GDELT sentiment data for {commodity}...")

    conn = sql.connect(
        server_hostname=os.environ['DATABRICKS_HOST'],
        http_path=os.environ['DATABRICKS_HTTP_PATH'],
        access_token=os.environ['DATABRICKS_TOKEN']
    )

    cursor = conn.cursor()

    # Query GDELT data (use lowercase commodity name)
    commodity_lower = commodity.lower()
    # Use explicit CAST to avoid Parquet type mismatch errors
    query = f"""
        SELECT
            article_date as date,
            commodity,
            -- Supply chain sentiment
            CAST(group_SUPPLY_count AS BIGINT) as group_SUPPLY_count,
            CAST(group_SUPPLY_tone_avg AS DOUBLE) as group_SUPPLY_tone_avg,
            CAST(group_SUPPLY_tone_positive AS DOUBLE) as group_SUPPLY_tone_positive,
            CAST(group_SUPPLY_tone_negative AS DOUBLE) as group_SUPPLY_tone_negative,
            CAST(group_SUPPLY_tone_polarity AS DOUBLE) as group_SUPPLY_tone_polarity,
            -- Logistics sentiment
            CAST(group_LOGISTICS_count AS BIGINT) as group_LOGISTICS_count,
            CAST(group_LOGISTICS_tone_avg AS DOUBLE) as group_LOGISTICS_tone_avg,
            CAST(group_LOGISTICS_tone_positive AS DOUBLE) as group_LOGISTICS_tone_positive,
            CAST(group_LOGISTICS_tone_negative AS DOUBLE) as group_LOGISTICS_tone_negative,
            CAST(group_LOGISTICS_tone_polarity AS DOUBLE) as group_LOGISTICS_tone_polarity,
            -- Trade sentiment
            CAST(group_TRADE_count AS BIGINT) as group_TRADE_count,
            CAST(group_TRADE_tone_avg AS DOUBLE) as group_TRADE_tone_avg,
            CAST(group_TRADE_tone_positive AS DOUBLE) as group_TRADE_tone_positive,
            CAST(group_TRADE_tone_negative AS DOUBLE) as group_TRADE_tone_negative,
            CAST(group_TRADE_tone_polarity AS DOUBLE) as group_TRADE_tone_polarity,
            -- Market sentiment
            CAST(group_MARKET_count AS BIGINT) as group_MARKET_count,
            CAST(group_MARKET_tone_avg AS DOUBLE) as group_MARKET_tone_avg,
            CAST(group_MARKET_tone_positive AS DOUBLE) as group_MARKET_tone_positive,
            CAST(group_MARKET_tone_negative AS DOUBLE) as group_MARKET_tone_negative,
            CAST(group_MARKET_tone_polarity AS DOUBLE) as group_MARKET_tone_polarity,
            -- Policy sentiment
            CAST(group_POLICY_count AS BIGINT) as group_POLICY_count,
            CAST(group_POLICY_tone_avg AS DOUBLE) as group_POLICY_tone_avg,
            CAST(group_POLICY_tone_positive AS DOUBLE) as group_POLICY_tone_positive,
            CAST(group_POLICY_tone_negative AS DOUBLE) as group_POLICY_tone_negative,
            CAST(group_POLICY_tone_polarity AS DOUBLE) as group_POLICY_tone_polarity,
            -- Core commodity sentiment
            CAST(group_CORE_count AS BIGINT) as group_CORE_count,
            CAST(group_CORE_tone_avg AS DOUBLE) as group_CORE_tone_avg,
            CAST(group_CORE_tone_positive AS DOUBLE) as group_CORE_tone_positive,
            CAST(group_CORE_tone_negative AS DOUBLE) as group_CORE_tone_negative,
            CAST(group_CORE_tone_polarity AS DOUBLE) as group_CORE_tone_polarity,
            -- Other sentiment
            CAST(group_OTHER_count AS BIGINT) as group_OTHER_count,
            CAST(group_OTHER_tone_avg AS DOUBLE) as group_OTHER_tone_avg,
            CAST(group_OTHER_tone_positive AS DOUBLE) as group_OTHER_tone_positive,
            CAST(group_OTHER_tone_negative AS DOUBLE) as group_OTHER_tone_negative,
            CAST(group_OTHER_tone_polarity AS DOUBLE) as group_OTHER_tone_polarity
        FROM commodity.silver.gdelt_wide
        WHERE commodity = '{commodity_lower}'
        ORDER BY article_date
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    df = pd.DataFrame.from_records(rows, columns=columns)

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Fill NULLs with 0 (no news = neutral sentiment)
    sentiment_cols = [col for col in df.columns if col not in ['date', 'commodity']]
    df[sentiment_cols] = df[sentiment_cols].fillna(0)

    print(f"Downloaded {len(df):,} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()

    # Save to parquet
    os.makedirs('data', exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"Saved to {output_file}")

    conn.close()

    return df


def merge_with_unified_data(
    unified_data_file='data/unified_data.parquet',
    gdelt_file='data/gdelt_coffee.parquet',
    output_file='data/unified_with_sentiment.parquet',
    region='Bahia_Brazil'
):
    """
    Merge GDELT sentiment data with unified price/weather data.

    Args:
        unified_data_file: Path to unified_data parquet
        gdelt_file: Path to GDELT parquet
        output_file: Output file path
        region: Region to filter (if None, aggregates)

    Returns:
        Merged DataFrame
    """
    print("=" * 80)
    print("Merging GDELT Sentiment with Price/Weather Data")
    print("=" * 80)
    print()

    # Load unified data
    print(f"Loading unified data from {unified_data_file}...")
    df_unified = pd.read_parquet(unified_data_file)

    # Filter to commodity and region
    df_unified = df_unified[df_unified['commodity'] == 'Coffee']
    if region:
        df_unified = df_unified[df_unified['region'] == region]
        print(f"Filtered to Coffee, {region}")
    else:
        # Aggregate across regions
        df_unified = df_unified.groupby('date').agg({
            'close': 'mean',
            'volume': 'sum',
            'temp_mean_c': 'mean',
            'precipitation_mm': 'mean',
            'wind_speed_max_kmh': 'mean',
            'humidity_mean_pct': 'mean',
            'temp_max_c': 'mean',
            'temp_min_c': 'mean',
            'rain_mm': 'mean'
        }).reset_index()
        print("Aggregated across all regions")

    print(f"Unified data: {len(df_unified):,} rows")
    print()

    # Load GDELT data
    print(f"Loading GDELT data from {gdelt_file}...")
    df_gdelt = pd.read_parquet(gdelt_file)
    print(f"GDELT data: {len(df_gdelt):,} rows")
    print()

    # Merge on date
    print("Merging on date...")
    df_merged = df_unified.merge(
        df_gdelt.drop(columns=['commodity'], errors='ignore'),
        on='date',
        how='left'
    )

    # Fill missing sentiment values with 0 (no news on that day)
    sentiment_cols = [col for col in df_gdelt.columns if col not in ['date', 'commodity']]
    df_merged[sentiment_cols] = df_merged[sentiment_cols].fillna(0)

    print(f"Merged data: {len(df_merged):,} rows")
    print(f"Date range: {df_merged['date'].min()} to {df_merged['date'].max()}")
    print()

    # Summary statistics
    print("Sentiment Coverage:")
    for group in ['SUPPLY', 'LOGISTICS', 'TRADE', 'MARKET', 'POLICY', 'CORE', 'OTHER']:
        count_col = f'group_{group}_count'
        if count_col in df_merged.columns:
            pct_with_news = (df_merged[count_col] > 0).mean() * 100
            avg_articles = df_merged[df_merged[count_col] > 0][count_col].mean()
            print(f"  {group:<12}: {pct_with_news:.1f}% days with news (avg {avg_articles:.1f} articles/day)")
    print()

    # Save merged data
    os.makedirs('data', exist_ok=True)
    df_merged.to_parquet(output_file, index=False)
    print(f"Saved to {output_file}")
    print()

    return df_merged


def get_sentiment_feature_names():
    """Get list of all GDELT sentiment feature names for use in model training."""
    groups = ['SUPPLY', 'LOGISTICS', 'TRADE', 'MARKET', 'POLICY', 'CORE', 'OTHER']
    metrics = ['count', 'tone_avg', 'tone_positive', 'tone_negative', 'tone_polarity']

    features = []
    for group in groups:
        for metric in metrics:
            features.append(f'group_{group}_{metric}')

    return features


if __name__ == '__main__':
    import sys

    print("\n" + "=" * 80)
    print("GDELT SENTIMENT DATA LOADER")
    print("=" * 80)
    print()

    # Download GDELT data
    try:
        df_gdelt = download_gdelt_data(commodity='Coffee')
        print()
    except Exception as e:
        print(f"❌ Error downloading GDELT data: {e}")
        print("\nMake sure you have SELECT permissions on commodity.silver.gdelt_wide")
        sys.exit(1)

    # Merge with unified data
    try:
        df_merged = merge_with_unified_data(region='Bahia_Brazil')
    except Exception as e:
        print(f"❌ Error merging data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 80)
    print("DATA LOADING COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Use 'data/unified_with_sentiment.parquet' for training")
    print("2. Add sentiment features to model covariates:")
    print()
    print("   from load_gdelt_data import get_sentiment_feature_names")
    print("   sentiment_features = get_sentiment_feature_names()")
    print("   all_features = weather_features + sentiment_features")
    print()

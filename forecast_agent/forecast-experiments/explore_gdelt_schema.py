"""
Explore GDELT Sentiment Data Schema

Investigate commodity.silver GDELT tables to understand:
- Table structure
- Available sentiment metrics
- Date coverage
- Commodity/topic coverage
- Best features for forecasting
"""

import os
from databricks import sql
import pandas as pd


def get_databricks_connection():
    """Create Databricks connection."""
    return sql.connect(
        server_hostname=os.environ['DATABRICKS_HOST'],
        http_path=os.environ['DATABRICKS_HTTP_PATH'],
        access_token=os.environ['DATABRICKS_TOKEN']
    )


def explore_gdelt_tables():
    """Find all GDELT-related tables in commodity.silver."""
    with get_databricks_connection() as conn:
        cursor = conn.cursor()

        # Find tables matching GDELT pattern
        cursor.execute("""
            SHOW TABLES IN commodity.silver LIKE '*gdelt*'
        """)

        tables = cursor.fetchall()
        print("=" * 80)
        print("GDELT Tables in commodity.silver")
        print("=" * 80)
        for table in tables:
            print(f"  - {table[1]}")

        print()
        return [table[1] for table in tables]


def explore_table_schema(table_name):
    """Get schema and sample data for a GDELT table."""
    with get_databricks_connection() as conn:
        cursor = conn.cursor()

        full_table_name = f"commodity.silver.{table_name}"

        print("=" * 80)
        print(f"Table: {full_table_name}")
        print("=" * 80)
        print()

        # Get schema
        print("Schema:")
        cursor.execute(f"DESCRIBE {full_table_name}")
        schema = cursor.fetchall()
        for col in schema:
            print(f"  - {col[0]:<30} {col[1]:<20} {col[2]}")
        print()

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {full_table_name}")
        count = cursor.fetchone()[0]
        print(f"Total rows: {count:,}")
        print()

        # Get date range
        cursor.execute(f"""
            SELECT
                MIN(date) as min_date,
                MAX(date) as max_date,
                COUNT(DISTINCT date) as unique_dates
            FROM {full_table_name}
            WHERE date IS NOT NULL
        """)
        date_info = cursor.fetchone()
        if date_info:
            print(f"Date range: {date_info[0]} to {date_info[1]}")
            print(f"Unique dates: {date_info[2]:,}")
            print()

        # Sample data
        print("Sample data (first 5 rows):")
        cursor.execute(f"SELECT * FROM {full_table_name} LIMIT 5")
        df_sample = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        for row in df_sample:
            print("\n" + "-" * 80)
            for col, val in zip(columns, row):
                print(f"  {col}: {val}")

        print("\n" + "=" * 80)
        print()


def analyze_sentiment_features(table_name):
    """Analyze sentiment features for coffee forecasting."""
    with get_databricks_connection() as conn:
        cursor = conn.cursor()

        full_table_name = f"commodity.silver.{table_name}"

        print("=" * 80)
        print(f"Sentiment Feature Analysis: {table_name}")
        print("=" * 80)
        print()

        # Check if commodity column exists
        cursor.execute(f"DESCRIBE {full_table_name}")
        schema = cursor.fetchall()
        column_names = [col[0].lower() for col in schema]

        # Look for sentiment-related columns
        sentiment_cols = [col for col in column_names if any(
            keyword in col for keyword in ['sentiment', 'tone', 'goldstein', 'score', 'avg']
        )]

        print("Sentiment-related columns:")
        for col in sentiment_cols:
            print(f"  - {col}")
        print()

        # If commodity column exists, check Coffee coverage
        if 'commodity' in column_names:
            cursor.execute(f"""
                SELECT
                    commodity,
                    COUNT(*) as num_records,
                    COUNT(DISTINCT date) as num_dates
                FROM {full_table_name}
                GROUP BY commodity
                ORDER BY num_records DESC
            """)
            print("Records by commodity:")
            for row in cursor.fetchall():
                print(f"  - {row[0]}: {row[1]:,} records across {row[2]:,} dates")
            print()

            # Coffee-specific stats
            if sentiment_cols:
                print("Coffee sentiment statistics:")
                stats_query = f"""
                    SELECT
                        COUNT(*) as num_records,
                        COUNT(DISTINCT date) as num_dates,
                        MIN(date) as earliest_date,
                        MAX(date) as latest_date
                        {', AVG(' + sentiment_cols[0] + ') as avg_sentiment' if sentiment_cols else ''}
                    FROM {full_table_name}
                    WHERE commodity = 'Coffee'
                """
                cursor.execute(stats_query)
                stats = cursor.fetchone()
                print(f"  Records: {stats[0]:,}")
                print(f"  Unique dates: {stats[1]:,}")
                print(f"  Date range: {stats[2]} to {stats[3]}")
                if len(stats) > 4:
                    print(f"  Avg sentiment: {stats[4]:.4f}")
                print()


def main():
    """Main exploration function."""
    print("\n" + "=" * 80)
    print("GDELT SENTIMENT DATA EXPLORATION")
    print("=" * 80)
    print()

    # Find GDELT tables
    tables = explore_gdelt_tables()

    if not tables:
        print("No GDELT tables found in commodity.silver")
        return

    # Explore each table
    for table in tables:
        explore_table_schema(table)
        analyze_sentiment_features(table)

    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Identify which sentiment features are most relevant")
    print("2. Download GDELT data to local cache")
    print("3. Join with price/weather data in unified_data")
    print("4. Retrain models with sentiment features as covariates")
    print()


if __name__ == '__main__':
    main()

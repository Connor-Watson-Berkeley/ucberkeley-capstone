"""
Create Weather v2 Bronze Table using COPY INTO (more efficient)

This approach uses COPY INTO instead of INSERT OVERWRITE, which is
optimized for bulk loading from S3 and should be much faster.
"""

import os
import sys
from databricks import sql
from dotenv import load_dotenv

# Load credentials
load_dotenv()
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")

if not all([DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH]):
    print("ERROR: Missing environment variables!")
    exit(1)


def create_table_with_copy_into():
    """Create weather_v2 table using COPY INTO for efficient loading"""
    print("=" * 80)
    print("Creating commodity.bronze.weather_v2 with COPY INTO")
    print("=" * 80)

    connection = sql.connect(
        server_hostname=DATABRICKS_HOST.replace("https://", ""),
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )
    cursor = connection.cursor()

    # Step 1: Create table structure
    print("\nStep 1: Creating table structure...")
    create_sql = """
    CREATE TABLE IF NOT EXISTS commodity.bronze.weather_v2 (
        region STRING,
        commodity STRING,
        country STRING,
        date DATE,
        latitude DOUBLE,
        longitude DOUBLE,
        elevation_m DOUBLE,
        temp_max_c DOUBLE,
        temp_min_c DOUBLE,
        temp_mean_c DOUBLE,
        precipitation_mm DOUBLE,
        rain_mm DOUBLE,
        snowfall_cm DOUBLE,
        wind_speed_max_kmh DOUBLE,
        humidity_mean_pct DOUBLE,
        description STRING,
        data_version STRING,
        coordinate_source STRING,
        ingest_ts TIMESTAMP,
        year INT,
        month INT,
        day INT
    )
    USING DELTA
    PARTITIONED BY (year, month, day)
    LOCATION 's3://groundtruth-capstone/delta/bronze/weather_v2/'
    """

    try:
        cursor.execute(create_sql)
        print("✅ Table created")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("⚠️  Table already exists")
        else:
            print(f"❌ Error: {e}")
            return False

    # Step 2: Use COPY INTO to load data
    print("\nStep 2: Loading data with COPY INTO...")
    print("  This is more efficient than INSERT OVERWRITE")
    print("  Source: s3://groundtruth-capstone/landing/weather_v2/")

    copy_into_sql = """
    COPY INTO commodity.bronze.weather_v2
    FROM (
        SELECT
            region,
            commodity,
            country,
            CAST(date AS DATE) as date,
            CAST(latitude AS DOUBLE) as latitude,
            CAST(longitude AS DOUBLE) as longitude,
            CAST(elevation_m AS DOUBLE) as elevation_m,
            CAST(temp_max_c AS DOUBLE) as temp_max_c,
            CAST(temp_min_c AS DOUBLE) as temp_min_c,
            CAST(temp_mean_c AS DOUBLE) as temp_mean_c,
            CAST(precipitation_mm AS DOUBLE) as precipitation_mm,
            CAST(rain_mm AS DOUBLE) as rain_mm,
            CAST(snowfall_cm AS DOUBLE) as snowfall_cm,
            CAST(wind_speed_max_kmh AS DOUBLE) as wind_speed_max_kmh,
            CAST(humidity_mean_pct AS DOUBLE) as humidity_mean_pct,
            description,
            data_version,
            coordinate_source,
            CAST(ingest_ts AS TIMESTAMP) as ingest_ts,
            CAST(YEAR(CAST(date AS DATE)) AS INT) as year,
            CAST(MONTH(CAST(date AS DATE)) AS INT) as month,
            CAST(DAY(CAST(date AS DATE)) AS INT) as day
        FROM 's3://groundtruth-capstone/landing/weather_v2/'
    )
    FILEFORMAT = JSON
    FORMAT_OPTIONS ('recursiveFileLookup' = 'true')
    COPY_OPTIONS ('mergeSchema' = 'false')
    """

    try:
        print("  Running COPY INTO (this may take 5-10 minutes)...")
        cursor.execute(copy_into_sql)
        result = cursor.fetchall()
        print("✅ Data loaded successfully")
        if result:
            print(f"  Loaded {result[0][0] if result[0] else 'unknown'} rows")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        cursor.close()
        connection.close()
        return False

    # Step 3: Optimize
    print("\nStep 3: Optimizing table...")
    try:
        cursor.execute("OPTIMIZE commodity.bronze.weather_v2 ZORDER BY (region, date)")
        print("✅ Table optimized")
    except Exception as e:
        print(f"⚠️  Optimization warning: {e}")

    # Step 4: Statistics
    print("\nStep 4: Computing statistics...")
    try:
        cursor.execute("ANALYZE TABLE commodity.bronze.weather_v2 COMPUTE STATISTICS")
        print("✅ Statistics computed")
    except Exception as e:
        print(f"⚠️  Statistics warning: {e}")

    # Step 5: Verify
    print("\nStep 5: Verifying data...")

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(DISTINCT region) as regions,
            MIN(date) as earliest,
            MAX(date) as latest
        FROM commodity.bronze.weather_v2
    """)
    row = cursor.fetchone()

    print(f"  Total records: {row.total:,}")
    print(f"  Distinct regions: {row.regions}")
    print(f"  Date range: {row.earliest} to {row.latest}")

    # Sample
    cursor.execute("""
        SELECT region, date, temp_min_c, temp_max_c, latitude, longitude
        FROM commodity.bronze.weather_v2
        WHERE region LIKE '%Minas%'
        ORDER BY date DESC
        LIMIT 1
    """)
    sample = cursor.fetchone()
    if sample:
        print(f"\n  Sample record (Sul de Minas):")
        print(f"    Region: {sample.region}")
        print(f"    Date: {sample.date}")
        print(f"    Temp: {sample.temp_min_c}°C to {sample.temp_max_c}°C")
        print(f"    Coordinates: ({sample.latitude}, {sample.longitude})")

    print("\n" + "=" * 80)
    print("✅ SUCCESS!")
    print("=" * 80)
    print(f"Weather v2 bronze table created with {row.total:,} records")
    print("\nNext steps:")
    print("  1. Run: python tests/validate_july2021_frost.py")
    print("  2. Update unified_data to use weather_v2")
    print("=" * 80)

    cursor.close()
    connection.close()
    return True


if __name__ == "__main__":
    success = create_table_with_copy_into()
    sys.exit(0 if success else 1)

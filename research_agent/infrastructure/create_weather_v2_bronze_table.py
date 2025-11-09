"""
Create Weather v2 Bronze Table from S3 Landing Data

This script creates the commodity.bronze.weather_v2 table by ingesting
the corrected weather data from s3://groundtruth-capstone/landing/weather_v2/

Key differences from v1:
- Correct coordinates for actual growing regions (not state capitals)
- Includes latitude/longitude columns for transparency
- Higher quality data for model training
"""

import os
from databricks import sql
from dotenv import load_dotenv

# Load credentials from .env (NO HARDCODED CREDENTIALS!)
load_dotenv()
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")

if not all([DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH]):
    print("ERROR: Missing environment variables!")
    print("Create research_agent/infrastructure/.env with credentials")
    exit(1)


def create_weather_v2_table():
    """Create weather_v2 bronze table from S3 landing data"""
    print("=" * 80)
    print("Creating commodity.bronze.weather_v2 Table")
    print("=" * 80)

    connection = sql.connect(
        server_hostname=DATABRICKS_HOST.replace("https://", ""),
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )
    cursor = connection.cursor()

    # Step 1: Create external table pointing to S3
    print("\nStep 1: Creating external table from S3 landing data...")
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS commodity.bronze.weather_v2 (
        region STRING COMMENT 'Coffee growing region name',
        commodity STRING COMMENT 'Commodity type (Coffee, etc.)',
        country STRING COMMENT 'Country name',
        date DATE COMMENT 'Date of weather observation',
        latitude DOUBLE COMMENT 'Latitude of weather observation point',
        longitude DOUBLE COMMENT 'Longitude of weather observation point',
        elevation_m DOUBLE COMMENT 'Elevation in meters',
        temp_max_c DOUBLE COMMENT 'Maximum temperature (Celsius)',
        temp_min_c DOUBLE COMMENT 'Minimum temperature (Celsius)',
        temp_mean_c DOUBLE COMMENT 'Mean temperature (Celsius)',
        precipitation_mm DOUBLE COMMENT 'Total precipitation (millimeters)',
        rain_mm DOUBLE COMMENT 'Total rainfall (millimeters)',
        snowfall_cm DOUBLE COMMENT 'Total snowfall (centimeters)',
        wind_speed_max_kmh DOUBLE COMMENT 'Maximum wind speed (km/h)',
        humidity_mean_pct DOUBLE COMMENT 'Mean relative humidity (percent)',
        description STRING COMMENT 'Region description',
        data_version STRING COMMENT 'Data version (v2_corrected_coordinates)',
        coordinate_source STRING COMMENT 'Source of coordinates',
        ingest_ts TIMESTAMP COMMENT 'Timestamp when data was ingested',
        year INT COMMENT 'Year (partition column)',
        month INT COMMENT 'Month (partition column)',
        day INT COMMENT 'Day (partition column)'
    )
    USING DELTA
    PARTITIONED BY (year, month, day)
    LOCATION 's3://groundtruth-capstone/delta/bronze/weather_v2/'
    COMMENT 'Weather data v2 with corrected growing region coordinates - Bronze layer'
    """

    try:
        cursor.execute(create_table_sql)
        print("✅ Table structure created")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("⚠️  Table already exists")
        else:
            print(f"❌ Error creating table: {e}")
            cursor.close()
            connection.close()
            return False

    # Step 2: Load data from S3 landing zone
    print("\nStep 2: Loading data from S3 landing zone...")
    print("  Source: s3://groundtruth-capstone/landing/weather_v2/")
    print("  Target: commodity.bronze.weather_v2")

    # Use SQL to read JSON and insert into Delta table
    insert_sql = """
    INSERT OVERWRITE TABLE commodity.bronze.weather_v2
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
    FROM read_files('s3://groundtruth-capstone/landing/weather_v2/*/*/*/*/*.jsonl', format => 'json')
    """

    try:
        print("  Running INSERT OVERWRITE (this may take several minutes)...")
        cursor.execute(insert_sql)
        print("✅ Data loaded successfully")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        cursor.close()
        connection.close()
        return False

    # Step 3: Optimize table and gather statistics
    print("\nStep 3: Optimizing table...")
    optimize_sql = "OPTIMIZE commodity.bronze.weather_v2 ZORDER BY (region, date)"

    try:
        cursor.execute(optimize_sql)
        print("✅ Table optimized with Z-ORDER")
    except Exception as e:
        print(f"⚠️  Optimization warning: {e}")

    # Step 4: Gather table statistics
    print("\nStep 4: Gathering table statistics...")
    stats_sql = "ANALYZE TABLE commodity.bronze.weather_v2 COMPUTE STATISTICS"

    try:
        cursor.execute(stats_sql)
        print("✅ Statistics computed")
    except Exception as e:
        print(f"⚠️  Statistics warning: {e}")

    # Step 5: Verify data
    print("\nStep 5: Verifying loaded data...")

    # Count total records
    cursor.execute("SELECT COUNT(*) FROM commodity.bronze.weather_v2")
    total_records = cursor.fetchone()[0]
    print(f"  Total records: {total_records:,}")

    # Count distinct regions
    cursor.execute("SELECT COUNT(DISTINCT region) FROM commodity.bronze.weather_v2")
    distinct_regions = cursor.fetchone()[0]
    print(f"  Distinct regions: {distinct_regions}")

    # Date range
    cursor.execute("SELECT MIN(date), MAX(date) FROM commodity.bronze.weather_v2")
    min_date, max_date = cursor.fetchone()
    print(f"  Date range: {min_date} to {max_date}")

    # Sample record with coordinates
    cursor.execute("""
        SELECT region, date, latitude, longitude, temp_min_c, temp_max_c
        FROM commodity.bronze.weather_v2
        WHERE region LIKE '%Minas%'
        ORDER BY date DESC
        LIMIT 1
    """)
    sample = cursor.fetchone()
    if sample:
        region, date, lat, lon, temp_min, temp_max = sample
        print(f"\n  Sample record:")
        print(f"    Region: {region}")
        print(f"    Date: {date}")
        print(f"    Coordinates: ({lat}, {lon})")
        print(f"    Temperature: {temp_min}°C to {temp_max}°C")

    # Summary
    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print(f"✅ commodity.bronze.weather_v2 table created and loaded")
    print(f"   - {total_records:,} weather observations")
    print(f"   - {distinct_regions} growing regions")
    print(f"   - Date range: {min_date} to {max_date}")
    print(f"   - Includes lat/lon for coordinate transparency")
    print("\n📝 Next Steps:")
    print("   1. Validate July 2021 frost: python validate_july2021_frost.py")
    print("   2. Update unified_data to use weather_v2")
    print("   3. Train new SARIMAX models with corrected weather")
    print("=" * 80)

    cursor.close()
    connection.close()
    return True


if __name__ == "__main__":
    create_weather_v2_table()

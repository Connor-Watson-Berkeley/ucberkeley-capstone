-- Create Weather v2 Bronze Table with Corrected Coordinates
-- Run this in Databricks SQL Editor for better performance

-- Step 1: Drop existing table if needed (optional)
-- DROP TABLE IF EXISTS commodity.bronze.weather_v2;

-- Step 2: Create the table structure
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
COMMENT 'Weather data v2 with corrected growing region coordinates - Bronze layer';

-- Step 3: Load data using COPY INTO (more efficient than INSERT)
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
PATTERN = '*.jsonl'
FORMAT_OPTIONS ('recursiveFileLookup' = 'true')
COPY_OPTIONS ('mergeSchema' = 'false');

-- Step 4: Optimize the table
OPTIMIZE commodity.bronze.weather_v2 ZORDER BY (region, date);

-- Step 5: Gather statistics
ANALYZE TABLE commodity.bronze.weather_v2 COMPUTE STATISTICS;

-- Step 6: Verify the data
SELECT
    COUNT(*) as total_records,
    COUNT(DISTINCT region) as distinct_regions,
    MIN(date) as earliest_date,
    MAX(date) as latest_date
FROM commodity.bronze.weather_v2;

-- Step 7: Sample data check
SELECT *
FROM commodity.bronze.weather_v2
WHERE region LIKE '%Minas%'
ORDER BY date DESC
LIMIT 5;

-- Weather v2 Delta Table Migration Strategy
--
-- Purpose: Handle weather data migration from v1 (wrong coordinates) to v2 (correct coordinates)
-- Author: Research Agent
-- Date: 2025-11-05
--
-- Strategy: Create parallel tables, validate, then migrate unified_data
-- This avoids duplicates and preserves v1 for accuracy comparison

-- ============================================================================
-- STEP 1: Create weather_v2 bronze table (corrected coordinates)
-- ============================================================================

CREATE TABLE IF NOT EXISTS commodity.bronze.weather_v2 (
  Date DATE COMMENT 'Date of weather observation',
  Region STRING COMMENT 'Growing region name (e.g., Minas_Gerais_Brazil)',
  Commodity STRING COMMENT 'Coffee or Sugar',

  -- Temperature
  Temp_Max_C DOUBLE COMMENT 'Maximum temperature (°C)',
  Temp_Min_C DOUBLE COMMENT 'Minimum temperature (°C)',
  Temp_Mean_C DOUBLE COMMENT 'Mean temperature (°C)',

  -- Precipitation
  Precipitation_mm DOUBLE COMMENT 'Total precipitation (mm)',
  Rain_mm DOUBLE COMMENT 'Rainfall only (mm)',
  Snowfall_cm DOUBLE COMMENT 'Snowfall (cm)',

  -- Other weather
  Wind_Speed_Max_kmh DOUBLE COMMENT 'Maximum wind speed (km/h)',
  Humidity_Mean_Pct DOUBLE COMMENT 'Mean relative humidity (%)',

  -- NEW: Coordinate transparency
  Latitude DOUBLE COMMENT '⭐ CORRECTED latitude of growing region',
  Longitude DOUBLE COMMENT '⭐ CORRECTED longitude of growing region',

  -- Metadata
  Country STRING COMMENT 'Country',
  Elevation_m INT COMMENT 'Elevation in meters',
  Description STRING COMMENT 'Region description (e.g., "Sul de Minas coffee region")',
  Ingest_Ts TIMESTAMP COMMENT 'When data was ingested',
  Data_Version STRING COMMENT 'v2_corrected_coordinates',
  Coordinate_Source STRING COMMENT 'region_coordinates.json'
)
USING DELTA
PARTITIONED BY (Date)
LOCATION 's3://groundtruth-capstone/delta/bronze/weather_v2'
COMMENT '⭐ Weather data with CORRECTED coordinates (actual growing regions, not state capitals)';

-- ============================================================================
-- STEP 2: Load data from S3 landing zone
-- ============================================================================

-- Copy from S3 landing/weather_v2/
COPY INTO commodity.bronze.weather_v2
FROM 's3://groundtruth-capstone/landing/weather_v2/'
FILEFORMAT = JSON
FORMAT_OPTIONS ('inferSchema' = 'true', 'mergeSchema' = 'true')
COPY_OPTIONS ('mergeSchema' = 'true');

-- Or use explicit schema loading:
-- CREATE OR REPLACE TEMPORARY VIEW weather_v2_staging
-- USING json
-- OPTIONS (
--   path 's3://groundtruth-capstone/landing/weather_v2/',
--   inferSchema 'true',
--   mergeSchema 'true'
-- );
--
-- INSERT INTO commodity.bronze.weather_v2
-- SELECT
--   CAST(date AS DATE) as Date,
--   region as Region,
--   commodity as Commodity,
--   temp_max_c as Temp_Max_C,
--   temp_min_c as Temp_Min_C,
--   temp_mean_c as Temp_Mean_C,
--   precipitation_mm as Precipitation_mm,
--   rain_mm as Rain_mm,
--   snowfall_cm as Snowfall_cm,
--   wind_speed_max_kmh as Wind_Speed_Max_kmh,
--   humidity_mean_pct as Humidity_Mean_Pct,
--   latitude as Latitude,
--   longitude as Longitude,
--   country as Country,
--   elevation_m as Elevation_m,
--   description as Description,
--   CAST(ingest_ts AS TIMESTAMP) as Ingest_Ts,
--   data_version as Data_Version,
--   coordinate_source as Coordinate_Source
-- FROM weather_v2_staging;

-- ============================================================================
-- STEP 3: Validate - Check for duplicates
-- ============================================================================

-- Should return 0 duplicates
SELECT
  Date,
  Region,
  COUNT(*) as count
FROM commodity.bronze.weather_v2
GROUP BY Date, Region
HAVING COUNT(*) > 1
ORDER BY count DESC;

-- ============================================================================
-- STEP 4: Validate - July 2021 Frost Detection
-- ============================================================================

-- Check if July 2021 frost is now detected (temps < 0°C)
SELECT
  Date,
  Region,
  Temp_Min_C,
  Temp_Max_C,
  Temp_Mean_C,
  Latitude,
  Longitude,
  Description,
  CASE
    WHEN Temp_Min_C < 0 THEN '✅ FROST DETECTED'
    WHEN Temp_Min_C < 5 THEN '⚠️ Near freezing'
    ELSE '❌ No frost'
  END as frost_status
FROM commodity.bronze.weather_v2
WHERE Region = 'Minas_Gerais_Brazil'
  AND Date BETWEEN '2021-07-15' AND '2021-07-25'
ORDER BY Date;

-- Expected: July 20-21 should show Temp_Min_C between -4°C and 0°C

-- ============================================================================
-- STEP 5: Compare v1 vs v2 coordinates
-- ============================================================================

-- Show coordinate differences (v1 vs v2)
-- NOTE: v1 doesn't have lat/lon columns, so we'll compare via description
SELECT
  v1.Region,
  v1.Date,
  v1.Temp_Min_C as v1_temp_min,
  v2.Temp_Min_C as v2_temp_min,
  v2.Latitude,
  v2.Longitude,
  v2.Description as v2_location,
  ROUND(ABS(v1.Temp_Min_C - v2.Temp_Min_C), 2) as temp_diff_c
FROM commodity.bronze.weather v1
JOIN commodity.bronze.weather_v2 v2
  ON v1.Date = v2.Date
  AND v1.Region = v2.Region
WHERE v1.Region = 'Minas_Gerais_Brazil'
  AND v1.Date BETWEEN '2021-07-15' AND '2021-07-25'
ORDER BY v1.Date;

-- ============================================================================
-- STEP 6: Update unified_data to use weather_v2
-- ============================================================================

-- Option A: Create unified_data_v2 (parallel table for testing)
CREATE OR REPLACE TABLE commodity.silver.unified_data_v2 AS
SELECT
  -- Existing unified_data columns
  u.*,

  -- Replace weather columns with v2 data
  w2.Temp_Max_C as temp_max_c,
  w2.Temp_Min_C as temp_min_c,
  w2.Temp_Mean_C as temp_mean_c,
  w2.Precipitation_mm as precipitation_mm,
  w2.Rain_mm as rain_mm,
  w2.Snowfall_cm as snowfall_cm,
  w2.Wind_Speed_Max_kmh as wind_speed_max_kmh,
  w2.Humidity_Mean_Pct as humidity_mean_pct,

  -- NEW: Add coordinate transparency
  w2.Latitude as weather_latitude,
  w2.Longitude as weather_longitude,
  w2.Description as weather_location_description,

  -- Flag to identify v2 data
  'v2_corrected_coordinates' as weather_data_version

FROM commodity.silver.unified_data u
LEFT JOIN commodity.bronze.weather_v2 w2
  ON u.date = w2.Date
  AND u.commodity = w2.Commodity
  AND u.region = w2.Region;

-- Option B: Update existing unified_data in place (use MERGE)
-- MERGE INTO commodity.silver.unified_data AS target
-- USING commodity.bronze.weather_v2 AS w2
-- ON target.date = w2.Date
--    AND target.region = w2.Region
--    AND target.commodity = w2.Commodity
-- WHEN MATCHED THEN UPDATE SET
--   target.temp_max_c = w2.Temp_Max_C,
--   target.temp_min_c = w2.Temp_Min_C,
--   target.temp_mean_c = w2.Temp_Mean_C,
--   target.precipitation_mm = w2.Precipitation_mm,
--   target.rain_mm = w2.Rain_mm,
--   target.snowfall_cm = w2.Snowfall_cm,
--   target.wind_speed_max_kmh = w2.Wind_Speed_Max_kmh,
--   target.humidity_mean_pct = w2.Humidity_Mean_Pct;

-- ============================================================================
-- STEP 7: Verify no duplicates in unified_data_v2
-- ============================================================================

SELECT
  date,
  region,
  commodity,
  COUNT(*) as count
FROM commodity.silver.unified_data_v2
GROUP BY date, region, commodity
HAVING COUNT(*) > 1;

-- Should return 0 rows

-- ============================================================================
-- STEP 8: Compare data quality metrics
-- ============================================================================

-- Count records with valid weather data
SELECT
  'v1_weather' as version,
  COUNT(*) as total_records,
  COUNT(temp_mean_c) as records_with_weather,
  ROUND(100.0 * COUNT(temp_mean_c) / COUNT(*), 2) as pct_coverage
FROM commodity.silver.unified_data
WHERE date >= '2015-07-07'

UNION ALL

SELECT
  'v2_weather' as version,
  COUNT(*) as total_records,
  COUNT(temp_mean_c) as records_with_weather,
  ROUND(100.0 * COUNT(temp_mean_c) / COUNT(*), 2) as pct_coverage
FROM commodity.silver.unified_data_v2
WHERE date >= '2015-07-07';

-- ============================================================================
-- STEP 9: Archive v1 and promote v2 (After validation)
-- ============================================================================

-- After confirming v2 is correct and models show improvement:

-- 1. Rename v1 for archival
ALTER TABLE commodity.bronze.weather RENAME TO commodity.bronze.weather_v1_archive;
ALTER TABLE commodity.silver.unified_data RENAME TO commodity.silver.unified_data_v1_archive;

-- 2. Promote v2 to production names
ALTER TABLE commodity.bronze.weather_v2 RENAME TO commodity.bronze.weather;
ALTER TABLE commodity.silver.unified_data_v2 RENAME TO commodity.silver.unified_data;

-- 3. Update table comments
COMMENT ON TABLE commodity.bronze.weather IS
  'Weather data with CORRECTED coordinates (migrated from v2 on 2025-11-05)';
COMMENT ON TABLE commodity.silver.unified_data IS
  'Unified data using weather v2 (corrected coordinates)';

-- ============================================================================
-- ROLLBACK PLAN (If needed)
-- ============================================================================

-- If v2 has issues, rollback:
-- ALTER TABLE commodity.bronze.weather RENAME TO commodity.bronze.weather_v2_rollback;
-- ALTER TABLE commodity.bronze.weather_v1_archive RENAME TO commodity.bronze.weather;
--
-- ALTER TABLE commodity.silver.unified_data RENAME TO commodity.silver.unified_data_v2_rollback;
-- ALTER TABLE commodity.silver.unified_data_v1_archive RENAME TO commodity.silver.unified_data;

-- ============================================================================
-- CLEANUP (After capstone presentation)
-- ============================================================================

-- After project is complete and we're confident in v2:
-- DROP TABLE IF EXISTS commodity.bronze.weather_v1_archive;
-- DROP TABLE IF EXISTS commodity.silver.unified_data_v1_archive;

-- ============================================================================
-- Summary
-- ============================================================================

/*
Migration Steps:
1. ✅ Create weather_v2 table
2. ✅ Load from S3 landing/weather_v2/
3. ✅ Validate no duplicates
4. ✅ Validate July 2021 frost detection
5. ✅ Compare v1 vs v2 temperature differences
6. ✅ Create unified_data_v2 with corrected weather
7. ✅ Train SARIMAX models on both v1 and v2
8. ✅ Document accuracy improvements
9. ✅ Promote v2 to production (after validation)
10. ✅ Archive v1 for documentation

Key Design Decisions:
- Parallel tables (v1 and v2) to enable comparison
- Delta Lake partitioning by Date for efficiency
- Lat/lon columns for coordinate transparency
- MERGE strategy available for future updates
- Archive v1 data for capstone documentation
- No duplicates possible with separate table strategy
*/

-- Quick Bronze Layer Setup using COPY INTO
-- Run this in Databricks SQL Editor

USE CATALOG commodity;

-- Create market data table
CREATE TABLE IF NOT EXISTS commodity.landing.market_data_inc (
  date DATE,
  commodity STRING,
  close DOUBLE,
  source_file STRING,
  ingest_ts TIMESTAMP
) USING DELTA;

COPY INTO commodity.landing.market_data_inc
FROM (
  SELECT 
    *,
    _metadata.file_path as source_file,
    current_timestamp() as ingest_ts
  FROM 's3://groundtruth-capstone/landing/market_data/'
)
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
COPY_OPTIONS ('mergeSchema' = 'true');

-- Create macro data table
CREATE TABLE IF NOT EXISTS commodity.landing.macro_data_inc
USING DELTA
AS SELECT *, 
  '_metadata.file_path' as source_file,
  current_timestamp() as ingest_ts
FROM read_files(
  's3://groundtruth-capstone/landing/macro_data/*.csv',
  format => 'csv',
  header => true,
  inferSchema => true
);

-- Create VIX data table  
CREATE TABLE IF NOT EXISTS commodity.landing.vix_data_inc
USING DELTA
AS SELECT *,
  '_metadata.file_path' as source_file,
  current_timestamp() as ingest_ts
FROM read_files(
  's3://groundtruth-capstone/landing/vix_data/*.csv',
  format => 'csv',
  header => true,
  inferSchema => true
);

-- Create weather data table
CREATE TABLE IF NOT EXISTS commodity.landing.weather_data_inc
USING DELTA
AS SELECT *,
  '_metadata.file_path' as source_file,
  current_timestamp() as ingest_ts
FROM read_files(
  's3://groundtruth-capstone/landing/weather_data/*.csv',
  format => 'csv',
  header => true,
  inferSchema => true
);

-- Create CFTC data table
CREATE TABLE IF NOT EXISTS commodity.landing.cftc_data_inc
USING DELTA
AS SELECT *,
  '_metadata.file_path' as source_file,
  current_timestamp() as ingest_ts
FROM read_files(
  's3://groundtruth-capstone/landing/cftc_data/*.csv',
  format => 'csv',
  header => true,
  inferSchema => true
);

-- Verify
SHOW TABLES IN commodity.landing;

-- Check row counts
SELECT 'market_data_inc' as table, COUNT(*) as rows FROM commodity.landing.market_data_inc
UNION ALL
SELECT 'macro_data_inc', COUNT(*) FROM commodity.landing.macro_data_inc
UNION ALL
SELECT 'vix_data_inc', COUNT(*) FROM commodity.landing.vix_data_inc
UNION ALL
SELECT 'weather_data_inc', COUNT(*) FROM commodity.landing.weather_data_inc
UNION ALL
SELECT 'cftc_data_inc', COUNT(*) FROM commodity.landing.cftc_data_inc;

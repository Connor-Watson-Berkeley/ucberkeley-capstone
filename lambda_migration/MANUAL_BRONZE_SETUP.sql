-- ================================================================
-- Manual Bronze Layer Setup
-- Run these commands in Databricks SQL Editor or Notebook
-- ================================================================

-- Step 1: Create Schemas
-- ================================================================
USE CATALOG commodity;

CREATE SCHEMA IF NOT EXISTS bronze
COMMENT 'Bronze layer - raw data with deduplication views';

CREATE SCHEMA IF NOT EXISTS silver
COMMENT 'Silver layer - cleaned and joined data';

CREATE SCHEMA IF NOT EXISTS landing
COMMENT 'Landing layer - raw ingestion from S3';

SHOW SCHEMAS IN commodity;

-- Step 2: Create Auto Loader Streams (Run in Databricks Notebook)
-- ================================================================
-- Copy this Python code into a Databricks notebook cell:

/*
%python
from pyspark.sql.functions import expr, current_timestamp

# Market Data
spark.readStream \
    .format("cloudFiles") \
    .option("cloudFiles.format", "csv") \
    .option("cloudFiles.inferColumnTypes", "true") \
    .option("cloudFiles.schemaLocation", "s3://groundtruth-capstone/_schemas/market_data") \
    .option("header", "true") \
    .load("s3://groundtruth-capstone/landing/market_data/") \
    .withColumn("source_file", expr("_metadata.file_path")) \
    .withColumn("ingest_ts", current_timestamp()) \
    .writeStream \
    .format("delta") \
    .option("checkpointLocation", "s3://groundtruth-capstone/_checkpoints/market_data") \
    .option("mergeSchema", "true") \
    .trigger(availableNow=True) \
    .toTable("commodity.landing.market_data_inc")

print("✓ Market data stream started")

# VIX Data
spark.readStream \
    .format("cloudFiles") \
    .option("cloudFiles.format", "csv") \
    .option("cloudFiles.inferColumnTypes", "true") \
    .option("cloudFiles.schemaLocation", "s3://groundtruth-capstone/_schemas/vix_data") \
    .option("header", "true") \
    .load("s3://groundtruth-capstone/landing/vix_data/") \
    .withColumn("source_file", expr("_metadata.file_path")) \
    .withColumn("ingest_ts", current_timestamp()) \
    .writeStream \
    .format("delta") \
    .option("checkpointLocation", "s3://groundtruth-capstone/_checkpoints/vix_data") \
    .option("mergeSchema", "true") \
    .trigger(availableNow=True) \
    .toTable("commodity.landing.vix_data_inc")

print("✓ VIX data stream started")

# Macro/FX Data
spark.readStream \
    .format("cloudFiles") \
    .option("cloudFiles.format", "csv") \
    .option("cloudFiles.inferColumnTypes", "true") \
    .option("cloudFiles.schemaLocation", "s3://groundtruth-capstone/_schemas/macro_data") \
    .option("header", "true") \
    .load("s3://groundtruth-capstone/landing/macro_data/") \
    .withColumn("source_file", expr("_metadata.file_path")) \
    .withColumn("ingest_ts", current_timestamp()) \
    .writeStream \
    .format("delta") \
    .option("checkpointLocation", "s3://groundtruth-capstone/_checkpoints/macro_data") \
    .option("mergeSchema", "true") \
    .trigger(availableNow=True) \
    .toTable("commodity.landing.macro_data_inc")

print("✓ Macro data stream started")

# Weather Data
spark.readStream \
    .format("cloudFiles") \
    .option("cloudFiles.format", "csv") \
    .option("cloudFiles.schemaLocation", "s3://groundtruth-capstone/_schemas/weather_data") \
    .option("cloudFiles.inferColumnTypes", "true") \
    .option("header", "true") \
    .option("pathGlobFilter", "*.csv") \
    .load("s3://groundtruth-capstone/landing/weather_data/") \
    .withColumn("source_file", expr("_metadata.file_path")) \
    .withColumn("ingest_ts", current_timestamp()) \
    .writeStream \
    .format("delta") \
    .option("checkpointLocation", "s3://groundtruth-capstone/_checkpoints/weather_data") \
    .option("mergeSchema", "true") \
    .trigger(availableNow=True) \
    .toTable("commodity.landing.weather_data_inc")

print("✓ Weather data stream started")

# CFTC Data (with column sanitization)
import re

def sanitize(name: str) -> str:
    n = re.sub(r"[ ,;{}\(\)\n\t=]+", "_", name.strip())
    n = re.sub(r"_+", "_", n).strip("_").lower()
    return n

def sanitize_cols(df):
    safe = []
    seen = set()
    for c in df.columns:
        s = sanitize(c)
        i, base = 1, s
        while s in seen:
            i += 1
            s = f"{base}_{i}"
        seen.add(s)
        safe.append(s)
    return df.toDF(*safe)

spark.readStream \
    .format("cloudFiles") \
    .option("cloudFiles.format", "csv") \
    .option("cloudFiles.inferColumnTypes", "true") \
    .option("cloudFiles.schemaLocation", "s3://groundtruth-capstone/_schemas/cftc_data") \
    .option("header", "true") \
    .option("pathGlobFilter", "*.csv") \
    .load("s3://groundtruth-capstone/landing/cftc_data/") \
    .transform(sanitize_cols) \
    .withColumn("source_file", expr("_metadata.file_path")) \
    .withColumn("ingest_ts", current_timestamp()) \
    .writeStream \
    .format("delta") \
    .option("checkpointLocation", "s3://groundtruth-capstone/_checkpoints/cftc_data") \
    .option("mergeSchema", "true") \
    .trigger(availableNow=True) \
    .toTable("commodity.landing.cftc_data_inc")

print("✓ CFTC data stream started")
print("\nAll Auto Loader streams initiated. Check back in 5-10 minutes.")
*/

-- Step 3: Wait for Auto Loader to complete (~5-10 minutes)
-- Then verify tables were created:

SHOW TABLES IN commodity.landing;

-- Check row counts
SELECT 'market_data_inc' as table_name, COUNT(*) as row_count FROM commodity.landing.market_data_inc
UNION ALL
SELECT 'vix_data_inc', COUNT(*) FROM commodity.landing.vix_data_inc
UNION ALL
SELECT 'macro_data_inc', COUNT(*) FROM commodity.landing.macro_data_inc
UNION ALL
SELECT 'weather_data_inc', COUNT(*) FROM commodity.landing.weather_data_inc
UNION ALL
SELECT 'cftc_data_inc', COUNT(*) FROM commodity.landing.cftc_data_inc;

-- Step 4: Create Bronze Views with Deduplication
-- ================================================================

-- Market Data Bronze View
CREATE OR REPLACE VIEW commodity.bronze.v_market_data_all AS
SELECT DISTINCT *
FROM commodity.landing.market_data_inc
ORDER BY date;

-- VIX Data Bronze View
CREATE OR REPLACE VIEW commodity.bronze.v_vix_data_all AS
SELECT DISTINCT date, vix
FROM commodity.landing.vix_data_inc
ORDER BY date;

-- Macro Data Bronze View (deduplicate by latest ingest)
CREATE OR REPLACE VIEW commodity.bronze.v_macro_data_all AS
SELECT
  date,
  vnd_usd, cop_usd, idr_usd, etb_usd, hnl_usd, ugx_usd, pen_usd, xaf_usd,
  gtq_usd, gnf_usd, nio_usd, crc_usd, tzs_usd, kes_usd, lak_usd, pkr_usd,
  php_usd, egp_usd, ars_usd, rub_usd, try_usd, uah_usd, irr_usd, byn_usd
FROM commodity.landing.macro_data_inc
QUALIFY ROW_NUMBER() OVER (PARTITION BY date ORDER BY ingest_ts DESC) = 1;

-- Weather Data Bronze View
CREATE OR REPLACE VIEW commodity.bronze.v_weather_data_all AS
SELECT
  Date AS date,
  Max_Temp_C AS max_temp_c,
  Min_Temp_C AS min_temp_c,
  Precipitation_mm AS precipitation_mm,
  Humidity_perc AS humidity_pct,
  Region AS region,
  Commodity AS commodity
FROM commodity.landing.weather_data_inc
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY date, region, commodity
  ORDER BY ingest_ts DESC
) = 1;

-- CFTC Data Bronze View
CREATE OR REPLACE VIEW commodity.bronze.v_cftc_data_all AS
SELECT DISTINCT *
FROM commodity.landing.cftc_data_inc
ORDER BY as_of_date_in_form_yymmdd;

-- Step 5: Create GDELT Bronze Table
-- ================================================================

CREATE TABLE IF NOT EXISTS commodity.bronze.bronze_gkg
USING CSV
OPTIONS (
  path 's3://groundtruth-capstone/landing/gdelt/filtered/',
  header 'true',
  inferSchema 'true',
  recursiveFileLookup 'true'
)
COMMENT 'GDELT news sentiment data (filtered by gdelt-processor Lambda)';

-- Refresh to pick up files
REFRESH TABLE commodity.bronze.bronze_gkg;

-- Step 6: Final Verification
-- ================================================================

SHOW TABLES IN commodity.bronze;

-- Check all bronze views/tables
SELECT 'v_market_data_all' as table_name, COUNT(*) as row_count FROM commodity.bronze.v_market_data_all
UNION ALL
SELECT 'v_vix_data_all', COUNT(*) FROM commodity.bronze.v_vix_data_all
UNION ALL
SELECT 'v_macro_data_all', COUNT(*) FROM commodity.bronze.v_macro_data_all
UNION ALL
SELECT 'v_weather_data_all', COUNT(*) FROM commodity.bronze.v_weather_data_all
UNION ALL
SELECT 'v_cftc_data_all', COUNT(*) FROM commodity.bronze.v_cftc_data_all
UNION ALL
SELECT 'bronze_gkg', COUNT(*) FROM commodity.bronze.bronze_gkg
ORDER BY table_name;

-- Sample data from each view
SELECT * FROM commodity.bronze.v_market_data_all LIMIT 5;
SELECT * FROM commodity.bronze.v_vix_data_all LIMIT 5;
SELECT * FROM commodity.bronze.v_macro_data_all LIMIT 5;

-- ================================================================
-- DONE! Bronze layer is now set up.
-- ================================================================

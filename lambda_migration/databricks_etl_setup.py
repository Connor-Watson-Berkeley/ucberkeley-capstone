# Databricks notebook source
# MAGIC %md
# MAGIC # Commodity Data Pipeline - Complete ETL Setup
# MAGIC
# MAGIC This notebook sets up the complete data pipeline:
# MAGIC 1. Creates Unity Catalog structure (commodity.bronze, commodity.silver)
# MAGIC 2. Sets up Auto Loader streaming jobs to ingest Lambda-generated data from S3
# MAGIC 3. Creates bronze views with deduplication
# MAGIC 4. Sets up GDELT bronze table
# MAGIC
# MAGIC **S3 Bucket**: groundtruth-capstone
# MAGIC **Cluster**: general-purpose-mid-compute

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Unity Catalog Structure

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create catalog
# MAGIC CREATE CATALOG IF NOT EXISTS commodity
# MAGIC COMMENT 'Commodity price forecasting data catalog';
# MAGIC
# MAGIC USE CATALOG commodity;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create bronze schema for raw/landing data
# MAGIC CREATE SCHEMA IF NOT EXISTS commodity.bronze
# MAGIC COMMENT 'Bronze layer - raw data from S3 (market, VIX, weather, GDELT)';
# MAGIC
# MAGIC -- Create silver schema for curated data
# MAGIC CREATE SCHEMA IF NOT EXISTS commodity.silver
# MAGIC COMMENT 'Silver layer - cleaned and joined data for forecasting';
# MAGIC
# MAGIC -- Create landing schema for incremental data
# MAGIC CREATE SCHEMA IF NOT EXISTS commodity.landing
# MAGIC COMMENT 'Landing layer - raw ingestion from Lambda functions';
# MAGIC
# MAGIC SHOW SCHEMAS IN commodity;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Set Up Auto Loader Streaming Jobs
# MAGIC
# MAGIC These jobs continuously ingest new CSV files written by Lambda functions

# COMMAND ----------

# MAGIC %md
# MAGIC ### Market Data (Coffee & Sugar Prices)
# MAGIC Lambda: market-data-fetcher

# COMMAND ----------

from pyspark.sql.functions import input_file_name, current_timestamp, expr

# Market data configuration
market_path = "s3://groundtruth-capstone/landing/market_data/"
market_bronze_tbl = "commodity.landing.market_data_inc"
market_checkpoint = "s3://groundtruth-capstone/_checkpoints/market_data"
market_schema_loc = "s3://groundtruth-capstone/_schemas/market_data"

# Start Auto Loader stream
(
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.inferColumnTypes", "true")
    .option("cloudFiles.schemaLocation", market_schema_loc)
    .option("header", "true")
    .load(market_path)
    .withColumn("source_file", expr("_metadata.file_path"))
    .withColumn("ingest_ts", current_timestamp())
    .writeStream
    .format("delta")
    .option("checkpointLocation", market_checkpoint)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)  # Process all available files and stop
    .toTable(market_bronze_tbl)
)

print(f"✓ Market data stream configured: {market_bronze_tbl}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### VIX Data (Volatility Index)
# MAGIC Lambda: vix-data-fetcher

# COMMAND ----------

# VIX data configuration
vix_path = "s3://groundtruth-capstone/landing/vix_data/"
vix_bronze_tbl = "commodity.landing.vix_data_inc"
vix_checkpoint = "s3://groundtruth-capstone/_checkpoints/vix_data"
vix_schema_loc = "s3://groundtruth-capstone/_schemas/vix_data"

(
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.inferColumnTypes", "true")
    .option("cloudFiles.schemaLocation", vix_schema_loc)
    .option("header", "true")
    .load(vix_path)
    .withColumn("source_file", expr("_metadata.file_path"))
    .withColumn("ingest_ts", current_timestamp())
    .writeStream
    .format("delta")
    .option("checkpointLocation", vix_checkpoint)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(vix_bronze_tbl)
)

print(f"✓ VIX data stream configured: {vix_bronze_tbl}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Macro Data (Exchange Rates: COP/USD, etc.)
# MAGIC Lambda: fx-calculator-fetcher

# COMMAND ----------

# Macro/FX data configuration
macro_path = "s3://groundtruth-capstone/landing/macro_data/"
macro_bronze_tbl = "commodity.landing.macro_data_inc"
macro_checkpoint = "s3://groundtruth-capstone/_checkpoints/macro_data"
macro_schema_loc = "s3://groundtruth-capstone/_schemas/macro_data"

(
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.inferColumnTypes", "true")
    .option("cloudFiles.schemaLocation", macro_schema_loc)
    .option("header", "true")
    .load(macro_path)
    .withColumn("source_file", expr("_metadata.file_path"))
    .withColumn("ingest_ts", current_timestamp())
    .writeStream
    .format("delta")
    .option("checkpointLocation", macro_checkpoint)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(macro_bronze_tbl)
)

print(f"✓ Macro data stream configured: {macro_bronze_tbl}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weather Data
# MAGIC Lambda: weather-data-fetcher

# COMMAND ----------

from pyspark.sql.functions import col, to_date

# Weather data configuration
weather_path = "s3://groundtruth-capstone/landing/weather_data/"
weather_bronze_tbl = "commodity.landing.weather_data_inc"
weather_checkpoint = "s3://groundtruth-capstone/_checkpoints/weather_data"
weather_schema_loc = "s3://groundtruth-capstone/_schemas/weather_data"

(
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.schemaLocation", weather_schema_loc)
    .option("cloudFiles.inferColumnTypes", "true")
    .option("header", "true")
    .option("pathGlobFilter", "*.csv")
    .load(weather_path)
    .withColumn("source_file", expr("_metadata.file_path"))
    .withColumn("ingest_ts", current_timestamp())
    .writeStream
    .format("delta")
    .option("checkpointLocation", weather_checkpoint)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(weather_bronze_tbl)
)

print(f"✓ Weather data stream configured: {weather_bronze_tbl}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CFTC Data (Commitment of Traders)
# MAGIC Lambda: cftc-data-fetcher

# COMMAND ----------

import re

def sanitize(name: str) -> str:
    """Sanitize column names"""
    n = re.sub(r"[ ,;{}\(\)\n\t=]+", "_", name.strip())
    n = re.sub(r"_+", "_", n).strip("_").lower()
    return n

def sanitize_cols(df):
    """Apply sanitization to all columns"""
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

# CFTC data configuration
cftc_path = "s3://groundtruth-capstone/landing/cftc_data/"
cftc_bronze_tbl = "commodity.landing.cftc_data_inc"
cftc_checkpoint = "s3://groundtruth-capstone/_checkpoints/cftc_data"
cftc_schema_loc = "s3://groundtruth-capstone/_schemas/cftc_data"

(
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.inferColumnTypes", "true")
    .option("cloudFiles.schemaLocation", cftc_schema_loc)
    .option("header", "true")
    .option("pathGlobFilter", "*.csv")
    .load(cftc_path)
    .transform(sanitize_cols)
    .withColumn("source_file", expr("_metadata.file_path"))
    .withColumn("ingest_ts", current_timestamp())
    .writeStream
    .format("delta")
    .option("checkpointLocation", cftc_checkpoint)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(cftc_bronze_tbl)
)

print(f"✓ CFTC data stream configured: {cftc_bronze_tbl}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Bronze Views
# MAGIC
# MAGIC These views provide deduplicated access to the data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Market Data Bronze View
# MAGIC CREATE OR REPLACE VIEW commodity.bronze.v_market_data_all AS
# MAGIC SELECT DISTINCT *
# MAGIC FROM commodity.landing.market_data_inc
# MAGIC ORDER BY date;
# MAGIC
# MAGIC SELECT COUNT(*) as row_count FROM commodity.bronze.v_market_data_all;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- VIX Data Bronze View
# MAGIC CREATE OR REPLACE VIEW commodity.bronze.v_vix_data_all AS
# MAGIC SELECT DISTINCT date, vix
# MAGIC FROM commodity.landing.vix_data_inc
# MAGIC ORDER BY date;
# MAGIC
# MAGIC SELECT COUNT(*) as row_count FROM commodity.bronze.v_vix_data_all;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Macro Data Bronze View (with deduplication)
# MAGIC CREATE OR REPLACE VIEW commodity.bronze.v_macro_data_all AS
# MAGIC SELECT
# MAGIC   date,
# MAGIC   vnd_usd, cop_usd, idr_usd, etb_usd, hnl_usd, ugx_usd, pen_usd, xaf_usd,
# MAGIC   gtq_usd, gnf_usd, nio_usd, crc_usd, tzs_usd, kes_usd, lak_usd, pkr_usd,
# MAGIC   php_usd, egp_usd, ars_usd, rub_usd, try_usd, uah_usd, irr_usd, byn_usd
# MAGIC FROM commodity.landing.macro_data_inc
# MAGIC QUALIFY ROW_NUMBER() OVER (PARTITION BY date ORDER BY ingest_ts DESC) = 1;
# MAGIC
# MAGIC SELECT COUNT(*) as row_count FROM commodity.bronze.v_macro_data_all;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Weather Data Bronze View
# MAGIC CREATE OR REPLACE VIEW commodity.bronze.v_weather_data_all AS
# MAGIC SELECT
# MAGIC   Date AS date,
# MAGIC   Max_Temp_C AS max_temp_c,
# MAGIC   Min_Temp_C AS min_temp_c,
# MAGIC   Precipitation_mm AS precipitation_mm,
# MAGIC   Humidity_perc AS humidity_pct,
# MAGIC   Region AS region,
# MAGIC   Commodity AS commodity
# MAGIC FROM commodity.landing.weather_data_inc
# MAGIC QUALIFY ROW_NUMBER() OVER (
# MAGIC   PARTITION BY date, region, commodity
# MAGIC   ORDER BY ingest_ts DESC
# MAGIC ) = 1;
# MAGIC
# MAGIC SELECT COUNT(*) as row_count FROM commodity.bronze.v_weather_data_all;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- CFTC Data Bronze View
# MAGIC CREATE OR REPLACE VIEW commodity.bronze.v_cftc_data_all AS
# MAGIC SELECT DISTINCT *
# MAGIC FROM commodity.landing.cftc_data_inc
# MAGIC ORDER BY date;
# MAGIC
# MAGIC SELECT COUNT(*) as row_count FROM commodity.bronze.v_cftc_data_all;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. GDELT Bronze Table
# MAGIC
# MAGIC GDELT processor writes filtered data to S3, we create a table to read it

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create external table for GDELT filtered data
# MAGIC CREATE TABLE IF NOT EXISTS commodity.bronze.bronze_gkg
# MAGIC USING CSV
# MAGIC OPTIONS (
# MAGIC   path 's3://groundtruth-capstone/landing/gdelt/filtered/',
# MAGIC   header 'true',
# MAGIC   inferSchema 'true',
# MAGIC   recursiveFileLookup 'true'
# MAGIC )
# MAGIC COMMENT 'GDELT news sentiment data (filtered by gdelt-processor Lambda)';
# MAGIC
# MAGIC -- Refresh to pick up any new files
# MAGIC REFRESH TABLE commodity.bronze.bronze_gkg;
# MAGIC
# MAGIC SELECT COUNT(*) as row_count FROM commodity.bronze.bronze_gkg;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify Bronze Layer

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Show all bronze tables and views
# MAGIC SHOW TABLES IN commodity.bronze;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Get row counts for all bronze views/tables
# MAGIC SELECT 'v_market_data_all' as table_name, COUNT(*) as row_count FROM commodity.bronze.v_market_data_all
# MAGIC UNION ALL
# MAGIC SELECT 'v_vix_data_all', COUNT(*) FROM commodity.bronze.v_vix_data_all
# MAGIC UNION ALL
# MAGIC SELECT 'v_macro_data_all', COUNT(*) FROM commodity.bronze.v_macro_data_all
# MAGIC UNION ALL
# MAGIC SELECT 'v_weather_data_all', COUNT(*) FROM commodity.bronze.v_weather_data_all
# MAGIC UNION ALL
# MAGIC SELECT 'v_cftc_data_all', COUNT(*) FROM commodity.bronze.v_cftc_data_all
# MAGIC UNION ALL
# MAGIC SELECT 'bronze_gkg', COUNT(*) FROM commodity.bronze.bronze_gkg;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Sample Data Verification

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify market data structure
# MAGIC SELECT * FROM commodity.bronze.v_market_data_all LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify VIX data structure
# MAGIC SELECT * FROM commodity.bronze.v_vix_data_all LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify macro/FX data structure
# MAGIC SELECT * FROM commodity.bronze.v_macro_data_all LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify weather data structure
# MAGIC SELECT * FROM commodity.bronze.v_weather_data_all LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify GDELT data structure
# MAGIC SELECT * FROM commodity.bronze.bronze_gkg LIMIT 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Complete!
# MAGIC
# MAGIC **Bronze Layer Created**:
# MAGIC - commodity.bronze.v_market_data_all
# MAGIC - commodity.bronze.v_vix_data_all
# MAGIC - commodity.bronze.v_macro_data_all
# MAGIC - commodity.bronze.v_weather_data_all
# MAGIC - commodity.bronze.v_cftc_data_all
# MAGIC - commodity.bronze.bronze_gkg
# MAGIC
# MAGIC **Auto Loader Streams**:
# MAGIC - Market data: s3://groundtruth-capstone/landing/market_data/
# MAGIC - VIX data: s3://groundtruth-capstone/landing/vix_data/
# MAGIC - Macro data: s3://groundtruth-capstone/landing/macro_data/
# MAGIC - Weather data: s3://groundtruth-capstone/landing/weather_data/
# MAGIC - CFTC data: s3://groundtruth-capstone/landing/cftc_data/
# MAGIC
# MAGIC **Next Steps**:
# MAGIC 1. Set up daily refresh job (see setup_daily_refresh_job.py)
# MAGIC 2. Create unified_data in silver layer (run create_gdelt_unified_data.py)
# MAGIC 3. Deploy forecast agent

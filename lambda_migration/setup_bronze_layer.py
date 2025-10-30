# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer Setup - Data Ingestion from S3
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Creates schemas in the commodity catalog
# MAGIC 2. Sets up Auto Loader streams to ingest S3 data
# MAGIC 3. Creates bronze views with deduplication
# MAGIC
# MAGIC **Prerequisites**: commodity catalog exists and is connected to s3://groundtruth-capstone

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Schemas

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG commodity;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS bronze
# MAGIC COMMENT 'Bronze layer - raw data with deduplication';
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS silver
# MAGIC COMMENT 'Silver layer - cleaned and joined data';
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS landing
# MAGIC COMMENT 'Landing layer - raw ingestion from S3';
# MAGIC
# MAGIC SHOW SCHEMAS IN commodity;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Set Up Auto Loader - Market Data

# COMMAND ----------

from pyspark.sql.functions import expr, current_timestamp

market_path = "s3://groundtruth-capstone/landing/market_data/"
market_table = "commodity.landing.market_data_inc"
market_checkpoint = "s3://groundtruth-capstone/_checkpoints/market_data"
market_schema = "s3://groundtruth-capstone/_schemas/market_data"

print(f"Ingesting market data from: {market_path}")

(
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.inferColumnTypes", "true")
    .option("cloudFiles.schemaLocation", market_schema)
    .option("header", "true")
    .load(market_path)
    .withColumn("source_file", expr("_metadata.file_path"))
    .withColumn("ingest_ts", current_timestamp())
    .writeStream
    .format("delta")
    .option("checkpointLocation", market_checkpoint)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(market_table)
)

print(f"✓ Market data ingested to {market_table}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) as row_count FROM commodity.landing.market_data_inc;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Set Up Auto Loader - VIX Data

# COMMAND ----------

vix_path = "s3://groundtruth-capstone/landing/vix_data/"
vix_table = "commodity.landing.vix_data_inc"
vix_checkpoint = "s3://groundtruth-capstone/_checkpoints/vix_data"
vix_schema = "s3://groundtruth-capstone/_schemas/vix_data"

print(f"Ingesting VIX data from: {vix_path}")

(
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.inferColumnTypes", "true")
    .option("cloudFiles.schemaLocation", vix_schema)
    .option("header", "true")
    .load(vix_path)
    .withColumn("source_file", expr("_metadata.file_path"))
    .withColumn("ingest_ts", current_timestamp())
    .writeStream
    .format("delta")
    .option("checkpointLocation", vix_checkpoint)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(vix_table)
)

print(f"✓ VIX data ingested to {vix_table}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) as row_count FROM commodity.landing.vix_data_inc;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Set Up Auto Loader - Macro/FX Data

# COMMAND ----------

macro_path = "s3://groundtruth-capstone/landing/macro_data/"
macro_table = "commodity.landing.macro_data_inc"
macro_checkpoint = "s3://groundtruth-capstone/_checkpoints/macro_data"
macro_schema = "s3://groundtruth-capstone/_schemas/macro_data"

print(f"Ingesting macro/FX data from: {macro_path}")

(
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.inferColumnTypes", "true")
    .option("cloudFiles.schemaLocation", macro_schema)
    .option("header", "true")
    .load(macro_path)
    .withColumn("source_file", expr("_metadata.file_path"))
    .withColumn("ingest_ts", current_timestamp())
    .writeStream
    .format("delta")
    .option("checkpointLocation", macro_checkpoint)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(macro_table)
)

print(f"✓ Macro data ingested to {macro_table}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) as row_count FROM commodity.landing.macro_data_inc;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Set Up Auto Loader - Weather Data

# COMMAND ----------

weather_path = "s3://groundtruth-capstone/landing/weather_data/"
weather_table = "commodity.landing.weather_data_inc"
weather_checkpoint = "s3://groundtruth-capstone/_checkpoints/weather_data"
weather_schema = "s3://groundtruth-capstone/_schemas/weather_data"

print(f"Ingesting weather data from: {weather_path}")

(
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.schemaLocation", weather_schema)
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
    .toTable(weather_table)
)

print(f"✓ Weather data ingested to {weather_table}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) as row_count FROM commodity.landing.weather_data_inc;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Set Up Auto Loader - CFTC Data

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

cftc_path = "s3://groundtruth-capstone/landing/cftc_data/"
cftc_table = "commodity.landing.cftc_data_inc"
cftc_checkpoint = "s3://groundtruth-capstone/_checkpoints/cftc_data"
cftc_schema = "s3://groundtruth-capstone/_schemas/cftc_data"

print(f"Ingesting CFTC data from: {cftc_path}")

(
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.inferColumnTypes", "true")
    .option("cloudFiles.schemaLocation", cftc_schema)
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
    .toTable(cftc_table)
)

print(f"✓ CFTC data ingested to {cftc_table}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) as row_count FROM commodity.landing.cftc_data_inc;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Create Bronze Views with Deduplication

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Market Data Bronze View
# MAGIC CREATE OR REPLACE VIEW commodity.bronze.v_market_data_all AS
# MAGIC SELECT DISTINCT *
# MAGIC FROM commodity.landing.market_data_inc
# MAGIC ORDER BY date;
# MAGIC
# MAGIC SELECT 'v_market_data_all' as view_name, COUNT(*) as row_count
# MAGIC FROM commodity.bronze.v_market_data_all;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- VIX Data Bronze View
# MAGIC CREATE OR REPLACE VIEW commodity.bronze.v_vix_data_all AS
# MAGIC SELECT DISTINCT date, vix
# MAGIC FROM commodity.landing.vix_data_inc
# MAGIC ORDER BY date;
# MAGIC
# MAGIC SELECT 'v_vix_data_all' as view_name, COUNT(*) as row_count
# MAGIC FROM commodity.bronze.v_vix_data_all;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Macro Data Bronze View (with deduplication by latest ingest)
# MAGIC CREATE OR REPLACE VIEW commodity.bronze.v_macro_data_all AS
# MAGIC SELECT
# MAGIC   date,
# MAGIC   vnd_usd, cop_usd, idr_usd, etb_usd, hnl_usd, ugx_usd, pen_usd, xaf_usd,
# MAGIC   gtq_usd, gnf_usd, nio_usd, crc_usd, tzs_usd, kes_usd, lak_usd, pkr_usd,
# MAGIC   php_usd, egp_usd, ars_usd, rub_usd, try_usd, uah_usd, irr_usd, byn_usd
# MAGIC FROM commodity.landing.macro_data_inc
# MAGIC QUALIFY ROW_NUMBER() OVER (PARTITION BY date ORDER BY ingest_ts DESC) = 1;
# MAGIC
# MAGIC SELECT 'v_macro_data_all' as view_name, COUNT(*) as row_count
# MAGIC FROM commodity.bronze.v_macro_data_all;

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
# MAGIC SELECT 'v_weather_data_all' as view_name, COUNT(*) as row_count
# MAGIC FROM commodity.bronze.v_weather_data_all;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- CFTC Data Bronze View
# MAGIC CREATE OR REPLACE VIEW commodity.bronze.v_cftc_data_all AS
# MAGIC SELECT DISTINCT *
# MAGIC FROM commodity.landing.cftc_data_inc
# MAGIC ORDER BY as_of_date_in_form_yymmdd;
# MAGIC
# MAGIC SELECT 'v_cftc_data_all' as view_name, COUNT(*) as row_count
# MAGIC FROM commodity.bronze.v_cftc_data_all;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Create GDELT Bronze Table

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
# MAGIC -- Refresh to pick up files
# MAGIC REFRESH TABLE commodity.bronze.bronze_gkg;
# MAGIC
# MAGIC SELECT 'bronze_gkg' as table_name, COUNT(*) as row_count
# MAGIC FROM commodity.bronze.bronze_gkg;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Verification Summary

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Summary of all bronze tables/views
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
# MAGIC SELECT 'bronze_gkg', COUNT(*) FROM commodity.bronze.bronze_gkg
# MAGIC ORDER BY table_name;

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✓ Bronze Layer Setup Complete!
# MAGIC
# MAGIC **Created**:
# MAGIC - commodity.landing.* (5 Delta tables with Auto Loader)
# MAGIC - commodity.bronze.v_* (5 views with deduplication)
# MAGIC - commodity.bronze.bronze_gkg (GDELT table)
# MAGIC
# MAGIC **Next Steps**:
# MAGIC 1. Set up daily refresh job
# MAGIC 2. Create silver.unified_data
# MAGIC 3. Deploy Lambda functions for daily updates

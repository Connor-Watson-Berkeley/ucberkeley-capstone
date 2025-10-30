# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer Setup - Auto Loader & Views
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Creates Auto Loader streams to ingest S3 CSV files
# MAGIC 2. Creates bronze views with deduplication
# MAGIC 3. Verifies data loaded successfully

# COMMAND ----------

from pyspark.sql.functions import expr, current_timestamp

print("="*60)
print("Starting Bronze Layer Setup")
print("="*60)

# COMMAND ----------

# MAGIC %md ## 1. Create Landing Tables via Auto Loader

# COMMAND ----------

# Market Data
print("\n[1/5] Market Data - Starting Auto Loader...")
try:
    (spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaLocation", "s3://groundtruth-capstone/_schemas/market_data")
        .option("header", "true")
        .load("s3://groundtruth-capstone/landing/market_data/")
        .withColumn("source_file", expr("_metadata.file_path"))
        .withColumn("ingest_ts", current_timestamp())
        .writeStream
        .format("delta")
        .option("checkpointLocation", "s3://groundtruth-capstone/_checkpoints/market_data")
        .option("mergeSchema", "true")
        .trigger(availableNow=True)
        .toTable("commodity.landing.market_data_inc")
        .awaitTermination())
    print("✓ Market data ingested")
except Exception as e:
    print(f"✗ Market data error: {str(e)}")

# COMMAND ----------

# VIX Data
print("\n[2/5] VIX Data - Starting Auto Loader...")
try:
    (spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaLocation", "s3://groundtruth-capstone/_schemas/vix_data")
        .option("header", "true")
        .load("s3://groundtruth-capstone/landing/vix_data/")
        .withColumn("source_file", expr("_metadata.file_path"))
        .withColumn("ingest_ts", current_timestamp())
        .writeStream
        .format("delta")
        .option("checkpointLocation", "s3://groundtruth-capstone/_checkpoints/vix_data")
        .option("mergeSchema", "true")
        .trigger(availableNow=True)
        .toTable("commodity.landing.vix_data_inc")
        .awaitTermination())
    print("✓ VIX data ingested")
except Exception as e:
    print(f"✗ VIX data error: {str(e)}")

# COMMAND ----------

# Macro/FX Data
print("\n[3/5] Macro/FX Data - Starting Auto Loader...")
try:
    (spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaLocation", "s3://groundtruth-capstone/_schemas/macro_data")
        .option("header", "true")
        .load("s3://groundtruth-capstone/landing/macro_data/")
        .withColumn("source_file", expr("_metadata.file_path"))
        .withColumn("ingest_ts", current_timestamp())
        .writeStream
        .format("delta")
        .option("checkpointLocation", "s3://groundtruth-capstone/_checkpoints/macro_data")
        .option("mergeSchema", "true")
        .trigger(availableNow=True)
        .toTable("commodity.landing.macro_data_inc")
        .awaitTermination())
    print("✓ Macro data ingested")
except Exception as e:
    print(f"✗ Macro data error: {str(e)}")

# COMMAND ----------

# Weather Data
print("\n[4/5] Weather Data - Starting Auto Loader...")
try:
    (spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.schemaLocation", "s3://groundtruth-capstone/_schemas/weather_data")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("header", "true")
        .option("pathGlobFilter", "*.csv")
        .load("s3://groundtruth-capstone/landing/weather_data/")
        .withColumn("source_file", expr("_metadata.file_path"))
        .withColumn("ingest_ts", current_timestamp())
        .writeStream
        .format("delta")
        .option("checkpointLocation", "s3://groundtruth-capstone/_checkpoints/weather_data")
        .option("mergeSchema", "true")
        .trigger(availableNow=True)
        .toTable("commodity.landing.weather_data_inc")
        .awaitTermination())
    print("✓ Weather data ingested")
except Exception as e:
    print(f"✗ Weather data error: {str(e)}")

# COMMAND ----------

# CFTC Data
print("\n[5/5] CFTC Data - Starting Auto Loader...")
try:
    (spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.schemaLocation", "s3://groundtruth-capstone/_schemas/cftc_data")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("header", "true")
        .option("pathGlobFilter", "*.csv")
        .load("s3://groundtruth-capstone/landing/cftc_data/")
        .withColumn("source_file", expr("_metadata.file_path"))
        .withColumn("ingest_ts", current_timestamp())
        .writeStream
        .format("delta")
        .option("checkpointLocation", "s3://groundtruth-capstone/_checkpoints/cftc_data")
        .option("mergeSchema", "true")
        .trigger(availableNow=True)
        .toTable("commodity.landing.cftc_data_inc")
        .awaitTermination())
    print("✓ CFTC data ingested")
except Exception as e:
    print(f"✗ CFTC data error: {str(e)}")

# COMMAND ----------

# MAGIC %md ## 2. Verify Landing Tables

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN commodity.landing

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'market_data_inc' as table, COUNT(*) as rows FROM commodity.landing.market_data_inc
# MAGIC UNION ALL
# MAGIC SELECT 'vix_data_inc', COUNT(*) FROM commodity.landing.vix_data_inc
# MAGIC UNION ALL
# MAGIC SELECT 'macro_data_inc', COUNT(*) FROM commodity.landing.macro_data_inc
# MAGIC UNION ALL
# MAGIC SELECT 'weather_data_inc', COUNT(*) FROM commodity.landing.weather_data_inc
# MAGIC UNION ALL
# MAGIC SELECT 'cftc_data_inc', COUNT(*) FROM commodity.landing.cftc_data_inc

# COMMAND ----------

# MAGIC %md ## 3. Create Bronze Views

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Market Data View
# MAGIC CREATE OR REPLACE VIEW commodity.bronze.v_market_data_all AS
# MAGIC SELECT *
# MAGIC FROM commodity.landing.market_data_inc
# MAGIC QUALIFY ROW_NUMBER() OVER (PARTITION BY date, commodity ORDER BY ingest_ts DESC) = 1

# COMMAND ----------

# MAGIC %sql
# MAGIC -- VIX Data View
# MAGIC CREATE OR REPLACE VIEW commodity.bronze.v_vix_data_all AS
# MAGIC SELECT date, vix
# MAGIC FROM commodity.landing.vix_data_inc
# MAGIC QUALIFY ROW_NUMBER() OVER (PARTITION BY date ORDER BY ingest_ts DESC) = 1

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Macro/FX Data View
# MAGIC CREATE OR REPLACE VIEW commodity.bronze.v_macro_data_all AS
# MAGIC SELECT *
# MAGIC FROM commodity.landing.macro_data_inc
# MAGIC QUALIFY ROW_NUMBER() OVER (PARTITION BY date ORDER BY ingest_ts DESC) = 1

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Weather Data View
# MAGIC CREATE OR REPLACE VIEW commodity.bronze.v_weather_data_all AS
# MAGIC SELECT *
# MAGIC FROM commodity.landing.weather_data_inc
# MAGIC QUALIFY ROW_NUMBER() OVER (PARTITION BY date, region, commodity ORDER BY ingest_ts DESC) = 1

# COMMAND ----------

# MAGIC %sql
# MAGIC -- CFTC Data View
# MAGIC CREATE OR REPLACE VIEW commodity.bronze.v_cftc_data_all AS
# MAGIC SELECT *
# MAGIC FROM commodity.landing.cftc_data_inc
# MAGIC QUALIFY ROW_NUMBER() OVER (PARTITION BY date, commodity ORDER BY ingest_ts DESC) = 1

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW VIEWS IN commodity.bronze

# COMMAND ----------

print("\n" + "="*60)
print("Bronze Layer Setup Complete!")
print("="*60)
print("\nVerify:")
print("  SELECT * FROM commodity.bronze.v_market_data_all LIMIT 10")
print("  SELECT * FROM commodity.bronze.v_macro_data_all LIMIT 10")

-- Databricks notebook source
-- MAGIC %md ##1. Create a unified view (hist + incremental)

-- COMMAND ----------

CREATE OR REPLACE VIEW commodity.bronze.v_market_data_all AS
SELECT * FROM commodity.landing.market_data_hist
UNION ALL
SELECT * FROM commodity.landing.market_data_inc;
DESCRIBE EXTENDED commodity.bronze.v_market_data_all;
SELECT * FROM commodity.bronze.v_market_data_all;

-- COMMAND ----------

CREATE OR REPLACE VIEW commodity.bronze.v_macro_data_all AS
WITH combined AS (
  SELECT
    *,
    1 AS priority -- Higher priority for incremental data
  FROM commodity.landing.macro_data_inc
  
  UNION ALL 
  
  SELECT
    *,
    2 AS priority -- Lower priority for historical data
  FROM commodity.landing.macro_data_hist
)
SELECT
  date,
  vnd_usd, cop_usd, idr_usd, etb_usd, hnl_usd, ugx_usd, pen_usd, xaf_usd,
  gtq_usd, gnf_usd, nio_usd, crc_usd, tzs_usd, kes_usd, lak_usd, pkr_usd,
  php_usd, egp_usd, ars_usd, rub_usd, try_usd,
  uah_usd,
  irr_usd, byn_usd
FROM combined
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY date
    ORDER BY priority ASC -- Select the incremental data (priority 1) first
) = 1;

DESCRIBE EXTENDED commodity.bronze.v_macro_data_all;
SELECT * FROM commodity.bronze.v_macro_data_all;


-- COMMAND ----------

CREATE OR REPLACE VIEW commodity.bronze.v_vix_data_all AS
SELECT DISTINCT date, vix
FROM (
SELECT * FROM commodity.landing.vix_data_hist
UNION
SELECT * FROM commodity.landing.vix_data_inc)
ORDER BY date;
DESCRIBE EXTENDED commodity.bronze.v_vix_data_all;
SELECT * FROM commodity.bronze.v_vix_data_all;

-- COMMAND ----------

CREATE OR REPLACE VIEW commodity.bronze.v_cftc_data_all AS
SELECT * FROM commodity.landing.cftc_data_hist
UNION ALL
SELECT * FROM commodity.landing.cftc_data_inc;
DESCRIBE EXTENDED commodity.bronze.v_cftc_data_all;
SELECT * FROM commodity.bronze.v_cftc_data_all;

-- COMMAND ----------

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
UNION ALL
SELECT
  date,
  max_temp_c,
  min_temp_c,
  precipitation_mm,
  humidity_pct,
  region,
  commodity
FROM commodity.landing.weather_data_hist;

DESCRIBE EXTENDED commodity.bronze.v_weather_data_all;
SELECT * FROM commodity.bronze.v_weather_data_all;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 2. One-time backfill of historical CSVs

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Macro Historical Data

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import input_file_name, current_timestamp, expr
-- MAGIC
-- MAGIC hist_path = "s3://berkeley-datasci210-capstone/landing/macro_data_hist"
-- MAGIC bronze_tbl = "commodity.bronze.macro_data_raw"
-- MAGIC checkpoint_hist = "s3://berkeley-datasci210-capstone/_checkpoints/macro_data_hist"
-- MAGIC schema_loc = "s3://berkeley-datasci210-capstone/_schemas/macro_data"
-- MAGIC
-- MAGIC (
-- MAGIC   spark.readStream
-- MAGIC     .format("cloudFiles")
-- MAGIC     .option("cloudFiles.format", "csv")
-- MAGIC     .option("cloudFiles.inferColumnTypes", "true")
-- MAGIC     .option("cloudFiles.schemaLocation", schema_loc)      # where inferred schema is stored
-- MAGIC     .option("header", "true")
-- MAGIC     .load(hist_path)
-- MAGIC     .withColumn("source_file", expr("_metadata.file_path"))
-- MAGIC     .withColumn("ingest_ts", current_timestamp())
-- MAGIC     .writeStream
-- MAGIC     .format("delta")
-- MAGIC     .option("checkpointLocation", checkpoint_hist)
-- MAGIC     .trigger(availableNow=True)                            # process everything and stop
-- MAGIC     .toTable(bronze_tbl)                                   # managed Delta in UC
-- MAGIC )
-- MAGIC

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dbutils.fs.ls("s3://berkeley-datasci210-capstone/")                  
-- MAGIC # see landing/ ?
-- MAGIC dbutils.fs.ls("s3://berkeley-datasci210-capstone/landing/")         
-- MAGIC # see market_data_hist or market_data_hist.csv ?
-- MAGIC dbutils.fs.ls("s3://berkeley-datasci210-capstone/landing/market_data_hist/")  # without the space
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Yahoo Finance Historical Market Data

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import input_file_name, current_timestamp, expr
-- MAGIC
-- MAGIC hist_path = "s3://berkeley-datasci210-capstone/landing/market_data_hist"
-- MAGIC bronze_tbl = "commodity.bronze.market_data_raw"
-- MAGIC checkpoint_hist = "s3://berkeley-datasci210-capstone/_checkpoints/market_data_hist"
-- MAGIC schema_loc = "s3://berkeley-datasci210-capstone/_schemas/market_data"
-- MAGIC
-- MAGIC (
-- MAGIC   spark.readStream
-- MAGIC     .format("cloudFiles")
-- MAGIC     .option("cloudFiles.format", "csv")
-- MAGIC     .option("cloudFiles.inferColumnTypes", "true")
-- MAGIC     .option("cloudFiles.schemaLocation", schema_loc)
-- MAGIC     .option("header", "true")
-- MAGIC     .load(hist_path)
-- MAGIC     .withColumn("source_file", expr("_metadata.file_path"))
-- MAGIC     .withColumn("ingest_ts", current_timestamp())
-- MAGIC     .writeStream
-- MAGIC     .format("delta")
-- MAGIC     .option("checkpointLocation", checkpoint_hist)
-- MAGIC     .trigger(availableNow=True)
-- MAGIC     .toTable(bronze_tbl)
-- MAGIC )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## VIX Volatility Historical Data

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import input_file_name, current_timestamp, expr
-- MAGIC
-- MAGIC hist_path = "s3://berkeley-datasci210-capstone/landing/vix_data_hist"
-- MAGIC bronze_tbl = "commodity.bronze.vix_data_raw"
-- MAGIC checkpoint_hist = "s3://berkeley-datasci210-capstone/_checkpoints/vix_data_hist"
-- MAGIC schema_loc = "s3://berkeley-datasci210-capstone/_schemas/vix_data"
-- MAGIC
-- MAGIC (
-- MAGIC   spark.readStream
-- MAGIC     .format("cloudFiles")
-- MAGIC     .option("cloudFiles.format", "csv")
-- MAGIC     .option("cloudFiles.inferColumnTypes", "true")
-- MAGIC     .option("cloudFiles.schemaLocation", schema_loc)
-- MAGIC     .option("header", "true")
-- MAGIC     .load(hist_path)
-- MAGIC     .withColumn("source_file", expr("_metadata.file_path"))
-- MAGIC     .withColumn("ingest_ts", current_timestamp())
-- MAGIC     .writeStream
-- MAGIC     .format("delta")
-- MAGIC     .option("checkpointLocation", checkpoint_hist)
-- MAGIC     .trigger(availableNow=True)
-- MAGIC     .toTable(bronze_tbl)
-- MAGIC )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Historical Trade Data

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import input_file_name, current_timestamp, expr
-- MAGIC import re
-- MAGIC
-- MAGIC def sanitize(name: str) -> str:
-- MAGIC     # replace spaces and forbidden chars [ ,;{}()\n\t=] with _
-- MAGIC     n = re.sub(r"[ ,;{}\(\)\n\t=]+", "_", name.strip())
-- MAGIC     # collapse repeats and lowercase
-- MAGIC     n = re.sub(r"_+", "_", n).strip("_").lower()
-- MAGIC     return n
-- MAGIC
-- MAGIC def sanitize_cols(df):
-- MAGIC     safe = []
-- MAGIC     seen = set()
-- MAGIC     for c in df.columns:
-- MAGIC         s = sanitize(c)
-- MAGIC         # avoid accidental duplicates after sanitizing
-- MAGIC         i, base = 1, s
-- MAGIC         while s in seen:
-- MAGIC             i += 1
-- MAGIC             s = f"{base}_{i}"
-- MAGIC         seen.add(s)
-- MAGIC         safe.append(s)
-- MAGIC     return df.toDF(*safe)
-- MAGIC
-- MAGIC hist_path = "s3://berkeley-datasci210-capstone/landing/faostat_data_hist"
-- MAGIC bronze_tbl = "commodity.bronze.trade_data_raw"
-- MAGIC checkpoint_hist = "s3://berkeley-datasci210-capstone/_checkpoints/trade_data_hist"
-- MAGIC schema_loc = "s3://berkeley-datasci210-capstone/_schemas/trade_data"
-- MAGIC
-- MAGIC (
-- MAGIC   spark.readStream
-- MAGIC     .format("cloudFiles")
-- MAGIC     .option("cloudFiles.format", "csv")
-- MAGIC     .option("cloudFiles.inferColumnTypes", "true")
-- MAGIC     .option("cloudFiles.schemaLocation", schema_loc)
-- MAGIC     .option("header", "true")
-- MAGIC     .option("pathGlobFilter", "*.csv")
-- MAGIC     .load(hist_path)
-- MAGIC     .transform(sanitize_cols)
-- MAGIC     .withColumn("source_file", expr("_metadata.file_path"))
-- MAGIC     .withColumn("ingest_ts", current_timestamp())
-- MAGIC     .writeStream
-- MAGIC     .format("delta")
-- MAGIC     .option("checkpointLocation", checkpoint_hist)
-- MAGIC     .option("mergeSchema", "true")
-- MAGIC     .trigger(availableNow=True)
-- MAGIC     .toTable(bronze_tbl)
-- MAGIC )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Open Meteo Historical Weather Data

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import col, to_date, current_timestamp, expr
-- MAGIC
-- MAGIC hist_path = "s3://berkeley-datasci210-capstone/landing/weather_data_hist"
-- MAGIC bronze_tbl = "commodity.bronze.weather_data_raw"
-- MAGIC checkpoint_hist = "s3://berkeley-datasci210-capstone/_checkpoints/weather_data_hist"
-- MAGIC schema_loc = "s3://berkeley-datasci210-capstone/_schemas/weather_data"
-- MAGIC
-- MAGIC (
-- MAGIC     spark.readStream
-- MAGIC     .format("cloudFiles")
-- MAGIC     .option("cloudFiles.format", "csv")
-- MAGIC     .option("cloudFiles.schemaLocation", schema_loc)
-- MAGIC     .option("cloudFiles.inferColumnTypes", "true")
-- MAGIC     .option("header", "true")
-- MAGIC     .option("pathGlobFilter", "*.csv")
-- MAGIC     .load(hist_path)
-- MAGIC     .withColumn("source_file", expr("_metadata.file_path"))
-- MAGIC     .withColumn("ingest_ts", current_timestamp())
-- MAGIC     .writeStream
-- MAGIC     .format("delta")
-- MAGIC     .option("checkpointLocation", checkpoint_hist)
-- MAGIC     .option("mergeSchema", "true")
-- MAGIC     .trigger(availableNow=True)
-- MAGIC     .toTable(bronze_tbl)
-- MAGIC )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Commodity Futures Trading Commision Historical Data

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import input_file_name, current_timestamp, expr
-- MAGIC import re
-- MAGIC
-- MAGIC def sanitize(name: str) -> str:
-- MAGIC     # replace spaces and forbidden chars [ ,;{}()\n\t=] with _
-- MAGIC     n = re.sub(r"[ ,;{}\(\)\n\t=]+", "_", name.strip())
-- MAGIC     # collapse repeats and lowercase
-- MAGIC     n = re.sub(r"_+", "_", n).strip("_").lower()
-- MAGIC     return n
-- MAGIC
-- MAGIC def sanitize_cols(df):
-- MAGIC     safe = []
-- MAGIC     seen = set()
-- MAGIC     for c in df.columns:
-- MAGIC         s = sanitize(c)
-- MAGIC         # avoid accidental duplicates after sanitizing
-- MAGIC         i, base = 1, s
-- MAGIC         while s in seen:
-- MAGIC             i += 1
-- MAGIC             s = f"{base}_{i}"
-- MAGIC         seen.add(s)
-- MAGIC         safe.append(s)
-- MAGIC     return df.toDF(*safe)
-- MAGIC
-- MAGIC hist_path = "s3://berkeley-datasci210-capstone/landing/cftc_data_hist"
-- MAGIC bronze_tbl = "commodity.bronze.cftc_data_raw"
-- MAGIC checkpoint_hist = "s3://berkeley-datasci210-capstone/_checkpoints/cftc_data_hist"
-- MAGIC schema_loc = "s3://berkeley-datasci210-capstone/_schemas/cftc_data"
-- MAGIC
-- MAGIC (
-- MAGIC   spark.readStream
-- MAGIC     .format("cloudFiles")
-- MAGIC     .option("cloudFiles.format", "csv")
-- MAGIC     .option("cloudFiles.inferColumnTypes", "true")
-- MAGIC     .option("cloudFiles.schemaLocation", schema_loc)
-- MAGIC     .option("header", "true")
-- MAGIC     .option("pathGlobFilter", "*.csv")
-- MAGIC     .load(hist_path)
-- MAGIC     .transform(sanitize_cols)
-- MAGIC     .withColumn("source_file", expr("_metadata.file_path"))
-- MAGIC     .withColumn("ingest_ts", current_timestamp())
-- MAGIC     .writeStream
-- MAGIC     .format("delta")
-- MAGIC     .option("checkpointLocation", checkpoint_hist)
-- MAGIC     .option("mergeSchema", "true")
-- MAGIC     .trigger(availableNow=True)
-- MAGIC     .toTable(bronze_tbl)
-- MAGIC )
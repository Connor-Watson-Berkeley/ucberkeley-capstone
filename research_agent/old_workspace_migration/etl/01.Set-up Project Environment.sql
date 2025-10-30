-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Set-up the project environment for Commodity Data Lakehouse

-- COMMAND ----------

-- MAGIC %md ## 1. Access the bucket berkeley-datasci210-capstone 

-- COMMAND ----------

-- MAGIC %fs ls s3://berkeley-datasci210-capstone 

-- COMMAND ----------

-- MAGIC %md ## 2. Create the catalog - commodity

-- COMMAND ----------

show catalogs;

-- COMMAND ----------

CREATE CATALOG IF NOT EXISTS commodity
  MANAGED LOCATION 's3://berkeley-datasci210-capstone/'
  COMMENT 'This is a catalog for the Commodity Data Lakehouse';

-- COMMAND ----------

-- MAGIC %md ##3. Create Schemas
-- MAGIC
-- MAGIC       1. Landing
-- MAGIC       2. Bronze
-- MAGIC       3. Silver
-- MAGIC       4. Gold
-- MAGIC
-- MAGIC

-- COMMAND ----------

use catalog commodity;

CREATE SCHEMA IF NOT EXISTS landing
    MANAGED LOCATION 's3://berkeley-datasci210-capstone/landing';

CREATE SCHEMA IF NOT EXISTS bronze
    MANAGED LOCATION 's3://berkeley-datasci210-capstone/bronze';

CREATE SCHEMA IF NOT EXISTS silver
    MANAGED LOCATION 's3://berkeley-datasci210-capstone/silver';

CREATE SCHEMA IF NOT EXISTS gold
    MANAGED LOCATION 's3://berkeley-datasci210-capstone/gold';

-- COMMAND ----------

USE CATALOG commodity;
DROP SCHEMA IF EXISTS landing CASCADE; 
CREATE SCHEMA IF NOT EXISTS landing 
  MANAGED LOCATION 's3://berkeley-datasci210-capstone/landing-managed/';

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 4. Alter an External Location

-- COMMAND ----------

SHOW EXTERNAL LOCATIONS;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 5. Create Tables

-- COMMAND ----------

CREATE TABLE IF NOT EXISTS commodity.landing.macro_data_inc
USING CSV
OPTIONS (
  header = true,
  inferSchema = true
)
LOCATION 's3://berkeley-datasci210-capstone/landing/macro_data/';

CREATE TABLE IF NOT EXISTS commodity.landing.macro_data_hist
USING CSV
OPTIONS (
  header = true,
  inferSchema = true
)
LOCATION 's3://berkeley-datasci210-capstone/landing/macro_data_hist/';

-- COMMAND ----------

CREATE TABLE IF NOT EXISTS commodity.landing.market_data_inc
USING CSV
OPTIONS (
  header = true,
  inferSchema = true
)
LOCATION 's3://berkeley-datasci210-capstone/landing/market_data/';

CREATE TABLE IF NOT EXISTS commodity.landing.market_data_hist (
  date DATE,
  close DOUBLE,
  high DOUBLE,
  low DOUBLE,
  open DOUBLE,
  volume DOUBLE,
  commodity STRING
)
USING CSV
OPTIONS (
  header = true
)
LOCATION 's3://berkeley-datasci210-capstone/landing/market_data_hist/';

-- COMMAND ----------

ALTER TABLE commodity.landing.macro_data_hist
  ADD COLUMNS (uah_usd DOUBLE);

-- COMMAND ----------

CREATE TABLE IF NOT EXISTS commodity.landing.vix_data_inc
USING CSV
OPTIONS (
  header = true,
  inferSchema = true
)
LOCATION 's3://berkeley-datasci210-capstone/landing/vix_data/';

CREATE TABLE IF NOT EXISTS commodity.landing.vix_data_hist
USING CSV
OPTIONS (
  header = true,
  inferSchema = true
)
LOCATION 's3://berkeley-datasci210-capstone/landing/vix_data_hist/';

-- COMMAND ----------

CREATE TABLE IF NOT EXISTS commodity.landing.cftc_data_inc
USING CSV
OPTIONS (
  header = true,
  inferSchema = true
)
LOCATION 's3://berkeley-datasci210-capstone/landing/cftc_data/';

CREATE TABLE IF NOT EXISTS commodity.landing.cftc_data_hist
USING CSV
OPTIONS (
  header = true,
  inferSchema = true
)
LOCATION 's3://berkeley-datasci210-capstone/landing/cftc_data_hist/';

-- COMMAND ----------

DROP TABLE IF EXISTS commodity.landing.weather_data_inc;

CREATE TABLE commodity.landing.weather_data_inc
USING CSV
OPTIONS (
  header = true,
  inferSchema = true
)
LOCATION 's3://berkeley-datasci210-capstone/landing/weather_data/';

DROP TABLE IF EXISTS commodity.landing.weather_data_hist;

CREATE TABLE commodity.landing.weather_data_hist
USING CSV
OPTIONS (
  header = true,
  inferSchema = true
)
LOCATION 's3://berkeley-datasci210-capstone/landing/weather_data_hist/';

-- COMMAND ----------

USE CATALOG commodity;
USE SCHEMA silver;
DROP TABLE IF EXISTS bronze_gkg;
DROP TABLE IF EXISTS gold_theme_cooccurrence;
DROP TABLE IF EXISTS gold_daily_commodity_sentiment;
DROP TABLE IF EXISTS gold_daily_theme_counts;
DROP TABLE IF EXISTS gold_daily_geo_counts;
DROP TABLE IF EXISTS gold_daily_actor_counts;
DROP TABLE IF EXISTS gold_driver_specific_sentiment;
DROP TABLE IF EXISTS gold_ml_features_daily;
DROP TABLE IF EXISTS gold_ml_features_hourly;
DROP TABLE IF EXISTS gold_ml_features_hourly_agg;
DROP TABLE IF EXISTS gold_weekly_rolling_sentiment;
DROP TABLE IF EXISTS commodity.gold.gold_sugar_weekly_merged;
DROP TABLE IF EXISTS commodity.silver.coffee_weekly;
DROP TABLE IF EXISTS commodity.silver.coffee_weekly_merged;
DROP TABLE IF EXISTS commodity.silver.silver_sugar_monthly;
DROP TABLE IF EXISTS commodity.silver.silver_sugar_weekly;
DROP TABLE IF EXISTS commodity.silver.sugar_daily_merged;
DROP TABLE IF EXISTS commodity.silver.sugar_monthly;
DROP TABLE IF EXISTS commodity.silver.sugar_weekly;
DROP TABLE IF EXISTS commodity.silver.sugar_weekly_merged;
DROP TABLE IF EXISTS commodity.silver.coffee_monthly;
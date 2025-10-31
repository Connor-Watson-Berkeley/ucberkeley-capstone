-- ============================================
-- DATABRICKS BRONZE LAYER - Deduplication Views
-- ============================================
-- Creates views with automatic deduplication
-- Uses QUALIFY to keep only latest version of each record

USE CATALOG commodity;

-- Market Data (Coffee & Sugar) - Full OHLCV data
CREATE OR REPLACE VIEW commodity.bronze.v_market_data_all AS
SELECT date, commodity, open, high, low, close, volume
FROM commodity.landing.market_data_inc
QUALIFY ROW_NUMBER() OVER (PARTITION BY date, commodity ORDER BY ingest_ts DESC) = 1;

-- Macro Data (FX rates)
CREATE OR REPLACE VIEW commodity.bronze.v_macro_data_all AS
SELECT *
FROM commodity.landing.macro_data_inc
QUALIFY ROW_NUMBER() OVER (PARTITION BY date ORDER BY ingest_ts DESC) = 1;

-- VIX Data (volatility)
CREATE OR REPLACE VIEW commodity.bronze.v_vix_data_all AS
SELECT date, vix
FROM commodity.landing.vix_data_inc
QUALIFY ROW_NUMBER() OVER (PARTITION BY date ORDER BY ingest_ts DESC) = 1;

-- Weather Data
CREATE OR REPLACE VIEW commodity.bronze.v_weather_data_all AS
SELECT *
FROM commodity.landing.weather_data_inc
QUALIFY ROW_NUMBER() OVER (PARTITION BY date, region, commodity ORDER BY ingest_ts DESC) = 1;

-- CFTC Data (trader positioning)
CREATE OR REPLACE VIEW commodity.bronze.v_cftc_data_all AS
SELECT *
FROM commodity.landing.cftc_data_inc
QUALIFY ROW_NUMBER() OVER (PARTITION BY date, market_name ORDER BY ingest_ts DESC) = 1;

-- GDELT News Sentiment (all records, no deduplication)
CREATE OR REPLACE VIEW commodity.bronze.v_gdelt_sentiment_all AS
SELECT
  date,
  source_url,
  themes,
  locations,
  persons,
  organizations,
  tone,
  all_names
FROM commodity.landing.gdelt_sentiment_inc;

-- Verify views created
SHOW VIEWS IN commodity.bronze;

-- Sample query: Coffee prices over time
SELECT * FROM commodity.bronze.v_market_data_all
WHERE commodity = 'Coffee'
ORDER BY date DESC
LIMIT 10;

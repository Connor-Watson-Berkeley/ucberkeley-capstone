-- ============================================================================
-- Weather Forecast Tables Setup
-- ============================================================================
-- Creates landing, bronze, and integration for 14-day weather forecasts
--
-- ⚠️  CRITICAL DATA LIMITATION - DATA LEAKAGE IN SYNTHETIC FORECASTS:
--     Weather forecasts 2015-2025 are SYNTHETIC (generated from actuals + error)
--     ⚠️  HAS DATA LEAKAGE - May overestimate forecast utility
--     Real forecasts from 2025-11-06+ (no data leakage)
--     Models using synthetic data MUST document the limitation
--     See research_agent/DATA_SOURCES.md and WEATHER_FORECAST_LIMITATION.md
--
-- Run this in Databricks SQL Editor
-- ============================================================================

-- ============================================================================
-- STEP 1: Create landing table (external table pointing to S3)
-- ============================================================================

-- NOTE: Update LOCATION path to your actual S3 bucket
CREATE EXTERNAL TABLE IF NOT EXISTS commodity.landing.weather_forecast_inc (
  forecast_date DATE COMMENT 'Date the forecast was made',
  target_date DATE COMMENT 'Date being forecasted',
  days_ahead INT COMMENT 'Days ahead (1-16)',
  region STRING COMMENT 'Region name (e.g., Antigua_Guatemala)',
  temp_max_c DECIMAL(5,2) COMMENT 'Forecasted maximum temperature (°C)',
  temp_min_c DECIMAL(5,2) COMMENT 'Forecasted minimum temperature (°C)',
  temp_mean_c DECIMAL(5,2) COMMENT 'Forecasted mean temperature (°C)',
  precipitation_mm DECIMAL(6,2) COMMENT 'Forecasted precipitation (mm)',
  humidity_pct DECIMAL(5,2) COMMENT 'Forecasted humidity (%)',
  wind_speed_kmh DECIMAL(5,2) COMMENT 'Forecasted wind speed (km/h)',
  ingest_ts TIMESTAMP COMMENT 'When this forecast was ingested',
  is_synthetic BOOLEAN COMMENT '⚠️ TRUE if synthetic forecast (has data leakage), FALSE if real',
  has_data_leakage BOOLEAN COMMENT '⚠️ TRUE if forecast was generated from actuals (synthetic)',
  generation_method STRING COMMENT 'How forecast was generated (e.g., actual_weather_plus_realistic_error, open_meteo_api)'
)
USING DELTA
PARTITIONED BY (forecast_date)
LOCATION 's3://commodity-data/weather-forecast/'
COMMENT '⚠️ WARNING: Forecasts 2015-2025 are SYNTHETIC with DATA LEAKAGE. Real forecasts from 2025-11-06+. See DATA_SOURCES.md';

-- ============================================================================
-- STEP 2: Create bronze table (deduplicated forecasts)
-- ============================================================================

CREATE OR REPLACE TABLE commodity.bronze.weather_forecast
COMMENT '⚠️ DATA LEAKAGE: Forecasts 2015-2025 are SYNTHETIC (actual+error), 2025-11-06+ are REAL. Models MUST document synthetic usage. See DATA_SOURCES.md'
AS
SELECT
  forecast_date,
  target_date,
  days_ahead,
  region,
  temp_max_c,
  temp_min_c,
  temp_mean_c,
  precipitation_mm,
  humidity_pct,
  wind_speed_kmh,
  ingest_ts,
  COALESCE(is_synthetic, FALSE) as is_synthetic,
  COALESCE(has_data_leakage, FALSE) as has_data_leakage,
  COALESCE(generation_method, 'open_meteo_api') as generation_method
FROM commodity.landing.weather_forecast_inc
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY forecast_date, target_date, region
  ORDER BY ingest_ts DESC
) = 1;

-- Add partitioning for performance
ALTER TABLE commodity.bronze.weather_forecast
SET TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact' = 'true'
);

-- ============================================================================
-- STEP 3: Validation queries
-- ============================================================================

-- Check row counts
SELECT
  'landing' as layer,
  COUNT(*) as row_count,
  COUNT(DISTINCT region) as regions,
  COUNT(DISTINCT forecast_date) as forecast_dates,
  MIN(forecast_date) as earliest_forecast,
  MAX(forecast_date) as latest_forecast
FROM commodity.landing.weather_forecast_inc
UNION ALL
SELECT
  'bronze' as layer,
  COUNT(*) as row_count,
  COUNT(DISTINCT region) as regions,
  COUNT(DISTINCT forecast_date) as forecast_dates,
  MIN(forecast_date) as earliest_forecast,
  MAX(forecast_date) as latest_forecast
FROM commodity.bronze.weather_forecast;

-- Check data quality
SELECT
  forecast_date,
  COUNT(DISTINCT region) as regions_with_forecasts,
  COUNT(DISTINCT target_date) as days_forecasted,
  AVG(temp_mean_c) as avg_temp_forecasted,
  SUM(precipitation_mm) as total_precip_forecasted,
  COUNT(*) as total_records
FROM commodity.bronze.weather_forecast
GROUP BY forecast_date
ORDER BY forecast_date DESC
LIMIT 10;

-- Check forecast horizon distribution
SELECT
  days_ahead,
  COUNT(*) as forecasts,
  COUNT(DISTINCT region) as regions,
  AVG(temp_mean_c) as avg_temp
FROM commodity.bronze.weather_forecast
WHERE forecast_date = (SELECT MAX(forecast_date) FROM commodity.bronze.weather_forecast)
GROUP BY days_ahead
ORDER BY days_ahead;

-- ============================================================================
-- STEP 4: Sample integration with unified_data
-- ============================================================================

-- Example: Join unified_data with 14-day ahead forecast
-- This shows how to incorporate forecast data into your modeling pipeline

SELECT
  ud.date,
  ud.commodity,
  ud.region,
  ud.close as actual_price,
  ud.temp_mean_c as actual_temp,
  wf.temp_mean_c as forecasted_temp_14d_ago,
  wf.precipitation_mm as forecasted_precip_14d_ago,
  (ud.temp_mean_c - wf.temp_mean_c) as temp_forecast_error
FROM commodity.silver.unified_data ud
LEFT JOIN commodity.bronze.weather_forecast wf
  ON ud.date = wf.target_date
  AND ud.region = wf.region
  AND wf.days_ahead = 14  -- Join on 14-day ahead forecast
WHERE ud.date >= '2024-01-01'
  AND ud.commodity = 'Coffee'
  AND wf.forecast_date IS NOT NULL  -- Only where forecast existed
LIMIT 100;

-- ============================================================================
-- STEP 5: Create view for forecast accuracy analysis
-- ============================================================================

CREATE OR REPLACE VIEW commodity.bronze.weather_forecast_accuracy AS
SELECT
  wf.forecast_date,
  wf.target_date,
  wf.days_ahead,
  wf.region,
  wf.temp_mean_c as forecasted_temp,
  w.temp_mean_c as actual_temp,
  ABS(wf.temp_mean_c - w.temp_mean_c) as temp_error,
  wf.precipitation_mm as forecasted_precip,
  w.rain_mm + COALESCE(w.snowfall_cm * 10, 0) as actual_precip,
  ABS(wf.precipitation_mm - (w.rain_mm + COALESCE(w.snowfall_cm * 10, 0))) as precip_error
FROM commodity.bronze.weather_forecast wf
INNER JOIN commodity.bronze.weather w
  ON wf.target_date = w.date
  AND wf.region = w.region
WHERE wf.target_date <= CURRENT_DATE()  -- Only compare for dates that have occurred
COMMENT 'Weather forecast accuracy analysis - compare forecasts to actuals';

-- Check forecast accuracy
SELECT
  days_ahead,
  COUNT(*) as forecasts_evaluated,
  AVG(temp_error) as mae_temp,
  AVG(precip_error) as mae_precip,
  PERCENTILE(temp_error, 0.5) as median_temp_error,
  PERCENTILE(precip_error, 0.5) as median_precip_error
FROM commodity.bronze.weather_forecast_accuracy
WHERE forecast_date >= DATE_SUB(CURRENT_DATE(), 30)  -- Last 30 days
GROUP BY days_ahead
ORDER BY days_ahead;

-- ============================================================================
-- STEP 6: Grant permissions (if using Unity Catalog access control)
-- ============================================================================

-- Grant read access to data science team
-- GRANT SELECT ON TABLE commodity.landing.weather_forecast_inc TO `data-science-team`;
-- GRANT SELECT ON TABLE commodity.bronze.weather_forecast TO `data-science-team`;
-- GRANT SELECT ON VIEW commodity.bronze.weather_forecast_accuracy TO `data-science-team`;

-- ============================================================================
-- FINAL STATE
-- ============================================================================
-- Landing: commodity.landing.weather_forecast_inc (raw incremental data from S3)
-- Bronze: commodity.bronze.weather_forecast (deduplicated forecasts)
-- View: commodity.bronze.weather_forecast_accuracy (forecast error analysis)
-- ============================================================================

-- Show summary
SELECT
  'Landing Table' as table_name,
  COUNT(*) as records,
  MIN(forecast_date) as earliest,
  MAX(forecast_date) as latest
FROM commodity.landing.weather_forecast_inc
UNION ALL
SELECT
  'Bronze Table' as table_name,
  COUNT(*) as records,
  MIN(forecast_date) as earliest,
  MAX(forecast_date) as latest
FROM commodity.bronze.weather_forecast;

-- =============================================================================
-- CREATE GOLD.UNIFIED_DATA - Array-based Multi-Regional Architecture
-- =============================================================================
-- Purpose: Unified commodity data with weather/GDELT as arrays of structs
-- Grain: (date, commodity) - ~7k rows
-- Benefits:
--   - 90% fewer rows than silver.unified_data (~7k vs ~75k)
--   - Models can aggregate regions flexibly (mean, weighted, separate features)
--   - Clean array structure for PySpark transformers
--   - Forward-fill handles missing GDELT dates (not every day has articles)
-- =============================================================================

-- Create gold schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS commodity.gold
COMMENT 'Gold layer: Production-ready aggregated data for ML models';

-- Create the unified_data table
CREATE OR REPLACE TABLE commodity.gold.unified_data AS

-- =============================================================================
-- STEP 1: CREATE COMPLETE DATE SPINE
-- =============================================================================
WITH date_spine AS (
  SELECT date_add('2015-07-07', x - 1) as date
  FROM (SELECT explode(sequence(1, 10000)) as x)
  WHERE date_add('2015-07-07', x - 1) <= current_date()
),

-- =============================================================================
-- STEP 2: DEDUPLICATE GLOBAL DATA (Same as silver.unified_data)
-- =============================================================================

-- Market Data: Full OHLCV data
market_clean AS (
  SELECT date, commodity, open, high, low, close, volume
  FROM commodity.bronze.market
  WHERE date >= '2015-07-07'
),

-- VIX: Simple DISTINCT (all duplicates are identical values)
vix_clean AS (
  SELECT DISTINCT date, vix
  FROM commodity.bronze.vix
  WHERE date >= '2015-07-07'
),

-- Macro: Pick row with most non-null columns
macro_ranked AS (
  SELECT
    date,
    vnd_usd, cop_usd, idr_usd, etb_usd, hnl_usd, ugx_usd, pen_usd, xaf_usd,
    gtq_usd, gnf_usd, nio_usd, crc_usd, tzs_usd, kes_usd, lak_usd, pkr_usd,
    php_usd, egp_usd, ars_usd, rub_usd, try_usd, uah_usd, irr_usd, byn_usd,
    ROW_NUMBER() OVER (
      PARTITION BY date
      ORDER BY
        (CASE WHEN vnd_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN cop_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN idr_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN etb_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN hnl_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN ugx_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN pen_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN xaf_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN gtq_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN gnf_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN nio_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN crc_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN tzs_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN kes_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN lak_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN pkr_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN php_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN egp_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN ars_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN rub_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN try_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN uah_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN irr_usd IS NOT NULL THEN 1 ELSE 0 END +
         CASE WHEN byn_usd IS NOT NULL THEN 1 ELSE 0 END) DESC
    ) as rn
  FROM commodity.bronze.macro
  WHERE date >= '2015-07-07'
),

macro_clean AS (
  SELECT
    date, vnd_usd, cop_usd, idr_usd, etb_usd, hnl_usd, ugx_usd, pen_usd, xaf_usd,
    gtq_usd, gnf_usd, nio_usd, crc_usd, tzs_usd, kes_usd, lak_usd, pkr_usd,
    php_usd, egp_usd, ars_usd, rub_usd, try_usd, uah_usd, irr_usd, byn_usd
  FROM macro_ranked
  WHERE rn = 1
),

-- =============================================================================
-- STEP 3: IDENTIFY TRADING DAYS
-- =============================================================================

trading_days AS (
  SELECT DISTINCT
    date,
    commodity,
    1 as is_trading_day
  FROM market_clean
),

-- =============================================================================
-- STEP 4: FORWARD FILL SCALAR DATA ONTO DATE SPINE
-- =============================================================================

commodities AS (
  SELECT 'Coffee' as commodity UNION ALL SELECT 'Sugar' as commodity
),

date_commodity_spine AS (
  SELECT ds.date, c.commodity
  FROM date_spine ds
  CROSS JOIN commodities c
),

market_filled AS (
  SELECT
    dcs.date,
    dcs.commodity,
    LAST_VALUE(mc.open, true) OVER (PARTITION BY dcs.commodity ORDER BY dcs.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as open,
    LAST_VALUE(mc.high, true) OVER (PARTITION BY dcs.commodity ORDER BY dcs.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as high,
    LAST_VALUE(mc.low, true) OVER (PARTITION BY dcs.commodity ORDER BY dcs.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as low,
    LAST_VALUE(mc.close, true) OVER (PARTITION BY dcs.commodity ORDER BY dcs.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as close,
    LAST_VALUE(mc.volume, true) OVER (PARTITION BY dcs.commodity ORDER BY dcs.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as volume
  FROM date_commodity_spine dcs
  LEFT JOIN market_clean mc ON dcs.date = mc.date AND dcs.commodity = mc.commodity
),

vix_filled AS (
  SELECT
    ds.date,
    LAST_VALUE(vc.vix, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as vix
  FROM date_spine ds
  LEFT JOIN vix_clean vc ON ds.date = vc.date
),

macro_filled AS (
  SELECT
    ds.date,
    LAST_VALUE(mc.vnd_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as vnd_usd,
    LAST_VALUE(mc.cop_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as cop_usd,
    LAST_VALUE(mc.idr_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as idr_usd,
    LAST_VALUE(mc.etb_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as etb_usd,
    LAST_VALUE(mc.hnl_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as hnl_usd,
    LAST_VALUE(mc.ugx_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as ugx_usd,
    LAST_VALUE(mc.pen_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as pen_usd,
    LAST_VALUE(mc.xaf_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as xaf_usd,
    LAST_VALUE(mc.gtq_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as gtq_usd,
    LAST_VALUE(mc.gnf_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as gnf_usd,
    LAST_VALUE(mc.nio_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as nio_usd,
    LAST_VALUE(mc.crc_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as crc_usd,
    LAST_VALUE(mc.tzs_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as tzs_usd,
    LAST_VALUE(mc.kes_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as kes_usd,
    LAST_VALUE(mc.lak_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as lak_usd,
    LAST_VALUE(mc.pkr_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as pkr_usd,
    LAST_VALUE(mc.php_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as php_usd,
    LAST_VALUE(mc.egp_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as egp_usd,
    LAST_VALUE(mc.ars_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as ars_usd,
    LAST_VALUE(mc.rub_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as rub_usd,
    LAST_VALUE(mc.try_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as try_usd,
    LAST_VALUE(mc.uah_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as uah_usd,
    LAST_VALUE(mc.irr_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as irr_usd,
    LAST_VALUE(mc.byn_usd, true) OVER (ORDER BY ds.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as byn_usd
  FROM date_spine ds
  LEFT JOIN macro_clean mc ON ds.date = mc.date
),

-- =============================================================================
-- STEP 5: WEATHER DATA AS ARRAY OF STRUCTS (Multi-Regional)
-- =============================================================================

weather_with_forward_fill AS (
  SELECT
    date,
    region,
    commodity,
    -- Forward fill all weather fields
    LAST_VALUE(temp_max_c, true) OVER (PARTITION BY region, commodity ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as temp_max_c,
    LAST_VALUE(temp_min_c, true) OVER (PARTITION BY region, commodity ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as temp_min_c,
    LAST_VALUE(temp_mean_c, true) OVER (PARTITION BY region, commodity ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as temp_mean_c,
    LAST_VALUE(precipitation_mm, true) OVER (PARTITION BY region, commodity ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as precipitation_mm,
    LAST_VALUE(rain_mm, true) OVER (PARTITION BY region, commodity ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as rain_mm,
    LAST_VALUE(snowfall_cm, true) OVER (PARTITION BY region, commodity ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as snowfall_cm,
    LAST_VALUE(humidity_mean_pct, true) OVER (PARTITION BY region, commodity ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as humidity_mean_pct,
    LAST_VALUE(wind_speed_max_kmh, true) OVER (PARTITION BY region, commodity ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as wind_speed_max_kmh
  FROM commodity.bronze.weather_v2
  WHERE date >= '2015-07-07'
),

-- Aggregate weather data into array of structs (one per region)
weather_array AS (
  SELECT
    date,
    commodity,
    collect_list(
      struct(
        region,
        temp_max_c,
        temp_min_c,
        temp_mean_c,
        precipitation_mm,
        rain_mm,
        snowfall_cm,
        humidity_mean_pct,
        wind_speed_max_kmh
      )
    ) as weather_data
  FROM weather_with_forward_fill
  GROUP BY date, commodity
),

-- =============================================================================
-- STEP 6: GDELT SENTIMENT AS ARRAY OF STRUCTS (Theme Groups)
-- =============================================================================

-- Convert wide format GDELT to array of structs
gdelt_long AS (
  SELECT
    article_date as date,
    commodity,
    stack(7,
      'SUPPLY',     group_SUPPLY_count,     group_SUPPLY_tone_avg,     group_SUPPLY_tone_positive,     group_SUPPLY_tone_negative,     group_SUPPLY_tone_polarity,
      'LOGISTICS',  group_LOGISTICS_count,  group_LOGISTICS_tone_avg,  group_LOGISTICS_tone_positive,  group_LOGISTICS_tone_negative,  group_LOGISTICS_tone_polarity,
      'TRADE',      group_TRADE_count,      group_TRADE_tone_avg,      group_TRADE_tone_positive,      group_TRADE_tone_negative,      group_TRADE_tone_polarity,
      'MARKET',     group_MARKET_count,     group_MARKET_tone_avg,     group_MARKET_tone_positive,     group_MARKET_tone_negative,     group_MARKET_tone_polarity,
      'POLICY',     group_POLICY_count,     group_POLICY_tone_avg,     group_POLICY_tone_positive,     group_POLICY_tone_negative,     group_POLICY_tone_polarity,
      'CORE',       group_CORE_count,       group_CORE_tone_avg,       group_CORE_tone_positive,       group_CORE_tone_negative,       group_CORE_tone_polarity,
      'OTHER',      group_OTHER_count,      group_OTHER_tone_avg,      group_OTHER_tone_positive,      group_OTHER_tone_negative,      group_OTHER_tone_polarity
    ) AS (theme_group, article_count, tone_avg, tone_positive, tone_negative, tone_polarity)
  FROM commodity.silver.gdelt_wide
  WHERE article_date >= '2015-07-07'
),

-- Aggregate GDELT into array of structs (one per theme group)
gdelt_array AS (
  SELECT
    date,
    commodity,
    collect_list(
      struct(
        theme_group,
        article_count,
        tone_avg,
        tone_positive,
        tone_negative,
        tone_polarity
      )
    ) as gdelt_themes
  FROM gdelt_long
  GROUP BY date, commodity
),

-- Forward-fill GDELT data (not every day has articles)
gdelt_filled AS (
  SELECT
    dcs.date,
    dcs.commodity,
    LAST_VALUE(ga.gdelt_themes, true) OVER (
      PARTITION BY dcs.commodity
      ORDER BY dcs.date
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as gdelt_themes
  FROM date_commodity_spine dcs
  LEFT JOIN gdelt_array ga ON dcs.date = ga.date AND dcs.commodity = ga.commodity
),

-- =============================================================================
-- STEP 7: FINAL JOIN - Combine all sources
-- =============================================================================

final_join AS (
  SELECT
    mf.date,
    mf.commodity,
    COALESCE(td.is_trading_day, 0) as is_trading_day,

    -- Market data (scalar)
    mf.open,
    mf.high,
    mf.low,
    mf.close,
    mf.volume,

    -- VIX (scalar)
    vf.vix,

    -- Exchange rates (scalar - 24 columns)
    macf.vnd_usd, macf.cop_usd, macf.idr_usd, macf.etb_usd, macf.hnl_usd,
    macf.ugx_usd, macf.pen_usd, macf.xaf_usd, macf.gtq_usd, macf.gnf_usd,
    macf.nio_usd, macf.crc_usd, macf.tzs_usd, macf.kes_usd, macf.lak_usd,
    macf.pkr_usd, macf.php_usd, macf.egp_usd, macf.ars_usd, macf.rub_usd,
    macf.try_usd, macf.uah_usd, macf.irr_usd, macf.byn_usd,

    -- Weather data (array of structs - one per region)
    wa.weather_data,

    -- GDELT sentiment (array of structs - one per theme group)
    gf.gdelt_themes

  FROM market_filled mf
  INNER JOIN vix_filled vf ON mf.date = vf.date
  INNER JOIN macro_filled macf ON mf.date = macf.date
  LEFT JOIN trading_days td ON mf.date = td.date AND mf.commodity = td.commodity
  LEFT JOIN weather_array wa ON mf.date = wa.date AND mf.commodity = wa.commodity
  LEFT JOIN gdelt_filled gf ON mf.date = gf.date AND mf.commodity = gf.commodity
)

-- =============================================================================
-- FINAL SELECT
-- =============================================================================

SELECT * FROM final_join
ORDER BY commodity, date;

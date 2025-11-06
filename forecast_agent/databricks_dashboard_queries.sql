-- ============================================================================
-- DATABRICKS FORECAST EVALUATION DASHBOARD
-- ============================================================================
-- Purpose: Visualize model performance across 40 historical forecast windows
-- Tables: commodity.forecast.distributions
-- Last Updated: 2025-11-01
-- ============================================================================

-- ============================================================================
-- QUERY 1: Model Comparison - Overall MAE by Model
-- ============================================================================
-- Chart Type: Bar Chart
-- X-axis: model_version
-- Y-axis: mae_7d
-- Description: Compare 7-day ahead forecast accuracy across all 5 models

WITH forecasts AS (
  SELECT
    model_version,
    forecast_start_date,
    AVG(day_7) as forecast_mean_day7
  FROM commodity.forecast.distributions
  WHERE commodity = 'Coffee'
    AND is_actuals = FALSE
    AND has_data_leakage = FALSE
  GROUP BY model_version, forecast_start_date
),
actuals AS (
  SELECT
    forecast_start_date,
    day_7 as actual_day7
  FROM commodity.forecast.distributions
  WHERE commodity = 'Coffee'
    AND path_id = 0
    AND is_actuals = TRUE
)
SELECT
  f.model_version,
  COUNT(*) as n_forecasts,
  AVG(ABS(f.forecast_mean_day7 - a.actual_day7)) as mae_7d,
  SQRT(AVG(POWER(f.forecast_mean_day7 - a.actual_day7, 2))) as rmse_7d,
  AVG(ABS(f.forecast_mean_day7 - a.actual_day7) / a.actual_day7 * 100) as mape_7d
FROM forecasts f
JOIN actuals a ON f.forecast_start_date = a.forecast_start_date
GROUP BY f.model_version
ORDER BY mae_7d ASC;


-- ============================================================================
-- QUERY 2: Forecast Accuracy Over Time
-- ============================================================================
-- Chart Type: Line Chart (one line per model)
-- X-axis: forecast_start_date
-- Y-axis: mae_7d
-- Series: model_version
-- Description: Track how each model's accuracy changes over time

WITH forecasts AS (
  SELECT
    model_version,
    forecast_start_date,
    AVG(day_7) as forecast_mean_day7
  FROM commodity.forecast.distributions
  WHERE commodity = 'Coffee'
    AND is_actuals = FALSE
    AND has_data_leakage = FALSE
  GROUP BY model_version, forecast_start_date
),
actuals AS (
  SELECT
    forecast_start_date,
    day_7 as actual_day7
  FROM commodity.forecast.distributions
  WHERE commodity = 'Coffee'
    AND path_id = 0
    AND is_actuals = TRUE
)
SELECT
  f.forecast_start_date,
  f.model_version,
  ABS(f.forecast_mean_day7 - a.actual_day7) as mae_7d
FROM forecasts f
JOIN actuals a ON f.forecast_start_date = a.forecast_start_date
ORDER BY f.forecast_start_date, f.model_version;


-- ============================================================================
-- QUERY 3: Multi-Horizon Performance (1-day, 7-day, 14-day)
-- ============================================================================
-- Chart Type: Grouped Bar Chart
-- X-axis: model_version
-- Y-axis: mae
-- Series: horizon (1d, 7d, 14d)
-- Description: Compare short vs long-term forecast accuracy

WITH forecasts AS (
  SELECT
    model_version,
    forecast_start_date,
    AVG(day_1) as forecast_1d,
    AVG(day_7) as forecast_7d,
    AVG(day_14) as forecast_14d
  FROM commodity.forecast.distributions
  WHERE commodity = 'Coffee'
    AND is_actuals = FALSE
    AND has_data_leakage = FALSE
  GROUP BY model_version, forecast_start_date
),
actuals AS (
  SELECT
    forecast_start_date,
    day_1 as actual_1d,
    day_7 as actual_7d,
    day_14 as actual_14d
  FROM commodity.forecast.distributions
  WHERE commodity = 'Coffee'
    AND path_id = 0
    AND is_actuals = TRUE
)
SELECT
  f.model_version,
  'Day 1' as horizon,
  AVG(ABS(f.forecast_1d - a.actual_1d)) as mae
FROM forecasts f
JOIN actuals a ON f.forecast_start_date = a.forecast_start_date
GROUP BY f.model_version

UNION ALL

SELECT
  f.model_version,
  'Day 7' as horizon,
  AVG(ABS(f.forecast_7d - a.actual_7d)) as mae
FROM forecasts f
JOIN actuals a ON f.forecast_start_date = a.forecast_start_date
GROUP BY f.model_version

UNION ALL

SELECT
  f.model_version,
  'Day 14' as horizon,
  AVG(ABS(f.forecast_14d - a.actual_14d)) as mae
FROM forecasts f
JOIN actuals a ON f.forecast_start_date = a.forecast_start_date
GROUP BY f.model_version

ORDER BY model_version, horizon;


-- ============================================================================
-- QUERY 4: Prediction Interval Coverage (95% Calibration)
-- ============================================================================
-- Chart Type: Bar Chart
-- X-axis: model_version
-- Y-axis: coverage_95
-- Description: Check if 95% prediction intervals contain 95% of actuals
-- Target: Should be ~95% for well-calibrated models

WITH forecasts AS (
  SELECT
    model_version,
    forecast_start_date,
    PERCENTILE(day_7, 0.025) as lower_95,
    PERCENTILE(day_7, 0.975) as upper_95
  FROM commodity.forecast.distributions
  WHERE commodity = 'Coffee'
    AND is_actuals = FALSE
    AND has_data_leakage = FALSE
  GROUP BY model_version, forecast_start_date
),
actuals AS (
  SELECT
    forecast_start_date,
    day_7 as actual_day7
  FROM commodity.forecast.distributions
  WHERE commodity = 'Coffee'
    AND path_id = 0
    AND is_actuals = TRUE
)
SELECT
  f.model_version,
  COUNT(*) as total_forecasts,
  SUM(CASE
    WHEN a.actual_day7 BETWEEN f.lower_95 AND f.upper_95 THEN 1
    ELSE 0
  END) as actuals_in_95_interval,
  SUM(CASE
    WHEN a.actual_day7 BETWEEN f.lower_95 AND f.upper_95 THEN 1
    ELSE 0
  END) * 100.0 / COUNT(*) as coverage_95
FROM forecasts f
JOIN actuals a ON f.forecast_start_date = a.forecast_start_date
GROUP BY f.model_version
ORDER BY coverage_95 DESC;


-- ============================================================================
-- QUERY 5: Forecast vs Actual (Latest Window)
-- ============================================================================
-- Chart Type: Line Chart with confidence bands
-- X-axis: day (1-14)
-- Y-axis: price
-- Series: forecast_mean, actual, lower_95, upper_95
-- Description: Detailed view of most recent forecast performance

WITH latest_date AS (
  SELECT MAX(forecast_start_date) as max_date
  FROM commodity.forecast.distributions
  WHERE commodity = 'Coffee'
),
forecast_stats AS (
  SELECT
    1 as day, AVG(day_1) as forecast_mean, PERCENTILE(day_1, 0.025) as lower_95, PERCENTILE(day_1, 0.975) as upper_95
  FROM commodity.forecast.distributions
  WHERE model_version = 'sarimax_auto_weather_v1'
    AND commodity = 'Coffee'
    AND is_actuals = FALSE
    AND has_data_leakage = FALSE
    AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 2, AVG(day_2), PERCENTILE(day_2, 0.025), PERCENTILE(day_2, 0.975)
  FROM commodity.forecast.distributions
  WHERE model_version = 'sarimax_auto_weather_v1' AND commodity = 'Coffee' AND is_actuals = FALSE
    AND has_data_leakage = FALSE AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 3, AVG(day_3), PERCENTILE(day_3, 0.025), PERCENTILE(day_3, 0.975)
  FROM commodity.forecast.distributions
  WHERE model_version = 'sarimax_auto_weather_v1' AND commodity = 'Coffee' AND is_actuals = FALSE
    AND has_data_leakage = FALSE AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 4, AVG(day_4), PERCENTILE(day_4, 0.025), PERCENTILE(day_4, 0.975)
  FROM commodity.forecast.distributions
  WHERE model_version = 'sarimax_auto_weather_v1' AND commodity = 'Coffee' AND is_actuals = FALSE
    AND has_data_leakage = FALSE AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 5, AVG(day_5), PERCENTILE(day_5, 0.025), PERCENTILE(day_5, 0.975)
  FROM commodity.forecast.distributions
  WHERE model_version = 'sarimax_auto_weather_v1' AND commodity = 'Coffee' AND is_actuals = FALSE
    AND has_data_leakage = FALSE AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 6, AVG(day_6), PERCENTILE(day_6, 0.025), PERCENTILE(day_6, 0.975)
  FROM commodity.forecast.distributions
  WHERE model_version = 'sarimax_auto_weather_v1' AND commodity = 'Coffee' AND is_actuals = FALSE
    AND has_data_leakage = FALSE AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 7, AVG(day_7), PERCENTILE(day_7, 0.025), PERCENTILE(day_7, 0.975)
  FROM commodity.forecast.distributions
  WHERE model_version = 'sarimax_auto_weather_v1' AND commodity = 'Coffee' AND is_actuals = FALSE
    AND has_data_leakage = FALSE AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 8, AVG(day_8), PERCENTILE(day_8, 0.025), PERCENTILE(day_8, 0.975)
  FROM commodity.forecast.distributions
  WHERE model_version = 'sarimax_auto_weather_v1' AND commodity = 'Coffee' AND is_actuals = FALSE
    AND has_data_leakage = FALSE AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 9, AVG(day_9), PERCENTILE(day_9, 0.025), PERCENTILE(day_9, 0.975)
  FROM commodity.forecast.distributions
  WHERE model_version = 'sarimax_auto_weather_v1' AND commodity = 'Coffee' AND is_actuals = FALSE
    AND has_data_leakage = FALSE AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 10, AVG(day_10), PERCENTILE(day_10, 0.025), PERCENTILE(day_10, 0.975)
  FROM commodity.forecast.distributions
  WHERE model_version = 'sarimax_auto_weather_v1' AND commodity = 'Coffee' AND is_actuals = FALSE
    AND has_data_leakage = FALSE AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 11, AVG(day_11), PERCENTILE(day_11, 0.025), PERCENTILE(day_11, 0.975)
  FROM commodity.forecast.distributions
  WHERE model_version = 'sarimax_auto_weather_v1' AND commodity = 'Coffee' AND is_actuals = FALSE
    AND has_data_leakage = FALSE AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 12, AVG(day_12), PERCENTILE(day_12, 0.025), PERCENTILE(day_12, 0.975)
  FROM commodity.forecast.distributions
  WHERE model_version = 'sarimax_auto_weather_v1' AND commodity = 'Coffee' AND is_actuals = FALSE
    AND has_data_leakage = FALSE AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 13, AVG(day_13), PERCENTILE(day_13, 0.025), PERCENTILE(day_13, 0.975)
  FROM commodity.forecast.distributions
  WHERE model_version = 'sarimax_auto_weather_v1' AND commodity = 'Coffee' AND is_actuals = FALSE
    AND has_data_leakage = FALSE AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 14, AVG(day_14), PERCENTILE(day_14, 0.025), PERCENTILE(day_14, 0.975)
  FROM commodity.forecast.distributions
  WHERE model_version = 'sarimax_auto_weather_v1' AND commodity = 'Coffee' AND is_actuals = FALSE
    AND has_data_leakage = FALSE AND forecast_start_date = (SELECT max_date FROM latest_date)
),
actuals_unpivot AS (
  SELECT 1 as day, day_1 as actual FROM commodity.forecast.distributions
  WHERE path_id = 0 AND is_actuals = TRUE AND commodity = 'Coffee'
    AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 2, day_2 FROM commodity.forecast.distributions
  WHERE path_id = 0 AND is_actuals = TRUE AND commodity = 'Coffee'
    AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 3, day_3 FROM commodity.forecast.distributions
  WHERE path_id = 0 AND is_actuals = TRUE AND commodity = 'Coffee'
    AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 4, day_4 FROM commodity.forecast.distributions
  WHERE path_id = 0 AND is_actuals = TRUE AND commodity = 'Coffee'
    AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 5, day_5 FROM commodity.forecast.distributions
  WHERE path_id = 0 AND is_actuals = TRUE AND commodity = 'Coffee'
    AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 6, day_6 FROM commodity.forecast.distributions
  WHERE path_id = 0 AND is_actuals = TRUE AND commodity = 'Coffee'
    AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 7, day_7 FROM commodity.forecast.distributions
  WHERE path_id = 0 AND is_actuals = TRUE AND commodity = 'Coffee'
    AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 8, day_8 FROM commodity.forecast.distributions
  WHERE path_id = 0 AND is_actuals = TRUE AND commodity = 'Coffee'
    AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 9, day_9 FROM commodity.forecast.distributions
  WHERE path_id = 0 AND is_actuals = TRUE AND commodity = 'Coffee'
    AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 10, day_10 FROM commodity.forecast.distributions
  WHERE path_id = 0 AND is_actuals = TRUE AND commodity = 'Coffee'
    AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 11, day_11 FROM commodity.forecast.distributions
  WHERE path_id = 0 AND is_actuals = TRUE AND commodity = 'Coffee'
    AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 12, day_12 FROM commodity.forecast.distributions
  WHERE path_id = 0 AND is_actuals = TRUE AND commodity = 'Coffee'
    AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 13, day_13 FROM commodity.forecast.distributions
  WHERE path_id = 0 AND is_actuals = TRUE AND commodity = 'Coffee'
    AND forecast_start_date = (SELECT max_date FROM latest_date)

  UNION ALL SELECT 14, day_14 FROM commodity.forecast.distributions
  WHERE path_id = 0 AND is_actuals = TRUE AND commodity = 'Coffee'
    AND forecast_start_date = (SELECT max_date FROM latest_date)
)
SELECT
  f.day,
  f.forecast_mean,
  f.lower_95,
  f.upper_95,
  a.actual
FROM forecast_stats f
LEFT JOIN actuals_unpivot a ON f.day = a.day
ORDER BY f.day;


-- ============================================================================
-- QUERY 6: Error Distribution by Model
-- ============================================================================
-- Chart Type: Box Plot or Violin Plot
-- X-axis: model_version
-- Y-axis: error (forecast - actual)
-- Description: Show distribution of errors to identify bias and variance

WITH forecasts AS (
  SELECT
    model_version,
    forecast_start_date,
    AVG(day_7) as forecast_mean_day7
  FROM commodity.forecast.distributions
  WHERE commodity = 'Coffee'
    AND is_actuals = FALSE
    AND has_data_leakage = FALSE
  GROUP BY model_version, forecast_start_date
),
actuals AS (
  SELECT
    forecast_start_date,
    day_7 as actual_day7
  FROM commodity.forecast.distributions
  WHERE commodity = 'Coffee'
    AND path_id = 0
    AND is_actuals = TRUE
)
SELECT
  f.model_version,
  f.forecast_start_date,
  f.forecast_mean_day7 - a.actual_day7 as error_7d,
  ABS(f.forecast_mean_day7 - a.actual_day7) as abs_error_7d
FROM forecasts f
JOIN actuals a ON f.forecast_start_date = a.forecast_start_date
ORDER BY f.model_version, f.forecast_start_date;


-- ============================================================================
-- QUERY 7: Forecast Summary Table
-- ============================================================================
-- Chart Type: Table
-- Description: Comprehensive performance metrics for all models

WITH forecasts AS (
  SELECT
    model_version,
    forecast_start_date,
    AVG(day_1) as f_1d, AVG(day_7) as f_7d, AVG(day_14) as f_14d
  FROM commodity.forecast.distributions
  WHERE commodity = 'Coffee' AND is_actuals = FALSE AND has_data_leakage = FALSE
  GROUP BY model_version, forecast_start_date
),
actuals AS (
  SELECT
    forecast_start_date,
    day_1 as a_1d, day_7 as a_7d, day_14 as a_14d
  FROM commodity.forecast.distributions
  WHERE commodity = 'Coffee' AND path_id = 0 AND is_actuals = TRUE
),
metrics AS (
  SELECT
    f.model_version,
    AVG(ABS(f.f_1d - a.a_1d)) as mae_1d,
    AVG(ABS(f.f_7d - a.a_7d)) as mae_7d,
    AVG(ABS(f.f_14d - a.a_14d)) as mae_14d,
    SQRT(AVG(POWER(f.f_7d - a.a_7d, 2))) as rmse_7d,
    AVG(ABS(f.f_7d - a.a_7d) / a.a_7d * 100) as mape_7d
  FROM forecasts f
  JOIN actuals a ON f.forecast_start_date = a.forecast_start_date
  GROUP BY f.model_version
)
SELECT
  model_version,
  ROUND(mae_1d, 2) as mae_1day,
  ROUND(mae_7d, 2) as mae_7day,
  ROUND(mae_14d, 2) as mae_14day,
  ROUND(rmse_7d, 2) as rmse_7day,
  ROUND(mape_7d, 2) as mape_7day_pct
FROM metrics
ORDER BY mae_7d ASC;

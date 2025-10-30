# Risk/Trading Agent

**Owner**: Tony
**Status**: Ready - Forecasts are production-ready

## Purpose
Convert forecast distributions into trading signals with risk management.

## Inputs
- `commodity.silver.point_forecasts` - 14-day forecasts with actuals (backfill)
- `commodity.silver.distributions` - 2,000 Monte Carlo paths (path_id=0 is actuals)
- `commodity.silver.forecast_actuals` - Historical actuals table

## Outputs
- Trading signals (long/short/neutral)
- Position sizing recommendations
- Risk metrics (VaR, CVaR)

## Interface

### Point Forecasts (with Actuals)
```sql
-- Get forecasts with actuals for evaluation
SELECT
  forecast_date,
  day_ahead,
  forecast_mean,
  forecast_std,
  lower_95,
  upper_95,
  actual_close,  -- NULL for future dates
  has_data_leakage  -- Should always be FALSE (data quality check)
FROM commodity.silver.point_forecasts
WHERE day_ahead = 7
  AND model_version = 'sarimax_weather_v1'
  AND has_data_leakage = FALSE  -- Filter out any bad data
ORDER BY forecast_date DESC
```

### Distributions (for VaR and Risk Metrics)

```sql
-- Calculate VaR from Monte Carlo paths (exclude actuals path)
SELECT
  forecast_start_date,
  PERCENTILE(day_7, 0.05) as var_95,
  PERCENTILE(day_7, 0.01) as cvar_99,
  AVG(day_7) as mean_price,
  STDDEV(day_7) as price_volatility
FROM commodity.silver.distributions
WHERE model_version = 'sarimax_weather_v1'
  AND is_actuals = FALSE  -- Exclude path_id=0 (actuals)
  AND has_data_leakage = FALSE
GROUP BY forecast_start_date
ORDER BY forecast_start_date DESC
```

```sql
-- Get actuals from distributions (path_id=0)
SELECT
  forecast_start_date,
  day_1, day_2, day_3, day_4, day_5, day_6, day_7,
  day_8, day_9, day_10, day_11, day_12, day_13, day_14
FROM commodity.silver.distributions
WHERE path_id = 0  -- Actuals row
  AND is_actuals = TRUE
ORDER BY forecast_start_date DESC
```

### Compare Forecast vs Actual

**Actuals are stored directly in point_forecasts** (actual_close column).

```sql
-- Compare forecast vs actual (no join needed!)
SELECT
  forecast_date,
  day_ahead,
  forecast_mean,
  actual_close,
  forecast_mean - actual_close as error,
  ABS(forecast_mean - actual_close) as abs_error,
  CASE
    WHEN actual_close IS NOT NULL THEN 'Backfill'
    ELSE 'Future Forecast'
  END as data_type
FROM commodity.silver.point_forecasts
WHERE day_ahead = 7
  AND model_version = 'sarimax_weather_v1'
  AND has_data_leakage = FALSE
ORDER BY forecast_date DESC
```

**Alternative: Use forecast_actuals table** (separate table for safety):
```sql
-- Join with forecast_actuals table
SELECT
  pf.forecast_date,
  pf.forecast_mean,
  a.actual_close,
  pf.forecast_mean - a.actual_close as error
FROM commodity.silver.point_forecasts pf
JOIN commodity.silver.forecast_actuals a
  ON pf.forecast_date = a.forecast_date
  AND pf.commodity = a.commodity
WHERE pf.day_ahead = 7
```

### Colombian Trader Use Case (with COP/USD)

```sql
-- Include forex rate for local currency value
SELECT
  pf.forecast_date,
  pf.forecast_mean * u.cop_usd as forecast_value_cop,
  a.actual_close * u.cop_usd as actual_value_cop
FROM commodity.silver.point_forecasts pf
JOIN commodity.silver.forecast_actuals a
  ON pf.forecast_date = a.forecast_date AND pf.commodity = a.commodity
JOIN commodity.silver.unified_data u
  ON a.forecast_date = u.date AND a.commodity = u.commodity
WHERE pf.commodity = 'Coffee'
  AND u.is_trading_day = 1
LIMIT 1  -- One row per date
```

## Risk Metrics

- **VaR (95%)**: Value at Risk from 2,000 Monte Carlo paths (exclude path_id=0)
- **CVaR (99%)**: Conditional Value at Risk (worst 1% of scenarios)
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Information Ratio**: Target > 0.5
- **Sharpe Ratio**: Risk-adjusted returns
- **Directional Accuracy from Day 0**: 69.5% ± 27.7% (SARIMAX+Weather)

## Production Model

**SARIMAX+Weather** (`sarimax_weather_v1`):
- MAE: $3.10
- Dir Day0: 69.5% ± 27.7%
- Features: temp_c, humidity_pct, precipitation_mm
- Evaluated on 30 walk-forward windows (420 days)

## Schema Details

**point_forecasts**:
- `actual_close` - Realized price (NULL for future dates)
- `has_data_leakage` - Data quality flag (should be FALSE)
- One row per (forecast_date, model_version, day_ahead)

**distributions**:
- `path_id=0` - Actuals row (when available)
- `is_actuals` - TRUE for path_id=0, FALSE for forecast paths
- `has_data_leakage` - Data quality flag
- 2,000 forecast paths + 1 actuals path = 2,001 total rows per forecast

**forecast_actuals**:
- Separate table with realized prices
- One row per (forecast_date, commodity)

See `../project_overview/DATA_CONTRACTS.md` for complete schema specifications.

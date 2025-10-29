# Risk/Trading Agent

**Owner**: Tony
**Status**: ⏳ Waiting on stable forecasts

## Purpose
Convert forecast distributions into trading signals with risk management.

## Inputs
- `commodity.silver.point_forecasts`
- `commodity.silver.distributions`
- `commodity.silver.forecast_actuals`

## Outputs
- Trading signals (long/short/neutral)
- Position sizing recommendations
- Risk metrics (VaR, CVaR)

## Interface

### Point Forecasts
```sql
SELECT forecast_date, forecast_mean, lower_95, upper_95
FROM commodity.silver.point_forecasts
WHERE day_ahead = 7
  AND model_version = 'production_v1'
  AND data_cutoff_date < forecast_date
```

### Distributions (for VaR)

```sql
-- Calculate VaR from forecasts
SELECT forecast_start_date,
       PERCENTILE(day_7, 0.05) as var_95,
       AVG(day_7) as mean_price
FROM commodity.silver.distributions
WHERE model_version = 'production_v1'
  AND data_cutoff_date < forecast_start_date
GROUP BY forecast_start_date
```

### Compare Forecast vs Actual

**Actuals are in separate table** (`forecast_actuals`) for safety and clarity.

```sql
-- Simple join to get forecast vs actual
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
  AND pf.data_cutoff_date < pf.forecast_date
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

- **VaR (95%)**: Value at Risk (from forecasts, path_id > 0)
- **CVaR**: Conditional Value at Risk
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Information Ratio**: Target > 0.5
- **Sharpe Ratio**: Risk-adjusted returns

## Dependencies

Waiting on:
- Stable forecast schema (Connor)
- Multiple model versions for comparison
- Historical backtested forecasts with actuals

See `agent_instructions/DATA_CONTRACTS.md` for schemas.

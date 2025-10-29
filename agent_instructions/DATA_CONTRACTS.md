# Data Contracts

**Critical**: These schemas define interfaces between agents. Changes require team alignment.

## Input: commodity.silver.unified_data

**Owner**: Research Agent (Francisco)
**Grain**: One row per (date, commodity, region)
**Rows**: ~75k (as of Oct 2024)

### Schema

| Column | Type | Description | Nulls |
|--------|------|-------------|-------|
| `date` | DATE | Calendar date | No |
| `is_trading_day` | INT | 1 = trading day, 0 = weekend/holiday | No |
| `commodity` | STRING | 'Coffee' or 'Sugar' | No |
| `close` | FLOAT | Futures close price (USD) | No |
| `high` | FLOAT | Daily high | No |
| `low` | FLOAT | Daily low | No |
| `open` | FLOAT | Daily open | No |
| `volume` | FLOAT | Trading volume | Some |
| `vix` | FLOAT | Volatility index | Some |
| `region` | STRING | Producing region (e.g., 'Bugisu_Uganda') | No |
| `temp_c` | FLOAT | Temperature (Celsius) | Some |
| `humidity_pct` | FLOAT | Humidity percentage | Some |
| `precipitation_mm` | FLOAT | Precipitation (mm) | Some |
| `*_usd` | FLOAT | Exchange rates (vnd_usd, cop_usd, etc.) | Some |

**Key Columns for Forecasting**:
- `close`: Primary target variable
- `is_trading_day`: Filter for model training (only use trading days)
- `region`: Enables hierarchical/regional models
- Weather columns: Important covariates
- `cop_usd`: Critical for Colombian trader use case

**Data Quality**:
- Forward-filled to handle non-trading days
- No duplicates on (date, commodity, region)
- ~65 regions covered

**Usage Pattern**:
```python
# Load only trading days for Coffee
df = spark.table("commodity.silver.unified_data") \
    .filter("commodity = 'Coffee' AND is_trading_day = 1")
```

## Output 1: commodity.silver.point_forecasts

**Owner**: Forecast Agent (Connor - YOU)
**Grain**: One row per (forecast_date, model_version, day_ahead)

### Schema

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `forecast_date` | DATE | Target date being forecasted | Yes |
| `data_cutoff_date` | DATE | Last training date (< forecast_date) | Yes |
| `generation_timestamp` | TIMESTAMP | When forecast was generated | Yes |
| `day_ahead` | INT | Horizon (1-14 days) | Yes |
| `forecast_mean` | FLOAT | Point forecast (cents/lb) | Yes |
| `forecast_std` | FLOAT | Standard error | Yes |
| `lower_95` | FLOAT | 95% CI lower bound | Yes |
| `upper_95` | FLOAT | 95% CI upper bound | Yes |
| `model_version` | STRING | Model identifier (e.g., 'sarimax_v0') | Yes |
| `commodity` | STRING | 'Coffee' or 'Sugar' | Yes |
| `model_success` | BOOLEAN | Did model converge? | Yes |

**Partitioning**: `model_version`, `commodity`

**Critical Invariants**:
- `data_cutoff_date < forecast_date` (prevents data leakage)
- Daily forecasts create overlapping predictions
- Each target date has up to 14 forecasts (from different cutoff dates)

**Usage by Trading Agent**:
```sql
-- Get 7-day ahead forecasts for backtesting
SELECT forecast_date, forecast_mean, lower_95, upper_95
FROM commodity.silver.point_forecasts
WHERE day_ahead = 7
  AND data_cutoff_date < forecast_date
  AND model_version = 'production_v1'
  AND forecast_date BETWEEN '2023-01-01' AND '2023-12-31'
```

## Output 2: commodity.silver.distributions

**Owner**: Forecast Agent (Connor - YOU)
**Grain**: One row per (forecast_start_date, model_version, path_id)

### Schema

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `path_id` | INT | Sample path ID (1-2000) | Yes |
| `forecast_start_date` | DATE | First day of forecast | Yes |
| `data_cutoff_date` | DATE | Last training date | Yes |
| `generation_timestamp` | TIMESTAMP | When generated | Yes |
| `model_version` | STRING | Model identifier | Yes |
| `commodity` | STRING | 'Coffee' or 'Sugar' | Yes |
| `day_1` to `day_14` | FLOAT | Forecasted prices | Yes |

**Partitioning**: `model_version`, `commodity`

**Purpose**: Monte Carlo paths for risk analysis (VaR, CVaR)

**Typical Usage**:
```sql
-- Calculate 95% VaR for day 7
SELECT forecast_start_date,
       PERCENTILE(day_7, 0.05) as var_95,
       AVG(day_7) as mean_price
FROM commodity.silver.distributions
WHERE forecast_start_date = '2024-01-15'
  AND data_cutoff_date < '2024-01-15'
  AND model_version = 'production_v1'
GROUP BY forecast_start_date
```

## Output 3: commodity.silver.forecast_actuals

**Owner**: Forecast Agent (Connor - YOU)
**Grain**: One row per (forecast_date, commodity)

### Schema

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `forecast_date` | DATE | Target date | Yes |
| `commodity` | STRING | 'Coffee' or 'Sugar' | Yes |
| `actual_close` | FLOAT | Realized close price | Yes |

**Partitioning**: `commodity`

**Purpose**: Store realized prices for backtesting and evaluation

**What is "actual"?**
- Currently: **Close price** from futures market (industry standard for commodity traders)
- Future consideration: VWAP or estimated VWAP for better execution price representation

**Why separate table?**
- Prevents accidental inclusion in forecast statistics (no path_id confusion)
- Simple, clean joins
- Easy to understand and maintain

**Typical Usage - Compare Forecast vs Actual**:
```sql
-- Calculate forecast errors
SELECT
  pf.forecast_date,
  pf.forecast_mean,
  a.actual_close,
  pf.forecast_mean - a.actual_close as error,
  ABS(pf.forecast_mean - a.actual_close) as abs_error
FROM commodity.silver.point_forecasts pf
JOIN commodity.silver.forecast_actuals a
  ON pf.forecast_date = a.forecast_date
  AND pf.commodity = a.commodity
WHERE pf.day_ahead = 7
  AND pf.data_cutoff_date < pf.forecast_date
```

**For Colombian Trader Use Case - Include COP/USD**:
```sql
-- Forecast value in Colombian Pesos
SELECT
  pf.forecast_date,
  pf.forecast_mean * u.cop_usd as forecast_value_cop,
  a.actual_close * u.cop_usd as actual_value_cop,
  (pf.forecast_mean - a.actual_close) * u.cop_usd as error_cop
FROM commodity.silver.point_forecasts pf
JOIN commodity.silver.forecast_actuals a
  ON pf.forecast_date = a.forecast_date AND pf.commodity = a.commodity
JOIN commodity.silver.unified_data u
  ON a.forecast_date = u.date AND a.commodity = u.commodity
WHERE u.is_trading_day = 1
  AND pf.commodity = 'Coffee'
  AND pf.day_ahead = 7
LIMIT 1  -- One row per date (multiple regions in unified_data)
```

**Implementation Notes**:
- Connor populates this table during backtesting
- Lookup actual close prices from `unified_data` WHERE `is_trading_day = 1`
- Only write actuals for dates where forecasts were generated

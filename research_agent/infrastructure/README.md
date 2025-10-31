# Research Infrastructure

Data collection pipeline for commodity price forecasting.

## Architecture

```
Lambda (AWS) → S3 (groundtruth-capstone) → Databricks (commodity catalog)
```

## Setup

### 1. Deploy Lambda Functions
```bash
cd lambda
./deploy_all_functions.sh
./backfill_historical_data.sh  # Optional: Load historical data
```

### 2. Configure EventBridge Schedules
```bash
cd eventbridge
./setup_all_eventbridge_schedules.sh
```

### 3. Create Databricks Tables
```sql
-- In Databricks SQL Editor
source databricks/01_create_landing_tables.sql
source databricks/02_create_bronze_views.sql
```

## Data Sources

| Function | Data | Schedule | S3 Path |
|----------|------|----------|---------|
| market-data-fetcher | Coffee/Sugar prices (Yahoo Finance) | Daily 2AM UTC | landing/market_data/ |
| weather-data-fetcher | Growing region weather (Open-Meteo) | Daily 2AM UTC | landing/weather_data/ |
| vix-data-fetcher | VIX volatility index (FRED) | Daily 2AM UTC | landing/vix_data/ |
| fx-calculator-fetcher | FX rates (FRED) | Daily 2AM UTC | landing/macro_data/ |
| cftc-data-fetcher | Trader positioning (CFTC) | Daily 2AM UTC | landing/cftc_data/ |
| gdelt-processor | News sentiment (GDELT GKG) | Daily 2AM UTC | landing/gdelt/filtered/ |

## Databricks Tables

**Landing** (`commodity.landing.*_inc`): Raw S3 data with `ingest_ts`
**Bronze** (`commodity.bronze.v_*_all`): Deduplicated views using `QUALIFY ROW_NUMBER()`

Query example:
```sql
SELECT * FROM commodity.bronze.v_market_data_all
WHERE commodity = 'Coffee'
ORDER BY date DESC LIMIT 100;
```

# Bronze Layer Setup - Run in Databricks

## Status
✅ **Catalog & Schemas Created**
- commodity.bronze
- commodity.silver  
- commodity.landing

## Next: Create Tables & Views

### Option 1: Via Databricks SQL Editor (Recommended)

1. Go to: https://dbc-fd7b00f3-7a6d.cloud.databricks.com
2. Click **SQL Editor** in sidebar
3. Copy/paste and run each block below

```sql
USE CATALOG commodity;

-- Market Data
CREATE OR REPLACE TABLE commodity.landing.market_data_inc
USING DELTA
LOCATION 's3://groundtruth-capstone/tables/landing/market_data'
AS SELECT 
  CAST(date AS DATE) as date,
  commodity,
  CAST(`close` AS DOUBLE) as close,
  current_timestamp() as ingest_ts
FROM read_files(
  's3://groundtruth-capstone/landing/market_data/*.csv',
  format => 'csv',
  header => true
);

-- Macro Data
CREATE OR REPLACE TABLE commodity.landing.macro_data_inc
USING DELTA
LOCATION 's3://groundtruth-capstone/tables/landing/macro_data'
AS SELECT *,
  current_timestamp() as ingest_ts
FROM read_files(
  's3://groundtruth-capstone/landing/macro_data/*.csv',
  format => 'csv',
  header => true
);

-- VIX Data
CREATE OR REPLACE TABLE commodity.landing.vix_data_inc
USING DELTA
LOCATION 's3://groundtruth-capstone/tables/landing/vix_data'
AS SELECT *,
  current_timestamp() as ingest_ts
FROM read_files(
  's3://groundtruth-capstone/landing/vix_data/*.csv',
  format => 'csv',
  header => true
);

-- Weather Data
CREATE OR REPLACE TABLE commodity.landing.weather_data_inc
USING DELTA
LOCATION 's3://groundtruth-capstone/tables/landing/weather_data'
AS SELECT *,
  current_timestamp() as ingest_ts
FROM read_files(
  's3://groundtruth-capstone/landing/weather_data/*.csv',
  format => 'csv',
  header => true
);

-- CFTC Data
CREATE OR REPLACE TABLE commodity.landing.cftc_data_inc
USING DELTA
LOCATION 's3://groundtruth-capstone/tables/landing/cftc_data'
AS SELECT *,
  current_timestamp() as ingest_ts
FROM read_files(
  's3://groundtruth-capstone/landing/cftc_data/*.csv',
  format => 'csv',
  header => true
);

-- Verify
SHOW TABLES IN commodity.landing;

-- Check row counts
SELECT 'market_data_inc' as `table`, COUNT(*) as rows FROM commodity.landing.market_data_inc
UNION ALL
SELECT 'macro_data_inc', COUNT(*) FROM commodity.landing.macro_data_inc
UNION ALL
SELECT 'vix_data_inc', COUNT(*) FROM commodity.landing.vix_data_inc  
UNION ALL
SELECT 'weather_data_inc', COUNT(*) FROM commodity.landing.weather_data_inc
UNION ALL
SELECT 'cftc_data_inc', COUNT(*) FROM commodity.landing.cftc_data_inc;
```

### Then Create Bronze Views

```sql
-- Market Data View
CREATE OR REPLACE VIEW commodity.bronze.v_market_data_all AS
SELECT date, commodity, close
FROM commodity.landing.market_data_inc
QUALIFY ROW_NUMBER() OVER (PARTITION BY date, commodity ORDER BY ingest_ts DESC) = 1;

-- Macro Data View
CREATE OR REPLACE VIEW commodity.bronze.v_macro_data_all AS
SELECT *
FROM commodity.landing.macro_data_inc
QUALIFY ROW_NUMBER() OVER (PARTITION BY date ORDER BY ingest_ts DESC) = 1;

-- VIX Data View
CREATE OR REPLACE VIEW commodity.bronze.v_vix_data_all AS
SELECT date, vix
FROM commodity.landing.vix_data_inc
QUALIFY ROW_NUMBER() OVER (PARTITION BY date ORDER BY ingest_ts DESC) = 1;

-- Weather Data View
CREATE OR REPLACE VIEW commodity.bronze.v_weather_data_all AS
SELECT *
FROM commodity.landing.weather_data_inc
QUALIFY ROW_NUMBER() OVER (PARTITION BY date, region, commodity ORDER BY ingest_ts DESC) = 1;

-- CFTC Data View
CREATE OR REPLACE VIEW commodity.bronze.v_cftc_data_all AS
SELECT *
FROM commodity.landing.cftc_data_inc
QUALIFY ROW_NUMBER() OVER (PARTITION BY date, commodity ORDER BY ingest_ts DESC) = 1;

-- Verify views
SHOW VIEWS IN commodity.bronze;
```

### Test with Union Data SQL

```sql
-- This should work now
SELECT * FROM commodity.bronze.v_market_data_all LIMIT 10;
SELECT * FROM commodity.bronze.v_macro_data_all LIMIT 10;
```

## Troubleshooting

### "read_files not found"
- Make sure you're using SQL Warehouse (not cluster)
- Or switch to notebook with cluster attached

### "S3 permission denied"
- Check that the warehouse/cluster has S3 instance profile
- Verify IAM role has S3 access to groundtruth-capstone bucket

### "No data"
- Verify S3 files exist: `aws s3 ls s3://groundtruth-capstone/landing/market_data/ --region us-west-2`

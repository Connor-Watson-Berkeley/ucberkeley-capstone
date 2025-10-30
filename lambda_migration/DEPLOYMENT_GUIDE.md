# Complete Data Pipeline Deployment Guide

## Overview

This guide walks through deploying the complete commodity forecasting data pipeline from scratch:

1. **AWS Lambda Functions** - Fetch data daily from external sources
2. **S3 Storage** - Store raw data files
3. **Databricks Bronze Layer** - Auto-ingest S3 data into Delta tables
4. **Databricks Silver Layer** - Create unified dataset for forecasting
5. **Forecast Agent** - Generate predictions

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ DAILY DATA INGESTION (2-3 AM UTC)                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ Lambda Functions (AWS us-west-2)                                    │
│ ├── market-data-fetcher     → s3://groundtruth-capstone/landing/   │
│ ├── weather-data-fetcher    → s3://groundtruth-capstone/landing/   │
│ ├── vix-data-fetcher        → s3://groundtruth-capstone/landing/   │
│ ├── fx-calculator-fetcher   → s3://groundtruth-capstone/landing/   │
│ ├── cftc-data-fetcher       → s3://groundtruth-capstone/landing/   │
│ └── gdelt-processor (2 AM)  → s3://groundtruth-capstone/landing/   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────┐
│ DATABRICKS AUTO-INGESTION (3 AM UTC)                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ Auto Loader Streams (Databricks)                                    │
│ ├── S3 → commodity.landing.market_data_inc (Delta)                  │
│ ├── S3 → commodity.landing.vix_data_inc (Delta)                     │
│ ├── S3 → commodity.landing.macro_data_inc (Delta)                   │
│ ├── S3 → commodity.landing.weather_data_inc (Delta)                 │
│ └── S3 → commodity.landing.cftc_data_inc (Delta)                    │
│                                                                      │
│ Bronze Views (Deduplicated)                                         │
│ ├── commodity.bronze.v_market_data_all                              │
│ ├── commodity.bronze.v_vix_data_all                                 │
│ ├── commodity.bronze.v_macro_data_all                               │
│ ├── commodity.bronze.v_weather_data_all                             │
│ ├── commodity.bronze.v_cftc_data_all                                │
│ └── commodity.bronze.bronze_gkg (GDELT)                             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────┐
│ SILVER LAYER - UNIFIED DATA                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ research_agent/create_gdelt_unified_data.py                         │
│ └── commodity.silver.unified_data                                   │
│     (Joins bronze tables, adds GDELT sentiment, weather features)   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────┐
│ FORECASTING                                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ forecast_agent/databricks_quickstart.py                             │
│ ├── commodity.silver.point_forecasts                                │
│ ├── commodity.silver.distributions                                  │
│ └── commodity.silver.forecast_actuals                               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- [x] AWS CLI configured with account 534150427458
- [x] AWS credentials with Lambda and S3 permissions
- [x] Databricks workspace (dbc-fd7b00f3-7a6d)
- [x] Databricks cluster: general-purpose-mid-compute (with S3 permissions)
- [x] Git repository cloned in Databricks Repos

---

## Phase 1: Deploy Lambda Functions (AWS)

### Step 1.1: Configure AWS CLI

```bash
aws configure
# Account ID: 534150427458
# Region: us-west-2
```

### Step 1.2: Deploy All Lambda Functions

```bash
cd migrated_functions/
./deploy_all_functions.sh
```

**What this does**:
- Creates IAM role `groundtruth-lambda-execution-role`
- Deploys 6 Lambda functions with API keys
- Configures S3 write permissions

**Time**: ~5-10 minutes

### Step 1.3: Set Up EventBridge Schedule

```bash
./setup_eventbridge_schedule.sh
```

**What this does**:
- Creates EventBridge rule for GDELT processor
- Schedule: Daily at 2 AM UTC
- Triggers incremental GDELT data ingestion

### Step 1.4: Test Lambda Functions

```bash
# Test market data fetcher
aws lambda invoke \
    --function-name market-data-fetcher \
    --region us-west-2 \
    response.json

cat response.json

# Check if data was written to S3
aws s3 ls s3://groundtruth-capstone/landing/ --recursive | tail -20
```

**Expected S3 structure after Lambda runs**:
```
s3://groundtruth-capstone/
├── landing/
│   ├── market_data/
│   ├── vix_data/
│   ├── macro_data/
│   ├── weather_data/
│   ├── cftc_data/
│   └── gdelt/
│       ├── raw/
│       └── filtered/
```

---

## Phase 2: Set Up Databricks Bronze Layer

### Step 2.1: Upload ETL Notebook to Databricks

**Option A: Via Databricks Repos (Recommended)**

1. Go to Databricks workspace: https://dbc-fd7b00f3-7a6d.cloud.databricks.com
2. Navigate to **Repos**
3. Clone repository: `https://github.com/Connor-Watson-Berkeley/ucberkeley-capstone`
4. Open: `lambda_migration/databricks_etl_setup.py`

**Option B: Manual Upload**

1. Go to **Workspace** → **Create** → **Notebook**
2. Copy contents of `databricks_etl_setup.py`
3. Paste into new notebook

### Step 2.2: Run ETL Setup Notebook

1. Attach notebook to cluster: **general-purpose-mid-compute**
2. Run all cells (Cmd/Ctrl + Shift + Enter)

**What this does**:
- Creates `commodity` catalog
- Creates `bronze`, `silver`, `landing` schemas
- Sets up Auto Loader streaming jobs
- Creates bronze views with deduplication
- Creates GDELT bronze table

**Time**: ~5-10 minutes

### Step 2.3: Verify Bronze Tables

```sql
-- Check all bronze tables exist
SHOW TABLES IN commodity.bronze;

-- Verify row counts
SELECT 'v_market_data_all' as table_name, COUNT(*) as row_count FROM commodity.bronze.v_market_data_all
UNION ALL
SELECT 'v_vix_data_all', COUNT(*) FROM commodity.bronze.v_vix_data_all
UNION ALL
SELECT 'v_macro_data_all', COUNT(*) FROM commodity.bronze.v_macro_data_all
UNION ALL
SELECT 'v_weather_data_all', COUNT(*) FROM commodity.bronze.v_weather_data_all
UNION ALL
SELECT 'bronze_gkg', COUNT(*) FROM commodity.bronze.bronze_gkg;
```

---

## Phase 3: Set Up Daily Auto-Refresh

### Step 3.1: Get Cluster ID

```bash
# In Databricks, go to Compute → general-purpose-mid-compute
# Copy the Cluster ID (looks like: 1234-567890-abcd1234)
```

### Step 3.2: Create Daily Job

**Option A: Via Script** (Requires Databricks CLI setup)

```bash
# Edit setup_databricks_daily_job.sh
# Replace <YOUR_CLUSTER_ID> and <YOUR_USERNAME>

chmod +x setup_databricks_daily_job.sh
./setup_databricks_daily_job.sh
```

**Option B: Via Databricks UI** (Recommended)

1. Go to **Workflows** → **Create Job**
2. Job name: `Commodity Data - Daily Bronze Refresh`
3. Task configuration:
   - **Type**: Notebook
   - **Path**: `/Workspace/Repos/<YOUR_USERNAME>/ucberkeley-capstone/lambda_migration/databricks_etl_setup`
   - **Cluster**: general-purpose-mid-compute
4. Schedule:
   - **Trigger**: Scheduled
   - **Cron**: `0 3 * * *` (3 AM UTC daily)
   - **Timezone**: UTC
5. Click **Create**

**What this does**:
- Runs Auto Loader daily at 3 AM UTC
- Picks up new CSV files from Lambda functions (which run at 2 AM)
- Updates bronze Delta tables
- Refreshes bronze views

---

## Phase 4: Create Silver Layer (Unified Data)

### Step 4.1: Verify Bronze Data Available

```sql
-- Quick check that we have data in all bronze tables
SELECT * FROM commodity.bronze.v_market_data_all LIMIT 5;
SELECT * FROM commodity.bronze.v_vix_data_all LIMIT 5;
SELECT * FROM commodity.bronze.v_macro_data_all LIMIT 5;
SELECT * FROM commodity.bronze.bronze_gkg LIMIT 5;
```

### Step 4.2: Create Unified Data

**Option A: Run in Databricks Notebook**

1. Open `research_agent/create_gdelt_unified_data.py` in Databricks
2. Adapt for Databricks:
   - Replace pandas with PySpark
   - Read from `commodity.bronze.*` tables
   - Write to `commodity.silver.unified_data`
3. Run all cells

**Option B: Use Existing Historical Data**

If you already have `unified_data.csv` from the migration:

```python
# Upload unified_data.csv to DBFS
dbutils.fs.cp("file:/path/to/unified_data.csv", "dbfs:/tmp/unified_data.csv")

# Create table
df = spark.read.csv("dbfs:/tmp/unified_data.csv", header=True, inferSchema=True)
df.write.mode("overwrite").saveAsTable("commodity.silver.unified_data")
```

### Step 4.3: Verify Unified Data

```sql
SELECT COUNT(*) as row_count FROM commodity.silver.unified_data;
SELECT * FROM commodity.silver.unified_data LIMIT 10;

-- Check date range
SELECT MIN(date) as earliest_date, MAX(date) as latest_date
FROM commodity.silver.unified_data;
```

---

## Phase 5: Deploy Forecast Agent

### Step 5.1: Run Forecast Agent Quickstart

1. Open `forecast_agent/databricks_quickstart.py` in Databricks
2. Update cell 4 with your username:
   ```python
   sys.path.insert(0, '/Workspace/Repos/<YOUR_USERNAME>/ucberkeley-capstone/forecast_agent')
   ```
3. Run all cells

**What this does**:
- Trains SARIMAX+Weather model
- Generates 14-day forecasts
- Creates:
  - `commodity.silver.point_forecasts`
  - `commodity.silver.distributions`
- Exports for trading agent

---

## Phase 6: Monitoring & Maintenance

### Daily Data Flow (Automated)

```
2:00 AM UTC → Lambda Functions Run
              ├── Fetch new market data
              ├── Fetch new weather data
              ├── Fetch new VIX/FX data
              └── Process new GDELT data

2:30 AM UTC → Lambda Functions Complete
              └── New CSV files in S3

3:00 AM UTC → Databricks Job Runs
              ├── Auto Loader picks up new CSVs
              ├── Updates bronze Delta tables
              └── Refreshes bronze views

3:30 AM UTC → Databricks Job Completes
              └── Bronze tables ready
```

### Monitoring Points

**1. Lambda Functions (AWS CloudWatch)**

```bash
# View logs for specific function
aws logs tail /aws/lambda/market-data-fetcher --follow --region us-west-2

# Check if Lambda ran successfully
aws lambda list-functions --region us-west-2 --query 'Functions[?starts_with(FunctionName, `market-data`)].FunctionName'
```

**2. S3 Data Freshness**

```bash
# Check latest files in S3
aws s3 ls s3://groundtruth-capstone/landing/ --recursive --region us-west-2 --human-readable | tail -20

# Verify today's data exists
aws s3 ls s3://groundtruth-capstone/landing/market_data/ --region us-west-2
```

**3. Databricks Tables**

```sql
-- Check when bronze tables were last updated
SELECT MAX(ingest_ts) as last_updated
FROM commodity.landing.market_data_inc;

-- Verify row counts are increasing
SELECT COUNT(*) FROM commodity.silver.unified_data;
```

**4. Databricks Jobs**

- Go to **Workflows** → **Commodity Data - Daily Bronze Refresh**
- Check **Run History** for failures
- Review logs if job fails

### Troubleshooting

**Issue: Lambda function fails**

```bash
# Check CloudWatch logs
aws logs tail /aws/lambda/<function-name> --region us-west-2

# Common issues:
# - API key expired → Update environment variables
# - S3 permission denied → Check IAM role
# - Timeout → Increase timeout in function config
```

**Issue: Databricks job fails**

```sql
-- Check if S3 files exist
SELECT COUNT(*) FROM delta.`s3://groundtruth-capstone/landing/market_data/`;

-- Common issues:
# - S3 access denied → Verify cluster has S3 instance profile
# - Schema changed → Run with mergeSchema=true
# - Checkpoint corrupted → Delete checkpoint and re-run
```

**Issue: Data not updating**

```sql
-- Check last ingestion time
SELECT MAX(ingest_ts) as last_update FROM commodity.landing.market_data_inc;

-- If stale, manually trigger job:
# Workflows → Daily Bronze Refresh → Run Now
```

---

## Cost Optimization

**Lambda**:
- Functions run daily for ~2-5 minutes each
- Estimated cost: <$5/month

**S3**:
- Storage: ~1-2 GB/month of CSV files
- Estimated cost: <$1/month

**Databricks**:
- Daily job runs ~10 minutes
- Estimated cost: ~$20-30/month (depending on cluster size)

**Total Monthly Cost**: ~$25-40

---

## Rollback Procedures

**Rollback Lambda Deployment**:
```bash
# Delete all functions
for func in market-data-fetcher weather-data-fetcher vix-data-fetcher fx-calculator-fetcher cftc-data-fetcher gdelt-processor; do
    aws lambda delete-function --function-name $func --region us-west-2
done
```

**Rollback Databricks Setup**:
```sql
-- Drop all tables
DROP SCHEMA IF EXISTS commodity.bronze CASCADE;
DROP SCHEMA IF EXISTS commodity.landing CASCADE;
DROP CATALOG IF EXISTS commodity CASCADE;
```

---

## Success Criteria

- [ ] All 6 Lambda functions deployed and running
- [ ] EventBridge schedule active for GDELT processor
- [ ] Data appearing in S3 daily
- [ ] All bronze tables showing data
- [ ] Databricks daily job running successfully
- [ ] commodity.silver.unified_data table populated
- [ ] Forecast agent generating predictions

---

## Support & Documentation

- **Lambda Migration**: `lambda_migration/migrated_functions/README.md`
- **Databricks Setup**: `infra/README.md`
- **Forecast Agent**: `forecast_agent/README.md`
- **Data Contracts**: `project_overview/DATA_CONTRACTS.md`

---

**Deployment Date**: October 2025
**Last Updated**: October 29, 2025

# Deployment Status - Commodity Data Pipeline Migration

**Last Updated**: October 29, 2025
**Migration Date**: October 2025
**Status**: 🟡 In Progress

---

## Overview

Migrating commodity forecasting data pipeline from old Databricks workspace to new environment due to access restrictions.

**Source**:
- AWS Account: 471112877453 (us-east-1)
- S3 Bucket: berkeley-datasci210-capstone
- Databricks: Old workspace (no admin access)

**Target**:
- AWS Account: 534150427458 (us-west-2)
- S3 Bucket: groundtruth-capstone
- Databricks: https://dbc-fd7b00f3-7a6d.cloud.databricks.com

---

## Deployment Checklist

### ✅ Phase 1: Infrastructure Setup
- [x] **Databricks workspace credentials secured** (infra/.databrickscfg)
- [x] **Git repository connected** to Databricks
- [x] **S3 bucket configured** (groundtruth-capstone)
- [x] **Cluster available** (general-purpose-mid-compute, ID: 1030-040527-3do4v2at)
- [x] **S3 instance profile attached** to cluster

### ✅ Phase 2: Code Migration
- [x] **Old workspace code extracted** from old_data_pipeline.zip
- [x] **ETL scripts organized** in research_agent/old_workspace_migration/
- [x] **Migration guide created** with table mappings
- [x] **Lambda functions inventoried** (19 total, 6 critical)
- [x] **Lambda code extracted and updated** for new bucket/region
- [x] **API keys retrieved** (FRED, OpenWeather)
- [x] **Deployment scripts created** (deploy_all_functions.sh)

### 🟡 Phase 3: Databricks Bronze Layer (In Progress)
- [x] **ETL setup notebook created** (databricks_etl_setup.py)
- [x] **Notebook uploaded to Databricks**
- [x] **Cluster started** (general-purpose-mid-compute)
- [🟡] **Notebook running** (Run ID: 434014329903945)
  - Status: PENDING → RUNNING
  - Expected duration: 5-10 minutes
  - Monitor: https://dbc-fd7b00f3-7a6d.cloud.databricks.com/#job/runs/434014329903945

**What the notebook will create**:
```
commodity (Unity Catalog)
├── landing (Delta tables)
│   ├── market_data_inc (Coffee & Sugar prices)
│   ├── vix_data_inc (VIX volatility)
│   ├── macro_data_inc (FX rates: COP/USD, etc.)
│   ├── weather_data_inc (Weather for producing regions)
│   └── cftc_data_inc (CFTC commitment of traders)
├── bronze (Views with deduplication)
│   ├── v_market_data_all
│   ├── v_vix_data_all
│   ├── v_macro_data_all
│   ├── v_weather_data_all
│   ├── v_cftc_data_all
│   └── bronze_gkg (GDELT news data)
└── silver (Curated datasets)
    ├── unified_data (to be created)
    ├── point_forecasts (forecast output)
    └── distributions (forecast output)
```

### ⏳ Phase 4: Lambda Functions Deployment (Ready)
- [ ] **Deploy Lambda functions to AWS**
  - Script: `./deploy_all_functions.sh`
  - Functions: market-data-fetcher, weather-data-fetcher, vix-data-fetcher, fx-calculator-fetcher, cftc-data-fetcher, gdelt-processor
  - Expected: 5-10 minutes
- [ ] **Set up EventBridge schedule**
  - Script: `./setup_eventbridge_schedule.sh`
  - Schedule: Daily at 2 AM UTC (GDELT processor)
- [ ] **Test Lambda functions**
  - Run test invocations
  - Verify S3 writes
  - Check CloudWatch logs

### ⏳ Phase 5: Daily Auto-Refresh (Ready)
- [ ] **Create Databricks daily job**
  - Script: `./setup_databricks_daily_job.sh` (or via UI)
  - Schedule: Daily at 3 AM UTC (1 hour after Lambda)
  - Job name: Commodity Data - Daily Bronze Refresh
- [ ] **Test daily refresh**
  - Manual trigger
  - Verify bronze tables update

### ⏳ Phase 6: Silver Layer (Pending)
- [ ] **Create unified_data table**
  - Adapt research_agent/create_gdelt_unified_data.py for Databricks
  - Join bronze tables with GDELT sentiment
  - Add weather features
- [ ] **Verify unified_data**
  - Check row counts
  - Validate date ranges
  - Test joins

### ⏳ Phase 7: Forecasting (Pending)
- [ ] **Deploy forecast agent**
  - Run forecast_agent/databricks_quickstart.py
  - Generate 14-day forecasts
  - Create point_forecasts and distributions tables
- [ ] **Verify forecasts**
  - Check forecast outputs
  - Validate predictions
  - Export for trading agent

---

## Current Status Details

### Databricks Bronze Layer Deployment

**Run ID**: 434014329903945
**Started**: ~6:50 AM UTC
**Expected completion**: ~7:00 AM UTC (5-10 minutes)

**Monitor with**:
```bash
cd lambda_migration
python3 monitor_databricks_run.py 434014329903945
```

**Or check directly**:
https://dbc-fd7b00f3-7a6d.cloud.databricks.com/#job/runs/434014329903945

**The notebook is executing**:
1. ✓ Creating Unity Catalog (commodity.bronze, commodity.silver, commodity.landing)
2. 🟡 Setting up Auto Loader streams (5 data sources)
   - Market data: s3://groundtruth-capstone/landing/market_data/
   - VIX data: s3://groundtruth-capstone/landing/vix_data/
   - Macro data: s3://groundtruth-capstone/landing/macro_data/
   - Weather data: s3://groundtruth-capstone/landing/weather_data/
   - CFTC data: s3://groundtruth-capstone/landing/cftc_data/
3. ⏳ Creating bronze views with deduplication
4. ⏳ Creating GDELT bronze table
5. ⏳ Running verification queries

---

## Next Steps (After Bronze Layer Completes)

### Option A: Deploy Lambda Functions First
This ensures daily data refresh starts immediately:

```bash
cd lambda_migration/migrated_functions
./deploy_all_functions.sh
./setup_eventbridge_schedule.sh
```

**Pros**: Data starts updating daily immediately
**Cons**: Bronze tables will be empty until Lambda runs

### Option B: Create Silver Layer First
This sets up the complete pipeline before daily updates:

```bash
# 1. Verify bronze tables
# 2. Create unified_data
# 3. Deploy forecast agent
# 4. Then deploy Lambda functions
```

**Pros**: Complete pipeline ready before daily updates
**Cons**: Relies on existing historical data in S3

**Recommended**: Option A - Deploy Lambda functions next to start daily data flow

---

## Verification Commands

### Check Bronze Tables (After notebook completes)

```sql
-- In Databricks SQL Editor:
SHOW TABLES IN commodity.bronze;

-- Check row counts
SELECT 'v_market_data_all' as table_name, COUNT(*) as row_count
FROM commodity.bronze.v_market_data_all
UNION ALL
SELECT 'v_vix_data_all', COUNT(*) FROM commodity.bronze.v_vix_data_all
UNION ALL
SELECT 'v_macro_data_all', COUNT(*) FROM commodity.bronze.v_macro_data_all
UNION ALL
SELECT 'v_weather_data_all', COUNT(*) FROM commodity.bronze.v_weather_data_all
UNION ALL
SELECT 'v_cftc_data_all', COUNT(*) FROM commodity.bronze.v_cftc_data_all
UNION ALL
SELECT 'bronze_gkg', COUNT(*) FROM commodity.bronze.bronze_gkg;
```

### Check S3 Data

```bash
# List files in S3 bucket
aws s3 ls s3://groundtruth-capstone/landing/ --recursive --region us-west-2 | head -20

# Check specific data folders
aws s3 ls s3://groundtruth-capstone/landing/market_data/ --region us-west-2
aws s3 ls s3://groundtruth-capstone/landing/gdelt/filtered/ --region us-west-2
```

### Check Lambda Functions (After deployment)

```bash
# List deployed functions
aws lambda list-functions --region us-west-2 --query 'Functions[?starts_with(FunctionName, `market-data`) || starts_with(FunctionName, `weather-data`) || starts_with(FunctionName, `vix-data`) || starts_with(FunctionName, `fx-calculator`) || starts_with(FunctionName, `cftc-data`) || starts_with(FunctionName, `gdelt-processor`)].FunctionName'

# Test a function
aws lambda invoke \
    --function-name market-data-fetcher \
    --region us-west-2 \
    response.json && cat response.json
```

---

## Architecture Summary

### Daily Data Flow (After Full Deployment)

```
2:00 AM UTC → Lambda Functions Execute
              ├── market-data-fetcher → Coffee/Sugar prices
              ├── weather-data-fetcher → Weather data
              ├── vix-data-fetcher → VIX volatility
              ├── fx-calculator-fetcher → FX rates
              ├── cftc-data-fetcher → CFTC data
              └── gdelt-processor → GDELT news
                    ↓
              Write CSV files to s3://groundtruth-capstone/landing/

3:00 AM UTC → Databricks Job Executes
              ├── Auto Loader picks up new CSVs
              ├── Ingests to Delta tables (commodity.landing.*)
              └── Refreshes bronze views (commodity.bronze.v_*)
                    ↓
              Bronze tables updated with latest data

Daily      → Forecast Agent Runs (manual or scheduled)
              ├── Reads commodity.silver.unified_data
              ├── Generates 14-day forecasts
              └── Writes to commodity.silver.point_forecasts
```

---

## Files Created

### Lambda Migration
- `lambda_migration/inventory_lambdas.py` - Function inventory script
- `lambda_migration/migrate_functions.py` - Migration script
- `lambda_migration/migrated_functions/` - Updated Lambda packages (6 functions)
  - `market-data-fetcher.zip` (49.17 MB)
  - `weather-data-fetcher.zip` (49.18 MB)
  - `vix-data-fetcher.zip` (49.18 MB)
  - `fx-calculator-fetcher.zip` (49.18 MB)
  - `cftc-data-fetcher.zip` (40.33 MB)
  - `gdelt-processor.zip` (0.01 MB)
- `lambda_migration/migrated_functions/deploy_all_functions.sh` - Deployment script
- `lambda_migration/migrated_functions/setup_eventbridge_schedule.sh` - Scheduler setup
- `lambda_migration/migrated_functions/README.md` - Deployment instructions

### Databricks Setup
- `lambda_migration/databricks_etl_setup.py` - Bronze layer notebook (~350 lines)
- `lambda_migration/setup_databricks_daily_job.sh` - Daily job setup
- `lambda_migration/upload_notebook_to_databricks.py` - Notebook deployment script
- `lambda_migration/monitor_databricks_run.py` - Run monitoring script

### Documentation
- `lambda_migration/DEPLOYMENT_GUIDE.md` - Complete deployment guide (~500 lines)
- `lambda_migration/DEPLOYMENT_STATUS.md` - This file
- `research_agent/old_workspace_migration/MIGRATION_GUIDE.md` - Table mappings

### Infrastructure
- `infra/databricks_config.yaml` - Workspace configuration
- `infra/.databrickscfg` - Databricks CLI config
- `infra/.env` - Environment variables
- `infra/README.md` - Infrastructure usage guide

---

## Troubleshooting

### Issue: Bronze layer notebook fails

**Check**:
1. Cluster has S3 instance profile attached
2. S3 bucket `groundtruth-capstone` exists and has data
3. Cluster has write permissions to Unity Catalog

**Solution**:
```bash
# View notebook logs in Databricks UI
# Check cluster event logs
# Verify S3 access: aws s3 ls s3://groundtruth-capstone/landing/
```

### Issue: Auto Loader not picking up files

**Check**:
1. CSV files exist in S3 landing folders
2. Checkpoint locations are accessible
3. Schema locations are accessible

**Solution**:
```python
# In Databricks, run:
display(dbutils.fs.ls("s3://groundtruth-capstone/landing/market_data/"))

# Check checkpoint
display(dbutils.fs.ls("s3://groundtruth-capstone/_checkpoints/market_data/"))
```

### Issue: Bronze views return 0 rows

**Possible causes**:
1. No data in S3 yet (Lambda functions haven't run)
2. Auto Loader streams haven't processed files
3. Schema mismatch

**Solution**:
```sql
-- Check landing tables first
SELECT COUNT(*) FROM commodity.landing.market_data_inc;

-- If 0, check S3:
-- aws s3 ls s3://groundtruth-capstone/landing/market_data/

-- If files exist but table is empty, rerun Auto Loader stream
```

---

## Cost Estimates

**AWS Lambda**: ~$5/month
- 6 functions running daily
- 2-5 minutes each
- Minimal compute

**S3 Storage**: ~$1/month
- 1-2 GB CSV files
- Standard storage class

**Databricks**: ~$20-30/month
- Daily job: 10 minutes
- Cluster: i3.xlarge autoscaling
- Development time extra

**Total**: ~$25-40/month for automated daily pipeline

---

## Support

**Current Run**: https://dbc-fd7b00f3-7a6d.cloud.databricks.com/#job/runs/434014329903945

**Monitor Script**:
```bash
cd lambda_migration
python3 monitor_databricks_run.py
```

**Documentation**:
- Deployment Guide: `lambda_migration/DEPLOYMENT_GUIDE.md`
- Migration Guide: `research_agent/old_workspace_migration/MIGRATION_GUIDE.md`
- Lambda README: `lambda_migration/migrated_functions/README.md`

---

## Timeline

- **Oct 29, 2025 - Morning**: Migration initiated
  - Old workspace code extracted
  - Databricks credentials secured
  - Lambda functions migrated

- **Oct 29, 2025 - 6:50 AM UTC**: Bronze layer deployment started
  - Notebook uploaded
  - Cluster started
  - ETL setup running

- **Expected - 7:00 AM UTC**: Bronze layer complete
  - All tables created
  - Ready for Lambda deployment

- **Expected - Later today**: Lambda functions deployed
  - Daily data refresh active
  - EventBridge schedule configured

- **Expected - This week**: Full pipeline operational
  - Silver layer created
  - Forecast agent deployed
  - Daily forecasts generating

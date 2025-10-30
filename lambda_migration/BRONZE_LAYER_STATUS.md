# Bronze Layer Setup - Status

**Last Updated**: October 30, 2025 - 7:15 AM UTC
**Status**: 🟢 IN PROGRESS

---

## Current Run

**Run ID**: 805674989503307
**Status**: RUNNING (3.7 minutes elapsed)
**Expected Duration**: 5-10 minutes
**Monitor**: https://dbc-fd7b00f3-7a6d.cloud.databricks.com/#job/runs/805674989503307

---

## What's Being Created

### 1. Schemas (In Progress)
```
commodity.bronze    - Raw data with deduplication views
commodity.silver    - Cleaned and joined data for forecasting
commodity.landing   - Raw Delta tables from S3
```

### 2. Auto Loader Streams (In Progress)
The notebook is setting up Auto Loader to ingest CSV files from S3:

```
s3://groundtruth-capstone/landing/market_data/  → commodity.landing.market_data_inc
s3://groundtruth-capstone/landing/vix_data/     → commodity.landing.vix_data_inc
s3://groundtruth-capstone/landing/macro_data/   → commodity.landing.macro_data_inc
s3://groundtruth-capstone/landing/weather_data/ → commodity.landing.weather_data_inc
s3://groundtruth-capstone/landing/cftc_data/    → commodity.landing.cftc_data_inc
```

### 3. Bronze Views (Pending)
Once data is ingested, these views will be created:

```
commodity.bronze.v_market_data_all   - Deduplicated market prices
commodity.bronze.v_vix_data_all      - Deduplicated VIX volatility
commodity.bronze.v_macro_data_all    - Deduplicated FX rates (latest by ingest_ts)
commodity.bronze.v_weather_data_all  - Deduplicated weather data
commodity.bronze.v_cftc_data_all     - Deduplicated CFTC trader positions
commodity.bronze.bronze_gkg          - GDELT news sentiment
```

---

## Technical Details

### Auto Loader Configuration
Each stream uses:
- **Format**: cloudFiles (Auto Loader)
- **File Format**: CSV with header
- **Schema Inference**: Enabled with schema evolution
- **Checkpointing**: Enabled for incremental processing
- **Trigger**: `availableNow=True` (batch mode - process all files and stop)

### Example Stream Setup (Market Data)
```python
spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.inferColumnTypes", "true")
    .option("cloudFiles.schemaLocation", "s3://groundtruth-capstone/_schemas/market_data")
    .option("header", "true")
    .load("s3://groundtruth-capstone/landing/market_data/")
    .withColumn("source_file", expr("_metadata.file_path"))
    .withColumn("ingest_ts", current_timestamp())
    .writeStream
    .format("delta")
    .option("checkpointLocation", "s3://groundtruth-capstone/_checkpoints/market_data")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable("commodity.landing.market_data_inc")
```

---

## Progress Tracking

**Completed**:
- [x] Commodity catalog exists (confirmed by user)
- [x] S3 bucket connected (groundtruth-capstone)
- [x] Bronze layer notebook created
- [x] Notebook uploaded to Databricks
- [x] Cluster started (general-purpose-mid-compute)
- [x] Notebook execution started

**In Progress** (Current Phase):
- [🔄] Creating schemas (bronze, silver, landing)
- [🔄] Setting up Auto Loader streams (5 data sources)
- [🔄] Ingesting CSV data from S3 to Delta tables
- [⏳] Creating bronze views with deduplication
- [⏳] Creating GDELT bronze table
- [⏳] Running verification queries

**Pending**:
- [ ] Verify row counts in all bronze tables
- [ ] Set up daily refresh job
- [ ] Deploy Lambda functions for daily updates

---

## Expected Outcome

Once the notebook completes successfully, you will have:

1. **6 Bronze Tables** in `commodity.landing.*`:
   - market_data_inc (Coffee & Sugar prices from Yahoo Finance)
   - vix_data_inc (VIX volatility from FRED)
   - macro_data_inc (Exchange rates: COP/USD, etc. from FRED)
   - weather_data_inc (Weather data from OpenWeather)
   - cftc_data_inc (CFTC commitment of traders data)

2. **6 Bronze Views** in `commodity.bronze.*`:
   - v_market_data_all (deduplicated)
   - v_vix_data_all (deduplicated)
   - v_macro_data_all (deduplicated by latest ingest)
   - v_weather_data_all (deduplicated by date/region/commodity)
   - v_cftc_data_all (deduplicated)
   - bronze_gkg (GDELT news data - external table)

3. **Schema Locations** for future ingestion:
   - Stored in s3://groundtruth-capstone/_schemas/
   - Auto-detected column types preserved

4. **Checkpoints** for resumable processing:
   - Stored in s3://groundtruth-capstone/_checkpoints/
   - Enables incremental data processing

---

## Monitoring Commands

### Check Run Status
```bash
# Set your Databricks token first:
# export DATABRICKS_TOKEN=$(grep '^token' infra/.databrickscfg | cut -d' ' -f3)

curl -s 'https://dbc-fd7b00f3-7a6d.cloud.databricks.com/api/2.1/jobs/runs/get?run_id=805674989503307' \
  -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
  | python3 -c "import json, sys; data=json.load(sys.stdin); state=data.get('state',{}); print(f\"Status: {state.get('life_cycle_state')}\"); print(f\"Result: {state.get('result_state', 'N/A')}\")"
```

### View Logs (After Completion)
Go to Databricks UI:
https://dbc-fd7b00f3-7a6d.cloud.databricks.com/#job/runs/805674989503307

### Verify Tables Created (After Completion)
```sql
-- In Databricks SQL Editor:
SHOW SCHEMAS IN commodity;
SHOW TABLES IN commodity.bronze;
SHOW TABLES IN commodity.landing;

-- Check row counts:
SELECT 'market_data_inc' as table, COUNT(*) as rows FROM commodity.landing.market_data_inc
UNION ALL
SELECT 'vix_data_inc', COUNT(*) FROM commodity.landing.vix_data_inc
UNION ALL
SELECT 'macro_data_inc', COUNT(*) FROM commodity.landing.macro_data_inc
UNION ALL
SELECT 'weather_data_inc', COUNT(*) FROM commodity.landing.weather_data_inc
UNION ALL
SELECT 'cftc_data_inc', COUNT(*) FROM commodity.landing.cftc_data_inc;
```

---

## Troubleshooting

### If the notebook fails:

**Check cluster logs**:
1. Go to Compute → general-purpose-mid-compute
2. Check Event Log for errors

**Check S3 access**:
```sql
-- In Databricks notebook:
%python
display(dbutils.fs.ls("s3://groundtruth-capstone/landing/"))
display(dbutils.fs.ls("s3://groundtruth-capstone/landing/market_data/"))
```

**Common issues**:
1. **No data in S3**: Lambda functions haven't run yet → Deploy Lambda functions first
2. **S3 permission denied**: Cluster missing S3 instance profile → Check IAM role attachment
3. **Schema mismatch**: CSV structure changed → Auto Loader with mergeSchema=true should handle this
4. **Checkpoint corruption**: Delete checkpoint and re-run → `dbutils.fs.rm("s3://groundtruth-capstone/_checkpoints/market_data", True)`

---

## Next Steps (After Completion)

### Immediate Next Steps:
1. **Verify data loaded**: Check row counts in all tables
2. **Inspect sample data**: SELECT * FROM commodity.bronze.v_market_data_all LIMIT 10
3. **Document data ranges**: Check MIN/MAX dates

### Daily Refresh Setup:
1. Create Databricks job to run this notebook daily at 3 AM UTC
2. Schedule: `0 3 * * *` (cron)
3. Runs after Lambda functions complete (2 AM UTC)

### Lambda Deployment (Parallel Track):
While bronze layer stabilizes, deploy Lambda functions:
```bash
cd lambda_migration/migrated_functions
./deploy_all_functions.sh
./setup_eventbridge_schedule.sh
```

### Silver Layer Creation:
Once bronze data is verified:
1. Run `research_agent/create_gdelt_unified_data.py` (adapted for Databricks)
2. Create `commodity.silver.unified_data` table
3. Join bronze tables with GDELT sentiment
4. Add weather features and lagged variables

---

## Success Criteria

Bronze layer setup is successful when:
- [  ] All 3 schemas exist (bronze, silver, landing)
- [  ] All 5 Auto Loader streams completed without errors
- [  ] All 5 landing tables have data (row_count > 0)
- [  ] All 6 bronze views/tables are queryable
- [  ] No errors in notebook execution
- [  ] Checkpoint and schema locations created in S3

---

## Files Created

### Notebooks
- `/Workspace/Users/ground.truth.datascience@gmail.com/setup_bronze_layer` - Bronze layer setup (deployed)
- `lambda_migration/setup_bronze_layer.py` - Local copy

### Scripts
- `lambda_migration/upload_notebook_to_databricks.py` - Deployment script
- `lambda_migration/monitor_databricks_run.py` - Monitoring script

### Documentation
- `lambda_migration/DEPLOYMENT_GUIDE.md` - Complete deployment guide
- `lambda_migration/BRONZE_LAYER_STATUS.md` - This file

---

**Estimated Completion**: ~7:20 AM UTC (2-3 more minutes)

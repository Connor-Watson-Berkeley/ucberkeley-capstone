# Forecast Backfill Status

**Last Updated:** 2025-11-22 17:10 PST

## Simple Goal

**✅ Train models → ✅ Load into `trained_models` table → ⏳ Run inference → ⏳ Load into `distributions` and `point_forecasts` tables**

### Workflow Details

1. **TRAIN** - Train models on historical data with semiannual frequency
   - Commodities: Coffee, Sugar
   - Models: naive, xgboost, sarimax_auto_weather
   - Date Range: 2018-01-01 to 2025-11-17
   - Output: `commodity.forecast.trained_models` table

2. **INFERENCE** - Use trained models to generate forecasts
   - Load pretrained models from database
   - Generate 2,000 Monte Carlo simulation paths per forecast
   - Output:
     - `commodity.forecast.distributions` (2000 paths × 14 days)
     - `commodity.forecast.point_forecasts` (14-day forecasts with intervals)

## Current Status

### ✅ Training COMPLETED Successfully!

**Databricks Job ID:** `773612620032262`
- **Status:** ✅ SUCCESS
- **Started:** 2025-11-22 16:49 PST
- **Completed:** 2025-11-22 17:07 PST (18 minutes)
- **Cluster:** Existing ML cluster `1121-061338-wj2kqadu`

**What was done:**
- Trained 199 models successfully saved to `commodity.forecast.trained_models`
  - 2 commodities (Coffee, Sugar)
  - 3 models per commodity (naive, xgboost, sarimax_auto_weather)
  - 16 semiannual training windows from 2018-01-01 to 2025-07-01
  - Models saved with version `v1.0`

**Incremental Testing Approach:**
- ✅ Test 1: Verified wheel package imports
- ✅ Test 2: Tested single model training
- ✅ Test 3: Tested database write with correct schema
- ✅ Full training: All 96 models trained successfully

**Fix Applied:**
- Created wheel package for ground_truth module
- Uploaded to DBFS at `/dbfs/FileStore/packages/ground_truth-0.1.0-py3-none-any.whl`
- Fixed database schema (training_cutoff_date vs training_window_end)
- Used existing ML cluster instead of creating new clusters

### ⏳ Next: Inference Backfill

Ready to run inference to populate `distributions` and `point_forecasts` tables

## Directory Structure

**Working Directory:** `/Users/connorwatson/Documents/Data Science/DS210-capstone/ucberkeley-capstone/forecast_agent`

**Key Files:**
- `databricks_train_simple.py` - Training notebook (currently running on Databricks)
- `backfill_rolling_window.py` - Inference script (will run after training)
- `../infra/.env` - Databricks credentials (DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH)

## Next Steps (After Training Completes)

### Option 1: Databricks Backfill (Recommended)

Submit inference job to Databricks:
```bash
cd "/Users/connorwatson/Documents/Data Science/DS210-capstone/ucberkeley-capstone/forecast_agent"
set -a && source ../infra/.env && set +a

# Create and run backfill submission script
python /tmp/submit_backfill_job.py  # (will need to be created)
```

### Option 2: Local Backfill

Run inference locally using pretrained models:
```bash
cd "/Users/connorwatson/Documents/Data Science/DS210-capstone/ucberkeley-capstone/forecast_agent"
set -a && source ../infra/.env && set +a

# Coffee
python backfill_rolling_window.py \
  --commodity Coffee \
  --models naive xgboost sarimax_auto_weather \
  --train-frequency semiannually \
  --start-date 2018-01-01 \
  --end-date 2025-11-17 \
  --model-version-tag v1.0

# Sugar
python backfill_rolling_window.py \
  --commodity Sugar \
  --models naive xgboost sarimax_auto_weather \
  --train-frequency semiannually \
  --start-date 2018-01-01 \
  --end-date 2025-11-17 \
  --model-version-tag v1.0
```

**Note:** Local backfill may fail due to NumPy/pmdarima binary incompatibility. Databricks is preferred.

## Key Technical Details

### Database Tables

**Input:**
- `commodity.silver.unified_data` - Historical commodity prices + weather + sentiment

**Output:**
- `commodity.forecast.trained_models` - Fitted model parameters (JSON or S3)
- `commodity.forecast.distributions` - 2,000 Monte Carlo paths (columns: day_1 to day_14, path_id 0-1999)
- `commodity.forecast.point_forecasts` - 14-day forecasts (columns: day_1 to day_14 with prediction intervals)

### Databricks Configuration

**Credentials Location:** `../infra/.env`
```bash
DATABRICKS_HOST=https://dbc-5e4780f4-fcec.cloud.databricks.com
DATABRICKS_TOKEN=dapi6272b22eb65...  (36 chars)
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/...
```

**Cluster Config:**
- Runtime: ML 14.3.x-cpu-ml-scala2.12 (includes scikit-learn, xgboost, statsmodels)
- Node Type: i3.xlarge
- Mode: Single-node (SINGLE_USER data security for Unity Catalog)
- Additional Library: pmdarima (via pip)

### Model Training Frequency

**Semiannual Training Windows:**
- Instead of training 2,875 models (one per date), we train 96 models (16 windows)
- Each model is reused for inference across multiple dates in its window
- Result: ~180x faster than train-per-date approach

## Monitoring Commands

### Check Training Status
```bash
cd "/Users/connorwatson/Documents/Data Science/DS210-capstone/ucberkeley-capstone/forecast_agent"
tail -f /tmp/training_monitor.log
```

### Check Databricks Job Manually
```bash
set -a && source ../infra/.env && set +a

# Quick status check
curl -X GET "${DATABRICKS_HOST}/api/2.1/jobs/runs/get?run_id=394970656151512" \
  -H "Authorization: Bearer ${DATABRICKS_TOKEN}" | jq '.state'
```

### Check Trained Models Count
Once training completes, verify models were saved:
```sql
-- In Databricks SQL Editor
SELECT
  commodity,
  model_name,
  COUNT(*) as num_models,
  MIN(training_window_end) as first_window,
  MAX(training_window_end) as last_window
FROM commodity.forecast.trained_models
WHERE model_version = 'v1.0'
GROUP BY commodity, model_name
ORDER BY commodity, model_name;

-- Expected: ~16 models per (commodity, model_name) combination
```

## Troubleshooting

### Local NumPy Error
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```
**Solution:** Use Databricks for both training and inference. Local environment has incompatible numpy/pmdarima versions.

### Databricks Import Error
```
ImportError: cannot import name 'sql' from 'databricks'
```
**Solution:** Inside Databricks notebooks, use `spark.sql()` instead of `from databricks import sql`. The training script is already fixed.

### Session Timeout
Databricks has 15-minute session timeout. The backfill script automatically reconnects every 50 forecasts.

## Reference Documents

- `DATABRICKS_CODING_GUIDE.md` - SQL connectivity patterns, cluster configs
- `CLAUDE.md` - Full architecture, train-once/inference-many pattern
- `TRAIN_ONCE_INFERENCE_MANY_SUMMARY.md` - Performance benchmarks
- `README_SPARK_BACKFILL.md` - Spark parallelization for massive backfills

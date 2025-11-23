# Databricks Training Quick Start

**The fastest way to train forecast models in Databricks.**

## TL;DR

```bash
# 1. Set up credentials
set -a && source ../infra/.env && set +a

# 2. Run training (inline version - no Git repo dependencies)
python /tmp/run_training_inline.py

# 3. Monitor at the URL printed (format: https://dbc-XXX.cloud.databricks.com/jobs/{job_id}/runs/{run_id}?o={org_id})
```

## What This Does

Trains Coffee and Sugar forecast models (naive, xgboost, sarimax_auto_weather) using the **train-once/inference-many** pattern:
- Semiannual training windows from 2018-01-01 to 2025-11-17
- Saves fitted models to `commodity.forecast.trained_models` table
- Expected duration: 30-60 minutes (SARIMAX is slow but accurate)

## Why Inline Feature Engineering?

**Problem**: Databricks notebooks can't import from Git repo structure (`ModuleNotFoundError: No module named 'ground_truth.features'`)

**Solution**: `databricks_train_fresh_models_inline.py` embeds all feature engineering functions directly in the notebook

## Files You Need

### 1. Training Notebook (already created)
- **databricks_train_fresh_models_inline.py** - Databricks notebook with inlined feature engineering

### 2. Submission Script (already in /tmp)
- **/tmp/run_training_inline.py** - Uploads notebook and submits job via API

## Step-by-Step Workflow

### 1. Load Databricks Credentials
```bash
cd forecast_agent/
set -a && source ../infra/.env && set +a
```

Required environment variables:
- `DATABRICKS_HOST` - Your workspace URL
- `DATABRICKS_TOKEN` - Personal access token

### 2. Submit Training Job
```bash
python /tmp/run_training_inline.py
```

This will:
1. Upload `databricks_train_fresh_models_inline.py` to your workspace
2. Submit a single-node job (i3.xlarge, Unity Catalog compatible)
3. Monitor for 2 minutes and print status
4. Save run ID to `/tmp/training_run_id_inline.txt`

### 3. Monitor Progress

The script prints a URL like:
```
https://dbc-5e4780f4-fcec.cloud.databricks.com/jobs/637644265989800/runs/975934593762469?o=2790149594734237
```

Click it to see live logs and progress.

### 4. Check Status Anytime
```bash
cd forecast_agent/
set -a && source ../infra/.env && set +a

python3 << 'EOF'
import urllib.request, json, os

host = os.environ['DATABRICKS_HOST']
token = os.environ['DATABRICKS_TOKEN']
run_id = open('/tmp/training_run_id_inline.txt').read().strip()

url = f"{host}/api/2.1/jobs/runs/get?run_id={run_id}"
req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'})

with urllib.request.urlopen(req) as resp:
    state = json.loads(resp.read())['state']
    print(f"Status: {state.get('life_cycle_state')} / {state.get('result_state', 'N/A')}")
EOF
```

## Expected Results

**SUCCESS**:
```
✅✅✅ TRAINING COMPLETED SUCCESSFULLY! ✅✅✅

All models trained and saved to commodity.forecast.trained_models
Next step: Run backfill to generate forecasts
```

**Trained models saved**:
- Table: `commodity.forecast.trained_models`
- Partitioned by: (commodity, model_name, year, month)
- Format: JSON (<1MB) or S3 path (≥1MB)

## Next Steps After Training

Once training completes successfully:

### 1. Verify Models Were Saved
```sql
SELECT
    commodity,
    model_name,
    training_window_end,
    COUNT(*) as model_count
FROM commodity.forecast.trained_models
WHERE model_version = 'v1.0'
GROUP BY commodity, model_name, training_window_end
ORDER BY commodity, model_name, training_window_end DESC
```

### 2. Run Backfill to Generate Forecasts
```bash
# Use pretrained models for fast inference
python backfill_rolling_window.py \
    --commodity Coffee \
    --models naive xgboost sarimax_auto_weather \
    --train-frequency semiannually \
    --start-date 2018-01-01 \
    --end-date 2025-11-17 \
    --model-version-tag v1.0
```

### 3. Validate Backfill Coverage
```bash
python check_backfill_coverage.py \
    --commodity Coffee \
    --models naive xgboost sarimax_auto_weather
```

## Troubleshooting

### Job Fails with "ModuleNotFoundError"
You're using the wrong notebook. Use `databricks_train_fresh_models_inline.py` (not `databricks_train_fresh_models.py`).

### Training Takes Longer Than Expected
SARIMAX with weather features is computationally intensive:
- Naive: ~2-5 minutes
- XGBoost: ~10-15 minutes
- SARIMAX (auto weather): ~40-60 minutes

Total: 50-80 minutes for all models.

### "HTTP Error 400: Bad Request"
Check your job configuration. Common issues:
- Invalid cluster config
- Missing notebook path
- Incorrect API endpoint

## Architecture

### Train-Once/Inference-Many Pattern
```
PHASE 1 (This workflow):
┌─────────────────────────────────────┐
│ Train models every 6 months         │
│ Save to trained_models table        │
│ ~16 training runs (2018-2025)       │
└─────────────────────────────────────┘

PHASE 2 (Next step):
┌─────────────────────────────────────┐
│ Load pretrained models from DB      │
│ Generate forecasts for all dates    │
│ ~2,875 forecasts (7.8 years daily)  │
│ 180x faster than train-per-date     │
└─────────────────────────────────────┘
```

### Why This Works
- **Databricks ML Runtime** pre-installs scikit-learn, xgboost, statsmodels
- **Single-node cluster** stays within AWS quota
- **Unity Catalog** enabled via `SINGLE_USER` mode
- **Inline feature engineering** eliminates Git repo dependencies

## Related Documentation

- **FEATURE_ENGINEERING_GUIDE.md** - Feature pipeline details
- **README_SPARK_BACKFILL.md** - Parallel backfill with Spark
- **TRAIN_ONCE_INFERENCE_MANY_SUMMARY.md** - Architecture overview
- **CLAUDE.md** - Complete reference for all workflows

## Success Metrics

After training completes, you should see:
- **~16 trained models per commodity** (semiannual windows from 2018-2025)
- **~96 total models** (Coffee + Sugar) × (naive + xgboost + sarimax_auto_weather) × 16 windows
- **Total table size**: ~50-100 MB (mostly SARIMAX models)

## Notes

- Training is idempotent - rerunning with same parameters overwrites existing models
- Use `model_version` tags to maintain multiple model versions
- SARIMAX models are large (~1-5 MB each) due to statsmodels pickle format
- XGBoost and naive models are tiny (~10-50 KB each)

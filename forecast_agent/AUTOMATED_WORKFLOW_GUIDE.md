# Automated Training & Backfill Workflow Guide

Complete guide for running automated model training and forecast backfilling in Databricks.

## Overview

This project implements a **train-once/inference-many** architecture with two automated workflows:

1. **Training Workflow**: Trains models on periodic windows (semiannually) and persists to database
2. **Backfill Workflow**: Loads pretrained models and generates forecasts for historical dates

Both workflows can run:
- **Manually** - Direct Python script execution (local or Databricks notebook)
- **Automated** - API-driven autonomous execution with error handling and monitoring

## Quick Start

### Option 1: Manual Execution (Databricks Notebook)

1. Open training notebook in Databricks Repos:
   ```
   /Repos/Project_Git/ucberkeley-capstone/forecast_agent/databricks_train_fresh_models
   ```

2. Select ML Runtime cluster (14.3.x-cpu-ml-scala2.12)

3. Run all cells

4. Training completes in 20-40 minutes for 96 models (Coffee + Sugar × 3 models × 16 windows)

### Option 2: Automated Execution (Local → Databricks API)

```bash
# Load credentials
cd forecast_agent
set -a && source ../infra/.env && set +a

# Run automated training (monitors and handles errors)
python /tmp/databricks_autonomous_training.py
```

The automated script:
- Uploads notebook to Databricks workspace
- Submits training job via API
- Monitors every 60 seconds with live status updates
- Auto-detects errors and suggests fixes
- Provides Databricks UI link for manual inspection

## Architecture

### Phase 1: Training (Periodic, Expensive)

**Purpose**: Train models once on fixed time windows

**Input**: `commodity.silver.unified_data` (historical price + features)

**Output**: `commodity.forecast.trained_models` (fitted model JSON or S3)

**Frequency**: Semiannual (every 6 months)

**Command**:
```bash
python train_models.py --commodity Coffee --models naive xgboost sarimax_auto_weather \
  --train-frequency semiannually --start-date 2018-01-01 --end-date 2025-11-17
```

**Performance**: ~16 training operations (vs 2,875 if training per-date) = **180x speedup**

**Database Schema**:
```sql
CREATE TABLE commodity.forecast.trained_models (
    commodity STRING,
    model_name STRING,
    model_version STRING,
    training_start_date DATE,      -- First date in training window
    training_cutoff_date DATE,     -- Last date in training window
    fitted_model_json STRING,      -- Serialized model (<1MB)
    fitted_model_s3_path STRING,   -- S3 location (≥1MB models)
    created_at TIMESTAMP,
    created_by STRING,
    year INT,                      -- Partition key
    month INT                      -- Partition key
) PARTITIONED BY (year, month);
```

### Phase 2: Backfill (Daily, Fast)

**Purpose**: Generate forecasts for all historical dates using pretrained models

**Input**:
- `commodity.forecast.trained_models` (pretrained models)
- `commodity.silver.unified_data` (last 90 days for inference)

**Output**:
- `commodity.forecast.distributions` (2,000 Monte Carlo paths × 14 days)
- `commodity.forecast.point_forecasts` (mean predictions + confidence intervals)
- `commodity.forecast.forecast_metadata` (performance metrics)

**Command**:
```bash
python backfill_rolling_window.py --commodity Coffee --models naive xgboost \
  --train-frequency semiannually --start-date 2018-01-01 --end-date 2025-11-17
```

**Performance**:
- Sequential: 10-20 hours for 2,800 forecasts
- Parallel (Databricks): 20-60 minutes with 200 Spark partitions

**Resumability**: Auto-skips existing forecasts, just rerun to continue

## Automated Training Workflow

### Architecture

```
Local Machine                 Databricks
     │                             │
     │  1. Upload Notebook         │
     ├───────────────────────────>│
     │                             │
     │  2. Submit Job              │
     ├───────────────────────────>│
     │                             │
     │                        ┌────┴────┐
     │                        │ Cluster │
     │                        │ Start   │
     │                        └────┬────┘
     │  3. Poll Status (60s)       │
     │<────────────────────────────┤
     │                             │
     │                        ┌────┴────┐
     │                        │ Execute │
     │                        │ Notebook│
     │                        └────┬────┘
     │                             │
     │  4. Get Status + Errors     │
     │<────────────────────────────┤
     │                             │
     └─────────────────────────────┘
```

### Implementation

**Script**: `/tmp/databricks_autonomous_training.py`

**Features**:
- Notebook upload to workspace via API
- Job submission with single-node ML Runtime cluster
- Live monitoring with state change detection
- Error retrieval from multi-task jobs
- URL generation for manual debugging
- Comprehensive logging with timestamps

**Key Functions**:

```python
class DatabricksJobMonitor:
    def upload_notebook(self, local_path, workspace_path):
        """Upload Python notebook to Databricks workspace"""
        # Reads local .py file
        # Base64 encodes content
        # Uploads via /api/2.0/workspace/import (v2.0, NOT v2.1!)

    def submit_job(self, notebook_path, cluster_config):
        """Submit training job with ephemeral cluster"""
        # Creates single-node ML Runtime cluster
        # Installs databricks-sql-connector + pmdarima
        # Submits job via /api/2.1/jobs/runs/submit
        # Returns run_id for monitoring

    def monitor_job(self, run_id, poll_interval=60):
        """Monitor job execution with live updates"""
        # Polls /api/2.1/jobs/runs/get every 60 seconds
        # Prints life_cycle_state changes (PENDING → RUNNING → TERMINATED)
        # Detects terminal states (SUCCESS, FAILED, INTERNAL_ERROR)
        # Retrieves errors on failure

    def get_error_logs(self, parent_run_id):
        """Retrieve errors from multi-task jobs"""
        # Gets task run IDs from parent run
        # Fetches output for each task via /api/2.1/jobs/runs/get-output
        # Prints error messages for failed tasks
```

### Usage Example

```bash
cd /path/to/forecast_agent
set -a && source ../infra/.env && set +a

python /tmp/databricks_autonomous_training.py
```

**Expected Output**:
```
================================================================================
Autonomous Databricks Training Job Executor
================================================================================

Step 1: Upload Notebook
================================================================================
Uploading notebook to /Users/user@domain.com/databricks_train_fresh_models...
✓ Notebook uploaded successfully

Step 2: Submit Job
================================================================================
Submitting training job with ephemeral cluster...
✓ Job submitted: run_id=123456789

View job at: https://dbc-xxxxx.cloud.databricks.com/#job/123456789/run/1

Step 3: Monitor Job
================================================================================
[0s] Status: PENDING
[120s] Status: RUNNING
[2400s] Status: TERMINATED

✅ Job completed successfully!

Summary:
  Trained: 96 models
  Duration: 40 minutes
  Run ID: 123456789
```

### Error Handling

The automated script detects common errors:

**Error Pattern 1: Import Error**
```
ImportError: cannot import name 'sql' from 'databricks'
```
**Auto-Fix**: Script suggests adding `databricks-sql-connector` to pip install

**Error Pattern 2: Parameter Mismatch**
```
naive_train() got an unexpected keyword argument 'horizon'
```
**Auto-Fix**: Script identifies training vs prediction parameter confusion

**Error Pattern 3: Column Not Found**
```
Column 'training_cutoff_date' cannot be resolved
```
**Auto-Fix**: Script suggests database schema migration

**Manual Intervention Required**: For unknown errors, script saves error logs to `/tmp/training_error_{attempt}.txt` and provides Databricks UI link

## Automated Backfill Workflow

### Local Execution (Resume-Capable)

```bash
# Start or resume Coffee backfill
set -a && source ../infra/.env && set +a
python -u backfill_rolling_window.py \
  --commodity Coffee \
  --models naive xgboost sarimax_auto_weather \
  --train-frequency semiannually \
  --start-date 2018-01-01 \
  --end-date 2025-11-17 \
  --model-version-tag v1.0 \
  2>&1 | tee coffee_backfill.log
```

**Features**:
- Auto-detects existing forecasts (skips completed dates)
- Reconnects every 50 forecasts (handles 15-min session timeout)
- Batch writes (50 forecasts/batch) for 10-20x speedup
- Progress logging every 50 forecasts
- Saves to file for monitoring

**Monitoring Progress**:
```bash
# Watch live progress
tail -f coffee_backfill.log

# Check completion
ps aux | grep backfill_rolling_window
```

### Databricks Spark Execution (Parallel)

For massive backfills (1000+ dates), use Spark parallelization:

**Notebook**: `backfill_rolling_window_spark.py`

**Setup**:
1. Open notebook in Databricks Repos
2. Select cluster with 4+ workers (e.g., 4 × i3.xlarge = 16 cores)
3. Update configuration in Cell 1:
   ```python
   commodities = ['Coffee', 'Sugar']
   models = ['naive', 'xgboost', 'sarimax_auto_weather']
   train_frequency = 'semiannually'
   start_date = '2018-01-01'
   end_date = '2025-11-17'
   num_partitions = 64  # 2-4x cluster cores
   ```
4. Run all cells

**Performance**: 20-60 minutes vs 10-20 hours local (20-30x speedup)

**Resource Sizing**:
```
Cluster Size         Partitions    Expected Duration
4 workers (16 cores)     64             45-60 min
8 workers (32 cores)    128             25-35 min
16 workers (64 cores)   256             15-25 min
```

See `README_SPARK_BACKFILL.md` for detailed Spark configuration guide.

## Workflow Comparison

| Feature | Manual (Notebook) | Automated (API) | Spark (Parallel) |
|---------|------------------|-----------------|------------------|
| **Execution** | Run All in UI | Python script | Notebook |
| **Monitoring** | Watch UI | Live stdout | Spark UI |
| **Error Handling** | Manual review | Auto-detect | Spark retry |
| **Resumability** | Manual rerun | Rerun script | Partition-level |
| **Speed (Training)** | 20-40 min | 20-40 min | N/A |
| **Speed (Backfill)** | 10-20 hrs | 10-20 hrs | 20-60 min |
| **Best For** | Ad-hoc runs | CI/CD integration | Large backfills |

## Production Job Configuration

### Job 1: Model Training (Periodic)

**Schedule**: Every 6 months (Jan 1, Jul 1)

**Purpose**: Train new models for recently added model keys in model_registry

**Cluster**: Single-node ML Runtime (cost-optimized)

**Configuration**:
```json
{
  "name": "Forecast Agent - Model Training",
  "schedule": {
    "quartz_cron_expression": "0 0 0 1 1,7 ? *",
    "timezone_id": "America/Los_Angeles"
  },
  "tasks": [{
    "task_key": "train_models",
    "notebook_task": {
      "notebook_path": "/Repos/Project_Git/ucberkeley-capstone/forecast_agent/databricks_train_fresh_models",
      "source": "WORKSPACE"
    },
    "new_cluster": {
      "spark_version": "14.3.x-cpu-ml-scala2.12",
      "node_type_id": "i3.xlarge",
      "num_workers": 0,
      "data_security_mode": "SINGLE_USER"
    },
    "libraries": [
      {"pypi": {"package": "databricks-sql-connector"}},
      {"pypi": {"package": "pmdarima"}}
    ],
    "timeout_seconds": 7200
  }]
}
```

**Outputs**:
- New models saved to `commodity.forecast.trained_models`
- Notification on completion/failure

### Job 2: Daily Inference (Automated)

**Schedule**: Daily at 8:00 AM (after unified_data updates)

**Purpose**: Generate forecasts for today using latest pretrained models

**Cluster**: 4-worker cluster (parallel execution)

**Configuration**:
```json
{
  "name": "Forecast Agent - Daily Inference",
  "schedule": {
    "quartz_cron_expression": "0 0 8 * * ? *",
    "timezone_id": "America/Los_Angeles"
  },
  "tasks": [{
    "task_key": "generate_forecasts",
    "notebook_task": {
      "notebook_path": "/Repos/Project_Git/ucberkeley-capstone/forecast_agent/daily_inference",
      "source": "WORKSPACE",
      "base_parameters": {
        "run_date": "{{job.run_date}}",
        "commodities": "Coffee,Sugar",
        "models": "naive,xgboost,sarimax_auto_weather"
      }
    },
    "existing_cluster_id": "1121-061338-wj2kqadu",
    "timeout_seconds": 3600
  }]
}
```

**Outputs**:
- Daily forecasts written to `commodity.forecast.distributions`
- Point forecasts to `commodity.forecast.point_forecasts`
- Trading agent consumes from distributions table

## Common Issues & Solutions

### Issue 1: Job Stuck in PENDING

**Symptoms**: Job shows PENDING for >5 minutes

**Causes**:
- Cluster provisioning delay (AWS capacity)
- Library installation timeout
- Cluster quota exceeded

**Solutions**:
1. Check cluster event log for errors
2. Use existing cluster instead of ephemeral
3. Reduce cluster size (single-node if over quota)

### Issue 2: Databricks Session Timeout

**Error**: `Connection closed after 15 minutes`

**Cause**: Databricks SQL connections timeout after 15 minutes inactivity

**Solution**: Already handled in backfill scripts (reconnect every 50 forecasts)

### Issue 3: Memory Errors During SARIMAX Training

**Error**: `MemoryError: Unable to allocate array with shape (10000000,)`

**Cause**: SARIMAX auto-search tries too many orders on large datasets

**Solutions**:
1. Reduce `max_p`, `max_q`, `max_P`, `max_Q` in auto_arima params
2. Increase cluster memory (larger node type)
3. Filter training data to last 3-5 years only

### Issue 4: Backfill Running Too Slowly

**Symptoms**: <10 forecasts/hour

**Causes**:
- Training from scratch each date (not using pretrained models)
- Loading full history instead of last 90 days
- Sequential execution instead of parallel

**Solutions**:
1. Verify `--train-frequency semiannually` is set
2. Check lookback optimization is enabled (line 145 in backfill_rolling_window.py)
3. Switch to Spark parallel backfill for 20-30x speedup

## Monitoring & Debugging

### Live Monitoring (Local)

```bash
# Watch backfill progress
tail -f coffee_backfill.log | grep "Progress:"

# Count completed forecasts
grep "✓ Forecast saved" coffee_backfill.log | wc -l

# Check for errors
grep "ERROR" coffee_backfill.log
```

### Databricks UI Monitoring

1. **Jobs Page**: View all job runs and their status
   - URL: `https://<workspace>/#job/list`

2. **Specific Job Run**: View logs and execution details
   - URL: `https://<workspace>/#job/<run_id>/run/1`

3. **Cluster Events**: Diagnose cluster startup failures
   - URL: `https://<workspace>/#setting/clusters/<cluster_id>/events`

### Database Validation

```sql
-- Check trained models
SELECT
    commodity,
    model_name,
    COUNT(*) as num_models,
    MIN(training_cutoff_date) as first_window,
    MAX(training_cutoff_date) as last_window
FROM commodity.forecast.trained_models
WHERE model_version = 'v1.0'
GROUP BY commodity, model_name
ORDER BY commodity, model_name;

-- Check forecast coverage
SELECT
    commodity,
    model_name,
    COUNT(DISTINCT forecast_date) as dates_covered,
    MIN(forecast_date) as first_forecast,
    MAX(forecast_date) as last_forecast
FROM commodity.forecast.distributions
WHERE model_version = 'v1.0'
  AND is_actuals = FALSE
GROUP BY commodity, model_name
ORDER BY commodity, model_name;

-- Identify gaps in forecasts
SELECT DISTINCT forecast_date
FROM (
    SELECT EXPLODE(SEQUENCE(DATE'2018-01-01', DATE'2025-11-17', INTERVAL 1 DAY)) as forecast_date
) all_dates
WHERE forecast_date NOT IN (
    SELECT DISTINCT forecast_date
    FROM commodity.forecast.distributions
    WHERE commodity = 'Coffee'
      AND model_name = 'Naive'
      AND model_version = 'v1.0'
)
ORDER BY forecast_date;
```

## Best Practices

### 1. Use Resume Mode for Long Backfills

Never start a backfill from scratch if it's already partially complete. The scripts automatically skip existing forecasts:

```bash
# This is safe - skips completed dates
python backfill_rolling_window.py --commodity Coffee --models naive \
  --start-date 2018-01-01 --end-date 2025-11-17
```

### 2. Run Training Before Backfill

Always train models first, then backfill:

```bash
# Step 1: Train models (20-40 min)
python train_models.py --commodity Coffee --models naive xgboost \
  --train-frequency semiannually

# Step 2: Backfill forecasts (uses pretrained models)
python backfill_rolling_window.py --commodity Coffee --models naive xgboost \
  --train-frequency semiannually
```

### 3. Use Spark for Large Backfills

For >500 dates or multiple commodities/models, always use Spark:

```
Sequential: 2,800 dates × 15 sec/date = 11.7 hours
Spark (64 partitions): 2,800 dates ÷ 64 ÷ 4 sec/date = 11 minutes
```

### 4. Monitor Resource Usage

**Single-node training**: ~$0.50/hour (sufficient for 96 models)

**4-worker backfill cluster**: ~$2.00/hour (20-60 min runtime = $0.70-$2.00/backfill)

**Cost optimization**: Terminate clusters after job completion (set `autotermination_minutes`)

### 5. Version Control Model Versions

Always use semantic versioning for models:

```python
model_version = 'v1.0'  # Initial production models
model_version = 'v1.1'  # Bug fix or minor improvement
model_version = 'v2.0'  # Major architecture change
```

This allows A/B testing and rollback:

```sql
-- Compare model performance across versions
SELECT
    model_version,
    model_name,
    AVG(mae) as avg_mae,
    AVG(dir_day0) as avg_directional_accuracy
FROM commodity.forecast.forecast_metadata
WHERE commodity = 'Coffee'
GROUP BY model_version, model_name
ORDER BY avg_mae;
```

## Next Steps

After running the automated workflows:

1. **Validate Results**: Run `check_backfill_coverage.py` to verify completeness
2. **Evaluate Performance**: Run `evaluate_historical_forecasts.py` for model metrics
3. **Set Up Production Jobs**: Configure Jobs 1 and 2 in Databricks UI
4. **Enable Trading Agent**: Verify trading agent reads from distributions table
5. **Monitor Daily**: Set up alerts for job failures and model degradation

## Additional Resources

- [CLAUDE.md](./CLAUDE.md) - Project overview and common commands
- [DATABRICKS_API_GUIDE.md](./docs/DATABRICKS_API_GUIDE.md) - API patterns and error handling
- [DATABRICKS_CLUSTER_SETUP.md](./DATABRICKS_CLUSTER_SETUP.md) - Cluster configuration
- [README_SPARK_BACKFILL.md](./README_SPARK_BACKFILL.md) - Spark parallelization guide
- [TRAIN_ONCE_INFERENCE_MANY_SUMMARY.md](./TRAIN_ONCE_INFERENCE_MANY_SUMMARY.md) - Architecture deep dive

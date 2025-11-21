# Aggressive Cleanup Plan - Backfill Scripts

Last Updated: 2025-11-20

## Critical Files - DO NOT DELETE

### Primary Production Scripts

**1. `backfill_rolling_window_spark.py`** - PRIMARY PRODUCTION SCRIPT
- Spark-parallelized backfill that runs entirely on Databricks cluster
- No SQL connection timeout issues (unlike local scripts)
- Populates all 3 forecast tables:
  - `commodity.forecast.distributions` - Monte Carlo simulation paths (2,000 paths)
  - `commodity.forecast.point_forecasts` - Point estimates with prediction intervals (computed from direct model inference)
  - `commodity.forecast.forecast_actuals` - Ground truth actuals (conditional, can be disabled)
- Has resume mode to avoid duplicate work
- Expected duration: 20-60 minutes with proper cluster sizing
- Recently updated (Nov 20) with:
  - Point forecast generation from direct model inference (NOT from Monte Carlo aggregation)
  - Resume mode using LEFT ANTI JOIN
  - Conditional actuals population (`write_actuals` flag)
  - FloatType schema compatibility with Delta tables

**2. `databricks_train_and_backfill.py`** - Databricks Notebook Wrapper
- Entry point for running Spark backfill on Databricks
- Imports and calls `backfill_all_models_spark()` from the Spark script
- Required for Databricks Job execution
- Recently updated (Nov 20)

**3. `backfill_actuals.py`**
- Separate workflow for populating actuals table
- Per architectural feedback: actuals should be separate workflow (not part of forecast backfill)
- Run once to backfill historical actuals

### Optional (Consider Keeping)

**4. `backfill_rolling_window.py`**
- Local sequential backfill using SQL connection
- Has 15-minute SQL connection timeout issues (KNOWN ISSUE)
- Useful as reference implementation or for small date-range backfills
- Contains correct logic for table population
- Can be removed if you're confident Spark version is production-ready

## Files to Consider for Removal

### Deprecated Backfill Scripts

1. **`backfill_spark.py`** (17KB, last modified Nov 17)
   - Older Spark implementation
   - Superseded by `backfill_rolling_window_spark.py`
   - **RECOMMENDATION: DELETE** if it doesn't have unique functionality

2. **`backfill_optimized.py`** (12KB, last modified Nov 17)
   - Optimization attempt (unclear if still used)
   - **RECOMMENDATION: Review for unique features, then DELETE**

3. **`backfill_daily_forecasts.py`** (16KB, last modified Nov 20)
   - Daily forecast backfill (may be different from rolling window)
   - **RECOMMENDATION: Review purpose before deleting**

4. **`backfill_distributions_historical.py`** (10KB, last modified Nov 1)
   - Historical distributions backfill (likely deprecated)
   - **RECOMMENDATION: DELETE** if Spark script handles this

5. **`databricks_spark_backfill.py`** (9KB, last modified Nov 20)
   - Another Databricks wrapper (unclear vs `databricks_train_and_backfill.py`)
   - **RECOMMENDATION: Check if duplicate, then DELETE**

### Monitoring Scripts (All Deprecated)

These were created for monitoring local backfills that kept timing out. Since you're moving to Databricks Spark execution, these are no longer needed:

1. **`monitor_backfills.sh`** (2.3KB)
2. **`monitor_backfills_v2.sh`** (2.4KB)
3. **`monitor_backfills_v3.sh`** (2.6KB)
4. **`monitor_backfills_v4.sh`** (2.5KB)
5. **`autonomous_monitor.sh`** (5.8KB)
6. **`completion_monitor.sh`** (4.4KB)
7. **`completion_monitor_v2.sh`** (4.1KB)
8. **`monitor_and_evaluate.sh`** (2.5KB)
9. **`check_progress.sh`** (0.5KB)

**RECOMMENDATION: DELETE ALL** - These were workarounds for local timeout issues

### Shell Scripts

1. **`complete_backfill.sh`** (2.3KB, last modified Nov 15)
   - Likely runs local backfills (which have timeout issues)
   - **RECOMMENDATION: DELETE** if using Databricks Spark exclusively

### Log Files

All log files from failed/incomplete local backfill runs:

1. `backfill_monitor.log` (15KB)
2. `coffee_backfill_fixed.log` (10KB)
3. `coffee_backfill_restart.log` (4KB)
4. `coffee_cluster_backfill.log` (2KB)
5. `sugar_backfill_fixed.log` (5KB)
6. `sugar_backfill_restart.log` (1KB)
7. `sugar_cluster_backfill.log` (2KB)
8. Plus any `autonomous_monitor.log`, `completion_monitor.log`, `monitor_output.log`

**RECOMMENDATION: DELETE ALL LOG FILES** - They're from failed attempts and debugging sessions

## Architecture Notes from Recent Session

### Key Architectural Decisions

1. **Point Forecasts Architecture** (CORRECTED Nov 20)
   - Point forecasts MUST use direct model inference output (mean_forecast, forecast_std)
   - DO NOT compute point forecasts by averaging Monte Carlo paths
   - This was a critical fix to `backfill_rolling_window_spark.py`

2. **Actuals Population** (Nov 20)
   - Actuals should be a separate workflow (not part of forecast backfill)
   - This is acknowledged tech debt in the current architecture
   - `write_actuals` flag in Spark script now defaults to False
   - Use `backfill_actuals.py` for separate actuals backfill

3. **Local vs Cluster Execution**
   - Local scripts use SQL connection (15-minute timeout limit)
   - Spark scripts run entirely on cluster (no timeout issues)
   - **RECOMMENDATION: Use Spark for all production backfills**

### Known Issues with Local Backfill

- Coffee backfill: Failed at 7/16 training periods due to SQL timeout
- Sugar backfill: Failed early due to SQL timeout
- Error: "Retry request would exceed Retry policy max retry duration of 900.0 seconds"
- Resume mode works correctly (both detected and skipped existing forecasts)

### Table Population Schema

All three forecast tables should be populated:

1. **`commodity.forecast.distributions`**
   - 2,000 Monte Carlo simulation paths per forecast
   - 14-day horizon (day_1 through day_14)
   - Uses FloatType for schema compatibility

2. **`commodity.forecast.point_forecasts`**
   - Point estimates with prediction intervals
   - Computed from direct model inference: mean_forecast[day_idx]
   - Prediction intervals use time-scaled volatility: `vol_scaled = forecast_std * sqrt(day_ahead)`
   - Uses FloatType for schema compatibility

3. **`commodity.forecast.forecast_actuals`**
   - Ground truth actuals
   - Hybrid convention: `model_version='actuals'` (primary) or `is_actuals=TRUE` (legacy)
   - Should be separate workflow (tech debt)

## Cleanup Command Suggestions

After reviewing the files, you can run these commands to clean up:

```bash
# 1. DELETE deprecated backfill scripts (review first!)
# rm backfill_spark.py backfill_optimized.py backfill_distributions_historical.py

# 2. DELETE all monitoring scripts
# rm monitor_backfills*.sh autonomous_monitor.sh completion_monitor*.sh monitor_and_evaluate.sh check_progress.sh complete_backfill.sh

# 3. DELETE all log files
# rm *.log

# 4. DELETE databricks_spark_backfill.py if duplicate
# rm databricks_spark_backfill.py

# 5. OPTIONAL: Delete local sequential backfill if confident in Spark version
# rm backfill_rolling_window.py
```

## Files to Review Before Deleting

1. `backfill_daily_forecasts.py` - Check if this serves a different purpose than rolling window
2. `databricks_spark_backfill.py` - Compare with `databricks_train_and_backfill.py` to see if duplicate
3. `complete_backfill.sh` - Check if it contains any useful orchestration logic

## Production Workflow Going Forward

```bash
# On Databricks (recommended):
from backfill_rolling_window_spark import backfill_all_models_spark

backfill_all_models_spark(
    commodities=['Coffee'],
    models=['naive', 'xgboost', 'sarimax_auto_weather'],
    train_frequency='semiannually',
    start_date='2018-01-01',
    end_date='2025-11-17',
    num_partitions=50  # 2-4x cluster cores
)

# For actuals (separate, run once):
python backfill_actuals.py --commodity Coffee --start-date 2018-01-01 --end-date 2025-11-17
```

## Summary

**KEEP:**
- `backfill_rolling_window_spark.py` (PRIMARY)
- `databricks_train_and_backfill.py` (wrapper)
- `backfill_actuals.py` (separate actuals workflow)

**OPTIONAL:**
- `backfill_rolling_window.py` (reference/testing)

**DELETE:**
- All monitoring scripts (9 files)
- All log files (10+ files)
- Deprecated backfill scripts (4-5 files)
- Old shell scripts (1 file)

**TOTAL SAVINGS:** ~20-30 files, cleaner codebase focused on production Spark workflow

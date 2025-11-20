# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Boundaries & Documentation Rules

### ❌ Don't Touch
**trading_agent/*** - The trading agent codebase is off-limits. You may only update documentation about the forecast_agent/trading_agent interface contract (e.g., explaining the `commodity.forecast.distributions` schema that the trading agent consumes).

### Documentation Preferences
- **Be conscientious about when documentation is needed** - don't create docs reflexively
- Sometimes scaffolding documentation is helpful during exploration, but clean it up and consolidate afterward
- When in doubt, ask the user if documentation would be valuable
- Always update existing docs as needed for code changes
- Prefer inline documentation (code comments and docstrings) for implementation details

### Credentials
- **NEVER hardcode credentials** (tokens, passwords, API keys)
- Always use environment variables via `os.environ` or load from `../infra/.env`
- Check for leaked credentials before committing: `grep -r "dapi" --include="*.py"`

### Execution Preference
- **Prefer Databricks over local** for training and backfilling
- **⚠️ CRITICAL: NEVER use Serverless SQL Warehouses for backfills - they cost $400+ for large jobs**
  - Always use All-Purpose Clusters with `existing_cluster_id` in Jobs API
  - SQL Warehouses are ONLY for quick read queries (< 5 minutes)
  - For backfills: Use Databricks Jobs with clusters or run notebooks directly on clusters
- Databricks provides scalable compute and avoids dependency/NumPy version conflicts
- Local execution is fine for development and testing small date ranges
- For production workloads: use Databricks notebooks or Databricks Jobs API with clusters

## Environment Setup

### Databricks Credentials
Load credentials from `../infra/.env`:
```bash
set -a && source ../infra/.env && set +a
```

Required environment variables:
- `DATABRICKS_HOST` - Databricks workspace URL
- `DATABRICKS_TOKEN` - Personal access token
- `DATABRICKS_HTTP_PATH` - SQL warehouse path (**⚠️ EXPENSIVE for backfills!** Use ONLY for quick queries <5min)
- `DATABRICKS_CLUSTER_HTTP_PATH` - All-purpose cluster path (use this for any long-running jobs)

### Long-Running Backfills: Use All-Purpose Cluster (Recommended)

**Problem**: SQL Warehouses have a hard 15-minute session timeout that causes remote backfills to stall frequently.

**Solution**: Create an all-purpose cluster with extended timeout settings:

1. **Create Cluster** (in Databricks UI):
   - Compute → Create Cluster
   - Name: `long-running-backfill-cluster`
   - Cluster Mode: Single Node (cheapest option)
   - Databricks Runtime: 14.3 LTS ML or later
   - **Auto Termination: 180 minutes** (3 hours - this is the key setting!)
   - Node Type: i3.xlarge or similar
   - Leave other settings as defaults

2. **Get HTTP Path**:
   - Click your cluster → Configuration → Advanced Options → JDBC/ODBC
   - Copy the HTTP Path (format: `/sql/protocolv1/o/ORG_ID/CLUSTER_ID`)

3. **Add to `../infra/.env`**:
   ```bash
   DATABRICKS_CLUSTER_HTTP_PATH=/sql/protocolv1/o/YOUR_ORG_ID/YOUR_CLUSTER_ID
   ```

4. **Update backfill scripts** to use `DATABRICKS_CLUSTER_HTTP_PATH` instead of `DATABRICKS_HTTP_PATH` for long-running jobs

**Benefits**:
- No 15-minute session timeout (cluster stays alive for hours)
- Faster compute than SQL Warehouses for intensive workloads
- Better for training models + backfilling in one session
- Auto-shuts down after 3 hours of inactivity (cost-effective)

**Cost Comparison**:
- ❌ SQL Warehouse (Serverless): $417 for 20-hour backfill
- ✅ All-Purpose Cluster (i3.xlarge): ~$10-20 for same workload with auto-termination
- **Always use clusters for any job longer than 5 minutes**

### Python Dependencies
```bash
pip install databricks-sql-connector scikit-learn xgboost statsmodels pmdarima prophet pandas numpy
```

## Core Architecture

### Train-Once/Inference-Many Pattern
This codebase implements a two-phase architecture for efficient forecasting at scale:

**Phase 1 - Training** (periodic, expensive):
```bash
python train_models.py --commodity Coffee --models naive xgboost sarimax_auto_weather \
  --train-frequency semiannually
```
- Trains models on fixed windows (e.g., every 6 months)
- Persists fitted models to `commodity.forecast.trained_models` table
- Models stored as JSON (<1MB) or S3 (≥1MB)
- ~16 trainings instead of ~2,875 (180x reduction)

**Phase 2 - Inference** (daily, fast):
```bash
python backfill_rolling_window.py --commodity Coffee --models naive xgboost \
  --train-frequency semiannually
```
- Loads pretrained models from database
- Generates 2,000 Monte Carlo paths per forecast
- Writes to `commodity.forecast.distributions` table
- Auto-resumes from last completed date

**Performance**: Semiannual training = ~180x speedup vs train-per-date

### Spark Parallelization (Optional)
For massive backfills (1000+ dates), use Databricks Spark:
```python
# In Databricks notebook
from backfill_rolling_window_spark import backfill_all_models_spark

backfill_all_models_spark(
    commodities=['Coffee'],
    models=['naive', 'xgboost'],
    train_frequency='semiannually',
    start_date='2018-01-01',
    end_date='2025-11-17',
    num_partitions=200  # 2-4x cluster cores
)
```
- 20-60 minutes vs 10-20 hours local
- See `README_SPARK_BACKFILL.md` for cluster sizing

## Data Contracts

### Input Table
**`commodity.silver.unified_data`**
- Unified commodity prices with weather, GDELT sentiment, VIX, exchange rates
- Daily continuous data (including weekends) with forward-fill
- See `../research_agent/UNIFIED_DATA_ARCHITECTURE.md` for details

### Output Tables (commodity.forecast schema)

**`point_forecasts`**
- 14-day forecasts with prediction intervals
- Columns: day_1 through day_14, actual_close

**`distributions`**
- 2,000 Monte Carlo paths for risk analysis
- Columns: day_1 through day_14, path_id (0-1999)
- Actuals stored with `model_version='actuals'` and `is_actuals=TRUE`

**`forecast_metadata`**
- Model performance metrics for backtesting
- MAE, RMSE, Dir Day0 (directional accuracy from day 0)

**`trained_models`**
- Persistent model storage for train-once pattern
- Partitioned by (year, month)
- Fields: fitted_model_json OR fitted_model_s3_path

### Actuals Storage Convention
Ground truth actuals use a **hybrid convention** for backwards compatibility:

**Primary** (use in new code):
```python
WHERE model_version = 'actuals'
```

**Legacy** (maintained for compatibility):
```python
WHERE is_actuals = TRUE AND path_id = 1
```

Backfill actuals:
```bash
python backfill_actuals.py --commodity Coffee --start-date 2018-01-01 --end-date 2025-11-17
```

## Model Implementation Pattern

All models implement three functions for train/predict separation:

```python
def my_model_train(df_pandas, target='close', **params) -> dict:
    """Train model and return fitted state (no forecasting)."""
    model = fit_model(df_pandas, target, **params)
    return {
        'fitted_model': model,
        'last_date': df_pandas.index[-1],
        'target': target,
        'model_type': 'my_model',
    }

def my_model_predict(fitted_model_dict, horizon=14, **params) -> pd.DataFrame:
    """Generate forecast using fitted model (no training)."""
    model = fitted_model_dict['fitted_model']
    return forecast_df  # columns: day_1 to day_14

def my_model_forecast_with_metadata(df_pandas, commodity, fitted_model=None, **params) -> dict:
    """Unified interface supporting both modes."""
    if fitted_model is None:
        fitted_model = my_model_train(df_pandas, **params)

    forecast_df = my_model_predict(fitted_model, **params)

    return {
        'forecast_df': forecast_df,
        'fitted_model': fitted_model,  # For persistence
    }
```

### Model Registry
Register models in `ground_truth/config/model_registry.py`:
```python
BASELINE_MODELS = {
    'my_new_model': {
        'name': 'My Model',
        'function': my_model.my_model_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_mean_c', 'vix'],
            'horizon': 14
        }
    }
}
```

## Common Commands

### Training Models
```bash
# Train semiannually (recommended for expensive models)
python train_models.py --commodity Coffee --models xgboost sarimax_auto_weather \
  --train-frequency semiannually

# Train monthly (for fast models)
python train_models.py --commodity Coffee --models naive arima_111 \
  --train-frequency monthly
```

### Backfilling Forecasts
```bash
# Full historical backfill (auto-resumes)
python backfill_rolling_window.py --commodity Coffee --models naive xgboost \
  --train-frequency semiannually

# Date-range backfill
python backfill_rolling_window.py --commodity Coffee --models naive \
  --start-date 2023-01-01 --end-date 2024-01-01 --train-frequency semiannually
```

### Checking Status
```bash
# Check backfill coverage
python check_backfill_coverage.py --commodity Coffee --models naive xgboost

# Verify specific model
python verify_backfill.py --commodity Coffee --model naive

# Quick evaluation
python quick_eval.py --commodity Coffee --model naive
```

### Evaluation
```bash
# Historical forecast evaluation
python evaluate_historical_forecasts.py --commodity Coffee --models naive xgboost

# Generate evaluation dashboard
python dashboard_forecast_evaluation.py
```

## Important Patterns

### Database Reconnection
Databricks has 15-minute session timeout. Scripts handle this automatically:
- Reconnect every 50 forecasts (before batch write)
- Resume mode skips existing forecasts
- Just rerun the same command to continue

### Batch Writing
Write forecasts in batches for 10-20x speedup:
- Default: 50 forecasts per batch
- Each batch: 2,000 paths + 14 point forecasts + actuals
- Progress logged every 50 forecasts

### Lookback Optimization
When using pretrained models, only load last 90 days of data:
```python
lookback_days = 90 if use_pretrained else None
training_df = load_training_data(connection, commodity, cutoff_date, lookback_days)
```
Result: 880x faster data loading

## Critical Findings

### ARIMA(auto) = Naive
`auto_arima` without exogenous variables selects order (0,1,0), which is mathematically equivalent to naive forecast. Always use exogenous features with SARIMAX models.

### Directional Accuracy from Day 0
Traditional day-to-day directional accuracy is misleading for trading. Use **Dir Day0** metric which measures whether day i > day 0 (trading signal quality).

### Column Name Conventions
Weather and feature columns in unified_data:
- `temp_mean_c` (NOT temp_c)
- `humidity_mean_pct` (NOT humidity_pct)
- `vix` (NOT vix_close)

## Production Model

**SARIMAX+Weather** (`sarimax_auto_weather`)
- MAE: $3.10
- Directional Accuracy from Day 0: 69.5% ± 27.7%
- Features: temp_mean_c, humidity_mean_pct, precipitation_mm
- Horizon: 14 days
- Training: Semiannual

Evaluated using 30-window walk-forward validation (420 days, non-overlapping).

## Project Structure

```
forecast_agent/
├── ground_truth/              # Production package
│   ├── config/
│   │   └── model_registry.py # 25 models (single source of truth)
│   ├── core/
│   │   ├── base_forecaster.py
│   │   ├── backtester.py
│   │   ├── data_loader.py
│   │   └── walk_forward_evaluator.py
│   ├── features/             # Feature engineering
│   ├── models/               # Model implementations
│   └── storage/
│       └── production_writer.py
│
├── utils/
│   ├── model_persistence.py  # save_model(), load_model()
│   └── monte_carlo_simulation.py
│
├── train_models.py           # Phase 1: Train and persist
├── backfill_rolling_window.py # Phase 2: Fast inference
├── backfill_rolling_window_spark.py # Spark parallel backfill
├── backfill_actuals.py       # Populate actuals table
├── evaluate_historical_forecasts.py
├── check_backfill_coverage.py
└── databricks_quickstart.py  # Databricks entry point
```

## Databricks Workflow

### Quick Start
1. Clone repo in Databricks Repos: `/Repos/<USERNAME>/ucberkeley-capstone`
2. Open `databricks_quickstart.py` notebook
3. Update path in cell 4: `/Workspace/Repos/<USERNAME>/ucberkeley-capstone/forecast_agent`
4. Run all cells

### Jobs API
```bash
# Upload notebook
curl -X POST "${DATABRICKS_HOST}/api/2.0/workspace/import" \
  -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
  -d '{"path": "/Repos/Project_Git/ucberkeley-capstone/forecast_agent/my_notebook", ...}'

# Run job
curl -X POST "${DATABRICKS_HOST}/api/2.0/jobs/run-now" \
  -d '{"job_id": 510355949628686}'
```

### NumPy Compatibility Issues
Databricks MLR may have NumPy binary incompatibility between training and inference. Solutions:

1. **Train+backfill in same job** (recommended for Databricks):
   - Ensures same NumPy version for both phases
   - See `databricks_train_and_backfill.py`

2. **Train locally, backfill on Databricks**:
   - Upload trained models to `commodity.forecast.trained_models`
   - Use Spark only for parallel inference

## Key Metrics

- **MAE** (Mean Absolute Error): Average prediction error in dollars
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **Dir Day0**: Directional accuracy from day 0 (primary trading metric)
- **Dir**: Day-to-day directional accuracy (less useful for trading)

## Testing & Validation

```bash
# Schema validation
python experiments/test_schema_updates.py

# Walk-forward evaluation
python experiments/run_walkforward_comprehensive.py
```

# Refined Approach Requirements

## Overview

This document captures all requirements for the refined forecast agent approach, designed to work seamlessly in Databricks and enable rapid experimentation.

## Core Requirements

### 1. Databricks Execution

**Requirement:** All code must run in Databricks notebooks using Databricks Repos (GitHub integration).

**Implementation:**
- Main notebooks: `.ipynb` files in sequence
- Python modules: `.py` files that can be imported from notebooks
- Use Spark SQL and Spark DataFrames natively
- No complex connection management

**References:**
- Databricks notebooks can import Python files from repos
- Use `spark.table()` for data access
- Use `spark.sql()` for queries

### 2. Training and Inference Separation

**Requirement:** Keep training and inference as separate phases.

**Implementation:**
- **Training notebooks**: Train models, save to `commodity.forecast.trained_models`
- **Inference notebooks**: Load saved models, generate forecasts, populate distributions table
- Models persist across sessions

### 3. Cross-Validation Approach

**Requirement:** Use expanding window cross-validation (primary), support rolling window (optional).

**Rationale:**
- Expanding window matches production (more data over time)
- More realistic for backtesting
- Can try rolling window for comparison

**Implementation:**
- `TimeSeriesDataLoader.create_walk_forward_folds()` - expanding window (default)
- `TimeSeriesDataLoader.create_temporal_folds()` - rolling window (optional)

### 4. Data Leakage Prevention

**Requirement:** Distributions table must ONLY contain data leakage-free forecasts.

**Definition:**
- Data leakage = forecast_start_date <= data_cutoff_date
- Only forecasts where `forecast_start_date > data_cutoff_date` should be in distributions table
- Use most recent trained model for each backtest date

**Implementation:**
- Track `data_cutoff_date` for each trained model
- Only generate forecasts for dates AFTER training cutoff
- Set `has_data_leakage=FALSE` for all distributions table entries
- Filter out any forecasts where `forecast_start_date <= data_cutoff_date`

### 5. Rapid Experimentation

**Requirement:** Enable quick experimentation with:
- Different featuresets
- Different feature engineering approaches
- Different model types
- GDELT sentiment data

**Implementation:**
- Modular feature engineering
- Feature set configurations
- Easy model swapping
- Experiment tracking

### 6. Unified Data Structure

**Requirement:** Handle `commodity.silver.unified_data` grain: `(date, commodity, region)`

**Structure:**
- Grain: One row per (date, commodity, region)
- ~65 regions per commodity
- Need to aggregate/pivot regions for time-series forecasting

**Aggregation Options:**
- Mean across regions (simple)
- Weighted by production
- Pivot regions as separate features
- Regional models (future)

**Implementation:**
- `TimeSeriesDataLoader.aggregate_regions()` - handles aggregation
- Support multiple aggregation methods
- Preserve flexibility for model-specific approaches

### 7. GDELT Sentiment Support

**Requirement:** Some featuresets will include GDELT sentiment data.

**Implementation:**
- Include GDELT columns in feature selection
- Handle missing GDELT data (not all dates have articles)
- Feature engineering for sentiment indicators

### 8. Model Persistence

**Requirement:** Save fitted models for reuse.

**Current Approach:**
- Table: `commodity.forecast.trained_models`
- Small models (<1MB): JSON in `fitted_model_json` column
- Large models (>=1MB): Pickle in S3, path in `fitted_model_s3_path` column

**Schema:**
```sql
CREATE TABLE commodity.forecast.trained_models (
    model_id STRING,
    commodity STRING,
    model_name STRING,
    model_version STRING,
    training_date DATE,  -- Cutoff date
    training_samples INT,
    training_start_date DATE,
    parameters STRING,  -- JSON
    fitted_model_json STRING,  -- Small models
    fitted_model_s3_path STRING,  -- Large models
    model_size_bytes BIGINT,
    created_at TIMESTAMP,
    created_by STRING,
    is_active BOOLEAN,
    year INT,
    month INT
) PARTITIONED BY (commodity, model_name, year, month)
```

**Implementation:**
- Use existing `utils/model_persistence.py` or create Spark-compatible version
- Support both JSON and S3 storage
- Track training metadata

### 9. Distributions Table Contract

**Requirement:** Populate `commodity.forecast.distributions` per trading agent expectations.

**Schema:**
```sql
CREATE TABLE commodity.forecast.distributions (
    path_id INT,  -- 0-2000 (0 = actuals)
    forecast_start_date DATE,  -- First day of forecast
    data_cutoff_date DATE,  -- Last training date
    generation_timestamp TIMESTAMP,
    model_version STRING,
    commodity STRING,
    day_1 to day_14 FLOAT,  -- 14 columns
    is_actuals BOOLEAN,
    has_data_leakage BOOLEAN  -- MUST be FALSE
) PARTITIONED BY (model_version, commodity)
```

**Requirements:**
- 2,000 Monte Carlo paths per forecast (path_id 1-2000)
- Optional actuals row (path_id=0, is_actuals=TRUE)
- All rows must have `has_data_leakage=FALSE`
- `forecast_start_date > data_cutoff_date` (no leakage)

**Trading Agent Expectations:**
- Query by commodity, model_version, date range
- Filter by `has_data_leakage=FALSE`
- Use path_id=0 for actuals comparison
- Use path_id=1-2000 for Monte Carlo analysis

### 10. Feature Engineering Flexibility

**Requirement:** Support different feature sets easily.

**Common Features:**
- Market: close, volume, high, low, open
- Weather: temp_mean_c, humidity_mean_pct, precipitation_mm
- VIX: vix
- FX: cop_usd, vnd_usd, etc.
- GDELT: sentiment scores, event counts (if available)

**Implementation:**
- Feature set configurations (dict/list)
- Easy to swap feature sets
- Handle missing features gracefully

### 11. Multiple Model Types

**Requirement:** Support various model types:
- Baseline: Naive, Random Walk
- Statistical: ARIMA, SARIMAX
- ML: XGBoost, Prophet
- Deep Learning: LSTM, TFT (future)

**Implementation:**
- `ModelPipeline` interface for all models
- Model registry for easy selection
- Consistent fit/predict interface

### 12. Notebook Sequence

**Requirement:** Sequential notebooks for workflow.

**Notebooks:**
1. **01_train_models.ipynb** - Train models, save to trained_models table
2. **02_generate_forecasts.ipynb** - Load models, generate forecasts, populate distributions
3. **03_evaluate_results.ipynb** (optional) - Evaluate and compare models

**Each notebook:**
- Can import from `.py` modules
- Uses Spark natively
- Can be run independently or in sequence

## Implied Requirements

### From Trading Agent Contracts

**Forecast Loading:**
- Trading agent expects specific schema in distributions table
- Needs to filter by `has_data_leakage=FALSE`
- Uses `model_version` string to identify models

**Model Selection:**
- Trading agent selects best model based on metrics
- Needs to query trained_models table for available models

### From Existing Code Patterns

**Backfill Strategy:**
- Train models periodically (semiannually/monthly)
- Generate forecasts for all dates between training windows
- Resume capability (skip existing forecasts)

**Performance Metrics:**
- MAE, RMSE, MAPE
- Directional accuracy (Day0 and day-to-day)
- Track per-window and aggregate

## Success Criteria

1. ✅ Distributions table populated with data leakage-free forecasts
2. ✅ Works entirely in Databricks notebooks
3. ✅ Training and inference separate workflows
4. ✅ Easy to experiment with features/models
5. ✅ Expanding window CV working
6. ✅ Models persist and can be reloaded
7. ✅ Supports region aggregation
8. ✅ Supports GDELT features
9. ✅ Trading agent can query distributions table successfully

## Non-Requirements (Avoid Over-Engineering)

- ❌ Complex scheduling infrastructure (use Databricks Jobs)
- ❌ Fancy experiment tracking (basic logging is fine)
- ❌ Real-time inference (batch is fine)
- ❌ Multi-tenant support (single team)
- ❌ Complex orchestration (simple notebooks are fine)

## Questions to Resolve

1. **Model versioning**: How to version models for experiments? (e.g., `v1.0`, `experiment_sentiment_v1`)
2. **Feature set naming**: How to name different feature combinations?
3. **Experiment tracking**: Minimal tracking needed? (just model_version?)
4. **Regional models**: Future requirement or skip for now?


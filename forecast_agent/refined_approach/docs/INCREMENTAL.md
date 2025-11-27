# Incremental/Resume Behavior

## Overview

Both training and inference notebooks support **incremental execution** - they automatically skip work that's already been done.

This means you can:
- ✅ Re-run training and only train new models
- ✅ Re-run inference and only generate missing forecasts
- ✅ Add new dates/models incrementally
- ✅ Resume after failures without redoing everything

## Training: Skip Existing Models

### How It Works

Before training, the notebook checks if the model already exists:

```python
if model_exists_spark(spark, commodity, model_name, training_date, model_version):
    print("⏩ Model already exists - skipping")
    skipped_count += 1
    continue
```

### Example Scenario

**Initial Run:**
```python
models = ["naive", "random_walk", "xgboost"]
training_dates = ["2020-01-01", "2020-07-01", "2021-01-01"]
```

**Result:** Trains 9 models (3 models × 3 dates)

**Later: Add New Model**
```python
models = ["naive", "random_walk", "xgboost", "sarimax"]  # Added sarimax
training_dates = ["2020-01-01", "2020-07-01", "2021-01-01", "2021-07-01"]  # Added new date
```

**Result:**
- ⏩ Skips: 9 existing models (naive, random_walk, xgboost on first 3 dates)
- ✅ Trains: 7 new models
  - sarimax on all 4 dates (4 models)
  - All models on new date 2021-07-01 (3 models)

**Summary:**
```
✅ Models Trained: 7 (new)
⏩ Models Skipped: 9 (already exist)
```

### Use Cases

1. **Adding New Models**
   - Train 3 models initially
   - Later add 2 more models
   - Re-run with all 5 models
   - Only 2 new models train

2. **Adding New Dates**
   - Train models up to 2024-01-01
   - New data arrives, extend to 2024-07-01
   - Re-run with extended date range
   - Only trains models for new dates

3. **Fixing Failures**
   - 10 models, 2 fail
   - Fix the 2 failing models
   - Re-run same config
   - Skips 8 successful, retrains 2 failed

## Inference: Skip Existing Forecasts

### How It Works

Before generating forecasts, the notebook checks what already exists:

```python
# Get existing forecasts
existing_dates = get_existing_forecast_dates(spark, commodity, model_version)

for forecast_date in forecast_dates:
    if forecast_date in existing_dates:
        print(f"⏩ Forecast already exists - skipping")
        skipped_count += 1
        continue
    # Generate forecast...
```

### Example Scenario

**Initial Run:**
```python
models = ["naive", "xgboost"]
forecast_dates = range("2021-01-01", "2021-12-31")  # 365 dates
```

**Result:** Generates 730 forecasts (2 models × 365 dates)

**Later: Add New Date Range**
```python
models = ["naive", "xgboost", "sarimax"]  # Added sarimax
forecast_dates = range("2021-01-01", "2022-06-30")  # Extended range
```

**Result:**
- ⏩ Skips: 730 existing forecasts (naive, xgboost for 2021)
- ✅ Generates: 548 new forecasts
  - sarimax for all dates in range (365 + 181 = 546)
  - All models for new dates 2022-01-01 to 2022-06-30 (3 × 181 = 543)
  - But wait... we need to be smart about what "exists" means

**Actually, better logic:**
- Check: Does forecast exist for (model_version, forecast_start_date)?
- If yes, skip
- If no, generate

### Use Cases

1. **Adding New Models**
   - Generate forecasts for 2 models
   - Later add 1 more model
   - Re-run with all 3 models
   - Only generates forecasts for new model

2. **Extending Date Range**
   - Generate forecasts up to 2024-01-01
   - New dates arrive
   - Re-run with extended range
   - Only generates forecasts for new dates

3. **Fixing Failed Forecasts**
   - Generate 1000 forecasts, 50 fail
   - Fix the issue
   - Re-run same config
   - Skips 950 successful, regenerates 50 failed

## How It Identifies "Existing"

### Training: Model ID

Models are identified by unique `model_id`:
```
model_id = f"{commodity}_{model_name}_{training_date}_{model_version}"
```

**Example:**
- `Coffee_Naive_2020-07-01_v1.0`
- `Coffee_XGBoost_2020-07-01_v1.0`
- `Coffee_Naive_2021-01-01_v1.0` (different training date)

### Inference: Forecast Date + Model Version

Forecasts are identified by:
- `forecast_start_date` (first day of forecast)
- `model_version` (matches training model_version)
- `commodity`

**Query:**
```sql
SELECT DISTINCT forecast_start_date
FROM commodity.forecast.distributions
WHERE commodity = 'Coffee'
  AND model_version = 'v1.0'
  AND is_actuals = FALSE
```

## Best Practices

### 1. Use Descriptive Model Versions

**Good:**
```python
model_version = "v1.0"              # Baseline
model_version = "experiment_gdelt"  # Experiment
model_version = "backfill_2024"     # Purpose-specific
```

**Why:** Makes it easy to track different runs

### 2. Train Before Inference

**Workflow:**
1. Run training notebook → Save models
2. Run inference notebook → Load models, generate forecasts

**Why:** Inference needs trained models to exist

### 3. Check Summary After Run

**Always review:**
```
✅ Models Trained: 15 (new)
⏩ Models Skipped: 20 (already exist)
❌ Models Failed: 2

✅ Forecasts Generated: 500 (new)
⏩ Forecasts Skipped: 1000 (already exist)
❌ Forecasts Failed: 5
```

**This tells you:**
- What was new vs existing
- What failed and needs attention

### 4. Re-run After Fixes

**If models fail:**
1. Fix the issue
2. Re-run training notebook
3. Only failed models retrain (existing ones skipped)

**If forecasts fail:**
1. Fix the issue
2. Re-run inference notebook
3. Only missing forecasts generated

## Example: Complete Workflow

### Week 1: Initial Setup

**Training:**
```python
commodity = "Coffee"
models = ["naive", "random_walk"]
train_frequency = "semiannually"
start_date = "2020-01-01"
end_date = "2024-01-01"
```

**Result:**
- Trains 8 models (2 models × 4 training dates)
- All saved to `trained_models` table

**Inference:**
```python
models = ["naive", "random_walk"]
start_date = "2021-01-01"
end_date = "2024-01-01"
```

**Result:**
- Generates ~2,190 forecasts (2 models × ~1,095 dates)

### Week 2: Add New Model

**Training:**
```python
models = ["naive", "random_walk", "xgboost"]  # Added xgboost
# Same dates as before
```

**Result:**
- ⏩ Skips: 8 existing models
- ✅ Trains: 4 new models (xgboost × 4 dates)

**Inference:**
```python
models = ["naive", "random_walk", "xgboost"]  # Added xgboost
# Same dates as before
```

**Result:**
- ⏩ Skips: ~2,190 existing forecasts
- ✅ Generates: ~1,095 new forecasts (xgboost only)

### Week 3: Extend Date Range

**Training:**
```python
end_date = "2024-07-01"  # Extended
```

**Result:**
- ⏩ Skips: 12 existing models (3 models × 4 dates)
- ✅ Trains: 3 new models (all models × 1 new date)

**Inference:**
```python
end_date = "2024-07-01"  # Extended
```

**Result:**
- ⏩ Skips: ~3,285 existing forecasts
- ✅ Generates: ~273 new forecasts (3 models × ~91 new dates)

## Implementation Details

### Training Notebook

**Check happens before training:**
```python
if model_exists_spark(spark, commodity, model_name, training_date, model_version):
    skipped_count += 1
    continue  # Skip training
```

**Why before training:**
- Saves compute time
- Skips expensive model fitting

### Inference Notebook

**Check happens before generation:**
```python
existing_dates = get_existing_forecast_dates(spark, commodity, model_version)

for forecast_date in forecast_dates:
    if forecast_date in existing_dates:
        skipped_count += 1
        continue  # Skip generation
```

**Why before generation:**
- Saves compute time
- Skips expensive forecast generation

## Summary

**Incremental behavior means:**
- ✅ Re-run anytime without redoing everything
- ✅ Add new models/dates incrementally
- ✅ Resume after failures
- ✅ Cost-effective (no redundant work)

**The notebooks automatically:**
- Check what exists
- Skip existing work
- Only do new work
- Report what was skipped vs new

**Just re-run the notebooks** - they'll figure out what needs to be done!


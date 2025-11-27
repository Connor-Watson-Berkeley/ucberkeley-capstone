# Workflow Flow

## Overview

Simple, fail-open workflow: Train as many models as possible, then generate forecasts for whatever succeeds.

## Training Flow (`01_train_models.py`)

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Load Data                                       │
│ - Load unified_data for commodity                       │
│ - Aggregate regions (mean/first)                        │
│ - Filter by cutoff_date                                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: Generate Training Dates                         │
│ - Based on frequency (semiannually, monthly, etc.)     │
│ - Example: 2020-01-01, 2020-07-01, 2021-01-01, ...     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: For Each Training Date                          │
│                                                          │
│   ┌──────────────────────────────────────────────┐     │
│   │ Filter data up to cutoff                     │     │
│   │ Check minimum training days                  │     │
│   │ Skip if insufficient data                    │     │
│   └──────────────────┬───────────────────────────┘     │
│                      │                                   │
│                      ▼                                   │
│   ┌──────────────────────────────────────────────┐     │
│   │ For Each Model (FAIL-OPEN)                   │     │
│   │                                              │     │
│   │   try:                                       │     │
│   │     - Create model                           │     │
│   │     - Fit model                              │     │
│   │     - Save to trained_models table           │     │
│   │     ✅ Count as trained                      │     │
│   │                                              │     │
│   │   except Exception:                          │     │
│   │     - Log error                              │     │
│   │     - Continue to next model                 │     │
│   │     ❌ Count as failed                       │     │
│   └──────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 4: Summary                                         │
│ - Report: trained, skipped, failed counts              │
│ - All successful models saved to database              │
└─────────────────────────────────────────────────────────┘
```

### Fail-Open Behavior

**Key Principle:** One model failure does NOT stop the process.

- ✅ Each model trained independently
- ✅ Exceptions caught and logged
- ✅ Continues to next model
- ✅ Continues to next training date
- ✅ Final summary shows what succeeded

**Example Output:**
```
Training Cutoff: 2020-07-01
  🔧 Training naive...
    ✅ Saved: Coffee_Naive_2020-07-01_v1.0
  🔧 Training xgboost...
    ❌ Failed: Insufficient data for XGBoost
  🔧 Training sarimax...
    ✅ Saved: Coffee_SARIMAX+Weather_2020-07-01_v1.0

[Continues to next training date...]

TRAINING COMPLETE
✅ Models Trained: 48
⏩ Models Skipped: 12
❌ Models Failed: 4
```

## Inference Flow (`02_generate_forecasts.py`)

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Load Trained Models                             │
│ - Query trained_models table                            │
│ - Filter by commodity, model_version                    │
│ - Get most recent model for each date                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: Generate Forecast Dates                         │
│ - Date range: start_date to end_date                    │
│ - One forecast per day                                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: For Each Forecast Date                          │
│                                                          │
│   ┌──────────────────────────────────────────────┐     │
│   │ Find most recent trained model               │     │
│   │ (training_date <= forecast_date)             │     │
│   └──────────────────┬───────────────────────────┘     │
│                      │                                   │
│                      ▼                                   │
│   ┌──────────────────────────────────────────────┐     │
│   │ For Each Model (FAIL-OPEN)                   │     │
│   │                                              │     │
│   │   try:                                       │     │
│   │     - Load model from table                  │     │
│   │     - Load data up to forecast_date          │     │
│   │     - Generate forecast (14 days)            │     │
│   │     - Generate 2000 Monte Carlo paths        │     │
│   │     ✅ Add to batch                          │     │
│   │                                              │     │
│   │   except Exception:                          │     │
│   │     - Log error                              │     │
│   │     - Continue to next model                 │     │
│   │     ❌ Count as failed                       │     │
│   └──────────────────┬───────────────────────────┘     │
│                      │                                   │
│                      ▼                                   │
│   ┌──────────────────────────────────────────────┐     │
│   │ Write Batch (if any forecasts)               │     │
│   │ - Filter data leakage                        │     │
│   │ - Only forecast_start_date > data_cutoff     │     │
│   │ - Write to distributions table               │     │
│   └──────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 4: Summary                                         │
│ - Report: forecasts generated, failed                   │
│ - Distributions table populated                         │
└─────────────────────────────────────────────────────────┘
```

## Simplicity Features

### 1. **No Complex Orchestration**
- Simple loops (for date, for model)
- No workflow engines
- No dependency graphs
- Easy to understand and modify

### 2. **Fail-Open Everywhere**
- Training: One model fails? Continue to next.
- Inference: One forecast fails? Continue to next.
- Data loading fails? Skip that date, continue.

### 3. **Clear Progress Tracking**
- Print statements show what's happening
- Counters track successes/failures
- Summary at end shows results

### 4. **Easy to Modify**
- Change model list? Edit one line.
- Change frequency? Edit one parameter.
- Add new model? Implement `ModelPipeline` interface.

## Example: Training 10 Models, 3 Fail

**Input:**
```python
models = ["naive", "random_walk", "arima", "sarimax", "xgboost",
          "prophet", "tft", "lstm", "ensemble", "baseline"]
training_dates = [2020-01-01, 2020-07-01, 2021-01-01]
```

**What Happens:**
1. Train naive on 2020-01-01 ✅
2. Train random_walk on 2020-01-01 ✅
3. Train arima on 2020-01-01 ✅
4. Train sarimax on 2020-01-01 ❌ (fails: no convergence)
   → Logs error, continues
5. Train xgboost on 2020-01-01 ✅
6. Train prophet on 2020-01-01 ❌ (fails: missing holiday data)
   → Logs error, continues
7. Train tft on 2020-01-01 ✅
8. Train lstm on 2020-01-01 ❌ (fails: GPU not available)
   → Logs error, continues
9. Train ensemble on 2020-01-01 ✅
10. Train baseline on 2020-01-01 ✅
11. Move to next training date (2020-07-01)
12. Repeat for all dates...

**Result:**
- 7 models × 3 dates = 21 successful trainings
- 3 models × 3 dates = 9 failures (logged)
- All 21 successful models saved and ready for inference

## Error Handling Details

### Training Failures

**What can fail:**
- Model creation (invalid parameters)
- Model fitting (convergence issues, data problems)
- Model saving (database errors, serialization issues)

**What happens:**
```python
try:
    model = create_model_from_registry(model_key)  # Can fail
    model.fit(training_df)                         # Can fail
    save_model_spark(...)                          # Can fail
    trained_count += 1
except Exception as e:
    print(f"❌ Failed: {str(e)[:100]}")
    failed_count += 1
    # Continues to next model
```

### Inference Failures

**What can fail:**
- Model loading (not found, deserialization error)
- Data loading (missing data, query error)
- Forecast generation (model error, insufficient data)
- Path generation (monte carlo simulation error)

**What happens:**
```python
try:
    model = load_model_spark(...)      # Can fail
    data = load_data(...)               # Can fail
    forecast = model.predict(...)       # Can fail
    paths = generate_paths(...)         # Can fail
    # Add to batch
except Exception as e:
    print(f"❌ Forecast failed: {str(e)[:100]}")
    # Continues to next model/date
```

## Benefits of This Approach

1. **Resilient:** Partial failures don't stop everything
2. **Transparent:** You see exactly what succeeded/failed
3. **Simple:** No complex error recovery logic
4. **Fast:** Don't waste time retrying failures
5. **Flexible:** Easy to add/remove models

## Running the Workflow

### Training
```python
# Set parameters
commodity = "Coffee"
models = ["naive", "random_walk", "xgboost"]  # Add/remove models easily
train_frequency = "semiannually"

# Run notebook - it will:
# - Train all models for all dates
# - Skip failures gracefully
# - Save successful models
# - Report summary
```

### Inference
```python
# Set parameters
commodity = "Coffee"
models = ["naive", "random_walk"]  # Only use models that succeeded
model_version = "v1.0"
start_date = "2021-01-01"
end_date = "2024-01-01"

# Run notebook - it will:
# - Load trained models
# - Generate forecasts for all dates
# - Skip failures gracefully
# - Write to distributions table
# - Report summary
```

## Incremental/Resume Behavior

### Training: Skip Existing Models

Before training each model, checks if it already exists:

```python
if model_exists_spark(spark, commodity, model_name, training_date, model_version):
    print("⏩ Model already exists - skipping")
    skipped_count += 1
    continue
```

**Benefits:**
- Re-run training → Only trains new models
- Add new models → Only trains those
- Add new dates → Only trains for those dates
- Fix failures → Only retrains failed models

### Inference: Skip Existing Forecasts

Before generating each forecast, checks if it already exists:

```python
existing_dates = get_existing_forecast_dates(spark, commodity, model_version)

for forecast_date in forecast_dates:
    if forecast_date in existing_dates:
        print("⏩ Forecast already exists - skipping")
        skipped_count += 1
        continue
```

**Benefits:**
- Re-run inference → Only generates missing forecasts
- Add new models → Only generates for those
- Extend date range → Only generates new dates
- Fix failures → Only regenerates failed forecasts

## Summary

**Flow is simple:**
1. Train → Save successful models (skips existing)
2. Inference → Generate forecasts from saved models (skips existing)

**Fail-open everywhere:**
- One failure doesn't stop the rest
- Log errors and continue
- Get summary of what worked

**Incremental execution:**
- Automatically skips existing work
- Re-run anytime without redoing everything
- Add new models/dates incrementally

**Easy to use:**
- Change parameters at top
- Run notebook
- Get results (new + skipped + failed)

No complex orchestration, no fragile dependencies - just simple loops that keep going!


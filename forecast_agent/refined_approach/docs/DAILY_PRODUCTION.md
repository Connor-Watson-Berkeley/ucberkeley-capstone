# Daily Production Workflow

## Overview

The refined approach supports **both backfilling and daily production** through the same notebooks and infrastructure.

## Two Use Cases

### 1. Backfilling (Historical Dates)

**Notebooks:**
- `01_train_models.py` - Train models for date range
- `02_generate_forecasts.py` - Generate forecasts for date range

**Usage:**
- Run once to populate historical data
- Date range: 2018-01-01 to 2024-01-01
- Generates all models for all dates in range

### 2. Daily Production (Today's Forecast)

**Notebook:**
- `00_daily_production.py` - Daily workflow

**Usage:**
- Run daily via Databricks Jobs
- Automatically checks retraining cadence
- Generates forecast for today only

## Daily Production Workflow

### What It Does

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Check Training Status                           │
│                                                          │
│   For each model:                                       │
│     ✅ Check if retraining needed (based on cadence)    │
│     ✅ Check if already trained for today               │
│     ✅ Train if needed                                  │
│                                                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: Generate Today's Forecast                       │
│                                                          │
│   For each model:                                       │
│     ✅ Find most recent trained model                   │
│     ✅ Load model                                       │
│     ✅ Generate forecast for today                      │
│     ✅ Write to distributions table                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### How Retraining Cadence Works

**Example: Semiannual Training**

1. **Day 1 (2024-01-01):**
   - No previous training → Train models
   - Generate forecast for 2024-01-01

2. **Day 2-180 (2024-01-02 to 2024-06-29):**
   - Check: Last training was 2024-01-01
   - Next training date: 2024-07-01 (6 months later)
   - Today < next training date → Skip training
   - Generate forecast using model from 2024-01-01

3. **Day 181 (2024-07-01):**
   - Check: Last training was 2024-01-01
   - Next training date: 2024-07-01
   - Today >= next training date → Retrain models
   - Generate forecast using new model from 2024-07-01

4. **Day 182-365 (2024-07-02 to 2024-12-31):**
   - Use model from 2024-07-01
   - Next retraining: 2025-01-01

### Incremental Behavior

**Training:**
- Checks if model already exists for today
- Skips if already trained
- Only trains if cadence says it's time

**Inference:**
- Checks if forecast already exists for today
- Skips if already generated
- Uses most recent trained model automatically

## Setting Up Daily Job

### Databricks Job Configuration

```
Job Name: daily-forecast-production
Schedule: Daily at 6:00 AM UTC
Notebook: refined_approach/notebooks/00_daily_production.py
Parameters:
  - commodity: Coffee
  - models: naive,random_walk,xgboost
  - train_frequency: semiannually
  - model_version: v1.0
  - forecast_date: (leave empty for today)
```

### What Happens Each Day

**Scenario: Regular Day (No Retraining)**

1. **6:00 AM:** Job starts
2. **Training Check:**
   - Checks last training date
   - Determines next training date based on cadence
   - Today < next training date → Skip training
3. **Forecast Generation:**
   - Finds most recent trained model
   - Generates forecast for today
   - Writes to distributions table
4. **6:05 AM:** Job completes ✅

**Scenario: Retraining Day**

1. **6:00 AM:** Job starts
2. **Training Check:**
   - Checks last training date
   - Today >= next training date → Train models
   - Saves new models to trained_models table
3. **Forecast Generation:**
   - Uses newly trained models
   - Generates forecast for today
   - Writes to distributions table
4. **6:30 AM:** Job completes ✅ (longer due to training)

## Key Features

### 1. Automatic Retraining Detection

```python
needs_training = should_retrain_today(
    spark=spark,
    commodity=commodity,
    model_name=model_name,
    train_frequency='semiannually',
    model_version='v1.0',
    today=date.today()
)

if needs_training:
    # Train model
else:
    # Skip training, use existing
```

**Logic:**
- Finds most recent training date
- Calculates next training date based on frequency
- Returns True if today >= next training date

### 2. Most Recent Model Selection

```python
model_info = get_most_recent_trained_model(
    spark=spark,
    commodity=commodity,
    model_name=model_name,
    forecast_date=today,
    model_version='v1.0'
)
```

**Logic:**
- Queries trained_models table
- Finds models where `training_date <= forecast_date`
- Returns most recent one

### 3. Fail-Open Execution

- One model fails to train? Continue with others
- One model fails to forecast? Continue with others
- Get summary of what worked vs failed

## Example Daily Run

### Input
```python
commodity = "Coffee"
models = ["naive", "xgboost", "sarimax"]
train_frequency = "semiannually"
forecast_date = date.today()  # 2024-07-15
```

### Execution

**Training Phase:**
```
Checking naive...
  ⏩ No retraining needed (last trained 2024-07-01, next 2025-01-01)

Checking xgboost...
  ⏩ No retraining needed (last trained 2024-07-01, next 2025-01-01)

Checking sarimax...
  ⏩ No retraining needed (last trained 2024-07-01, next 2025-01-01)
```

**Forecast Phase:**
```
Generating forecast for 2024-07-15...

naive...
  📦 Using model trained on 2024-07-01
  ✅ Generated forecast

xgboost...
  📦 Using model trained on 2024-07-01
  ✅ Generated forecast

sarimax...
  📦 Using model trained on 2024-07-01
  ✅ Generated forecast

💾 Writing 3 forecasts to distributions table...
✅ Wrote forecasts to commodity.forecast.distributions
```

### Output

```
DAILY PRODUCTION COMPLETE
Date: 2024-07-15
Commodity: Coffee

Training:
   ✅ Trained: 0
   ⏩ Skipped: 3

Forecasts:
   ✅ Generated: 3
   ❌ Failed: 0
```

## Benefits

### 1. Single Workflow for Both Cases

- **Backfilling:** Run `01_train_models.py` then `02_generate_forecasts.py`
- **Daily Production:** Run `00_daily_production.py` daily

Same infrastructure, different use cases.

### 2. Automatic Cadence Management

- No manual tracking of when to retrain
- System automatically determines if retraining needed
- Based on frequency (semiannually, monthly, etc.)

### 3. Uses Most Recent Models

- Automatically finds best model for each date
- No manual model selection
- Handles model updates seamlessly

### 4. Incremental and Fail-Open

- Skips existing forecasts
- Continues on failures
- Gets maximum coverage

## Comparison: Backfill vs Daily

| Aspect | Backfilling | Daily Production |
|--------|-------------|------------------|
| **Notebook** | `01_train_models.py`<br>`02_generate_forecasts.py` | `00_daily_production.py` |
| **Frequency** | Run once | Run daily (scheduled) |
| **Date Range** | Historical range | Today only |
| **Training** | All dates in range | Only if cadence says so |
| **Forecasts** | All dates in range | Today only |
| **Use Case** | Historical backfill | Production operations |

## Setup Checklist

### For Daily Production

- [ ] Create Databricks Job
- [ ] Set schedule (daily at desired time)
- [ ] Configure parameters (commodity, models, frequency)
- [ ] Test run manually
- [ ] Monitor first few runs
- [ ] Set up alerts for failures

### For Backfilling

- [ ] Run `01_train_models.py` with date range
- [ ] Run `02_generate_forecasts.py` with date range
- [ ] Verify distributions table populated
- [ ] Check data leakage flags

## Questions?

- How does it know when to retrain? → Checks last training date + cadence
- What if a model fails? → Logs error, continues with others
- What if forecast already exists? → Skips it
- Can I run both workflows? → Yes, they're independent


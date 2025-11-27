# Backfilling vs Daily Production

## Answer: Yes, Works for Both! ✅

The refined approach supports **both backfilling and daily production** through different notebooks.

## Two Workflows

### 1. Backfilling Workflow (Historical Data)

**Use Case:** Populate historical forecasts (2018-2024)

**Notebooks:**
- `01_train_models.py` - Train models for date range
- `02_generate_forecasts.py` - Generate forecasts for date range

**Run:** Once, manually or scheduled

### 2. Daily Production Workflow (Today's Forecast)

**Use Case:** Daily production operations

**Notebook:**
- `00_daily_production.py` - Combined training check + inference

**Run:** Daily via Databricks Jobs (automated)

## Daily Production: How It Works

### The Daily Job Does Two Things

**Phase 1: Training Check**
```
For each model:
  ✅ Check last training date
  ✅ Calculate next training date (based on cadence)
  ✅ If today >= next training date → Train new model
  ✅ If today < next training date → Skip (use existing)
```

**Phase 2: Forecast Generation**
```
For each model:
  ✅ Find most recent trained model (training_date <= today)
  ✅ Load model from database
  ✅ Generate forecast for today
  ✅ Write to distributions table
```

### Example: Semiannual Training

**Scenario:** Models retrain every 6 months (semiannually)

**Day 1 (2024-01-01):**
- Check: No previous training → Train all models
- Forecast: Generate for 2024-01-01 using new models

**Day 2-180 (Regular days):**
- Check: Last training was 2024-01-01, next is 2024-07-01
- Today < next training date → Skip training ✅
- Forecast: Generate using models from 2024-01-01

**Day 181 (2024-07-01 - Retraining Day):**
- Check: Last training was 2024-01-01, next is 2024-07-01
- Today >= next training date → Retrain all models ✅
- Forecast: Generate using new models from 2024-07-01

**Day 182-365:**
- Check: Last training was 2024-07-01, next is 2025-01-01
- Today < next training date → Skip training ✅
- Forecast: Generate using models from 2024-07-01

## Setting Up Daily Job

### Databricks Job Configuration

```
Job Name: daily-forecast-production
Type: Notebook
Notebook Path: refined_approach/notebooks/00_daily_production.py

Schedule:
  Type: Cron Schedule
  Cron: 0 6 * * *  (Daily at 6:00 AM UTC)
  
Parameters:
  - commodity: Coffee
  - models: naive,random_walk,xgboost
  - train_frequency: semiannually
  - model_version: v1.0
  - forecast_date: (leave empty = today)
```

### What Happens Each Day

**6:00 AM:** Job runs automatically

**Training Check:**
- Queries `trained_models` table for last training date
- Calculates if retraining needed based on cadence
- Trains if needed, skips if not

**Forecast Generation:**
- Finds most recent trained model for each model
- Loads from `trained_models` table
- Generates forecast for today
- Writes to `distributions` table (data leakage-free)

**6:05-30 AM:** Job completes (longer on retraining days)

## Key Functions

### Check Retraining Cadence

```python
from daily_production import should_retrain_today

needs_training = should_retrain_today(
    spark=spark,
    commodity='Coffee',
    model_name='Naive',
    train_frequency='semiannually',
    model_version='v1.0',
    today=date.today()
)

# Returns True if today >= next_training_date
```

### Find Most Recent Model

```python
from daily_production import get_most_recent_trained_model

model_info = get_most_recent_trained_model(
    spark=spark,
    commodity='Coffee',
    model_name='Naive',
    forecast_date=date.today(),
    model_version='v1.0'
)

# Returns model with training_date <= forecast_date (most recent)
```

## Example Daily Run

### Input
```python
commodity = "Coffee"
models = ["naive", "xgboost", "sarimax"]
train_frequency = "semiannually"
forecast_date = date.today()  # 2024-07-15
```

### Execution Log

```
PHASE 1: Training Check
========================
Checking naive...
  ⏩ No retraining needed (last: 2024-07-01, next: 2025-01-01)

Checking xgboost...
  ⏩ No retraining needed (last: 2024-07-01, next: 2025-01-01)

Checking sarimax...
  ⏩ No retraining needed (last: 2024-07-01, next: 2025-01-01)

Training Summary:
   ✅ Trained: 0
   ⏩ Skipped: 3

PHASE 2: Forecast Generation
=============================
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

Forecast Summary:
   ✅ Generated: 3
   ❌ Failed: 0
```

### On Retraining Day (e.g., 2025-01-01)

```
PHASE 1: Training Check
========================
Checking naive...
  🔧 Retraining needed (last: 2024-07-01, next: 2025-01-01)
  ✅ Trained and saved: Coffee_Naive_2025-01-01_v1.0

Checking xgboost...
  🔧 Retraining needed
  ✅ Trained and saved: Coffee_XGBoost_2025-01-01_v1.0

Checking sarimax...
  🔧 Retraining needed
  ✅ Trained and saved: Coffee_SARIMAX+Weather_2025-01-01_v1.0

Training Summary:
   ✅ Trained: 3
   ⏩ Skipped: 0

PHASE 2: Forecast Generation
=============================
[Uses newly trained models from 2025-01-01]
```

## Benefits

### 1. Single Infrastructure

- Same tables (`trained_models`, `distributions`)
- Same models
- Same data loading
- Works for both backfilling and daily

### 2. Automatic Cadence Management

- No manual tracking of retraining dates
- System calculates next training date automatically
- Based on frequency parameter

### 3. Uses Latest Data

- Forecast uses data up to today
- Model uses most recent training (within cadence)
- Always data leakage-free

### 4. Incremental and Resilient

- Skips existing forecasts (won't duplicate)
- Continues on failures (fail-open)
- Can rerun safely anytime

## Comparison Table

| Aspect | Backfilling | Daily Production |
|--------|-------------|------------------|
| **Notebook** | `01_train_models.py`<br>`02_generate_forecasts.py` | `00_daily_production.py` |
| **Run Frequency** | Once (manual) | Daily (automated) |
| **Date Range** | Historical range | Today only |
| **Training** | All dates in range | Only if cadence says so |
| **Forecasts** | All dates in range | Today only |
| **Use Case** | Historical backfill | Production operations |
| **Automation** | Manual or scheduled once | Scheduled daily |

## Workflow Summary

### For Daily Production

```
Daily Job (6 AM):
  1. Check retraining cadence
     - If time → Train models → Save to trained_models
     - If not → Skip training
  2. Generate today's forecast
     - Find most recent trained model
     - Load model
     - Generate forecast for today
     - Write to distributions table
  3. Done! ✅
```

### For Backfilling

```
Run Once:
  1. Train models for date range
     - All training dates in range
     - Save to trained_models
  2. Generate forecasts for date range
     - All dates in range
     - Use most recent trained model per date
     - Write to distributions table
  3. Done! ✅
```

## To Answer Your Question

**Q: Will it work for both backfilling and daily inference?**

**A: Yes! ✅**

- **Backfilling:** Use `01_train_models.py` + `02_generate_forecasts.py`
- **Daily Production:** Use `00_daily_production.py` (automated daily)

**Q: Will it check retraining cadence and train if needed?**

**A: Yes! ✅**

- `should_retrain_today()` checks last training date
- Calculates next training date based on frequency
- Trains if today >= next training date

**Q: Will it inference all models based on latest day's data?**

**A: Yes! ✅**

- Finds most recent trained model for each model
- Uses data up to today
- Generates forecast for today only
- Writes to distributions table

**Q: Can I set up a daily job?**

**A: Yes! ✅**

- Create Databricks Job pointing to `00_daily_production.py`
- Schedule daily (e.g., 6 AM UTC)
- Set parameters (commodity, models, frequency)
- Job handles everything automatically!

## Simple Answer

**Yes, it works for both!**

- **Backfilling:** Run training + inference notebooks for date ranges
- **Daily Production:** Run `00_daily_production.py` daily via job

The daily notebook automatically:
1. ✅ Checks if retraining needed (based on cadence)
2. ✅ Trains if needed
3. ✅ Generates forecast for today using latest data
4. ✅ Writes to distributions table

Just set up the job and it runs daily! 🚀


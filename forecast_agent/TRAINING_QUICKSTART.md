# Training Quickstart: 5 Simple Steps

**The simplest path to training forecast models.**

## Our Goal

Train commodity forecast models (Coffee & Sugar) using a simple, clear workflow:

1. **Read model configs** - Define which models to train
2. **Load training data** - Get historical commodity prices and features
3. **Fit models** - Train models on historical data
4. **Save fitted models** - Persist models to database
5. **Use fitted models for inference** - Generate forecasts (separate workflow)

No Spark complexity. No unnecessary abstractions. Just working code.

---

## Quick Start

### Option 1: Run Locally (Development)

```bash
# 1. Load credentials
cd forecast_agent/
set -a && source ../infra/.env && set +a

# 2. Run simple training script
python databricks_train_simple.py
```

This will train all models locally using your Databricks database for data/storage.

**Expected Output:**
```
================================================================================
STARTING TRAINING WORKFLOW
================================================================================

Training windows: 16
Total models to train: 96
  2 commodities × 3 models × 16 windows

================================================================================
COMMODITY: Coffee
================================================================================

Training Window: 2018-01-01
  Training naive for Coffee (cutoff: 2018-01-01)...
  Loading data for Coffee up to 2018-01-01...
    Loaded 1460 rows (2014-01-02 to 2018-01-01)
    ✓ Model trained successfully

  Saving naive for Coffee (cutoff: 2018-01-01)...
    ✓ Model saved to trained_models table

  Progress: 1/96 models trained
...
```

**Duration:** 1-2 hours (SARIMAX is slow but accurate)

### Option 2: Run on Databricks (Production)

```bash
# 1. Load credentials
cd forecast_agent/
set -a && source ../infra/.env && set +a

# 2. Submit to Databricks
python /tmp/run_training_simple.py
```

This uploads the notebook and runs it on a Databricks cluster.

**Monitor URL:** Printed in output (format: `https://dbc-XXX.cloud.databricks.com/#job/{run_id}/run/1`)

**Duration:** 30-60 minutes (Databricks is faster than local)

---

## The 5 Steps (Explained)

### Step 1: Read Model Configs

Models are defined in `ground_truth/config/model_registry.py`:

```python
BASELINE_MODELS = {
    'naive': {
        'name': 'naive',
        'function': naive_forecast_with_metadata,
        'params': {'target': 'close', 'horizon': 14}
    },
    'xgboost': {
        'name': 'xgboost',
        'function': xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_mean_c', 'humidity_mean_pct', 'precipitation_mm', 'vix'],
            'horizon': 14
        }
    },
    'sarimax_auto_weather': {
        'name': 'sarimax_auto_weather',
        'function': sarimax_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_mean_c', 'humidity_mean_pct', 'precipitation_mm'],
            'horizon': 14,
            'auto_arima': True
        }
    }
}
```

**What this means:**
- **Naive**: Simple baseline (last price + random walk)
- **XGBoost**: Gradient boosting with weather + market features
- **SARIMAX**: Statistical time series with weather features

### Step 2: Load Training Data

Data comes from `commodity.silver.unified_data` table (multi-region format):

```python
def load_training_data(commodity: str, cutoff_date: str, lookback_days: int = 1460):
    """Load 4 years of historical data up to cutoff_date"""

    query = f"""
        SELECT
            date,
            commodity,
            region,
            close,
            temp_mean_c,
            humidity_mean_pct,
            precipitation_mm,
            vix
        FROM commodity.silver.unified_data
        WHERE commodity = '{commodity}'
          AND date >= '{cutoff_date - 4 years}'
          AND date <= '{cutoff_date}'
        ORDER BY date
    """

    # Execute query via Databricks SQL connector
    # Returns: pd.DataFrame with ~1460 rows
```

**Data includes:**
- **Daily commodity prices** (close)
- **Weather features** (temp, humidity, precipitation)
- **Market features** (VIX)

### Step 3: Fit Models

Each model implements a standard interface:

```python
def train_model(commodity: str, model_key: str, cutoff_date: str):
    """Train a single model on historical data"""

    # 1. Get model config
    model_config = BASELINE_MODELS[model_key]
    model_function = model_config['function']
    params = model_config['params']

    # 2. Load and prepare data
    raw_df = load_training_data(commodity, cutoff_date)
    training_df = prepare_features(raw_df, model_config)

    # 3. Train model
    result = model_function(training_df, commodity, fitted_model=None, **params)
    fitted_model = result['fitted_model']

    return fitted_model
```

**Training process:**
- Load 4 years of historical data
- Extract required features
- Fit model on historical data
- Return fitted model object

### Step 4: Save Fitted Models

Models are saved to `commodity.forecast.trained_models` table:

```python
def save_fitted_model(model_data: dict):
    """Persist fitted model to database"""

    # Serialize model to JSON
    fitted_model_json = json.dumps({
        'model_type': model_data['model_type'],
        'fitted_model': str(model_data['fitted_model']),
        'last_date': str(model_data['last_date']),
        'target': model_data['target']
    })

    # Insert into database
    insert_sql = f"""
        INSERT INTO commodity.forecast.trained_models
        (commodity, model_name, model_version, training_window_end,
         year, month, fitted_model_json, created_at)
        VALUES (
            '{commodity}',
            '{model_name}',
            'v1.0',
            '{training_window_end}',
            {year},
            {month},
            '{fitted_model_json}',
            CURRENT_TIMESTAMP()
        )
    """
```

**Storage format:**
- **Small models** (<1MB): JSON in `fitted_model_json` column
- **Large models** (≥1MB): S3 with path in `fitted_model_s3_path` column
- **Partitioned by:** (year, month) for efficient loading

### Step 5: Use Fitted Models for Inference

This step is implemented in `backfill_rolling_window.py` (separate workflow):

```python
# 1. Load pretrained model from database
fitted_model = load_model_from_db(commodity, model_name, training_window_end)

# 2. Generate forecast using pretrained model
forecast_result = model_function(
    training_df,
    commodity,
    fitted_model=fitted_model,  # Use pretrained model!
    **params
)

# 3. Generate Monte Carlo paths
paths_df = generate_monte_carlo_paths(forecast_result['forecast_df'], num_paths=2000)

# 4. Save to commodity.forecast.distributions
write_distributions_to_db(commodity, model_name, forecast_date, paths_df)
```

**Why separate training and inference?**
- Train once (expensive, ~16 windows over 7 years)
- Infer many times (fast, ~2,875 dates)
- **180x speedup** vs train-per-date

---

## Training Configuration

### Models
- `naive` - Baseline model (~1 min per window)
- `xgboost` - Gradient boosting (~5 min per window)
- `sarimax_auto_weather` - Statistical time series (~30 min per window)

### Commodities
- `Coffee`
- `Sugar`

### Training Frequency
- `semiannually` - Train every 6 months (recommended)
- `monthly` - Train every month (for fast models only)

### Date Range
- **Start:** 2018-01-01
- **End:** 2025-11-17
- **Windows:** 16 (one every 6 months)

### Total Models
- 2 commodities × 3 models × 16 windows = **96 trained models**

---

## Verification

### Check Models Were Saved

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

**Expected output:**
```
Coffee  | naive                | 2025-07-01 | 1
Coffee  | naive                | 2025-01-01 | 1
Coffee  | naive                | 2024-07-01 | 1
Coffee  | xgboost              | 2025-07-01 | 1
Coffee  | sarimax_auto_weather | 2025-07-01 | 1
Sugar   | naive                | 2025-07-01 | 1
...
```

---

## Next Steps

After training completes successfully:

### 1. Run Backfill for Inference

```bash
python backfill_rolling_window.py \
    --commodity Coffee \
    --models naive xgboost sarimax_auto_weather \
    --train-frequency semiannually \
    --start-date 2018-01-01 \
    --end-date 2025-11-17 \
    --model-version-tag v1.0
```

This loads pretrained models and generates forecasts for all dates.

### 2. Verify Forecast Coverage

```bash
python check_backfill_coverage.py \
    --commodity Coffee \
    --models naive xgboost sarimax_auto_weather
```

### 3. Evaluate Performance

```bash
python evaluate_historical_forecasts.py \
    --commodity Coffee \
    --models naive xgboost sarimax_auto_weather
```

---

## Troubleshooting

### Training Takes Too Long Locally

Use Databricks instead:
```bash
python /tmp/run_training_simple.py
```

Databricks is 2-3x faster due to better hardware.

### "ModuleNotFoundError: No module named 'ground_truth'"

Make sure you're running from the `forecast_agent/` directory:
```bash
cd /path/to/ucberkeley-capstone/forecast_agent/
```

### Database Connection Timeout

Databricks sessions timeout after 15 minutes. Scripts automatically reconnect between training windows.

### Model Serialization Issues

For large models (SARIMAX), we use S3 storage automatically when JSON exceeds 1MB.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│                  SIMPLE 5-STEP WORKFLOW                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. READ MODEL CONFIGS                                   │
│     ├─ naive (baseline)                                  │
│     ├─ xgboost (gradient boosting)                       │
│     └─ sarimax_auto_weather (statistical TS)             │
│                                                          │
│  2. LOAD TRAINING DATA                                   │
│     ├─ commodity.silver.unified_data (multi-region)      │
│     ├─ 4 years of historical data per window             │
│     └─ Features: close, temp, humidity, precip, vix      │
│                                                          │
│  3. FIT MODELS                                           │
│     ├─ Train on historical data                          │
│     ├─ 16 training windows (semiannually)                │
│     └─ 3 models × 2 commodities = 6 per window           │
│                                                          │
│  4. SAVE FITTED MODELS                                   │
│     ├─ commodity.forecast.trained_models                 │
│     ├─ JSON (<1MB) or S3 (≥1MB)                          │
│     └─ Partitioned by (year, month)                      │
│                                                          │
│  5. USE FITTED MODELS FOR INFERENCE                      │
│     ├─ Load pretrained model from DB                     │
│     ├─ Generate 14-day forecasts                         │
│     ├─ Create 2,000 Monte Carlo paths                    │
│     └─ Write to commodity.forecast.distributions         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Key principle:** Train once, infer many times.

---

## Files

- `databricks_train_simple.py` - Main training script (5 steps)
- `/tmp/run_training_simple.py` - Databricks submission script
- `ground_truth/config/model_registry.py` - Model configurations
- `ground_truth/models/` - Model implementations
- `backfill_rolling_window.py` - Inference workflow (Step 5)

---

## Related Documentation

- `DATABRICKS_TRAINING_QUICKSTART.md` - Alternative Databricks workflow
- `CLAUDE.md` - Complete reference documentation
- `FEATURE_ENGINEERING_GUIDE.md` - Feature pipeline details

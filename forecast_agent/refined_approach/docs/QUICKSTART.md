# Quick Start Guide - Refined Approach

## Overview

This guide helps you get started with the refined forecast agent approach in Databricks.

## Setup

### 1. Clone Repository to Databricks Repos

1. In Databricks, go to **Repos**
2. Click **Add Repo**
3. Connect to your GitHub repository
4. The `forecast_agent/refined_approach` folder will be available

### 2. Verify Table Access

Ensure you have access to:
- `commodity.silver.unified_data` (input data)
- `commodity.forecast.trained_models` (model storage)
- `commodity.forecast.distributions` (forecast output)

### 3. Run Notebooks in Sequence

## Workflow

### Step 1: Train Models (`01_train_models.py`)

**Purpose:** Train forecasting models and save to `trained_models` table.

**Parameters:**
- `commodity`: Coffee or Sugar
- `models`: Comma-separated model keys (e.g., "naive,random_walk")
- `train_frequency`: semiannually, monthly, quarterly, etc.
- `model_version`: Version tag (e.g., "v1.0")
- `start_date`: Start of training period
- `end_date`: End of training period

**Output:**
- Models saved to `commodity.forecast.trained_models` table

**Example:**
```python
# Set widgets or modify code
commodity = "Coffee"
models = ["naive", "random_walk"]
train_frequency = "semiannually"
model_version = "v1.0"
start_date = "2020-01-01"
end_date = "2024-01-01"
```

### Step 2: Generate Forecasts (`02_generate_forecasts.py`)

**Purpose:** Load trained models, generate forecasts, populate distributions table.

**Parameters:**
- `commodity`: Coffee or Sugar
- `models`: Comma-separated model keys
- `model_version`: Version tag (must match training)
- `start_date`: First forecast date
- `end_date`: Last forecast date

**Output:**
- Forecasts written to `commodity.forecast.distributions` table
- Only data leakage-free forecasts (has_data_leakage=FALSE)

**Key Features:**
- Automatically uses most recent trained model for each date
- Filters out data leakage
- Generates 2000 Monte Carlo paths per forecast

## Key Concepts

### Data Leakage Prevention

**Critical:** Distributions table only contains forecasts where `forecast_start_date > data_cutoff_date`.

- `data_cutoff_date` = Last date in training data
- `forecast_start_date` = First day of forecast
- Only forecasts with `forecast_start_date > data_cutoff_date` are written

### Region Aggregation

`unified_data` has grain: `(date, commodity, region)`

Models need aggregated data (date level). Options:
- **Mean**: Average weather across regions
- **First**: Use first region's values (market data is identical)

Example:
```python
df = loader.load_to_pandas(
    commodity='Coffee',
    aggregate_regions=True,
    aggregation_method='mean'  # or 'first'
)
```

### Feature Sets

Easy to experiment with different features:

```python
# Basic features
features = ['close', 'temp_mean_c', 'humidity_mean_pct', 'precipitation_mm', 'vix']

# With GDELT
features = ['close', 'temp_mean_c', 'vix', 'gdelt_tone', 'gdelt_event_count']

# Different combinations
features = ['close', 'vix', 'cop_usd']  # Minimal set
```

### Expanding Window CV

Default is expanding window (more realistic for production):
- Each fold uses all data up to test period
- Mimics production scenario

To use rolling window instead:
```python
folds = loader.create_temporal_folds(
    df=df,
    n_folds=5,
    test_fold=True,
    min_train_size=365,
    step_size=14,
    horizon=14
)
```

## Troubleshooting

### Import Errors

If modules can't be imported:
1. Check that repo is properly connected
2. Verify path in notebook matches your repo structure
3. Try adding path manually:
   ```python
   import sys
   sys.path.insert(0, '/Workspace/Repos/<your-repo>/forecast_agent/refined_approach')
   ```

### No Data Leakage-Free Forecasts

If you see: `⚠️  No data leakage-free forecasts`

- Check that training dates are before forecast dates
- Ensure `forecast_start_date > data_cutoff_date`
- Review training window and forecast window overlap

### Region Aggregation Issues

If you see region-related errors:
- Ensure `aggregate_regions=True` when loading data
- Check that market data columns are aggregated correctly
- Verify feature columns exist in unified_data

## Next Steps

1. **Experiment with Features:**
   - Try different feature combinations
   - Add GDELT sentiment features
   - Test feature engineering approaches

2. **Compare Models:**
   - Train multiple models
   - Compare performance metrics
   - Select best model for production

3. **Backfill Historical Data:**
   - Generate forecasts for historical dates
   - Populate distributions table
   - Enable trading agent backtesting

## Questions?

- Check `REQUIREMENTS.md` for detailed requirements
- Review `README.md` for architecture overview
- See `COMPARISON.md` for differences from current approach


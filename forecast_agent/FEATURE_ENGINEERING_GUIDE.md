# Feature Engineering Pipeline Guide

## Overview

The feature engineering pipeline (`ground_truth/features/data_preparation.py`) provides flexible data aggregation strategies for different model types. It handles both **multi-region weather data** and **GDELT sentiment vectors** with configurable preprocessing.

## Data Structure

### Input: commodity.silver.unified_data

The unified data table contains **multiple rows per date** - one per geographic region:

```
date       commodity  region    close  temp_mean_c  humidity_mean_pct  ...
2020-01-01 Coffee     Brazil    100.5  25.0         70.0              ...
2020-01-01 Coffee     Colombia  100.5  20.0         80.0              ...
2020-01-02 Coffee     Brazil    101.2  26.0         68.0              ...
2020-01-02 Coffee     Colombia  101.2  21.0         78.0              ...
```

### GDELT Sentiment Data

GDELT data is already in **wide format** (one row per date) with theme-specific columns:

```
date       commodity  group_SUPPLY_tone_avg  group_LOGISTICS_tone_avg  group_MARKET_tone_avg  ...
2020-01-01 Coffee     0.50                   0.30                      0.40                  ...
```

**Available themes:**
- SUPPLY
- LOGISTICS
- TRADE
- MARKET
- POLICY
- CORE

**Metrics per theme:**
- `_count`: Number of articles
- `_tone_avg`: Average tone score
- `_tone_positive`: Positive sentiment
- `_tone_negative`: Negative sentiment
- `_tone_polarity`: Polarity score

## Aggregation Strategies

### 1. Aggregate Strategy (`region_strategy='aggregate'`)

**Use Case**: Traditional models that need single-row-per-date data (ARIMA, simple models)

**Behavior**: Average across all regions

**Input** (multi-row):
```
date       region    temp_mean_c  humidity_mean_pct
2020-01-01 Brazil    25.0         70.0
2020-01-01 Colombia  20.0         80.0
```

**Output** (single-row):
```
date       temp_mean_c  humidity_mean_pct
2020-01-01 22.5         75.0
```

**Example**:
```python
from ground_truth.features.data_preparation import prepare_data_for_model

df = prepare_data_for_model(
    raw_data=unified_data,
    commodity='Coffee',
    region_strategy='aggregate',  # Average across regions
    gdelt_strategy='aggregate'     # Average across GDELT themes
)
# Result: One row per date, all features averaged
```

### 2. Pivot Strategy (`region_strategy='pivot'`)

**Use Case**: SARIMAX, XGBoost, models that can handle region-specific features

**Behavior**: Create region-specific columns (brazil_temp_mean_c, colombia_temp_mean_c, etc.)

**Input** (multi-row):
```
date       region    temp_mean_c  humidity_mean_pct
2020-01-01 Brazil    25.0         70.0
2020-01-01 Colombia  20.0         80.0
```

**Output** (single-row, pivoted):
```
date       brazil_temp_mean_c  brazil_humidity_mean_pct  colombia_temp_mean_c  colombia_humidity_mean_pct
2020-01-01 25.0               70.0                      20.0                  80.0
```

**Example**:
```python
df = prepare_data_for_model(
    raw_data=unified_data,
    commodity='Coffee',
    region_strategy='pivot',          # Create region-specific columns
    gdelt_strategy='select',           # Use specific GDELT themes only
    gdelt_themes=['SUPPLY', 'MARKET']  # Keep only these themes
)
# Result: brazil_temp_mean_c, colombia_temp_mean_c, group_SUPPLY_tone_avg, group_MARKET_tone_avg
```

### 3. All Strategy (`region_strategy='all'`)

**Use Case**: Neural models (LSTM, Transformer) that can handle sequences and vectors

**Behavior**: Keep raw multi-row format

**Input** (multi-row):
```
date       region    temp_mean_c  humidity_mean_pct
2020-01-01 Brazil    25.0         70.0
2020-01-01 Colombia  20.0         80.0
2020-01-02 Brazil    26.0         68.0
2020-01-02 Colombia  21.0         78.0
```

**Output** (multi-row, unchanged):
```
date       region    temp_mean_c  humidity_mean_pct
2020-01-01 Brazil    25.0         70.0
2020-01-01 Colombia  20.0         80.0
2020-01-02 Brazil    26.0         68.0
2020-01-02 Colombia  21.0         78.0
```

**Example**:
```python
df = prepare_data_for_model(
    raw_data=unified_data,
    commodity='Coffee',
    region_strategy='all',  # Keep multi-row format
    gdelt_strategy='all'    # Keep all GDELT themes
)
# Result: Multiple rows per date, ready for sequence models
```

## Model Configuration in model_registry.py

### Example 1: SARIMAX with Aggregated Data

```python
BASELINE_MODELS = {
    'sarimax_auto_weather': {
        'name': 'SARIMAX+Weather',
        'function': sarimax.sarimax_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_mean_c', 'humidity_mean_pct', 'precipitation_mm'],
            'region_strategy': 'aggregate',  # Average across regions
            'gdelt_strategy': 'aggregate',   # Average across themes
            'horizon': 14
        }
    }
}
```

### Example 2: XGBoost with Region-Specific Columns

```python
BASELINE_MODELS = {
    'xgboost_regional': {
        'name': 'XGBoost (Regional)',
        'function': xgboost_model.xgboost_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': [
                'brazil_temp_mean_c', 'colombia_temp_mean_c',
                'brazil_humidity_mean_pct', 'colombia_humidity_mean_pct',
                'group_SUPPLY_tone_avg', 'group_MARKET_tone_avg'
            ],
            'region_strategy': 'pivot',                   # Create region columns
            'gdelt_strategy': 'select',                   # Use specific themes
            'gdelt_themes': ['SUPPLY', 'MARKET'],         # Only these themes
            'horizon': 14
        }
    }
}
```

### Example 3: LSTM with Full Vectors

```python
BASELINE_MODELS = {
    'lstm_multiregion': {
        'name': 'LSTM (Multi-Region)',
        'function': lstm_model.lstm_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_mean_c', 'humidity_mean_pct', 'precipitation_mm'],
            'region_strategy': 'all',        # Keep multi-row format
            'gdelt_strategy': 'all',         # Keep all GDELT themes
            'sequence_length': 30,
            'horizon': 14
        }
    }
}
```

## API Reference

### `prepare_data_for_model()`

```python
def prepare_data_for_model(
    raw_data: pd.DataFrame,
    commodity: str,
    region_strategy: str = 'aggregate',
    gdelt_strategy: Optional[str] = None,
    gdelt_themes: Optional[List[str]] = None,
    feature_columns: Optional[List[str]] = None
) -> pd.DataFrame
```

**Parameters:**

- **raw_data** (DataFrame): Raw data from `commodity.silver.unified_data` (multi-region format)
- **commodity** (str): 'Coffee' or 'Sugar'
- **region_strategy** (str): How to handle region data
  - `'aggregate'`: Average across regions → single value per date
  - `'pivot'`: Create region-specific columns → brazil_temp, colombia_temp, etc.
  - `'all'`: Keep multi-row format → for neural models
- **gdelt_strategy** (str, optional): How to handle GDELT themes (defaults to `region_strategy`)
  - `'aggregate'`: Average across all themes → gdelt_tone_avg_all
  - `'pivot'`: Keep theme-specific columns → group_SUPPLY_tone_avg, group_MARKET_tone_avg
  - `'select'`: Use only specific themes (requires `gdelt_themes` parameter)
  - `'all'`: Keep all GDELT theme columns as-is
- **gdelt_themes** (List[str], optional): List of themes when `gdelt_strategy='select'`
  - Example: `['SUPPLY', 'LOGISTICS', 'MARKET']`
- **feature_columns** (List[str], optional): Specific feature columns to include

**Returns:**
- **DataFrame**: Processed data ready for model training/prediction

## Integration with Training Pipeline

### Step 1: Update Data Loading

Modify `databricks_train_fresh_models.py`:

```python
def load_training_data_spark(commodity: str, cutoff_date, model_config: dict):
    """Load and prepare training data using Spark SQL"""
    # Load raw data (multi-region)
    query = f"""
        SELECT *
        FROM commodity.silver.unified_data
        WHERE commodity = '{commodity}'
        AND date <= '{cutoff_date}'
        ORDER BY date
    """
    spark_df = spark.sql(query)
    pandas_df = spark_df.toPandas()
    pandas_df['date'] = pd.to_datetime(pandas_df['date'])
    pandas_df = pandas_df.set_index('date')

    # Apply feature engineering based on model config
    from ground_truth.features.data_preparation import prepare_data_for_model

    params = model_config['params']
    processed_df = prepare_data_for_model(
        raw_data=pandas_df,
        commodity=commodity,
        region_strategy=params.get('region_strategy', 'aggregate'),
        gdelt_strategy=params.get('gdelt_strategy', None),
        gdelt_themes=params.get('gdelt_themes', None),
        feature_columns=params.get('exog_features', None)
    )

    return processed_df
```

### Step 2: Update Model Training

Models automatically use the preprocessed data:

```python
# Training loop
for model_key in model_keys:
    model_config = BASELINE_MODELS[model_key]

    # Load data with model-specific preprocessing
    training_df = load_training_data_spark(commodity, training_cutoff, model_config)

    # Train model (receives preprocessed data)
    train_func = train_functions[model_key]
    fitted_model_dict = train_func(training_df, **model_config['params'])
```

## Best Practices

### 1. Model-Specific Configuration

Don't hardcode aggregation logic in model code. Use parameters in `model_registry.py`:

**Bad**:
```python
# In xgboost_model.py
def xgboost_train(df, ...):
    # Hardcoded: always aggregate
    df_agg = df.groupby('date').mean()
```

**Good**:
```python
# In model_registry.py
'xgboost': {
    'params': {
        'region_strategy': 'pivot',  # Configurable!
        'gdelt_strategy': 'select',
        'gdelt_themes': ['SUPPLY', 'MARKET']
    }
}
```

### 2. Experiment with Different Strategies

Test multiple feature engineering approaches:

```python
# Test 1: Aggregated features
'xgboost_agg': {
    'params': {
        'region_strategy': 'aggregate',  # Single avg temp
        'gdelt_strategy': 'aggregate'    # Single avg sentiment
    }
}

# Test 2: Regional features
'xgboost_regional': {
    'params': {
        'region_strategy': 'pivot',      # brazil_temp, colombia_temp
        'gdelt_strategy': 'select',      # Specific themes only
        'gdelt_themes': ['SUPPLY', 'MARKET']
    }
}
```

### 3. GDELT Theme Selection

Start with relevant themes for commodity forecasting:

- **SUPPLY**: Supply chain disruptions, production issues
- **LOGISTICS**: Transportation, shipping, port activity
- **MARKET**: Market conditions, trading activity
- **TRADE**: Trade agreements, tariffs, international trade

Example:
```python
'sarimax_gdelt': {
    'params': {
        'exog_features': ['temp_mean_c', 'humidity_mean_pct'],
        'region_strategy': 'aggregate',
        'gdelt_strategy': 'select',
        'gdelt_themes': ['SUPPLY', 'LOGISTICS', 'MARKET']  # Most relevant
    }
}
```

## Troubleshooting

### Issue: "Column 'region' found in feature set"

**Cause**: Model code doesn't properly exclude region column after preprocessing

**Fix**: Update model to exclude region in feature selection:

```python
# In xgboost_model.py
feature_cols = [col for col in df.columns if col != target and col not in ['commodity', 'region']]
```

### Issue: "GDELT columns not found"

**Cause**: GDELT data not joined to unified_data

**Solution**: Ensure unified_data includes GDELT columns, or load separately and join:

```python
# Load GDELT
gdelt_df = spark.sql("SELECT * FROM commodity.silver.gdelt_wide WHERE commodity = 'Coffee'").toPandas()

# Join with unified_data
raw_data = raw_data.merge(gdelt_df, on=['date', 'commodity'], how='left')
```

### Issue: "Multiple rows per date after pivot"

**Cause**: Pivot strategy requires unique regions

**Fix**: Ensure data has one row per (date, region) before calling prepare_data_for_model

## Future Enhancements

1. **Weighted Aggregation**: Weight regions by production volume
2. **Time-Lagged Features**: Add lagged GDELT sentiment
3. **Interaction Features**: brazil_temp × supply_tone interactions
4. **Dimensionality Reduction**: PCA on GDELT themes for high-dimensional models

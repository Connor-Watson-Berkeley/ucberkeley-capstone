# ML Pipeline Library

**Clean, scalable PySpark ML approach for commodity price forecasting**

---

## Overview

This library replaces the legacy forecasting system with a unified PySpark ML Pipeline architecture. Inspired by the flight delay prediction pipeline, it emphasizes:
- **Simplicity**: One pipeline definition per model, clear separation of train/inference
- **Scalability**: Native PySpark parallelization, extensible transformer pattern
- **Reproducibility**: All models defined in registry, CV residuals stored for Monte Carlo

---

## Core Principles

### 1. Two-Stage Workflow

**Stage 1: Training** (`train.py`)
- Fit models with time-series cross-validation
- Save fitted pipelines to DBFS
- Track metrics (directional accuracy, MAE, RMSE) in metadata table
- Store CV residuals for Monte Carlo generation

**Stage 2: Inference** (`inference.py`)
- Load fitted pipelines from DBFS
- Generate point forecasts
- Generate 2,000 Monte Carlo paths using block bootstrap
- Write to `distributions` and `point_forecasts` tables

### 2. Pipeline Registry with Builder Functions

All models defined in `pipelines/pipeline_registry.py` using **builder functions** for dependency isolation:

```python
def build_xgboost_weather_pipeline():
    """Build XGBoost pipeline with weather features.

    Dependencies: pyspark.ml.regression.GBTRegressor
    """
    from ml_lib.transformers import WeatherFeaturesEstimator, LagFeaturesEstimator
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import GBTRegressor

    return Pipeline(stages=[
        WeatherFeaturesEstimator(),
        LagFeaturesEstimator(lags=[1, 7, 14]),
        VectorAssembler(
            inputCols=['temp_mean_c', 'humidity_mean_pct', 'lag_1', 'lag_7', 'lag_14'],
            outputCol='features'
        ),
        GBTRegressor(featuresCol='features', labelCol='close', maxIter=100)
    ])

def build_arima_pipeline():
    """Build ARIMA pipeline.

    Dependencies: statsmodels (only imported if this model is used)
    """
    from ml_lib.models import ARIMAEstimator
    from pyspark.ml import Pipeline

    return Pipeline(stages=[ARIMAEstimator(order=(1,1,1))])

PIPELINE_REGISTRY = {
    'xgboost_weather': {
        'name': 'XGBoost with Weather Features',
        'description': 'Gradient boosting with temp, humidity, precipitation',
        'builder': build_xgboost_weather_pipeline,  # Function reference
        'metadata': {
            'horizon': 14,
            'features': ['weather', 'lags'],
            'target_metric': 'directional_accuracy_day0',
            'dependencies': ['pyspark.ml']
        }
    },
    'arima': {
        'name': 'ARIMA',
        'description': 'AutoRegressive Integrated Moving Average',
        'builder': build_arima_pipeline,  # Function reference
        'metadata': {
            'horizon': 14,
            'features': ['lags'],
            'target_metric': 'directional_accuracy_day0',
            'dependencies': ['statsmodels']  # Optional dependency
        }
    }
}
```

**Usage:**
```python
from ml_lib.pipelines import get_pipeline

# Only imports dependencies for xgboost_weather model
pipeline, metadata = get_pipeline('xgboost_weather')
```

**Benefits:**
- Lazy loading: Dependencies only imported when model is requested
- Isolated: Each model's imports are contained in its builder function
- Flexible: Can have models with conflicting dependencies (e.g., TensorFlow vs PyTorch)
- Clear: Easy to see what each model needs by reading its builder function

### 3. Directional Accuracy as Primary Metric

**Trading insight:** Profit depends on getting direction right, not minimizing MAE.

**Primary metric:** Directional Accuracy from Day 0
- Is day_i > day_0? (for i=1..14)
- Measures trading signal quality

**Secondary metrics:** MAE, RMSE (for model diagnostics)

### 4. Universal Monte Carlo via Block Bootstrap

**Works for ANY model type** (ARIMA, XGBoost, LSTM):
1. Collect forecast residuals during CV
2. Bootstrap blocks of residuals (preserves autocorrelation)
3. Add to point forecasts to generate 2,000 realistic paths

**Why:** Simpler than quantile regression, more general than ARIMA-specific methods.

---

## Folder Structure

```
ml_lib/
├── README.md                       # This file
├── temp/
│   └── ARCHITECTURE_ANALYSIS.md    # Detailed trade-offs and decisions
│
├── transformers/                   # Custom PySpark transformers
│   ├── weather_features.py         # Add weather columns
│   ├── sentiment_features.py       # Add GDELT sentiment
│   ├── lag_features.py             # LagFeatureEstimator (fit finds optimal lags)
│   └── time_features.py            # Day of week, month, seasonality
│
├── models/                         # Model implementations
│   ├── baseline.py                 # Naive, RandomWalk
│   ├── linear.py                   # LinearRegression, Ridge, LASSO
│   └── xgboost_model.py            # XGBoost forecaster
│
├── cross_validation/               # Time-series CV
│   ├── time_series_cv.py           # TimeSeriesForecastCV class
│   └── data_loader.py              # Loads commodity.silver.unified_data
│
├── pipelines/                      # Pipeline definitions (model registry)
│   ├── pipeline_registry.py        # All model configs (builder functions)
│   └── pipeline_factory.py         # get_pipeline() - lazy load
│
├── monte_carlo/                    # Uncertainty quantification
│   └── path_generator.py           # Block bootstrap path generation
│
├── persistence/                    # Save/load pipelines
│   ├── pipeline_saver.py           # DBFS save/load
│   └── metadata_tracker.py         # SQL table tracking
│
├── train.py                        # Stage 1: Train models with CV
└── inference.py                    # Stage 2: Generate forecasts
```

---

## Quick Start

### Stage 1: Train Models

```bash
cd forecast_agent/ml_lib

# Train with 5-fold expanding window CV
python train.py \
  --commodity Coffee \
  --models xgboost_weather linear_baseline \
  --cv-folds 5 \
  --cv-window expanding

# Output:
# - Fitted pipelines: dbfs:/forecast_models/coffee_xgboost_weather_2024-12-05/
# - Metadata: commodity.forecast.ml_pipeline_metadata
# - CV residuals: dbfs:/forecast_residuals/coffee_xgboost_weather_2024-12-05.parquet
```

### Stage 2: Generate Forecasts

```bash
# Load fitted pipeline and generate forecasts
python inference.py \
  --commodity Coffee \
  --model xgboost_weather \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --n-paths 2000

# Output:
# - Point forecasts: commodity.forecast.point_forecasts
# - Monte Carlo paths: commodity.forecast.distributions
```

---

## Data Flow

```
commodity.silver.unified_data (continuous daily data)
  ↓
[TimeSeriesForecastCV]
  ├─ Expanding window splits (2018-2020, 2018-2021, etc.)
  ├─ Fit pipeline on each fold
  ├─ Collect residuals for uncertainty estimation
  └─ Save metrics to metadata table
  ↓
[train.py] → Fitted PipelineModel saved to DBFS
  ↓
[inference.py] → Load pipeline, generate forecasts
  ├─ Point forecasts (mean predictions)
  └─ Monte Carlo paths (block bootstrap on CV residuals)
  ↓
commodity.forecast.distributions (2,000 paths × dates)
commodity.forecast.point_forecasts (14-day predictions)
```

---

## Adding a New Model

### 1. Create Builder Function in Registry

Edit `pipelines/pipeline_registry.py`:
```python
def build_my_new_model_pipeline():
    """Build my new model.

    Dependencies: pyspark.ml, scikit-learn
    """
    # Import dependencies HERE (lazy loading)
    from ml_lib.transformers import WeatherFeaturesEstimator
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import RandomForestRegressor

    return Pipeline(stages=[
        WeatherFeaturesEstimator(),
        VectorAssembler(
            inputCols=['temp_mean_c', 'humidity_mean_pct'],
            outputCol='features'
        ),
        RandomForestRegressor(
            featuresCol='features',
            labelCol='close',
            numTrees=100
        )
    ])

# Add to registry
PIPELINE_REGISTRY['my_new_model'] = {
    'name': 'Random Forest with Weather',
    'description': 'RF using temperature and humidity',
    'builder': build_my_new_model_pipeline,  # Function reference
    'metadata': {
        'horizon': 14,
        'features': ['weather'],
        'target_metric': 'directional_accuracy_day0',
        'dependencies': ['pyspark.ml']
    }
}
```

**Key points:**
- Builder function name: `build_{model_name}_pipeline()`
- Import dependencies INSIDE the function (lazy loading)
- Docstring lists dependencies for documentation
- `metadata['dependencies']` tracks what's needed (for checking)

### 2. Train and Evaluate

```bash
python train.py --commodity Coffee --models my_new_model --cv-folds 5
```

### 3. Generate Forecasts

```bash
python inference.py --commodity Coffee --model my_new_model --start-date 2024-01-01
```

**That's it!** Dependencies are only loaded when you use this model.

---

## Model Persistence

**Method:** PySpark native `Pipeline.save()` + SQL metadata table

**Why:**
- Simple, no extra dependencies (no MLflow setup)
- Native PySpark API
- Easy migration to MLflow later if needed
- Metadata queryable in SQL for analysis

**Storage:**
- Fitted pipelines: `dbfs:/forecast_models/{commodity}_{model}_{date}/`
- Metadata: `commodity.forecast.ml_pipeline_metadata`
- CV residuals: `dbfs:/forecast_residuals/{commodity}_{model}_{date}.parquet`

---

## Time-Series Cross-Validation

**Expanding Window (Default):**
```
Fold 1: Train [2018-2020] → Validate [2021]
Fold 2: Train [2018-2021] → Validate [2022]
Fold 3: Train [2018-2022] → Validate [2023]
...
```
- Uses all historical data
- Better for non-stationary time series
- Recommended for commodity prices

**Rolling Window (Optional):**
```
Fold 1: Train [2018-2020] → Validate [2021]
Fold 2: Train [2019-2021] → Validate [2022]
Fold 3: Train [2020-2022] → Validate [2023]
...
```
- Fixed window size slides forward
- Better for stationary data with regime changes
- Configure with `--cv-window rolling`

---

## Evaluation Metrics

### Directional Accuracy from Day 0 (Primary)

**Definition:** For each forecast horizon i ∈ {1..14}, is the direction correct?
```python
actual_direction = actual_day_i > actual_day_0
forecast_direction = forecast_day_i > forecast_day_0
correct = (actual_direction == forecast_direction)
```

**Averaged across all horizons:** `mean(correct[day_1], ..., correct[day_14])`

**Why it matters:** This is what the trading agent needs to make buy/hold/sell decisions.

### MAE and RMSE (Secondary)

Used for model diagnostics and understanding prediction quality, but NOT the primary optimization target.

---

## Monte Carlo Path Generation

**Goal:** Generate 2,000 realistic autocorrelated paths for risk analysis.

**Method:** Block Bootstrap on CV Residuals

1. During CV, collect forecast errors: `residuals = actual - predicted`
2. To generate path:
   - Start with point forecast: `[day_1, ..., day_14]`
   - Sample blocks of residuals (size 3) to preserve autocorrelation
   - Add to point forecast: `path = forecast + sampled_residuals`
   - Repeat 2,000 times

**Why this works for all models:**
- ARIMA: Residuals capture stochastic component
- XGBoost: Residuals capture prediction uncertainty
- LSTM: Residuals capture model error patterns

**Key:** Use model-specific CV residuals, not generic historical volatility.

---

## Integration with Trading Agent

**Contract:** Trading agent expects these tables:

**`commodity.forecast.distributions`**
- Columns: `commodity`, `cutoff_date`, `model_name`, `path_id`, `day_1`...`day_14`
- 2,000 rows per (commodity, cutoff_date, model) combination
- Used for risk analysis and portfolio optimization

**`commodity.forecast.point_forecasts`**
- Columns: `commodity`, `cutoff_date`, `model_name`, `day_1`...`day_14`, `actual_close`
- 1 row per (commodity, cutoff_date, model) combination
- Used for simple trading strategies

**Our output matches this contract exactly.** No changes needed to trading agent.

---

## Dependency Management Pattern

**Problem:** Different models need different dependencies (e.g., statsmodels for ARIMA, PyTorch for LSTM).

**Solution:** Builder functions with lazy imports.

### Example: Optional statsmodels dependency

```python
# pipelines/pipeline_registry.py

def build_arima_pipeline():
    """Build ARIMA pipeline.

    Dependencies: statsmodels>=0.13.0
    """
    try:
        from ml_lib.models import ARIMAEstimator
    except ImportError:
        raise ImportError(
            "ARIMA model requires statsmodels. Install with: pip install statsmodels"
        )

    from pyspark.ml import Pipeline
    return Pipeline(stages=[ARIMAEstimator(order=(1,1,1))])
```

**Benefits:**
- Only fails if you actually try to use ARIMA
- Other models work fine without statsmodels
- Clear error message if dependency missing
- Can check `metadata['dependencies']` before attempting to load

---

## Migration Plan

### Phase 1: Proof of Concept (Current)
- [x] Create folder structure
- [x] Document architecture decisions
- [ ] Implement `TimeSeriesForecastCV`
- [ ] Implement block bootstrap path generator
- [ ] Migrate 2 models: Naive baseline, Linear Regression
- [ ] Test on small date range (Jan 2024)

### Phase 2: Expand Coverage
- [ ] Add XGBoost with full feature engineering
- [ ] Add ARIMA/SARIMAX (custom Estimator wrapper)
- [ ] Compare metrics across all models

### Phase 3: Production
- [ ] Wire up to Databricks jobs
- [ ] Backfill historical forecasts
- [ ] Deprecate legacy code

---

## Open Questions / TODOs

1. **ARIMA Integration:** Need custom `ARIMAEstimator` wrapper (statsmodels not native to Spark)
2. **Spark Parallelization:** Can parallelize inference across dates using `spark.parallelize(dates)`
3. **Feature Store:** Consider pre-computing expensive features later (not MVP)

---

## References

- **Architecture Analysis:** [temp/ARCHITECTURE_ANALYSIS.md](temp/ARCHITECTURE_ANALYSIS.md) - Full trade-offs and decisions
- **Current System:** [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) - Legacy train-once pattern
- **Inspiration:** Flight delay ML pipeline (DS261 project)

---

**Maintained by:** Connor Watson
**Last Updated:** 2024-12-05
**Status:** Active Development

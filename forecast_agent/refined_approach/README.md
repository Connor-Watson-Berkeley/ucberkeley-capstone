# Refined Approach - Streamlined Forecast Agent

A cleaner, more maintainable design for the forecast agent inspired by the DS261 project structure.

## 📚 Documentation

All documentation is organized in the [`docs/`](docs/) folder:

### Getting Started
- **[docs/DATABRICKS_DEPLOYMENT.md](docs/DATABRICKS_DEPLOYMENT.md)** - ⚡ **START HERE** - Deployment guide for Databricks
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Get started quickly in Databricks
- **[docs/PRIORITIES.md](docs/PRIORITIES.md)** - Priorities: Get it working first

### Core Concepts
- **[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)** - Complete requirements specification
- **[docs/FLOW.md](docs/FLOW.md)** - Workflow flow and fail-open behavior
- **[docs/INCREMENTAL.md](docs/INCREMENTAL.md)** - Incremental/resume execution (skip existing)
- **[docs/SIMPLICITY.md](docs/SIMPLICITY.md)** - Why this approach is simple

### Workflows
- **[docs/DAILY_PRODUCTION.md](docs/DAILY_PRODUCTION.md)** - Daily production workflow (backfill + daily)
- **[docs/BACKFILL_VS_DAILY.md](docs/BACKFILL_VS_DAILY.md)** - Backfilling vs daily production comparison

### Reference
- **[docs/SUMMARY.md](docs/SUMMARY.md)** - Implementation summary and status
- **[docs/COMPARISON.md](docs/COMPARISON.md)** - Comparison with current approach
- **[docs/MLFLOW_DECISION.md](docs/MLFLOW_DECISION.md)** - MLflow decision: Skip for now

## Overview

This refined approach addresses pain points in the current forecast_agent structure:

- **Overcomplicated structure** - Too many layers and abstractions
- **Databricks friction** - Complex Spark/connection handling scattered across files
- **Model interface inconsistency** - Multiple patterns for similar functionality
- **Cross-validation complexity** - Hard to understand and modify validation logic

## Design Philosophy

### Inspired by DS261 Flight Delay Project

The DS261 project demonstrates a clean OOP approach:
- **Simple classes** with single responsibilities
- **Clear separation**: DataLoader → Evaluator → CrossValidator
- **Model-agnostic**: Works with any estimator/pipeline
- **Spark-friendly**: Handles Databricks naturally

### Key Principles

1. **Separation of Concerns**
   - `TimeSeriesDataLoader` - Loads and prepares data, creates folds
   - `ForecastEvaluator` - Calculates metrics
   - `TimeSeriesCrossValidator` - Orchestrates CV workflow
   - `ModelPipeline` - Standardized model interface

2. **Simple and Explicit**
   - No hidden abstractions
   - Easy to understand flow
   - Minimal indirection

3. **Databricks Native**
   - Works seamlessly with Spark DataFrames
   - Handles both Spark and pandas modes
   - No complex connection management

## Architecture Comparison

### Current Approach

```
forecast_agent/
├── ground_truth/
│   ├── core/
│   │   ├── backtester.py          # Complex walk-forward logic
│   │   ├── walk_forward_evaluator.py  # Another evaluator?
│   │   ├── data_loader.py          # Spark-specific
│   │   └── base_forecaster.py      # Abstract base class
│   ├── models/                     # 12+ model files
│   └── config/
│       └── model_registry.py       # Function-based registry
├── train_models.py                 # 400+ lines, complex logic
└── backfill_rolling_window.py     # 1000+ lines, dual-mode (Spark/local)
```

**Issues:**
- Multiple overlapping evaluators
- Complex train/predict separation scattered across files
- Hard to understand data flow
- Difficult to add new models

### Refined Approach

```
refined_approach/
├── data_loader.py          # Simple, handles Spark/pandas
├── evaluator.py            # Single evaluator, all metrics
├── cross_validator.py      # Clean CV orchestration
├── model_pipeline.py       # Standardized interface
└── example_usage.py        # Clear examples
```

**Benefits:**
- **4 core files** instead of 20+
- **Single responsibility** per class
- **Easy to extend** - add new models by implementing `ModelPipeline`
- **Testable** - each component isolated

## Core Components

### 1. TimeSeriesDataLoader

Loads data and creates temporal folds.

```python
loader = TimeSeriesDataLoader(spark=spark)

# Load data
df = loader.load_to_pandas(commodity='Coffee', cutoff_date='2024-01-01')

# Create folds
folds = loader.create_walk_forward_folds(
    df=df,
    min_train_size=365,
    step_size=14,
    horizon=14
)
```

**Key Features:**
- Handles Spark and pandas seamlessly
- Simple fold generation (like DS261 `split.py`)
- Supports both sliding window and walk-forward

### 2. ForecastEvaluator

Calculates forecast metrics.

```python
evaluator = ForecastEvaluator(target_col='close', prediction_col='forecast')

metrics = evaluator.evaluate(actuals, forecasts)
# Returns: {'mae': 3.10, 'rmse': 4.25, 'dir_day0': 69.5, ...}
```

**Key Features:**
- All metrics in one place
- Null-safe calculations
- Directional accuracy (Day0 and day-to-day)

### 3. TimeSeriesCrossValidator

Orchestrates cross-validation.

```python
cv = TimeSeriesCrossValidator(
    data_loader=loader,
    evaluator=evaluator,
    folds=folds
)

# Run CV
metrics_df = cv.fit(model_fn=model, model_params={}, target_col='close', horizon=14)

# Test fold
test_metrics = cv.evaluate_test(model_fn=model, model_params={}, target_col='close', horizon=14)
```

**Key Features:**
- Clean separation of CV and test folds
- Works with any model function
- Returns structured metrics DataFrame

### 4. ModelPipeline

Standardized model interface.

```python
class NaivePipeline(ModelPipeline):
    def fit(self, train_df, target_col='close', **kwargs):
        self.last_value = train_df[target_col].iloc[-1]
        return self
    
    def predict(self, horizon=14, **kwargs):
        return pd.DataFrame({
            'date': forecast_dates,
            'forecast': [self.last_value] * horizon
        })

# Usage
model = NaivePipeline()
cv.fit(model_fn=model, model_params={})
```

**Key Features:**
- Simple `fit()` and `predict()` interface
- Works directly with cross-validator
- Can wrap existing model registry functions

## Usage Examples

### Basic Cross-Validation

```python
from refined_approach import TimeSeriesDataLoader, ForecastEvaluator, TimeSeriesCrossValidator
from refined_approach.model_pipeline import NaivePipeline

# Setup
loader = TimeSeriesDataLoader(spark=spark)
df = loader.load_to_pandas(commodity='Coffee', cutoff_date='2024-01-01')
folds = loader.create_walk_forward_folds(df, min_train_size=365, step_size=14, horizon=14)

evaluator = ForecastEvaluator()
cv = TimeSeriesCrossValidator(loader, evaluator, folds)

# Run CV
model = NaivePipeline()
metrics_df = cv.fit(model_fn=model, model_params={}, target_col='close', horizon=14)
print(metrics_df)
```

### Model Comparison

```python
from refined_approach.model_pipeline import NaivePipeline, RandomWalkPipeline

models = {
    'naive': NaivePipeline(),
    'random_walk': RandomWalkPipeline(lookback_days=30)
}

results = {}
for name, model in models.items():
    cv = TimeSeriesCrossValidator(loader, evaluator, folds)
    metrics_df = cv.fit(model_fn=model, model_params={}, target_col='close', horizon=14)
    results[name] = cv

# Compare
comparison = TimeSeriesCrossValidator.compare_models(
    results['naive'],
    results['random_walk'],
    name1="Naive",
    name2="Random Walk"
)
```

### Using Existing Model Registry

```python
from refined_approach.model_pipeline import create_model_from_registry

# Wrap existing model
model = create_model_from_registry('naive')

cv = TimeSeriesCrossValidator(loader, evaluator, folds)
metrics_df = cv.fit(model_fn=model, model_params={}, target_col='close', horizon=14)
```

## Benefits Over Current Approach

### 1. Simplicity

**Before:** 20+ files, complex interactions
**After:** 4 core files, clear flow

### 2. Databricks Compatibility

**Before:** Complex connection handling, Spark/pandas confusion
**After:** Clean Spark DataFrame handling, works in both modes

### 3. Model Development

**Before:** Multiple patterns (BaseForecaster, function-based, etc.)
**After:** Single `ModelPipeline` interface

### 4. Testing

**Before:** Hard to test due to complex dependencies
**After:** Each component isolated and testable

### 5. Extensibility

**Before:** Unclear how to add new models or metrics
**After:** Implement `ModelPipeline` or extend `ForecastEvaluator`

## Migration Path

### Phase 1: Evaluate (Current)

1. Review this refined approach
2. Run examples in `example_usage.py`
3. Compare with current workflow

### Phase 2: Pilot (Recommended)

1. Use refined approach for new models
2. Keep existing code running
3. Gradually migrate models to `ModelPipeline`

### Phase 3: Full Migration (Future)

1. Migrate all models to `ModelPipeline`
2. Replace current evaluators with `ForecastEvaluator`
3. Use `TimeSeriesCrossValidator` for all CV

## Comparison: DS261 vs Refined Approach

### DS261 Structure

```
cv.py:
├── FlightDelayDataLoader    → TimeSeriesDataLoader
├── FlightDelayEvaluator     → ForecastEvaluator  
└── FlightDelayCV            → TimeSeriesCrossValidator

split.py:
└── create_sliding_window_folds() → TimeSeriesDataLoader.create_temporal_folds()
```

### Key Differences

1. **Time-series specific**: Handles datetime indexes, horizons
2. **Forecasting metrics**: MAE, RMSE, directional accuracy
3. **Model interface**: `ModelPipeline` for standardized models
4. **Walk-forward support**: Both sliding window and expanding window

## Next Steps

1. **Test with existing models**: Wrap a few models and run CV
2. **Compare results**: Ensure metrics match current implementation
3. **Extend as needed**: Add model-specific pipelines (SARIMAX, XGBoost, etc.)
4. **Integrate with train-once pattern**: Add model persistence support

## Questions?

- Does this structure address your Databricks training/backtesting issues?
- What additional features do you need for your workflow?
- Should we migrate specific models first as a proof of concept?


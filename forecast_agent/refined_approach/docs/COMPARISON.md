# Comparison: Current vs Refined Approach

## Quick Reference

| Aspect | Current Approach | Refined Approach |
|--------|-----------------|------------------|
| **Files** | 20+ files, complex structure | 4 core files |
| **Data Loading** | Scattered across `data_loader.py`, `train_models.py`, `backfill_rolling_window.py` | Single `TimeSeriesDataLoader` |
| **Evaluation** | `backtester.py`, `walk_forward_evaluator.py`, `evaluator.py` | Single `ForecastEvaluator` |
| **Cross-Validation** | Complex logic in multiple places | Single `TimeSeriesCrossValidator` |
| **Model Interface** | `BaseForecaster` (abstract), function-based registry | `ModelPipeline` (simple ABC) |
| **Databricks** | Complex Spark/local handling, connection management | Clean Spark DataFrame handling |
| **Testability** | Hard to test due to dependencies | Each component isolated |

## Code Complexity Comparison

### Loading Data and Creating Folds

**Current Approach:**
```python
# Requires understanding multiple files:
# - ground_truth/core/data_loader.py
# - train_models.py (load_training_data function)
# - backfill_rolling_window.py (different implementation)
# - ground_truth/core/walk_forward_evaluator.py (generate_windows)

from ground_truth.core.data_loader import load_and_prepare
from ground_truth.core.walk_forward_evaluator import WalkForwardEvaluator

# Multiple steps, unclear flow
evaluator = WalkForwardEvaluator(data_df=df, horizon=14, min_train_size=365, step_size=14)
windows = evaluator.generate_windows(n_windows=30)
```

**Refined Approach:**
```python
# Single file, clear interface
from refined_approach.data_loader import TimeSeriesDataLoader

loader = TimeSeriesDataLoader(spark=spark)
df = loader.load_to_pandas(commodity='Coffee', cutoff_date='2024-01-01')
folds = loader.create_walk_forward_folds(df, min_train_size=365, step_size=14, horizon=14)
```

### Running Cross-Validation

**Current Approach:**
```python
# Complex setup with multiple abstractions
from ground_truth.core.walk_forward_evaluator import WalkForwardEvaluator
from ground_truth.models import naive

evaluator = WalkForwardEvaluator(data_df=df, horizon=14, min_train_size=365, step_size=14)
windows = evaluator.generate_windows(n_windows=30)

result = evaluator.evaluate_model_walk_forward(
    model_fn=naive.naive_forecast_with_metadata,
    model_params={'commodity': 'Coffee', 'target': 'close', 'horizon': 14},
    windows=windows,
    target='close'
)

# Metrics scattered in result dict
print(result['aggregate_metrics']['mae_mean'])
```

**Refined Approach:**
```python
# Clean, explicit flow
from refined_approach import TimeSeriesDataLoader, ForecastEvaluator, TimeSeriesCrossValidator
from refined_approach.model_pipeline import NaivePipeline

loader = TimeSeriesDataLoader(spark=spark)
df = loader.load_to_pandas(commodity='Coffee', cutoff_date='2024-01-01')
folds = loader.create_walk_forward_folds(df, min_train_size=365, step_size=14, horizon=14)

evaluator = ForecastEvaluator()
cv = TimeSeriesCrossValidator(loader, evaluator, folds)

model = NaivePipeline()
metrics_df = cv.fit(model_fn=model, model_params={}, target_col='close', horizon=14)

# Clear DataFrame output
print(metrics_df)
```

### Model Implementation

**Current Approach:**
```python
# Multiple patterns to choose from:
# Option 1: BaseForecaster (complex ABC with many methods)
class MyModel(StatisticalForecaster):
    def fit(self, df, target_col='close', exog_features=None, **kwargs):
        # Must implement 3+ abstract methods
        pass
    def forecast(self, horizon=14, exog_future=None, **kwargs):
        pass
    def forecast_with_intervals(self, horizon=14, exog_future=None, confidence_level=0.95, **kwargs):
        pass

# Option 2: Function-based (inconsistent interface)
def my_model_forecast_with_metadata(df_pandas, commodity, target='close', horizon=14, **params):
    # Custom return format
    return {'forecast_df': ..., 'fitted_model': ..., ...}
```

**Refined Approach:**
```python
# Single, simple pattern
from refined_approach.model_pipeline import ModelPipeline

class MyModel(ModelPipeline):
    def fit(self, train_df, target_col='close', **kwargs):
        # Store what you need
        self.last_value = train_df[target_col].iloc[-1]
        return self
    
    def predict(self, horizon=14, **kwargs):
        # Return DataFrame
        return pd.DataFrame({
            'date': forecast_dates,
            'forecast': [self.last_value] * horizon
        })
```

## Databricks Compatibility

### Current Approach

**Issues:**
- Complex connection handling (`reconnect_if_needed`, dual-mode execution)
- Spark vs SQL confusion (`execute_sql` function with multiple paths)
- Hard to debug connection issues

```python
# train_models.py - 400+ lines with complex connection management
connection = sql.connect(...)

# backfill_rolling_window.py - 1000+ lines with:
# - is_databricks() detection
# - execute_sql() dual-mode function
# - reconnect_if_needed() logic
# - Spark DataFrame vs SQL INSERT branches
```

### Refined Approach

**Benefits:**
- Natural Spark DataFrame handling
- Works in both Spark and local modes transparently
- No connection management needed

```python
# Clean Spark usage
loader = TimeSeriesDataLoader(spark=spark)  # Spark or None
df = loader.load_to_pandas(...)  # Handles both modes
```

## Key Improvements

### 1. Separation of Concerns

**Before:**
- `backfill_rolling_window.py` mixes data loading, model training, forecasting, and writing
- `walk_forward_evaluator.py` mixes fold generation and evaluation

**After:**
- `TimeSeriesDataLoader` - only data loading
- `ForecastEvaluator` - only metrics
- `TimeSeriesCrossValidator` - only orchestration

### 2. Model Interface Consistency

**Before:**
- Some models use `BaseForecaster`
- Others use function-based approach
- Inconsistent return formats

**After:**
- All models implement `ModelPipeline`
- Consistent `fit()` and `predict()` interface
- Standard DataFrame return format

### 3. Testability

**Before:**
- Hard to test due to Databricks dependencies
- Complex mocking required
- Integration tests only

**After:**
- Each component can be tested in isolation
- Easy to mock Spark sessions
- Unit tests possible

### 4. Extensibility

**Before:**
- Unclear where to add new metrics
- Multiple places to modify for new models
- Hard to understand what affects what

**After:**
- Add metrics: extend `ForecastEvaluator`
- Add models: implement `ModelPipeline`
- Clear dependency graph

## Migration Example

### Current: Training and Backfilling

```python
# train_models.py - complex setup
python train_models.py \
  --commodity Coffee \
  --models naive xgboost \
  --train-frequency semiannually

# backfill_rolling_window.py - separate script
python backfill_rolling_window.py \
  --commodity Coffee \
  --models naive xgboost \
  --train-frequency semiannually \
  --use-pretrained
```

### Refined: Single Flow

```python
# All in one script with clear separation
from refined_approach import TimeSeriesDataLoader, ForecastEvaluator, TimeSeriesCrossValidator

loader = TimeSeriesDataLoader(spark=spark)
df = loader.load_to_pandas(commodity='Coffee')
folds = loader.create_walk_forward_folds(df)

evaluator = ForecastEvaluator()
cv = TimeSeriesCrossValidator(loader, evaluator, folds)

# Train models
for model_name in ['naive', 'xgboost']:
    model = create_model_from_registry(model_name)
    metrics = cv.fit(model_fn=model, model_params={})
    # Save metrics, persist models, etc.
```

## Recommendation

The refined approach offers:
- ✅ **Simpler structure** - 4 files vs 20+
- ✅ **Better Databricks support** - Clean Spark handling
- ✅ **Easier testing** - Isolated components
- ✅ **Clearer model interface** - Single pattern

**Next Steps:**
1. Test refined approach with existing models
2. Migrate 1-2 models as proof of concept
3. Compare results and performance
4. Gradually migrate remaining models


# Development Guide for AI Assistants

**Purpose**: Help AI assistants work effectively on this project

## First Steps When Starting a Session

1. **Read PREFERENCES.md** - Understand Connor's working style
2. **Read PROJECT_OVERVIEW.md** - Get project context
3. **Read DATA_CONTRACTS.md** - Understand data schemas
4. **Check current directory** - Should be in `ucberkeley-capstone/`

## Working with Data

### Local Testing
```python
# Load local snapshot (468 KB - fast)
df = spark.read.parquet("data/unified_data_snapshot_all.parquet")

# Or with pandas (but prefer PySpark)
import pandas as pd
df = pd.read_parquet("data/unified_data_snapshot_all.parquet")
```

### Databricks Production
```python
# Connor runs this in Databricks
df = spark.table("commodity.silver.unified_data")
```

### Dual-Mode Pattern
```python
def load_data(local_mode=False):
    if local_mode:
        return spark.read.parquet("data/unified_data_snapshot_all.parquet")
    else:
        return spark.table("commodity.silver.unified_data")
```

## Code Organization

### When to Create New Files

**DO create**:
- New model implementations (`forecast_agent/ground_truth/models/`)
- Utility functions (`forecast_agent/ground_truth/core/`)
- Experiment notebooks (`forecast_agent/notebooks/experiments/`)

**DON'T create**:
- Unnecessary markdown files
- Duplicate documentation
- Files that should be combined

### Module Structure

```python
# forecast_agent/ground_truth/models/arima_forecaster.py

from ground_truth.core.base_forecaster import BaseForecaster

class ARIMAForecaster(BaseForecaster):
    """Simple ARIMA baseline model."""

    def __init__(self, hyperparameters):
        self.order = hyperparameters["order"]

    def fit(self, df_spark):
        """Train on PySpark DataFrame."""
        pass

    def predict(self, horizon=14):
        """Generate point forecasts."""
        pass

    def sample(self, n_paths=2000, horizon=14):
        """Generate Monte Carlo paths."""
        pass
```

### Configuration Pattern

```python
# forecast_agent/ground_truth/config/model_registry.py

from ground_truth.core import feature_engineering

MODELS = {
    "model_id": {
        "class": "ModelClassName",
        "hyperparameters": {...},
        "features": ["col1", "col2"],
        "commodity": "Coffee",
        "feature_fn": feature_engineering.function_name,
        "training_mode": "rolling"
    }
}
```

## PySpark Best Practices

### Prefer PySpark over Pandas
```python
# ✅ GOOD - PySpark
df_coffee = df.filter("commodity = 'Coffee'")
df_trading = df.filter("is_trading_day = 1")

# ❌ AVOID - Pandas (unless necessary)
df_coffee = df[df['commodity'] == 'Coffee']  # Not distributed
```

### Aggregations
```python
# ✅ GOOD - PySpark aggregation
from pyspark.sql.functions import avg, col

df_agg = df.groupBy("date", "commodity") \
    .agg(
        avg("temp_c").alias("avg_temp"),
        avg("humidity_pct").alias("avg_humidity")
    )

# Only convert to pandas when necessary (e.g., for statsmodels)
df_pd = df_agg.toPandas()
```

### Window Functions
```python
from pyspark.sql.window import Window
from pyspark.sql.functions import lag, lead

# Rolling features
window = Window.partitionBy("commodity").orderBy("date")
df = df.withColumn("prev_close", lag("close", 1).over(window))
```

## Common Tasks

### Add a New Model

1. Create file: `forecast_agent/ground_truth/models/{model_name}_forecaster.py`
2. Inherit from `BaseForecaster`
3. Implement `fit()`, `predict()`, `sample()`
4. Register in `config/model_registry.py`
5. Test locally with sample data
6. Connor runs in Databricks

### Modify Feature Engineering

1. Edit `forecast_agent/ground_truth/core/feature_engineering.py`
2. Add function with signature: `fn(df_spark, commodity, features, cutoff_date=None)`
3. Return transformed PySpark DataFrame
4. Update model configs to reference new function

### Run Backtesting

```python
# Parallel backtesting across dates
from pyspark.sql.functions import pandas_udf
from datetime import datetime, timedelta

# Generate cutoff dates
cutoff_dates = pd.date_range('2023-01-01', '2024-10-28', freq='D')

# Distribute via Spark
def backtest_forecast(cutoff_date):
    # Load data up to cutoff
    df = load_data().filter(f"date <= '{cutoff_date}'")
    # Train model
    # Generate forecast
    # Return results
    return results

# Parallel execution
results = spark.createDataFrame([(str(d),) for d in cutoff_dates], ["cutoff_date"]) \
    .rdd.map(lambda row: backtest_forecast(row.cutoff_date)) \
    .collect()
```

## File Naming Conventions

- **Python modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case()`
- **Constants**: `UPPER_CASE`
- **Notebooks**: `Descriptive Name.ipynb`

## Testing Strategy

### Local Testing
```bash
# From repo root
cd forecast_agent
python -c "from ground_truth.models.arima_forecaster import ARIMAForecaster; print('✓ Import works')"
```

### Databricks Testing
Connor will:
1. Upload code to Databricks Repos
2. Run test notebook
3. Share any errors for debugging

## Common Pitfalls to Avoid

1. **Data leakage**: Always ensure `data_cutoff_date < forecast_date`
2. **Pandas on large data**: Use PySpark, only convert for model training
3. **Over-engineering**: Start simple, add complexity when needed
4. **Forgetting trading days**: Filter `is_trading_day = 1` for training
5. **Not partitioning writes**: Always partition by `model_version`, `commodity`

## Debugging Tips

### Check Data Quality
```python
# Quick validation
df.filter("close IS NULL").count()  # Should be 0
df.filter("forecast_date <= data_cutoff_date").count()  # Should be 0
```

### Validate Schemas
```python
# Check schema matches contract
df.printSchema()
df.select("forecast_date", "data_cutoff_date", "forecast_mean").show(5)
```

### Performance Profiling
```python
# Cache frequently accessed data
df.cache()
df.count()  # Trigger caching

# Check query plan
df.explain()
```

## Communication with Connor

### Use File References
```
Found issue in data_loader.py:45 - missing filter for trading days
```

### Show Code Diffs
```python
# Before
df = spark.table("commodity.silver.unified_data")

# After
df = spark.table("commodity.silver.unified_data").filter("is_trading_day = 1")
```

### Propose, Don't Assume
```
"I can implement this two ways:
1. Simple aggregation (faster, less accurate)
2. Weighted by production volume (slower, potentially better)

Which do you prefer?"
```

## Next Steps After Session

When handing off to Connor:
1. Summarize what was done
2. List any decisions made
3. Highlight what needs testing in Databricks
4. Note any blockers or questions
5. Suggest immediate next steps

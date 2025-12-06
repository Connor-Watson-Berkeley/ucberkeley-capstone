# Forecast Agent - Forecasting Patterns

**Owner:** Connor

---

## Before Working

Read [README.md](README.md) → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first

---

## Critical Rules

**1. Data Sources - Use Gold Tables ONLY**
```python
# CORRECT
from ml_lib.cross_validation.data_loader import GoldDataLoader
loader = GoldDataLoader()
df = loader.load(commodity='Coffee')

# WRONG - NEVER query bronze or silver directly
df = spark.table('commodity.bronze.market')      # Has gaps!
df = spark.table('commodity.silver.unified_data') # Deprecated!
```

**Why:** Gold tables have continuous daily coverage (no gaps), 90% fewer rows, forward-filled data

**2. Always Cache After Imputation**
```python
df_imputed = imputer.transform(df_raw)
df_imputed.cache()  # CRITICAL for 2-3x speedup!
df_imputed.count()   # Materialize
```

**3. Package Deployment**
After code changes:
```bash
python infrastructure/databricks/clusters/deploy_package.py
```

**4. Testing**
```bash
pytest tests/
```

---

## Key Patterns

- **Train-once/inference-many** - See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **GoldDataLoader** - Standard data access
- **"Fit many, publish few"** - See [ml_lib/MODEL_SELECTION_STRATEGY.md](ml_lib/MODEL_SELECTION_STRATEGY.md)

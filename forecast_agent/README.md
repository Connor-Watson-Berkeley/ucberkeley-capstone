# Forecast Agent

**Owner**: Connor
**Status**: 🚧 In Progress

## Purpose
Generate probabilistic price forecasts for coffee and sugar futures.

## Architecture

```
ground_truth/              # Python package
├── config/
│   └── model_registry.py  # Model definitions
├── core/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── base_forecaster.py (TODO)
│   └── forecast_writer.py (TODO)
└── models/
    ├── arima_forecaster.py (TONIGHT)
    └── sarimax_forecaster.py

proof_of_concept/          # V1 prototype (54.95% accuracy)
notebooks/experiments/     # Model evaluation
tests/                     # Unit tests
```

## Inputs
- `commodity.silver.unified_data`

## Outputs
- `commodity.silver.point_forecasts` - 14-day ahead forecasts with confidence intervals
- `commodity.silver.distributions` - 2,000 Monte Carlo sample paths

## Quick Start

### Local Development
```python
# Load sample data
import pyspark.sql.SparkSession as SparkSession
spark = SparkSession.builder.appName("test").getOrCreate()
df = spark.read.parquet("../data/unified_data_snapshot_all.parquet")

# Test imports
from ground_truth.core import data_loader
from ground_truth.config.model_registry import MODELS
```

### Databricks
```python
# Load full dataset
df = spark.table("commodity.silver.unified_data")

# Run model
# (See notebooks/experiments/ for examples)
```

## Model Registry

Models defined in `config/model_registry.py`:
- `arima_baseline_v1`: Simple ARIMA on close price
- `sarimax_v1`: V1 champion (54.95% directional accuracy)

## Development Priorities

1. ✅ Repo structure
2. 🚧 Baseline ARIMA (tonight)
3. ⏳ Model bank framework
4. ⏳ Parallel training
5. ⏳ Evaluation framework

See `agent_instructions/` for detailed guidance.

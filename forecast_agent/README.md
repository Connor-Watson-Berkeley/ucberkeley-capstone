# Forecast Agent

Machine learning system for coffee price forecasting with 14-day horizon. Generates probabilistic forecasts (2,000 Monte Carlo paths) and point predictions with uncertainty quantification.

## Quick Start

**Before running any commands, review relevant documentation in [docs/](docs/):**
- New to the system? Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for train-once/inference-many pattern
- Running backfills at scale? See [docs/SPARK_BACKFILL_GUIDE.md](docs/SPARK_BACKFILL_GUIDE.md)
- Looking for general workflow guidelines? See root [../CLAUDE.md](../CLAUDE.md) for AI agent guidance

### Setup

```bash
# Load Databricks credentials from ../infra/.env
cd forecast_agent
set -a && source ../infra/.env && set +a

# Install dependencies
pip install databricks-sql-connector scikit-learn xgboost statsmodels pmdarima prophet pandas numpy
```

### Train Models (Phase 1)

Train models periodically and persist them for reuse:

```bash
python train_models.py \
  --commodity Coffee \
  --models naive xgboost sarimax_auto_weather \
  --train-frequency semiannually
```

Trains models every 6 months and saves to `commodity.forecast.trained_models` table.

### Generate Forecasts (Phase 2)

Load pretrained models and generate forecasts (180x faster):

```bash
python backfill_rolling_window.py \
  --commodity Coffee \
  --models naive xgboost \
  --train-frequency semiannually
```

Auto-resumes from last completed date. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for performance metrics.

### Check Status

```bash
# Coverage across models
python check_backfill_coverage.py --commodity Coffee --models naive xgboost

# Verify specific model
python verify_backfill.py --commodity Coffee --model naive

# Quick evaluation
python quick_eval.py --commodity Coffee --model naive
```

## Architecture Overview

### Train-Once/Inference-Many Pattern

**Problem**: Traditional systems retrain models for every forecast (2,875 trainings for 2018-2024 backfill).

**Solution**: Two-phase architecture:
- **Phase 1**: Train models semiannually (~16 trainings)
- **Phase 2**: Load pretrained models for fast inference (~880x faster data loading)
- **Result**: 24-48 hours → 1-2 hours for full backfill

**Read more**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed implementation patterns and performance metrics.

### Production Model

**SARIMAX+Weather** (`sarimax_auto_weather`)
- MAE: $3.10
- Directional Accuracy from Day 0: 69.5% ± 27.7%
- Features: temp_mean_c, humidity_mean_pct, precipitation_mm
- Horizon: 14 days
- Training: Semiannual

## Data Flow

```
commodity.silver.unified_data (input)
  ↓
Train models → commodity.forecast.trained_models (model storage)
  ↓
Load models → Generate forecasts
  ↓
commodity.forecast.distributions (2,000 Monte Carlo paths)
commodity.forecast.point_forecasts (14-day predictions)
commodity.forecast.forecast_metadata (performance metrics)
```

**Input data**: See `../research_agent/UNIFIED_DATA_ARCHITECTURE.md` for unified_data schema details.

## Model Registry

25+ models in `ground_truth/config/model_registry.py`:
- Baseline: Naive, Random Walk
- Statistical: ARIMA, SARIMAX (with/without weather)
- Machine Learning: XGBoost, Prophet

All models implement train/predict separation for efficient reuse. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for implementation pattern.

## Execution Environments

### Local Development

Good for testing small date ranges:
```bash
python backfill_rolling_window.py --commodity Coffee --models naive \
  --start-date 2024-01-01 --end-date 2024-01-31
```

### Databricks (Recommended for Production)

**CRITICAL**: Always use All-Purpose Clusters for long-running jobs, NOT SQL Warehouses.
- SQL Warehouses: $417 for 20-hour backfill
- All-Purpose Cluster: ~$10-20 for same workload

See "Environment Setup" section below for cluster setup instructions and cost optimization.

### Spark Parallelization (Large Scale)

For 1000+ date backfills: 20-60 minutes vs 10-20 hours local execution.

**Read more**: [docs/SPARK_BACKFILL_GUIDE.md](docs/SPARK_BACKFILL_GUIDE.md)

## Project Structure

```
forecast_agent/
├── README.md                    # This file
├── CLAUDE.md                    # AI assistant guide
├── docs/                        # Detailed documentation
│   ├── ARCHITECTURE.md          # Train-once pattern, performance
│   └── SPARK_BACKFILL_GUIDE.md  # Parallel processing at scale
│
├── ground_truth/                # Production package
│   ├── config/
│   │   └── model_registry.py   # 25 models (single source of truth)
│   ├── core/
│   │   ├── base_forecaster.py
│   │   ├── backtester.py
│   │   ├── data_loader.py
│   │   └── walk_forward_evaluator.py
│   ├── features/               # Feature engineering
│   ├── models/                 # Model implementations
│   └── storage/
│       └── production_writer.py
│
├── utils/
│   ├── model_persistence.py    # save_model(), load_model()
│   └── monte_carlo_simulation.py
│
├── train_models.py             # Phase 1: Train and persist
├── backfill_rolling_window.py  # Phase 2: Fast inference
├── backfill_rolling_window_spark.py  # Spark parallel backfill
├── backfill_actuals.py         # Populate actuals table
├── evaluate_historical_forecasts.py
├── check_backfill_coverage.py
└── databricks_quickstart.py    # Databricks entry point
```

## Common Commands

### Training

```bash
# Semiannual training (recommended for expensive models)
python train_models.py --commodity Coffee --models xgboost sarimax_auto_weather \
  --train-frequency semiannually

# Monthly training (for fast models)
python train_models.py --commodity Coffee --models naive arima_111 \
  --train-frequency monthly
```

### Backfilling

```bash
# Full historical backfill (auto-resumes)
python backfill_rolling_window.py --commodity Coffee --models naive xgboost \
  --train-frequency semiannually

# Date-range backfill
python backfill_rolling_window.py --commodity Coffee --models naive \
  --start-date 2023-01-01 --end-date 2024-01-01 --train-frequency semiannually

# Backfill actuals
python backfill_actuals.py --commodity Coffee --start-date 2018-01-01 --end-date 2025-11-17
```

### Evaluation

```bash
# Historical forecast evaluation
python evaluate_historical_forecasts.py --commodity Coffee --models naive xgboost

# Generate evaluation dashboard
python dashboard_forecast_evaluation.py
```

## Key Metrics

- **MAE** (Mean Absolute Error): Average prediction error in dollars
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **Dir Day0**: Directional accuracy from day 0 (primary trading metric)
  - Measures: Is day i > day 0? (trading signal quality)
- **Dir**: Day-to-day directional accuracy (less useful for trading)

## Critical Notes

### Database Sessions

Databricks SQL Warehouses have 15-minute session timeouts. Scripts handle this automatically:
- Reconnect every 50 forecasts (before batch write)
- Resume mode skips existing forecasts
- Just rerun the same command to continue

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for database reconnection strategy details.

### Column Name Conventions

Weather and feature columns in unified_data:
- `temp_mean_c` (NOT temp_c)
- `humidity_mean_pct` (NOT humidity_pct)
- `vix` (NOT vix_close)

### ARIMA Auto-Tuning

`auto_arima` without exogenous variables selects order (0,1,0) = naive forecast. **Always use exogenous features with SARIMAX models.**

## Output Tables

All tables in `commodity.forecast` schema:

**`distributions`**
- 2,000 Monte Carlo paths per forecast
- Columns: day_1 through day_14, path_id (0-1999)
- Actuals: `model_version='actuals'` and `is_actuals=TRUE` (hybrid convention)

**`point_forecasts`**
- 14-day forecasts with prediction intervals
- Columns: day_1 through day_14, actual_close

**`forecast_metadata`**
- Model performance metrics (MAE, RMSE, Dir Day0)

**`trained_models`**
- Persistent model storage
- Partitioned by (year, month)
- Storage: JSON (<1MB) or S3 (≥1MB)

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for data contracts and actuals storage conventions.

## Environment Setup

### Databricks Credentials
Load credentials from `../infra/.env`:
```bash
set -a && source ../infra/.env && set +a
```

Required environment variables:
- `DATABRICKS_HOST` - Databricks workspace URL
- `DATABRICKS_TOKEN` - Personal access token
- `DATABRICKS_HTTP_PATH` - SQL warehouse path (**Use clusters for long-running jobs!**)

### Python Dependencies
```bash
pip install databricks-sql-connector scikit-learn xgboost statsmodels pmdarima prophet pandas numpy
```

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Train-once/inference-many pattern, model persistence, performance metrics
- **[docs/SPARK_BACKFILL_GUIDE.md](docs/SPARK_BACKFILL_GUIDE.md)** - Parallel processing guide for large-scale backfills
- **`../research_agent/UNIFIED_DATA_ARCHITECTURE.md`** - Input data schema and architecture
- **[../CLAUDE.md](../CLAUDE.md)** - Root AI agent workflow guidelines

# Forecast Agent

**Owner**: Connor Watson
**Status**: Production Ready

Time series forecasting system for commodity price prediction with walk-forward evaluation, multiple model comparison, and production-ready deployment.

## Quick Start

### Databricks
1. Clone repo in Databricks Repos
2. Open `databricks_quickstart.py` notebook
3. Update path in cell 4: `/Workspace/Repos/<YOUR_USERNAME>/ucberkeley-capstone/forecast_agent`
4. Run all cells

The notebook will install dependencies, load data, run forecasts, and write to production tables.

### Local Development
```bash
# Production deployment
python run_production_deployment.py

# Outputs:
# - production_forecasts/point_forecasts.parquet
# - production_forecasts/distributions.parquet
# - trading_agent_forecast.json
```

## Architecture

```
forecast_agent/
├── .gitignore
├── README.md
├── databricks_quickstart.py      # Databricks notebook (start here!)
├── run_production_deployment.py  # Local production script
│
├── ground_truth/                 # Production package
│   ├── config/
│   │   └── model_registry.py    # 25 models (single source of truth)
│   ├── core/
│   │   ├── base_forecaster.py   # Base class for all models
│   │   ├── backtester.py
│   │   ├── data_loader.py
│   │   ├── evaluator.py
│   │   └── walk_forward_evaluator.py
│   ├── features/                # Feature engineering
│   ├── models/                  # ARIMA, SARIMAX, XGBoost, Prophet
│   ├── storage/
│   │   └── production_writer.py # Writes to commodity.silver schema
│   └── testing/
│       └── data_validation.py   # Schema validators
│
└── Local folders (not in git):
    ├── experiments/              # 13 scripts + dashboard code
    ├── dashboards/               # HTML visualizations
    ├── deprecated/               # Historical docs
    └── production_forecasts/     # Generated outputs

Data pipeline (create_gdelt_unified_data.py) located in ../research_agent/
```

## Data Contract

### Input
`commodity.silver.unified_data` - Unified commodity data with weather, GDELT sentiment, VIX, exchange rates

**⚠️ Important:** For details on unified_data architecture (forward-filling, date spine, data sources), see:
- `../research_agent/UNIFIED_DATA_ARCHITECTURE.md` - Complete architecture doc
- Why continuous daily data (including weekends)
- How forward-fill prevents data leakage
- Trading day indicators

### Output
Three tables in `commodity.forecast` schema:
- `point_forecasts` - 14-day forecasts with prediction intervals and actuals
- `distributions` - 2,000 Monte Carlo paths for risk analysis (actuals stored as `model_version='actuals'`)
- `forecast_metadata` - Performance metrics for model comparison and backtesting

### Actuals Storage Convention (Hybrid Approach)

**Ground truth actuals** are stored in the `distributions` table using a **hybrid convention** for backwards compatibility:

**Primary Convention** (use this for new code):
- `model_version = 'actuals'` - Actuals are a special "model" representing perfect hindsight
- Query pattern: `WHERE model_version = 'actuals'`

**Legacy Convention** (kept for compatibility):
- `is_actuals = TRUE` - Boolean flag marking actuals rows
- `path_id = 1` - Single ground truth path (not path_id=0)

**Why Hybrid?**
- **Cleaner Semantics**: "actuals" is just another model (the perfect one) rather than a special flag
- **Backwards Compatibility**: Existing queries using `is_actuals=TRUE` continue to work
- **Migration Path**: New code uses `model_version='actuals'`, old code gradually transitions

**Example: Loading Actuals for Evaluation**
```python
# New convention (recommended)
query = """
SELECT day_1, day_2, ..., day_14
FROM commodity.forecast.distributions
WHERE commodity = 'Coffee'
  AND model_version = 'actuals'
  AND forecast_start_date = '2024-01-15'
"""

# Legacy convention (still works)
query = """
SELECT day_1, day_2, ..., day_14
FROM commodity.forecast.distributions
WHERE commodity = 'Coffee'
  AND is_actuals = TRUE
  AND forecast_start_date = '2024-01-15'
"""
```

**Backfilling Actuals**:
```bash
# Populate actuals from unified_data (handles multiple regions per date)
python backfill_actuals.py --commodity Coffee --start-date 2018-01-01 --end-date 2025-11-17
python backfill_actuals.py --commodity Sugar --start-date 2018-01-01 --end-date 2025-11-17

# Output:
# - Inserts rows with model_version='actuals' and is_actuals=TRUE
# - Sources data from commodity.silver.unified_data (first close price per date)
# - Skips existing actuals (idempotent)
```

**Evaluation Scripts**: All evaluation scripts (`evaluate_historical_forecasts.py`, `quick_eval.py`) use the `model_version='actuals'` convention.

## Production Model

**SARIMAX+Weather** (`sarimax_weather_v1`)
- MAE: $3.10
- Directional Accuracy from Day 0: 69.5% ± 27.7%
- Features: temp_c, humidity_pct, precipitation_mm
- Horizon: 14 days

Evaluated using 30-window walk-forward validation (420 days, non-overlapping).

## Model Training Architecture

### Train-Once/Inference-Many Pattern

All models support **two-phase workflow** for efficient backtesting:

**Phase 1 - Training** (one-time setup):
```bash
python train_models.py --commodity Coffee --train-frequency semiannually --models naive xgboost
```
- Trains N models on fixed training windows
- Persists fitted models to `commodity.forecast.trained_models` table
- Model storage: JSON (small models) or S3 (large models)

**Phase 2 - Inference** (fast backfill):
```bash
python backfill_rolling_window.py --commodity Coffee --models naive xgboost
```
- Loads pre-trained models from database
- Generates ~2,875 forecasts using 16 models (180x faster)
- No retraining required

**Performance Impact**: Semiannual training = ~16 trainings instead of ~2,875 (one per forecast date)

### Model Implementation Pattern

All models implement train/inference separation:

```python
def my_model_train(df_pandas, target='close', **params) -> dict:
    """Train model and return fitted state."""
    # Training logic
    model = fit_model(df_pandas, target, **params)

    return {
        'fitted_model': model,
        'last_date': df_pandas.index[-1],
        'target': target,
        'model_type': 'my_model',
        # ... other metadata
    }

def my_model_predict(fitted_model_dict, horizon=14, **params) -> pd.DataFrame:
    """Generate forecast using fitted model (no training)."""
    model = fitted_model_dict['fitted_model']
    # Inference logic
    return forecast_df

def my_model_forecast_with_metadata(df_pandas, commodity, fitted_model=None, **params) -> dict:
    """Unified interface supporting both train+predict and inference-only."""
    if fitted_model is None:
        # Train mode
        fitted_model = my_model_train(df_pandas, **params)

    # Inference mode (always)
    forecast_df = my_model_predict(fitted_model, **params)

    return {
        'forecast_df': forecast_df,
        'fitted_model': fitted_model,  # Return for persistence
        # ... metadata
    }
```

**Updated Models**: naive, random_walk, arima, sarimax, xgboost, prophet

### Adding New Models

Register in `ground_truth/config/model_registry.py`:

```python
BASELINE_MODELS = {
    'my_new_model': {
        'name': 'My Model',
        'function': my_model.my_model_forecast_with_metadata,
        'params': {
            'target': 'close',
            'exog_features': ['temp_c', 'vix'],
            'horizon': 14
        },
        'description': 'My custom model'
    }
}
```

Implement following the train/predict pattern above.

## Key Features

### Base Forecaster Class
All models inherit from `BaseForecaster` providing:
- Standardized API: `fit()`, `forecast()`, `forecast_with_intervals()`
- Automatic sample path generation for distributions table
- Consistent metadata handling

### Schema Enhancements
- `actual_close` column in point_forecasts (NULL for future dates)
- **Actuals Storage**: Hybrid convention with `model_version='actuals'` (primary) and `is_actuals=TRUE` (legacy)
- `has_data_leakage` flag for data quality validation
- Ground truth stored in distributions table for consistent backtesting
- See "Actuals Storage Convention" section above for usage details

### Data Validators
- `UnifiedDataValidator` - Checks input data for duplicates, nulls, data quality
- `ForecastOutputValidator` - Validates output schema, data leakage, flag consistency

## Critical Findings

### 1. ARIMA(auto) = Naive
auto_arima without exogenous variables selects order (0,1,0), which is mathematically equivalent to naive forecast. Always use exogenous features with SARIMAX.

### 2. VIX and cop_usd Were Unused
- VIX: 75,354 values available but not used initially
- cop_usd: Critical for Colombian trader use case
- Added 6 new models incorporating these features

### 3. Directional Accuracy from Day 0 is Key Metric
Traditional day-to-day directional accuracy is misleading for trading. Use Dir Day0 which measures whether day i > day 0 (trading signal quality).

## Development vs Production

**Production files (in git)**:
- `ground_truth/` package
- `run_production_deployment.py`
- `README.md`

**Development files (not in git)**:
- `experiments/` - 13 experimental scripts + dashboard code
- `dashboards/` - HTML visualizations from walk-forward evaluation
- `deprecated/` - Historical documentation
- `project_overview/` - Project documentation and specifications
- `production_forecasts/` - Generated output tables
- `*.log`, `*.parquet`, `*.json`, `*.html` - Generated files

See `.gitignore` for complete exclusion list.

## Walk-Forward Evaluation

Method: Expanding window
- Initial training: 3 years (1095 days)
- Forecast horizon: 14 days
- Step size: 14 days (non-overlapping)
- Windows: 30 (420 days evaluation period)

Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Dir Day0 (Directional Accuracy from Day 0) - Primary metric for trading
- Dir (Day-to-Day Directional Accuracy)

## Deployment Workflow

```python
# 1. Load data
df = spark.table("commodity.silver.unified_data")

# 2. Train on full history
from ground_truth.storage.production_writer import ProductionForecastWriter
from ground_truth.models.sarimax import sarimax_forecast

writer = ProductionForecastWriter("production_forecasts")
result = sarimax_forecast(df_coffee, commodity='Coffee', horizon=14)

# 3. Write to production tables
writer.write_point_forecasts(
    forecast_df=result['forecast_df'],
    model_version='sarimax_weather_v1',
    commodity='Coffee',
    data_cutoff_date=df.index[-1]
)

# 4. Export for trading agent
writer.export_for_trading_agent(
    commodity='Coffee',
    model_version='sarimax_weather_v1',
    output_path='trading_agent_forecast.json'
)
```

Weekly retraining schedule recommended.

## Testing

```bash
# Validate schema
python experiments/test_schema_updates.py

# Run walk-forward evaluation
python experiments/run_walkforward_comprehensive.py
```

## Documentation

All essential information is in this README. Additional technical details:
- `../project_overview/DATA_CONTRACTS.md` - Schema specifications
- `deprecated/` - Historical development documentation (local only)
- Code comments and docstrings

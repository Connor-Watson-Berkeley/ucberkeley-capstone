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

### Output
Three tables in `commodity.silver` schema:
- `point_forecasts` - 14-day forecasts with prediction intervals and actuals
- `distributions` - 2,000 Monte Carlo paths for risk analysis (path_id=0 is actuals)
- `forecast_actuals` - Realized prices for backtesting

## Production Model

**SARIMAX+Weather** (`sarimax_weather_v1`)
- MAE: $3.10
- Directional Accuracy from Day 0: 69.5% ± 27.7%
- Features: temp_c, humidity_pct, precipitation_mm
- Horizon: 14 days

Evaluated using 30-window walk-forward validation (420 days, non-overlapping).

## Adding New Models

All models defined in single file: `ground_truth/config/model_registry.py`

```python
BASELINE_MODELS = {
    'my_new_model': {
        'name': 'My Model',
        'model_type': 'sarimax',  # or 'xgboost', 'prophet'
        'exog_features': ['temp_c', 'vix', 'cop_usd'],
        'model_params': {'order': (1, 1, 1)},
        'description': 'My custom model'
    }
}
```

Model implementation should inherit from `BaseForecaster`:

```python
from ground_truth.core.base_forecaster import StatisticalForecaster

class MyForecaster(StatisticalForecaster):
    def fit(self, df, target_col='close', exog_features=None, **kwargs):
        # Training logic
        pass

    def forecast(self, horizon=14, exog_future=None, **kwargs):
        # Return dict with 'forecast_df', 'prediction_intervals', 'model_success'
        pass
```

## Key Features

### Base Forecaster Class
All models inherit from `BaseForecaster` providing:
- Standardized API: `fit()`, `forecast()`, `forecast_with_intervals()`
- Automatic sample path generation for distributions table
- Consistent metadata handling

### Schema Enhancements
- `actual_close` column in point_forecasts (NULL for future dates)
- `is_actuals` flag in distributions (True for path_id=0)
- `has_data_leakage` flag for data quality validation
- Actuals stored alongside forecasts for backtesting

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

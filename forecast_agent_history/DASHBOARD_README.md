# Interactive Forecast Dashboard

## Overview

Comprehensive forecasting experiment system with visual analytics dashboard. Train multiple models, compare performance, and view results in an interactive HTML interface.

## Features

### 🎯 8 Baseline Models
1. **Naive** - Last value persistence
2. **Random Walk** - With drift detection
3. **ARIMA(1,1,1)** - Classical time series
4. **SARIMAX(auto)** - Auto-fitted without covariates
5. **SARIMAX+Weather** - With persisted weather projection
6. **SARIMAX+Weather(seasonal)** - With seasonal weather projection
7. **XGBoost** - With engineered features (lags, rolling stats)
8. **XGBoost+Weather** - With weather covariates

### 📊 Visualizations
- **Forecast vs Actuals** - All models overlaid with confidence intervals
- **Performance Comparison** - MAE, RMSE, MAPE, Directional Accuracy
- **Error Distribution** - Box plots and violin plots
- **Feature Importance** - For XGBoost models

### 📈 Performance Metrics
- **MAE** (Mean Absolute Error) - Primary metric
- **RMSE** (Root Mean Squared Error) - Emphasizes large errors
- **MAPE** (Mean Absolute Percentage Error) - Scale-independent
- **Directional Accuracy** - % correct up/down predictions

## Quick Start

### Local Execution

```bash
cd forecast_agent
python run_experiment_with_dashboard.py
```

The dashboard will automatically open in your browser!

### Custom Configuration

```python
from run_experiment_with_dashboard import run_experiment_with_dashboard

# Run specific models only
results = run_experiment_with_dashboard(
    commodity='Coffee',
    cutoff_date='2023-12-31',
    models_to_run=['naive', 'random_walk', 'xgboost', 'xgboost_weather'],
    show_progress=True
)

# Access results
performance_df = results['performance_df']
dashboard_path = results['dashboard_path']
```

### For Sugar

```python
results = run_experiment_with_dashboard(
    commodity='Sugar',
    cutoff_date='2023-12-31'
)
```

## Databricks Deployment

The experiment runner works in both local (pandas) and Databricks (PySpark) environments.

### Option 1: Using Databricks Connect (Local → Databricks)

**Prerequisites:**
- Databricks access token (see `DATABRICKS_ACCESS.md`)
- Configure `~/.databrickscfg`

```python
# Same code works!
python run_experiment_with_dashboard.py
```

### Option 2: Upload to Databricks Workspace

1. **Upload package to Databricks:**
   ```
   Workspace → Upload → ground_truth/ folder
   ```

2. **Create notebook:**
   ```python
   %run ./ground_truth/run_experiment_with_dashboard
   ```

3. **Or use PySpark data loader:**
   ```python
   from ground_truth.core.data_loader import load_and_prepare
   from ground_truth.config.model_registry import BASELINE_MODELS

   # Load from Delta table
   df = load_and_prepare(
       spark,
       commodity='Coffee',
       features=['close', 'temp_c', 'humidity_pct', 'precipitation_mm'],
       aggregation_method='mean',
       cutoff_date='2023-12-31'
   )

   # Train models...
   ```

## Experiment Results (Coffee - 14-day forecast)

**Latest Run:** 2025-10-29 00:32:00

| Model | MAE | RMSE | MAPE | Directional Accuracy |
|-------|-----|------|------|---------------------|
| **RandomWalk** | $3.67 | $4.10 | 2.02% | 46.2% |
| SARIMAX+Weather(seasonal) | $5.01 | $5.60 | 2.75% | 30.8% |
| Naive | $5.04 | $5.65 | 2.77% | 30.8% |
| SARIMAX(auto) | $5.04 | $5.65 | 2.77% | 30.8% |
| SARIMAX+Weather | $5.04 | $5.65 | 2.77% | 30.8% |
| ARIMA(1,1,1) | $5.24 | $5.87 | 2.88% | 23.1% |
| **XGBoost+Weather** | $6.43 | $7.08 | 3.53% | **61.5%** ⭐ |
| XGBoost | $7.94 | $8.61 | 4.35% | 46.2% |

### Key Insights

1. **Random Walk wins on MAE** - Capturing the -$0.19/day drift is critical
2. **XGBoost+Weather wins on directional accuracy (61.5%)** - Best for trading signals!
3. **Weather covariates** - Minimal MAE improvement, but boost directional accuracy
4. **Auto-ARIMA** selected (0,1,0) - Validates simple models for this data

### Trading Applications

**Best for:**
- **Price forecasting** → Random Walk (lowest MAE)
- **Trading signals** → XGBoost+Weather (highest directional accuracy)
- **Risk management** → Ensemble of top 3 models

## Output Files

After running an experiment, you'll find:

```
forecast_agent/results/
├── dashboard_Coffee_20251029_003200.html    # Interactive dashboard
├── performance_Coffee_20251029_003200.csv   # Performance metrics
├── forecast_naive_Coffee_*.csv              # Individual forecasts
├── forecast_random_walk_Coffee_*.csv
├── forecast_xgboost_Coffee_*.csv
├── ...
└── plots/
    ├── forecast_vs_actual.png
    ├── performance_comparison.png
    └── error_distribution.png
```

## Dashboard Components

### 1. Header
- Experiment metadata
- Commodity, timestamp, horizon

### 2. Best Model Card
- Highlighted best performer
- Key metrics at a glance

### 3. Performance Table
- All models ranked by MAE
- Sortable, interactive

### 4. Forecast Visualization
- All models overlaid on actuals
- Confidence intervals (80%, 95%)
- Zoomable, interactive

### 5. Performance Charts
- 4-panel comparison (MAE, RMSE, MAPE, Directional Accuracy)
- Easy visual comparison

### 6. Error Distribution
- Box plots and violin plots
- Understand model behavior

### 7. Feature Importance (XGBoost only)
- Which features matter most
- Lag vs rolling stats vs weather

## Customization

### Add New Models

1. **Create model file:**
   ```python
   # ground_truth/models/my_model.py
   def my_model_forecast_with_metadata(df_pandas, commodity, **kwargs):
       # Your implementation
       return {
           'forecast_df': forecast_df,
           'model_name': 'MyModel',
           # ... metadata
       }
   ```

2. **Register in model_registry.py:**
   ```python
   'my_model': {
       'name': 'MyModel',
       'function': my_model.my_model_forecast_with_metadata,
       'params': {...},
       'description': 'Description'
   }
   ```

3. **Run experiment:**
   ```python
   results = run_experiment_with_dashboard(
       models_to_run=['my_model', 'naive', 'random_walk']
   )
   ```

### Custom Preprocessing

Modify feature engineering in `ground_truth/features/`:
- `aggregators.py` - Regional aggregation strategies
- `covariate_projection.py` - Weather projection methods
- `transformers.py` - Lag, diff, rolling stats

### Custom Metrics

Add to `ground_truth/core/evaluator.py`:
```python
def my_custom_metric(actuals, forecasts):
    # Your metric
    return metric_value
```

## Troubleshooting

### Dashboard not opening automatically
```bash
# Manually open in browser
open results/dashboard_Coffee_*.html

# Or get path from output:
# 🌐 Open dashboard: file:///path/to/dashboard.html
```

### Databricks Connect issues
- See `DATABRICKS_ACCESS.md`
- Tokens disabled → Use Databricks notebook upload

### Memory errors with large datasets
```python
# Reduce data
df_train = df_agg[-365*2:]  # Last 2 years only
```

### Missing dependencies
```bash
pip install pandas numpy matplotlib seaborn xgboost statsmodels pmdarima scipy
```

## Performance Tips

### Faster iteration
```python
# Run fewer models during development
results = run_experiment_with_dashboard(
    models_to_run=['naive', 'random_walk', 'xgboost']
)
```

### Parallel model training
Currently sequential. For production, parallelize with:
```python
from joblib import Parallel, delayed
```

### Reduce auto-ARIMA search space
In `ground_truth/models/sarimax.py`:
```python
auto_model = auto_arima(
    max_p=2, max_q=2,  # Reduce from 3,3
    stepwise=True,
    n_fits=10  # Limit search
)
```

## Next Steps

1. ✅ Baseline models working
2. ✅ Interactive dashboard
3. ⏳ Test in Databricks (pending access token)
4. ⏳ Walk-forward validation (104 windows)
5. ⏳ Add TimesFM, Prophet, LSTM
6. ⏳ GDELT sentiment integration
7. ⏳ Weather API (14-day forecasts)

## Support

**Issues:**
- Local testing: Check `SESSION_SUMMARY.md`
- Databricks: See `DATABRICKS_ACCESS.md`
- Model questions: See `agent_instructions/ARCHITECTURE.md`

**Contact:**
Connor Watson - Forecast Agent Lead

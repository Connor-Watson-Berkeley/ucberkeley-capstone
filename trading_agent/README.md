# Trading Agent

**Status**: ✅ Production Ready
**Last Updated**: 2025-11-10

## Overview

The Trading Agent converts commodity price forecasts into actionable trading recommendations using advanced backtesting and strategy analysis. It supports real-time operational recommendations and multi-model comparison across 15+ forecasting models.

---

## Quick Start

### Daily Recommendations (Operational Use)

Generate trading recommendations using the latest forecasts:

```bash
cd trading_agent
source ../venv/bin/activate

# Single model
python operations/daily_recommendations.py \
  --commodity coffee \
  --model sarimax_auto_weather_v1

# All models
python operations/daily_recommendations.py \
  --commodity coffee \
  --all-models

# With JSON export (for WhatsApp/messaging)
python operations/daily_recommendations.py \
  --commodity coffee \
  --model sarimax_auto_weather_v1 \
  --output-json recommendations.json
```

**See:** [`operations/README.md`](operations/README.md) for complete guide

### Multi-Model Backtest Analysis

Run backtests across all models and commodities:

```bash
# Run the multi-model notebook in Databricks
# File: commodity_prediction_analysis/trading_prediction_analysis_multi_model.py
```

---

## Features

### ✅ Operational Tools
- **Daily Recommendations**: Real-time trading signals from latest forecasts
- **9 Trading Strategies**: 4 baseline + 5 prediction-based
- **WhatsApp Integration**: Structured JSON output for messaging services
- **Multi-Currency Support**: 15+ currencies with automatic FX rates

### ✅ Multi-Model Analysis
- **15+ Models**: 10 Coffee + 5 Sugar real models
- **Synthetic Testing**: 6 accuracy levels (50%, 60%, 70%, 80%, 90%, 100%)
- **Accuracy Threshold Analysis**: Determine minimum accuracy for profitability
- **Statistical Comparisons**: Bootstrap CI, t-tests, p-values

### ✅ Data Integration
- **Unity Catalog**: All data from Databricks (`commodity.forecast.*`)
- **Zero CSV Dependencies**: Complete database integration
- **Real-Time FX Rates**: Automatic currency conversion

---

## Architecture

### Data Sources (Unity Catalog)

| Table | Purpose | Example |
|-------|---------|---------|
| `commodity.forecast.distributions` | Forecast paths + actuals | 2000 paths × 14 days |
| `commodity.bronze.market_data` | Historical prices | Daily OHLCV data |
| `commodity.bronze.fx_rates` | Exchange rates | 15+ currency pairs |

**All queries use Databricks SQL connection** - no local files required.

### Trading Strategies

**Baseline (4):**
1. **Immediate Sale** - Sell at regular intervals
2. **Equal Batches** - Fixed-size periodic sales
3. **Price Threshold** - Sell when price exceeds threshold
4. **Moving Average** - Sell when price above MA

**Prediction-Based (5):**
1. **Consensus** - Follow majority of forecast paths
2. **Expected Value** - Maximize expected returns
3. **Risk-Adjusted** - Balance returns vs uncertainty
4. **Price Threshold Predictive** - Baseline + forecast
5. **Moving Average Predictive** - Baseline + forecast

---

## Directory Structure

```
trading_agent/
├── README.md                          # This file
├── FORECAST_API_GUIDE.md              # Unity Catalog data access guide
│
├── operations/                        # Operational tools
│   ├── README.md                      # Daily recommendations guide
│   ├── daily_recommendations.py       # Generate daily trading signals
│   └── __init__.py
│
├── data_access/                       # Unity Catalog interface
│   ├── forecast_loader.py             # Load predictions & actuals
│   └── __init__.py
│
├── commodity_prediction_analysis/     # Backtest notebooks
│   ├── trading_prediction_analysis_multi_model.py  # All models
│   └── trading_prediction_analysis.py              # Single model
│
├── docs/                              # Technical documentation
│   ├── ACCURACY_THRESHOLD_ANALYSIS.md  # Synthetic model analysis
│   ├── MULTI_MODEL_MODIFICATIONS.md    # Multi-model notebook changes
│   ├── DATA_FORMAT_VERIFICATION.md     # Format compatibility tests
│   └── backtest_results.md             # Historical backtest results
│
└── analysis/                          # Analysis framework
    ├── model_runner.py                # Multi-model orchestration
    └── __init__.py
```

---

## Key Capabilities

### 1. Real-Time Recommendations

Generate daily recommendations with:
- Current market price & 7-day trend
- 14-day forecast range
- Best sale window (3-day window with highest expected prices)
- Financial impact (sell now vs wait)
- Multi-currency pricing

**Output:** Console display + structured JSON

### 2. Multi-Model Backtesting

Test all forecasting models simultaneously:
- **Coffee**: 10 real models + 6 synthetic = 16 total
- **Sugar**: 5 real models + 6 synthetic = 11 total
- **Total**: 27 model/commodity combinations

Compare performance metrics:
- Net earnings, total revenue, transaction costs
- Strategy-specific performance
- Statistical significance tests

### 3. Accuracy Threshold Analysis

Synthetic models answer: *"What forecast accuracy is needed for profitability?"*

**Finding:** 70% directional accuracy is the minimum threshold for predictions to beat baseline strategies.

### 4. WhatsApp/Messaging Integration

Structured JSON output ready for messaging services:
```json
{
  "market": {
    "current_price_usd": 105.50,
    "local_prices": {"COP": 408967.50, "VND": 2473975.00, ...}
  },
  "recommendation": {
    "action": "HOLD",
    "financial_impact": {
      "usd": {"potential_gain": 142.00},
      "local_currency": {"COP": {"potential_gain": 550450.00}}
    }
  }
}
```

**See:** [`operations/README.md`](operations/README.md) for complete integration guide

---

## Documentation

### User Guides
- **[FORECAST_API_GUIDE.md](FORECAST_API_GUIDE.md)** - How to query forecast data from Unity Catalog
- **[operations/README.md](operations/README.md)** - Daily recommendations & WhatsApp integration

### Technical Documentation
- **[docs/ACCURACY_THRESHOLD_ANALYSIS.md](docs/ACCURACY_THRESHOLD_ANALYSIS.md)** - Synthetic prediction methodology
- **[docs/MULTI_MODEL_MODIFICATIONS.md](docs/MULTI_MODEL_MODIFICATIONS.md)** - Multi-model notebook changes
- **[docs/DATA_FORMAT_VERIFICATION.md](docs/DATA_FORMAT_VERIFICATION.md)** - Data format compatibility
- **[docs/backtest_results.md](docs/backtest_results.md)** - Historical backtest results (archive)

### Project Management
- **[../collaboration/TRADING_AGENT_TODO.md](../collaboration/TRADING_AGENT_TODO.md)** - Current status & roadmap

---

## Data Access

All data is accessed via Unity Catalog:

```python
from databricks import sql
from data_access.forecast_loader import (
    get_available_models,
    load_forecast_distributions,
    load_actuals_from_distributions,
    get_exchange_rates
)

# Connect to Databricks
connection = sql.connect(
    server_hostname=DATABRICKS_HOST,
    http_path=DATABRICKS_HTTP_PATH,
    access_token=DATABRICKS_TOKEN
)

# Load forecasts
models = get_available_models('Coffee', connection)
predictions = load_forecast_distributions('Coffee', 'sarimax_auto_weather_v1', connection)
actuals = load_actuals_from_distributions('Coffee', 'sarimax_auto_weather_v1', connection)
fx_rates = get_exchange_rates(connection)
```

**See:** [FORECAST_API_GUIDE.md](FORECAST_API_GUIDE.md) for complete examples

---

## Model Coverage

### Coffee (16 total)
**Real Models (10):**
- sarimax_auto_weather_v1, sarimax_auto_weather_v2, sarimax_manual_weather_v1
- prophet_v1, prophet_v2, xgboost_weather_v1, xgboost_weather_v2
- arima_auto_v1, arima_manual_v1, random_walk_v1

**Synthetic Models (6):**
- synthetic_50pct (random), synthetic_60pct, synthetic_70pct
- synthetic_80pct, synthetic_90pct, synthetic_perfect (oracle)

### Sugar (11 total)
**Real Models (5):**
- sarimax_auto_weather_v1, prophet_v1, xgboost_weather_v1
- arima_auto_v1, random_walk_v1

**Synthetic Models (6):** Same as Coffee

---

## Multi-Currency Support

Automatically supports 15+ currencies from `commodity.bronze.fx_rates`:

**Major Producers:**
- COP (Colombia), VND (Vietnam), BRL (Brazil)
- INR (India), THB (Thailand), IDR (Indonesia)
- ETB (Ethiopia), HNL (Honduras), UGX (Uganda)
- MXN (Mexico), PEN (Peru)

**Major Economies:**
- USD, EUR, GBP, JPY, CNY, AUD, CHF, KRW, ZAR

All prices and financial metrics automatically calculated in all available currencies.

---

## Next Steps

### For Production Deployment
1. **Schedule Daily Runs**: Set up Databricks job to run `daily_recommendations.py`
2. **Messaging Integration**: Implement WhatsApp service using JSON output
3. **Monitoring**: Track recommendation accuracy vs actual outcomes

### For Analysis
1. **Interactive Dashboard**: Build Streamlit/Dash dashboard for multi-model comparison
2. **Model Comparison**: Analyze which models perform best in different market conditions
3. **Strategy Optimization**: Fine-tune strategy parameters based on backtest results

---

## Support

- **Forecast API Issues**: See [FORECAST_API_GUIDE.md](FORECAST_API_GUIDE.md)
- **Operations Guide**: See [operations/README.md](operations/README.md)
- **Project Status**: See [../collaboration/TRADING_AGENT_TODO.md](../collaboration/TRADING_AGENT_TODO.md)
- **Questions**: Create issue in [../collaboration/REQUESTS.md](../collaboration/REQUESTS.md)

---

## Status Summary

✅ **Production Ready**
- Real-time daily recommendations
- Multi-model backtesting
- WhatsApp/messaging integration
- Multi-currency support
- Complete Databricks integration

⏸️ **Pending**
- Interactive dashboard
- Production deployment automation

Last Updated: 2025-11-10 (Session 3)

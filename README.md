# Ground Truth - Commodity Forecasting System

**Team**: Connor Watson, Stuart Holland, Francisco Munoz, Tony Gibbons
**Timeline**: Week 10-11 of 12 (Backtesting & Evaluation Phase)

## Quick Start

```bash
# Project structure
ucberkeley-capstone/
├── agent_instructions/    # AI assistant context (read this first!)
├── research_agent/        # Data pipeline → unified_data (Stuart & Francisco)
├── forecast_agent/        # Time series forecasting (Connor)
├── trading_agent/         # Risk/trading signals (Tony)
├── data/                  # Local snapshots (gitignored)
└── docs/                  # Project documentation
```

## Purpose

AI-driven forecasting for coffee & sugar futures to help Colombian traders optimize harvest sales.

**Key Insight**: Traders care about `Coffee Price (USD) × COP/USD Rate`, not just USD futures.

## Three-Agent System

```
Research → Forecast → Trading
(Stuart & Francisco)  (Connor)   (Tony)
```

- **Research Agent**: Creates `commodity.silver.unified_data` (✅ Complete)
- **Forecast Agent**: Generates forecasts + distributions (🚧 In Progress)
- **Trading Agent**: Risk management + signals (⏳ Waiting)

## Data Contracts

### Input: `commodity.silver.unified_data`
- Grain: (date, commodity, region)
- ~75k rows, 37 columns
- Market data + weather + macro + exchange rates

### Outputs:
- `commodity.silver.point_forecasts` - 14-day forecasts with confidence intervals
- `commodity.silver.distributions` - 2000 Monte Carlo paths for risk analysis
- `commodity.silver.forecast_actuals` - Realized close prices for backtesting

See `agent_instructions/DATA_CONTRACTS.md` for schemas.

## For AI Assistants

**Start here**:
1. Read `agent_instructions/PREFERENCES.md` - Connor's working style
2. Read `agent_instructions/PROJECT_OVERVIEW.md` - Project context
3. Read `agent_instructions/DATA_CONTRACTS.md` - Data schemas
4. Read `agent_instructions/DEVELOPMENT_GUIDE.md` - How to work on this codebase

## Current Focus: Forecast Agent

**Connor's Priorities**:
1. Baseline ARIMA model (tonight)
2. Scalable model bank framework
3. Parallel model training in Databricks
4. Evaluation framework
5. Stable outputs for trading agent

See `forecast_agent/README.md` for details.

## Success Metrics

- Information Ratio > 0.5
- AUC > 0.65 for directional predictions
- Brier Score < 0.20
- Baseline: 54.95% directional accuracy (V1 SARIMAX)

## Tech Stack

- **Platform**: Databricks (PySpark)
- **Storage**: Delta Lake
- **Modeling**: statsmodels, (future: LSTM, transformers)
- **Local Testing**: Parquet snapshots in `data/`

## Documentation

- `docs/COFFEE_FORECASTING_STRATEGIC_ROADMAP.md` - Full project vision
- `docs/Ground Truth Project Plan.pdf` - Original proposal
- `docs/next_steps.md` - Chat summary (from V1 development)

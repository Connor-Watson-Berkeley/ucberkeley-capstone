# Ground Truth - Project Overview

## Mission

**Primary Goal**: Demonstrate data science and ML engineering excellence through a scalable, modular forecasting system.

**Philosophy**: This project is about building a **foundation for future success**, not achieving perfect accuracy with limited data. We're showcasing:
- Scalable architecture that can grow with more/better data
- Modular design for easy model experimentation
- Rigorous evaluation and statistical testing
- Production-ready engineering practices

With better data (14-day weather forecasts, GDELT sentiment), bigger models (transformers, TimesFM), and refined trading strategies, this foundation will deliver strong results. For now, we focus on **demonstrating engineering excellence**.

## Target Use Case: Colombian Coffee Traders

**Problem**: Colombian coffee producers need to know the optimal time to sell their harvest.

**Key Insight**: They care about **local currency value**, not just USD futures price.

```
Harvest Value (COP) = Coffee Futures Price (USD/lb) × COP/USD Rate × Volume
```

**Decision Factors**:
1. Coffee futures close price (previous day) - from commodity markets
2. COP/USD exchange rate - forex markets
3. Weather patterns - affects harvest timing and quality
4. Global supply/demand - price drivers

## Three-Agent Architecture

```
Research Agent (Stuart & Francisco) → Forecast Agent (Connor) → Risk/Trading Agent (Tony)
        ↓                                     ↓                           ↓
   unified_data                       point_forecasts             trading_signals
                                      distributions
                                      forecast_actuals
```

### Agent S - Research Agent (Stuart & Francisco)
- **Input**: Raw market, weather, macro data
- **Output**: `commodity.silver.unified_data`
- **Responsibility**: Data pipeline, feature engineering
- **Status**: ✅ Complete - SQL provided

### Agent T - Forecast Agent (Connor - YOU)
- **Input**: `commodity.silver.unified_data`
- **Output**: `commodity.silver.point_forecasts`, `commodity.silver.distributions`, `commodity.silver.forecast_actuals`
- **Responsibility**: Time series forecasting, model bank, evaluation
- **Status**: 🚧 In progress - V1 prototype done, scaling now

### Agent R - Risk/Trading Agent (Tony)
- **Input**: Forecast distributions + actuals
- **Output**: Trading signals, position sizing
- **Responsibility**: Risk management, trading rules
- **Status**: ⏳ Waiting on stable forecast outputs

## Your Role: Forecast Agent Lead

**Primary**: Build scalable model bank for commodity forecasting
**Timeline**: Week 10-11 of 12 (backtesting & evaluation phase)
**Deadline**: ~4 weeks remaining

### Immediate Deliverables
1. Baseline ARIMA model (tonight)
2. Model bank framework (modular, scalable)
3. Multiple model training (parallel in Databricks)
4. Evaluation framework (compare performance)
5. Stable forecast outputs (for trading agent)

## Success Metrics

- **Information Ratio > 0.5** vs buy-and-hold
- **AUC > 0.65** for directional predictions
- **Brier Score < 0.20** (calibrated probabilities)
- **Directional accuracy**: Baseline 54.95% (from V1 SARIMAX)

## Technical Stack

- **Platform**: Databricks (PySpark)
- **Storage**: Delta Lake tables
- **Modeling**: statsmodels, (future: LSTM, transformers)
- **Orchestration**: Databricks Jobs/Workflows
- **Visualization**: Notebooks, (future: QuickSight)

## Key Constraints

1. **Data contract stability**: point_forecasts and distributions schemas must remain stable
2. **PySpark first**: Minimize pandas, parallelize everything possible
3. **Time pressure**: 4 weeks to deliver working system
4. **Team dependencies**: Can't change unified_data schema unilaterally

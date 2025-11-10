# Trading Agent TODO

Last Updated: 2025-11-10

## In Progress

### Phase 1: Data Source Migration (Priority: HIGH)
- [ ] Migrate trading_prediction_analysis.py to use `commodity.forecast.distributions` table
  - Replace CSV file reads with SQL queries to Unity Catalog
  - Remove hardcoded file paths (`/Volumes/.../distributions/*.csv`)
  - Update data loading functions to query Delta table

### Phase 2: Multi-Model Analysis Loop
- [ ] Implement NESTED loop structure (commodity → model)
  - **CRITICAL:** Must loop over BOTH commodity AND model_version
  - Query available models from `commodity.forecast.distributions` WHERE commodity = ?
  - Coffee: 10 models, Sugar: 5 models = **15 total backtest runs**

- [ ] Create data access functions
  - `get_available_models(commodity, connection)` - returns list of model versions
  - `load_forecast_distributions(commodity, model_version, connection)` - returns prediction matrices

- [ ] Update result storage structure
  - OLD: `results[commodity]`
  - NEW: `results[commodity][model_version]`
  - Store results for all 15 (commodity, model) combinations

- [ ] Update file outputs to include model identifier
  - OLD: `cumulative_returns_coffee.png`
  - NEW: `cumulative_returns_coffee_sarimax_auto_weather_v1.png`

### Phase 3: Interactive Dashboard Development (3-Tab Structure)

**Dashboard Structure:**
- Tab 1: Coffee (model leaderboard + detailed analysis)
- Tab 2: Sugar (model leaderboard + detailed analysis)
- Tab 3: Coffee vs Sugar (cross-commodity comparison)

**Coffee/Sugar Tab Components:**
- [ ] Top: Model comparison leaderboard (all models ranked)
- [ ] Middle: Model selector dropdown
- [ ] Bottom: Detailed analysis with 4 sub-tabs:
  - [ ] Sub-Tab 1: Performance (cumulative returns, timeline, earnings)
  - [ ] Sub-Tab 2: Statistical Analysis (bootstrap CI, t-tests, p-values) ⭐
  - [ ] Sub-Tab 3: Sensitivity Analysis (parameter/cost sensitivity)
  - [ ] Sub-Tab 4: Feature Analysis (importance, correlations)

**Coffee vs Sugar Tab:**
- [ ] Dual model selectors (one per commodity)
- [ ] Side-by-side metrics comparison
- [ ] Cross-commodity charts

**Statistical Analysis Components (ALL PRESERVED):**
- [ ] Bootstrap confidence intervals (1000 iterations)
- [ ] T-tests and p-values
- [ ] Significance stars (*, **, ***)
- [ ] Feature importance (Random Forest)
- [ ] Feature correlation heatmap
- [ ] Parameter sensitivity heatmaps
- [ ] Cost sensitivity analysis

## Completed

(Tracking completed tasks below)

## Pending

- [ ] Testing and validation of migrated code
- [ ] Documentation updates for new data source
- [ ] Performance optimization for multi-model runs
- [ ] Dashboard deployment to Databricks workspace

## Blocked

(Add any blockers or dependencies on Research Agent)

## Notes

- Forecast API Guide: `trading_agent/FORECAST_API_GUIDE.md`
- All example queries validated and working (2025-11-07)
- Forecast data location: `commodity.forecast.distributions`
- See REQUESTS.md for any questions about the data pipeline

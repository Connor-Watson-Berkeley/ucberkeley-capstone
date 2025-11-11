# Trading Agent TODO

Last Updated: 2025-11-10 (Session 2)

## 📊 Progress Summary

**Overall Status: Phase 2 - 75% Complete**

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Data Source Migration | ✅ Complete | 100% |
| Phase 2: Multi-Model Framework | 🔄 In Progress | 75% |
| Phase 3: Interactive Dashboard | ⏸️ Pending | 0% |

**Current Blocker:** Need to choose integration approach for backtest engine (Option A/B/C)

**What's Working:**
- ✅ Unity Catalog connection and data loading (15 models accessible)
- ✅ Nested loop framework for commodity → model iteration
- ✅ Data format 100% compatible with existing backtest engine
- ✅ End-to-end testing successful (4.2s per model)

**What's Missing:**
- ❌ Actual backtest engine integration (currently using mock results)
- ❌ Full 15-model analysis execution
- ❌ Interactive dashboard

**Next Action:** User decision needed on backtest integration approach

---

## ✅ Completed

### Phase 1: Data Source Migration ✅
- [x] Created data access layer (`trading_agent/data_access/`)
  - [x] `forecast_loader.py` with 8 functions for Unity Catalog queries
  - [x] `get_available_models(commodity, connection)` - returns list of model versions
  - [x] `load_forecast_distributions(commodity, model_version, connection)` - loads data
  - [x] `transform_to_prediction_matrices()` - converts to backtest format
  - [x] Verified 100% format compatibility with existing backtest engine
  - [x] Testing: 244,120 rows loaded in 4.2 seconds

- [x] Modified `trading_prediction_analysis.py` to use Unity Catalog
  - [x] Updated `load_prediction_matrices()` function (lines 264-355)
  - [x] Added `model_version` and `connection` parameters
  - [x] Maintains backward compatibility with local files
  - [x] Returns source string: `'UNITY_CATALOG:<model>'`

### Phase 2: Multi-Model Analysis Loop ✅
- [x] Created model runner framework (`trading_agent/analysis/`)
  - [x] `model_runner.py` with nested loop orchestration
  - [x] `run_analysis_for_model()` - single commodity/model combination
  - [x] `run_analysis_for_all_models()` - all models for one commodity
  - [x] `run_analysis_for_all_commodities()` - complete nested loop
  - [x] `compare_model_performance()` - cross-model comparison

- [x] Implemented NESTED loop structure (commodity → model)
  - [x] Queries available models from Unity Catalog
  - [x] Coffee: 10 models, Sugar: 5 models = **15 total runs**
  - [x] Result storage: `results[commodity][model_version]`

- [x] Created orchestration script
  - [x] `run_multi_model_analysis.py` - end-to-end workflow
  - [x] Configurable commodity parameters
  - [x] Model comparison summaries
  - [x] JSON output per model

- [x] End-to-end testing verified
  - [x] Unity Catalog connection working
  - [x] 15 models discovered and queryable
  - [x] Data format 100% compatible
  - [x] Single model test: 4.2s execution, 41 dates, 12K paths

### Phase 2 Status Tracking
- [x] Created `PHASE_2_STATUS.md` with detailed progress
- [x] Updated `.gitignore` for test scripts
- [x] Git commit pushed (7c26665)

## 🔄 In Progress

### Phase 2: Backtest Engine Integration (Priority: HIGH)
- [ ] **DECISION NEEDED:** Choose integration approach
  - Option A: Extract backtest classes to `trading_agent/backtest/` module ⭐ RECOMMENDED
  - Option B: Keep in notebook, use Databricks Jobs API
  - Option C: Create simplified backtest implementation

- [ ] Integrate actual backtest engine (currently using mock results)
  - [ ] Extract or integrate BacktestEngine class
  - [ ] Extract or integrate Strategy classes (9 strategies total)
  - [ ] Extract or integrate statistical analysis (bootstrap CI, t-tests)
  - [ ] Extract or integrate metrics calculation
  - [ ] Replace mock `run_backtest()` in `run_multi_model_analysis.py`

- [ ] Update file outputs to include model identifier
  - OLD: `cumulative_returns_coffee.png`
  - NEW: `cumulative_returns_coffee_sarimax_auto_weather_v1.png`

- [ ] Run full 15-model analysis
  - [ ] Execute for all Coffee models (10 models)
  - [ ] Execute for all Sugar models (5 models)
  - [ ] Estimated time: ~60 seconds per model = 15 minutes total

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

## 📋 Pending (Lower Priority)

- [ ] Performance optimization for multi-model runs
  - Current: 4.2s per model (acceptable)
  - Could add parallel processing if needed

- [ ] Dashboard deployment to Databricks workspace
  - Wait until dashboard is developed

- [ ] Documentation updates
  - Usage guide for `run_multi_model_analysis.py`
  - API documentation for `backtest/` module (after extraction)

## 🚫 Blocked

- **Backtest Integration:** Awaiting decision on approach (Option A/B/C)
  - Cannot proceed with full 15-model run until backtest engine is integrated
  - Cannot start dashboard until results are available

## Notes

- Forecast API Guide: `trading_agent/FORECAST_API_GUIDE.md`
- All example queries validated and working (2025-11-07)
- Forecast data location: `commodity.forecast.distributions`
- See REQUESTS.md for any questions about the data pipeline

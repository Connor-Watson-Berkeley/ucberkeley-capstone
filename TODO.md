# Ground Truth Project TODO

**Last Updated**: 2025-01-11

**Purpose**: Centralized task tracking for all three agents (Research, Forecast, Trading)

---

## ✅ Recently Completed

- [x] Schema migration: `commodity.silver` → `commodity.forecast`
- [x] Backtest evaluation (42 windows)
- [x] Dashboard SQL queries
- [x] Updated FORECAST_API_GUIDE.md with backtest results
- [x] Reorganized .gitignore with clear sections
- [x] Updated documentation philosophy in CLAUDE_GUIDE.md
- [x] Diagnosed Sugar data issue (2025-01-11)
- [x] Created backfill fix documentation and script
- [x] Backfilled Sugar market data (5,470 rows, 2015-2025) ✅
- [x] Attempted weather data backfill (partial - Coffee only)

---

## 🔄 In Progress

None currently

---

## P1 - Critical Priority

### Forecast Agent

- [ ] **Build Databricks evaluation dashboard for model performance**
  - Visualize forecast accuracy across 40 historical windows
  - Compare models (SARIMAX, Prophet, XGBoost, ARIMA, Random Walk)
  - Show MAE/RMSE/MAPE at 1-day, 7-day, 14-day horizons
  - Display prediction interval calibration (95% coverage)

- [ ] **Backfill forecast_metadata with performance metrics from 40 windows**
  - Calculate MAE/RMSE/MAPE for each model × window
  - Track training/inference timing data
  - Store hardware info for reproducibility

### Research Agent

- [ ] **Complete Sugar weather data backfill (PARTIALLY FIXED)**
  - **Status (2025-01-11 22:45)**:
    - Market Data: ✅ Sugar has 5,470 rows (2015-2025) in landing table
    - Weather Data: ❌ Sugar has only 380 rows (10 days) in landing table
    - Unified Data: ❌ Sugar has 380 rows (blocked by missing weather)
  - **Root Cause**: Weather Lambda never ran historical backfill for Sugar regions
  - **Actions Taken**:
    1. ✅ Backfilled market data via `market-data-fetcher` Lambda (HISTORICAL mode)
    2. ⏳ Invoked `weather-data-fetcher` Lambda with 2015-2025 range (async, status unknown)
    3. ⚠️  Weather backfill produced only Coffee data (79,161 rows), no Sugar yet
  - **Next Steps**:
    1. Check weather Lambda logs / S3 for Sugar data
    2. If failed: Try chunked yearly backfill (2015, 2016, ..., 2025)
    3. Once weather data exists: Run `rebuild_all_layers.py`
    4. Expected final result: ~140,000 Sugar rows in unified_data
  - **See**: `/SUGAR_BACKFILL_STATUS.md` for complete analysis

---

## P2 - Important

### Forecast Agent

- [ ] **Upload point_forecasts for 40 historical windows to Databricks**
  - Currently only distributions table is populated
  - Point forecasts needed for time-series charting
  - ~2,100 rows (42 dates × 5 models × 14 days × 1 mean forecast)

- [ ] **Extend pipeline to Sugar commodity (after data validation)**
  - Validate Sugar data availability in commodity.silver tables
  - Run backfill for Sugar: 40 windows × 5 models × 2,000 paths
  - Update FORECAST_API_GUIDE.md with Sugar examples

### All Agents

- [ ] **Design experiment tracking database**
  - Track model experiments: config, performance, artifacts
  - Enable pruning of poor-performing models from registry
  - Maintain experiment history for presentation/thesis
  - **Goal**: Show platform's experimentation & scale capabilities while keeping config clean

---

## P3 - Lower Priority

### Forecast Agent

- [ ] **Create training_infrastructure_experiments table for cost optimization**
  - Track: cluster config, training time, cost per model
  - Compare: local vs Databricks, Spark vs Pandas
  - Goal: Optimize cost/performance trade-off

- [ ] **Benchmark Spark vs Pandas for parallel model training**
  - Test multi-model training on Databricks cluster
  - Compare training time: sequential vs parallel
  - Document recommendations in DESIGN_DECISIONS.md

- [ ] **Test different Databricks cluster configs for training cost/speed**
  - Small cluster (2 workers) vs large (8 workers)
  - Spot instances vs on-demand
  - Auto-scaling policies

### Trading Agent

- [ ] **Build monitoring dashboard for pipeline data freshness**
  - Track: latest forecast_start_date per model
  - Alert if forecast > 24 hours old
  - Show data quality metrics (null rates, coverage)

---

## Blockers / Dependencies

- **Sugar data validation** must complete before pipeline extension (P1)
- **forecast_metadata table** needs population before eval dashboard is fully functional (P1)

---

## Notes

- All forecast code uses `commodity.forecast` schema (migration complete)
- 622,300 distribution rows for 40 historical windows (Coffee only)
- Recommended production model: `sarimax_auto_weather_v1`
- Trading agent interface ready: `FORECAST_API_GUIDE.md`, `forecast_client.py`

---

## Future Enhancements (Post-Capstone)

- Model pruning based on experiment database results
- Real-time alerting for data pipeline failures
- Automated model retraining on new data
- Multi-commodity expansion (Sugar, Corn, Wheat)
- COP/USD forecast integration for full Colombian trader use case

# Trading Agent TODO List

**Last Updated**: 2025-11-01

---

## ✅ Completed

- [x] Fix schema references: databricks_writer.py silver → forecast
- [x] Fix schema references: forecast_client.py silver → forecast
- [x] Fix schema references: FORECAST_API_GUIDE.md all SQL queries
- [x] Fix schema references: validate_forecast_pipeline.py
- [x] Recreate tables in commodity.forecast schema
- [x] Migrate 622K rows from commodity.silver to commodity.forecast
- [x] Run backtest_forecasts.py to evaluate model performance (42 windows)
- [x] Build Databricks evaluation dashboard SQL queries
- [x] Update FORECAST_API_GUIDE.md with backtest results and recommendations

---

## 🔄 In Progress

None currently

---

## P1 - Critical Priority

- [ ] **Build Databricks evaluation dashboard for model performance**
  - Visualize forecast accuracy across 40 historical windows
  - Compare models (SARIMAX, Prophet, XGBoost, ARIMA, Random Walk)
  - Show MAE/RMSE/MAPE at 1-day, 7-day, 14-day horizons
  - Display prediction interval calibration (95% coverage)

- [ ] **Backfill forecast_metadata with performance metrics from 40 windows**
  - Calculate MAE/RMSE/MAPE for each model × window
  - Track training/inference timing data
  - Store hardware info for reproducibility

- [ ] **Implement backtesting functionality for trading agent**
  - Create `backtest_forecasts.py` script
  - Load historical forecasts + actuals from commodity.forecast.distributions
  - Calculate performance metrics across all 42 forecast dates
  - Generate `backtest_results.md` report

---

## P2 - Important

- [ ] **Upload point_forecasts for 40 historical windows to Databricks**
  - Currently only distributions table is populated
  - Point forecasts needed for time-series charting
  - ~2,100 rows (42 dates × 5 models × 14 days × 1 mean forecast)

- [ ] **Extend pipeline to Sugar commodity (currently Coffee-only)**
  - Validate Sugar data availability in commodity.silver tables
  - Run backfill for Sugar: 40 windows × 5 models × 2,000 paths
  - Update FORECAST_API_GUIDE.md with Sugar examples

---

## P3 - Lower Priority

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

- [ ] **Build monitoring dashboard for pipeline data freshness**
  - Track: latest forecast_start_date per model
  - Alert if forecast > 24 hours old
  - Show data quality metrics (null rates, coverage)

- [ ] **Fix Sugar weather Lambda backfill (research_agent repo)**
  - Currently only 8 days of Sugar weather data
  - Should have 3,770 days (2015-2025)
  - Coordinate with research_agent team

---

## Blockers / Dependencies

- **Schema migration must complete** before backtesting can run (ETA: 30-60 min)
- **forecast_metadata table** needs to be populated before eval dashboard is fully functional
- **Sugar data validation** needed before extending pipeline

---

## Notes

- All forecast code now uses `commodity.forecast` schema (migration from `.silver` complete)
- 622,300 distribution rows generated for 40 historical windows (Coffee only)
- Recommended production model: `sarimax_auto_weather_v1`
- Trading agent interface files ready: `FORECAST_API_GUIDE.md`, `forecast_client.py`

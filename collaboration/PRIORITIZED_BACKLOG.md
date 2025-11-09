# Prioritized Backlog

Last Updated: 2025-11-07

## Priority 0: Critical Path (Blocking)

### Research Agent

- [ ] **Fix weather_v2 bronze table creation** (In Progress)
  - Issue: S3 path pattern was incorrect
  - Status: Path fixed, re-running now
  - Blocks: Weather v2 migration, unified_data update

### Trading Agent

- (None currently)

---

## Priority 1: High Priority (Core Features)

### Research Agent

- [ ] **Validate July 2021 frost event capture**
  - Why: Verify corrected coordinates work correctly
  - Dependencies: weather_v2 bronze table
  - Estimated time: 5 minutes

- [ ] **Update unified_data to use weather_v2**
  - Why: Enable models to use corrected weather data
  - Dependencies: weather_v2 bronze table, frost validation
  - Estimated time: 15 minutes
  - Files: `research_agent/sql/create_unified_data.sql`

- [ ] **Train new SARIMAX models with weather_v2**
  - Why: Improved forecast accuracy with correct coordinates
  - Dependencies: unified_data updated
  - Estimated time: 2-4 hours

### Trading Agent

- [ ] **Integrate forecast API into trading strategy**
  - API Guide: `trading_agent/FORECAST_API_GUIDE.md`
  - All queries validated (2025-11-07)
  - Data available: `commodity.forecast.distributions`

- [ ] **Implement backtesting framework**
  - Data available: `commodity.forecast.forecast_actuals`
  - Use for model validation

---

## Priority 2: Medium Priority (Infrastructure)

### Research Agent

- [ ] **Setup Databricks jobs for automated pipeline**
  - Manual via UI (Jobs API requires saved queries)
  - Jobs needed:
    - Daily Bronze Refresh (2 AM)
    - Silver Update (3 AM)
    - Data Quality Validation (4 AM)

- [ ] **Fix GDELT date column type**
  - Currently STRING, should be DATE or TIMESTAMP
  - Low impact (sparse data coverage)

### Trading Agent

- (Add medium priority tasks here)

---

## Priority 3: Nice to Have (Enhancements)

### Research Agent

- [ ] **Evaluate GDELT inclusion in models**
  - Only 32 days of data across 3 years
  - Determine if adds value or should be excluded
  - See: `collaboration/WARNINGS.md`

- [ ] **Add data freshness monitoring**
  - Alert when data is >7 days stale
  - Integrate with Databricks jobs

### Trading Agent

- (Add nice-to-have tasks here)

---

## Stretch Goals

### Research Agent

- [ ] **Add stock price data integration**
  - Source: Yahoo Finance / Alpha Vantage / other
  - Tickers: Related commodity ETFs, mining companies, etc.
  - Purpose: Additional features for trading models
  - Integration: New bronze table + add to unified_data
  - Estimated time: 3-5 hours
  - Notes:
    - Consider which stock tickers are relevant to commodity trading
    - May need rate-limited API calls
    - Historical data availability varies by source

- [ ] **Add sentiment analysis from news sources**
  - Supplement sparse GDELT data
  - Sources: Financial news APIs, Twitter/X, Reddit
  - NLP pipeline for sentiment scoring

- [ ] **Multi-horizon forecasts**
  - Current: 14-day forecasts
  - Stretch: 30-day, 90-day horizons
  - Evaluate model accuracy at longer horizons

### Trading Agent

- [ ] **Multi-commodity portfolio optimization**
  - Use forecast distributions for correlation analysis
  - Optimize across Coffee, Sugar, and future commodities
  - Risk-adjusted portfolio allocation

- (Add other stretch goals here)

---

## Completed

### Research Agent

- [x] Weather backfill v2 - 10+ years historical data (2015-2025)
- [x] Full pipeline validation - 20/20 tests passed
- [x] Forecast API Guide validation - 7/7 tests passed
- [x] New Databricks workspace setup (Unity Catalog)
- [x] Remove hardcoded credentials from git
- [x] Create collaboration folder for team coordination

### Trading Agent

- (Track completed tasks here)

---

## Blocked / On Hold

(No blocked items currently)

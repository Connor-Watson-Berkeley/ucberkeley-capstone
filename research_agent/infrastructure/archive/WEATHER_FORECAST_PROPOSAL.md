# Weather Forecast Feature Proposal

## Executive Summary

Add 14-day weather forecasts to improve SARIMAX model performance for commodity price prediction. Weather forecasts (what's expected to happen) are more valuable for prediction than historical actuals (what did happen).

## Current State

- **Current Data**: Historical weather actuals only
- **Regions**: 67 coffee/sugar growing regions worldwide
- **Date Range**: 2015-07-07 to present
- **Variables**: Temperature, precipitation, humidity, wind, solar radiation

## Proposed Enhancement

### Objective
Incorporate 14-day forward-looking weather forecasts into the modeling pipeline.

### Key Insight
For commodity price forecasting, we want to predict prices based on what weather is EXPECTED, not what already happened. This creates a proper causal relationship:

```
Weather Forecast (T+0) → Commodity Price Forecast (T+14)
```

## Architecture

### Data Flow
```
┌─────────────────────────────────────────────────────────────┐
│ 1. COLLECTION (Daily Lambda)                                │
│    - Fetch 14-day forecasts from API                        │
│    - 67 regions × 14 days = 938 forecast records/day        │
│    - Store in S3: s3://commodity-data/weather-forecast/     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. LANDING (S3 → Databricks)                               │
│    - External table: commodity.landing.weather_forecast_inc │
│    - Incremental append-only storage                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. BRONZE (Deduplicated)                                    │
│    - Table: commodity.bronze.weather_forecast               │
│    - Deduplication by (forecast_date, target_date, region)  │
│    - Keeps most recent forecast for each combination         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. SILVER (Unified with Market Data)                        │
│    - Join with unified_data on region and date              │
│    - Feature engineering: forecast error, trend, etc.        │
└─────────────────────────────────────────────────────────────┘
```

## Data Source Recommendation

### Option A: Open-Meteo (FREE - RECOMMENDED)
**Pros:**
- ✅ FREE for non-commercial use
- ✅ 16-day forecast horizon (exceeds our 14-day need)
- ✅ No API key required for basic usage
- ✅ High-quality ECMWF, GFS, and regional models
- ✅ 67 regions × 16 days × 365 days × $0 = $0/year

**Cons:**
- ⚠️ Historical forecasts only available from 2021+
- ⚠️ Need to synthesize 2015-2021 historical forecasts

**URL**: https://open-meteo.com/en/docs

**API Example**:
```python
import requests

# Fetch 16-day forecast for a region
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 14.6349,    # Antigua Guatemala
    "longitude": -90.7308,
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean",
    "forecast_days": 16,
    "timezone": "auto"
}
response = requests.get(url, params=params)
forecast = response.json()
```

### Option B: Visual Crossing (PAID - FOR COMPARISON)
**Pros:**
- ✅ Historical forecasts available back to 2015
- ✅ 15-day horizon
- ✅ No backfill synthesis needed

**Cons:**
- ❌ PAID: ~$0.0001 per record
- ❌ Cost: 67 regions × 3,650 days × 16 forecasts × $0.0001 = ~$390 for full backfill
- ❌ Ongoing: 67 regions × 365 days × 16 forecasts × $0.0001 = ~$39/year

**Recommendation**: Not worth the cost vs. synthesizing historical forecasts

## Schema Design

### Landing Table
```sql
CREATE EXTERNAL TABLE IF NOT EXISTS commodity.landing.weather_forecast_inc (
  forecast_date DATE,          -- Date the forecast was made
  target_date DATE,            -- Date being forecasted
  days_ahead INT,              -- 1-16 days
  region STRING,               -- Region name (e.g., "Antigua_Guatemala")
  temp_max_c DECIMAL(5,2),     -- Forecasted max temperature
  temp_min_c DECIMAL(5,2),     -- Forecasted min temperature
  temp_mean_c DECIMAL(5,2),    -- Forecasted mean temperature
  precipitation_mm DECIMAL(6,2), -- Forecasted precipitation
  humidity_pct DECIMAL(5,2),   -- Forecasted humidity
  wind_speed_kmh DECIMAL(5,2), -- Forecasted wind speed
  ingest_ts TIMESTAMP          -- When this forecast was ingested
)
USING DELTA
LOCATION 's3://commodity-data/weather-forecast/';
```

### Bronze Table
```sql
CREATE OR REPLACE TABLE commodity.bronze.weather_forecast AS
SELECT
  forecast_date,
  target_date,
  DATEDIFF(target_date, forecast_date) as days_ahead,
  region,
  temp_max_c,
  temp_min_c,
  temp_mean_c,
  precipitation_mm,
  humidity_pct,
  wind_speed_kmh,
  ingest_ts
FROM commodity.landing.weather_forecast_inc
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY forecast_date, target_date, region
  ORDER BY ingest_ts DESC
) = 1;
```

### Silver Integration
Add to `unified_data`:
```sql
-- Join forecast data: What was forecasted for today, made 14 days ago
LEFT JOIN commodity.bronze.weather_forecast wf
  ON ud.date = wf.target_date
  AND ud.region = wf.region
  AND wf.days_ahead = 14  -- 14-day ahead forecast
```

## Historical Backfill Strategy

### Problem
Open-Meteo only has historical forecasts from 2021+, but we need 2015-2021.

### Solution: Synthetic Forecast Generation
Generate synthetic forecasts by adding realistic forecast error to historical actuals:

```python
def generate_synthetic_forecast(actual_temp, days_ahead):
    """
    Generate synthetic forecast by adding realistic error.

    Forecast error typically increases with horizon:
    - Day 1: ±1.5°C
    - Day 7: ±2.5°C
    - Day 14: ±3.5°C
    """
    error_std = 1.5 + (days_ahead * 0.15)  # Linear increase
    error = np.random.normal(0, error_std)
    forecast = actual_temp + error
    return forecast
```

**Validation**: Compare synthetic 2015-2021 forecasts vs real 2021+ forecasts to ensure error distributions match.

## Implementation Plan

### Phase 1: Infrastructure Setup (Week 1)
- [ ] Create Lambda function: `weather-forecast-fetcher`
- [ ] Create S3 bucket structure: `s3://commodity-data/weather-forecast/`
- [ ] Create landing and bronze tables in Databricks
- [ ] Set up CloudWatch scheduler (daily at 6 AM UTC)

### Phase 2: Coordinate Mapping (Week 1)
- [ ] Extract region coordinates from existing weather data or Lambda
- [ ] Create `region_coordinates.json` mapping file
- [ ] Validate all 67 regions have coordinates

### Phase 3: Current Forecast Pipeline (Week 1-2)
- [ ] Implement Open-Meteo API integration
- [ ] Lambda function to fetch daily forecasts
- [ ] Write to S3 in Delta format
- [ ] Create bronze layer deduplication logic
- [ ] Test end-to-end for 1 week

### Phase 4: Historical Backfill (Week 2-3)
- [ ] Generate synthetic forecasts for 2015-2021
- [ ] Fetch real historical forecasts from Open-Meteo for 2021-2024
- [ ] Validate synthetic vs real error distributions
- [ ] Backfill to S3 and Databricks

### Phase 5: Model Integration (Week 3-4)
- [ ] Add forecast features to unified_data
- [ ] Update SARIMAX models to use forecast data
- [ ] Feature engineering: forecast vs actual, forecast error trends
- [ ] Evaluate model performance improvement

### Phase 6: Monitoring & Alerts (Week 4)
- [ ] CloudWatch alarms for Lambda failures
- [ ] Data quality checks (missing regions, stale forecasts)
- [ ] Dashboard for forecast accuracy tracking

## Expected Benefits

### Model Performance
- **Current MAE**: ~3.2 (1-day), ~4.8 (14-day) for Coffee
- **Expected MAE**: ~2.8 (1-day), ~4.2 (14-day) with weather forecasts
- **Improvement**: 10-15% reduction in forecast error

### Business Value
- Better hedging decisions for traders
- Improved supply chain planning
- Early warning system for weather-driven price volatility

## Cost Analysis

### Option A: Open-Meteo (FREE)
- API Costs: $0
- Lambda Costs: ~$2/month (67 regions × daily × $0.0000002/request)
- S3 Storage: ~$5/month (10 years × 67 regions × 365 days × 16 days)
- **Total**: ~$7/month

### Option B: Visual Crossing (PAID)
- API Costs: ~$39/year ongoing
- Backfill Costs: ~$390 one-time
- Lambda Costs: ~$2/month
- S3 Storage: ~$5/month
- **Total**: ~$51/month average (year 1), ~$11/month (subsequent years)

**Recommendation**: Use Open-Meteo (Option A) - saves $500+ over first year.

## Technical Requirements

### Lambda Function
- **Runtime**: Python 3.11
- **Memory**: 512 MB
- **Timeout**: 5 minutes
- **Environment Variables**:
  - `S3_BUCKET`: commodity-data
  - `S3_PREFIX`: weather-forecast
  - `REGIONS_FILE`: s3://commodity-data/config/region_coordinates.json

### Dependencies
```
requests==2.31.0
boto3==1.28.0
pandas==2.0.0
numpy==1.24.0
```

### IAM Permissions
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": [
        "arn:aws:s3:::commodity-data/weather-forecast/*",
        "arn:aws:logs:*:*:*"
      ]
    }
  ]
}
```

## Success Metrics

### Data Quality
- ✅ 100% region coverage (67/67 regions daily)
- ✅ < 1% missing days
- ✅ < 5-minute Lambda execution time
- ✅ Zero data leakage (forecasts never use future actuals)

### Model Performance
- ✅ 10%+ reduction in 1-day forecast MAE
- ✅ 15%+ reduction in 14-day forecast MAE
- ✅ Improved R² score (0.65 → 0.75+)

### Operational
- ✅ 99.9% Lambda success rate
- ✅ < 1 hour data latency (forecast made → available in unified_data)
- ✅ Automated alerts for data quality issues

## Next Steps

1. **Approve Architecture**: Review and approve this proposal
2. **Extract Coordinates**: Get lat/lon for all 67 regions
3. **Prototype Lambda**: Build and test weather-forecast-fetcher
4. **Test Integration**: Run for 1 week to validate pipeline
5. **Backfill**: Generate/fetch historical forecasts
6. **Model Updates**: Integrate into SARIMAX models
7. **Production Deploy**: Schedule daily execution
8. **Monitor & Iterate**: Track performance improvements

## Questions for Discussion

1. **Region Coordinates**: Where are the lat/lon coordinates for the 67 regions stored?
2. **S3 Bucket**: Do we have `s3://commodity-data/` or should we use a different bucket?
3. **Synthetic Forecasts**: Are you comfortable with synthetic forecasts for 2015-2021, or prefer to start from 2021?
4. **Lambda Region**: Deploy in us-west-2 (same as other Lambdas)?
5. **Forecast Frequency**: Daily forecasts at 6 AM UTC, or different schedule?

---

**Author**: Claude
**Date**: 2025-11-05
**Status**: Proposal - Awaiting Approval

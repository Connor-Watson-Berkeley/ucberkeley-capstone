# Future Improvements

**Purpose**: Document enhancements that would improve forecast accuracy and system capabilities.

**Philosophy**: Current focus is demonstrating ML engineering excellence with available data. These improvements show understanding of what would drive real-world success.

---

## Data Enhancements

### 1. 14-Day Weather Forecasts (HIGH IMPACT)

**Current State**: Roll forward last known weather values for 14-day horizon
```python
# Current: persist_last_value()
temp_forecast = [last_temp] * 14  # Unrealistic
```

**Proposed**: Integrate actual weather forecast API

**APIs to Consider**:
- **OpenWeather** (~$200/month for commercial)
- **Weather.gov** (Free, US-focused)
- **Visual Crossing** (~$100/month)
- **Tomorrow.io** (Advanced, ~$500/month)

**Impact**:
- Weather is a **top-3 covariate** in V1 model
- Current projection assumes static weather (unrealistic)
- 14-day forecasts provide actual expected conditions
- Could improve SARIMAX accuracy by 10-15%

**Effort**: Medium
- API integration: ~8 hours
- Caching strategy: ~4 hours
- Testing: ~4 hours
- **Total**: ~2 days

**Implementation**:
```python
# features/covariate_projection.py
def weather_forecast_api(df_spark, features, horizon=14, api="openweather"):
    """Fetch 14-day weather forecast for coffee/sugar regions"""

    regions = df_spark.select("region").distinct().collect()

    forecasts = []
    for region in regions:
        lat, lon = get_region_coords(region)
        forecast = requests.get(
            f"https://api.openweathermap.org/data/2.5/forecast",
            params={"lat": lat, "lon": lon, "appid": API_KEY}
        )
        forecasts.append(parse_weather_forecast(forecast))

    return spark.createDataFrame(forecasts)
```

---

### 2. GDELT News Sentiment Analysis ✨ DATA EXISTS!

**Current State**: GDELT data available in `commodity.bronze.bronze_gkg` but **not yet integrated**

**What is GDELT GKG?**
- Global Knowledge Graph from GDELT
- News sentiment/tone for global events
- Already loaded in Databricks!
- V2TONE: Sentiment score (-100 to +100)
- THEMES: Article topics/keywords

**Use Cases**:
- Frost warnings in Brazil → coffee prices spike
- Trade policy changes → sugar futures react
- Supply chain disruptions → commodity volatility

**Impact**:
- Capture market psychology not in price/weather data
- Early warning signals for price shocks
- Could improve directional accuracy by 5-10%

**Effort**: LOW-MEDIUM (data exists!)
- SQL integration to unified_data: ~4 hours (COMMENTED CODE READY!)
- Testing/validation: ~4 hours
- Backtesting with sentiment features: ~8 hours
- **Total**: ~2 days (vs 1 week if we had to fetch data)

**Implementation** (commented in `research_agent/sql/create_unified_data.sql`):
```sql
-- Already provided! Uncomment these lines:

gdelt_sentiment AS (
  SELECT
    DATE(SQLDATE) as date,
    CASE
      WHEN THEMES LIKE '%COFFEE%' THEN 'Coffee'
      WHEN THEMES LIKE '%SUGAR%' THEN 'Sugar'
    END as commodity,
    AVG(V2TONE) as avg_tone,  -- Sentiment (-100 to +100)
    COUNT(*) as article_count,
    STDDEV(V2TONE) as tone_volatility
  FROM commodity.bronze.bronze_gkg
  WHERE (THEMES LIKE '%COFFEE%' OR THEMES LIKE '%SUGAR%')
  GROUP BY DATE(SQLDATE), commodity
)

-- Then join to unified_data
-- Forward-fill for non-trading days
-- Add: gdelt_avg_tone, gdelt_article_count, gdelt_tone_volatility
```

**New Features Available** (7 total):
- `gdelt_tone`: Overall sentiment (-100 to +100, daily average)
- `gdelt_positive`: Positive sentiment score
- `gdelt_negative`: Negative sentiment score
- `gdelt_polarity`: Sentiment polarity measure
- `gdelt_tone_volatility`: Sentiment disagreement across articles (uncertainty)
- `gdelt_article_count`: News volume (attention/coverage)
- `gdelt_total_words`: Total word count (coverage depth)

**Use Cases by Feature**:
- **gdelt_tone**: Overall market sentiment
- **gdelt_polarity**: Controversy/disagreement (high = conflicting news)
- **gdelt_tone_volatility**: Market uncertainty
- **gdelt_article_count**: Breaking news events (spikes = important)
- **gdelt_negative** + **gdelt_positive**: Separate positive/negative signals

**Ready to use!** Just uncomment SQL in `research_agent/sql/create_unified_data.sql` and coordinate with Stuart/Francisco.

---

### 3. Production Volume Forecasts

**Current State**: Static regional weights (if using weighted aggregation)

**Proposed**: Integrate USDA, CONAB, ICO production forecasts

**Data Sources**:
- **USDA PSD** (Production, Supply, Distribution): Free, monthly
- **CONAB** (Brazil ag data): Free, quarterly
- **ICO** (International Coffee Org): Free, monthly
- **ISO** (International Sugar Org): Free, monthly

**Impact**:
- Weight regional weather by expected production
- Brazil frost matters more if they're main producer that year
- Modest improvement: 2-5% accuracy gain

**Effort**: Low
- API calls / web scraping: ~8 hours
- Data cleaning: ~4 hours
- Integration: ~4 hours
- **Total**: ~2 days

---

### 4. Logistics Data (Port Activity, Shipping)

**Current State**: No logistics data

**Proposed**: MarineTraffic, Cecafé export data

**What it provides**:
- Coffee shipments from Brazil ports
- Sugar exports from India
- Supply chain disruptions

**Impact**: Low-Medium (3-5%)
- Leading indicator of supply
- Catches disruptions before price impact

**Effort**: Medium (~1 week)

---

## Model Enhancements

### 5. Seasonal ARIMA (SARIMA)

**Current**: SARIMAX with `seasonal=False`

**Proposed**: Test seasonal components

```python
# Test seasonal patterns
"sarimax_seasonal_v1": {
    "hyperparameters": {
        "auto_order": True,
        "seasonal": True,  # ← Add this
        "m": 12  # Monthly seasonality
    }
}
```

**Impact**: Could capture coffee harvest cycles
**Effort**: Low (~1 day to test)

---

### 6. LSTM with Regional Pivots

**Current**: Only simple models (ARIMA, SARIMAX)

**Proposed**: LSTM that handles multi-regional weather

```python
# Pivot regions as separate features
df_pivot = pivot_regions_as_features(df)
# Each region becomes its own feature column

# LSTM can learn which regions matter most
# and temporal dependencies
```

**Impact**: Medium (5-10% improvement potential)
**Effort**: Medium (~1 week for implementation + tuning)

---

### 7. TimesFM with Dynamic Covariates

**Current**: N/A

**Proposed**: Google's TimesFM foundation model

**What is TimesFM?**
- Google's time series foundation model
- Pre-trained on diverse datasets
- Can handle covariates dynamically
- May learn better covariate projection than manual strategies

**Impact**: Potentially high (10-20% improvement)
- Foundation models often beat classical approaches
- Better at handling missing/irregular data

**Effort**: Medium-High
- Model integration: ~1 week
- Hyperparameter tuning: ~1 week
- Computational cost (may need GPU)

---

### 8. XGBoost with Engineered Features

**Current**: N/A

**Proposed**: Gradient boosting with lag/diff/rolling features

```python
# Add rich feature set
df = add_lags(df, lags=[1, 7, 14, 30])
df = add_rolling_stats(df, windows=[7, 30, 90])
df = add_differences(df)

# XGBoost on engineered features
```

**Impact**: Medium (5-10%)
**Effort**: Low-Medium (~3-4 days)

---

## Covariate Projection Strategies

### 9. Seasonal Average Projection

**Current**: `persist_last_value()` (copy last known)

**Proposed**: `seasonal_average()` (use historical patterns)

```python
def seasonal_average(df_spark, features, horizon=14, lookback_years=3):
    """
    For each forecast day, use average of that day-of-year over past N years

    E.g., forecasting Jan 15 temp → average of all Jan 15 temps from 2021-2023
    """
```

**Impact**: Low-Medium (2-5%)
- Better than persistence
- Captures seasonality
- Easy to implement

**Effort**: Low (~1 day)

---

### 10. Linear Trend Projection

**Current**: `persist_last_value()`

**Proposed**: Fit trend on recent data, extrapolate

```python
def linear_trend(df_spark, features, horizon=14, lookback_days=30):
    """Fit linear regression on last 30 days, project forward"""
```

**Impact**: Low (1-3%)
- Good for trending variables (e.g., warming temperatures)
- Can extrapolate unrealistically

**Effort**: Low (~1 day)

---

## System Enhancements

### 11. MLflow Integration

**Current**: No experiment tracking

**Proposed**: Log all experiments to MLflow

**Benefits**:
- Track hyperparameters, metrics, artifacts
- Compare models visually
- Model versioning and registry
- Reproducibility

**Effort**: Low (~2 days)

---

### 12. Real-Time Inference API

**Current**: Batch forecasts in Databricks

**Proposed**: REST API for on-demand forecasts

```
GET /forecast?commodity=Coffee&horizon=14
{
  "forecast_date": "2024-10-29",
  "forecast_mean": 167.23,
  "lower_95": 162.43,
  "upper_95": 172.03,
  "model_version": "sarimax_auto_v1"
}
```

**Effort**: Medium (~1 week)

---

### 13. Automated Retraining Pipeline

**Current**: Manual retraining

**Proposed**: Daily scheduled retraining job

**Workflow**:
1. Nightly: Fetch latest data
2. Retrain production model
3. Validate performance (vs last week)
4. If regression detected → alert, don't deploy
5. If improved → deploy new model
6. Write forecasts to Delta tables

**Effort**: Medium (~1 week)

---

## Priority Ranking

| Enhancement | Impact | Effort | ROI | Priority |
|-------------|--------|--------|-----|----------|
| **14-day weather API** | High | Medium | **High** | **1** |
| **GDELT sentiment** ✨ | Medium | **Low-Med** | **High** | **2** |
| **Seasonal average projection** | Medium | Low | **High** | **3** |
| **XGBoost with features** | Medium | Low-Med | **Med-High** | **4** |
| **LSTM regional** | Medium | Medium | **Medium** | **5** |
| **TimesFM** | High | Med-High | **Medium** | **6** |
| **Production forecasts** | Low-Med | Low | **Medium** | **7** |
| **SARIMA seasonal** | Low-Med | Low | **Medium** | **8** |
| **MLflow tracking** | Med | Low | **Medium** | **9** |
| **Linear trend projection** | Low | Low | **Low** | **10** |
| **Real-time API** | Med | Medium | **Low-Med** | **11** |
| **Auto-retraining** | Med | Medium | **Low-Med** | **12** |
| **Logistics data** | Low-Med | Medium | **Low** | **13** |

---

## Experiment Candidates for AI

When running autonomously, AI should prioritize:
1. Testing `seasonal_average()` vs `persist_last_value()`
2. Testing SARIMA seasonal components
3. Comparing different auto_arima hyperparameters (max_p, max_q)
4. Testing XGBoost with basic lag features

Low-hanging fruit that don't require new data sources.

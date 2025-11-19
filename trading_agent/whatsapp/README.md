# WhatsApp Trading Recommendations

Automated daily trading recommendations for Coffee and Sugar commodities, delivered via WhatsApp.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Databricks Unity Catalog                    │
│                                                          │
│  ┌────────────────────┐  ┌─────────────────────────┐   │
│  │ commodity.forecast │  │ commodity.bronze        │   │
│  │ .distributions     │  │ .market_data            │   │
│  │ (2000 MC paths)    │  │ (historical prices)     │   │
│  └────────────────────┘  └─────────────────────────┘   │
│                                                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ SQL queries
                     │
         ┌───────────▼──────────────┐
         │  generate_daily_          │
         │  recommendation.py        │
         │                           │
         │  - Availability-first     │
         │    model selection        │
         │  - Expected Value         │
         │    strategy               │
         │  - WhatsApp formatting    │
         └───────────┬───────────────┘
                     │
                     │ JSON output
                     │
         ┌───────────▼──────────────┐
         │  WhatsApp Business API    │
         │  (future integration)     │
         └───────────────────────────┘
```

## Two-Step Optimization Approach

**Q: "Where are you getting the best model from?"**

**A: We use a two-step optimization:**
1. **Select best forecast model** from `commodity.forecast.forecast_metadata` (forecast accuracy)
2. **Select best trading strategy** for that model from backtesting results

### Step 1: Forecast Model Selection

Query `commodity.forecast.forecast_metadata` for best performing model:

```python
def get_best_model_from_forecast_metadata(connection, commodity, metric='mae_14d'):
    """
    SELECT model_version, AVG(mae_14d) as avg_metric
    FROM commodity.forecast.forecast_metadata
    WHERE commodity = 'Coffee'
      AND mae_14d IS NOT NULL
    GROUP BY model_version
    ORDER BY avg_metric ASC  -- Lower MAE is better
    LIMIT 1
    """
```

**Metrics Available**:
- MAE (Mean Absolute Error): 1d, 7d, 14d horizons
- RMSE (Root Mean Squared Error): 1d, 7d, 14d
- CRPS (Continuous Ranked Probability Score): 1d, 7d, 14d
- Calibration scores and coverage metrics

**Fallback**: If best model has no recent forecast, try other models in order until one is found.

### Step 2: Trading Strategy Selection

Based on `trading_agent/EXECUTION_RESULTS_SUMMARY.md` backtesting results:

```python
def get_best_strategy_for_model(commodity, model_version):
    """
    Coffee: ExpectedValue strategy (proven +3.4% vs baseline)
    Sugar: ImmediateSale strategy (stable commodity, predictions add no value)

    TODO: Query commodity.trading.backtest_results table once created
    """
```

**Current Implementation**: Hardcoded based on backtest findings
- **Coffee**: All 12 models → Expected Value strategy ($751,641 net earnings)
- **Sugar**: All 9 models → Immediate Sale strategy ($50,071 net earnings)

**Future Enhancement**: When model diversity is confirmed (backtesting shows all models produce identical results - under investigation), this function will query a backtesting results table to get strategy performance per model.

## Expected Value Strategy

Uses the strategy proven best in backtesting (+3.4% gain for Coffee).

### Decision Logic

```python
for each future day (1-14):
    expected_price = median(2000 Monte Carlo paths)
    storage_cost = current_price × 0.025% × days_held
    transaction_cost = expected_price × 0.25%

    net_expected_value = expected_price - storage_cost - transaction_cost

if max(net_expected_value) - current_price > $50/ton:
    HOLD (sell on optimal day)
else:
    SELL NOW (expected gain too small)
```

### Parameters

- **Storage cost**: 0.025% of value per day
- **Transaction cost**: 0.25% of sale value
- **Minimum gain threshold**: $50/ton
- **Inventory size**: 50 tons (default)

These parameters are based on the backtesting configuration that achieved +3.4% improvement.

## Data Sources

### 1. commodity.forecast.distributions
- 2000 Monte Carlo simulation paths per forecast
- 14-day horizon
- Updated periodically (sparse data - not all models forecast daily)
- Columns: `day_1` through `day_14` (predicted prices)
- Filter: `is_actuals = FALSE` for forecasts

### 2. commodity.bronze.market_data
- Historical closing prices
- Updated daily
- Used for:
  - Current market price
  - 7-day trend calculation

### 3. commodity.silver.unified_data (future)
- Weather data (temperature, precipitation)
- FX rates (COP, VND, BRL, etc.)
- CFTC sentiment (Net Non-Commercial Position)
- Not currently used in recommendation, but available

## Output Format

### WhatsApp Message
```
────────────────────────────────────────
📊 Current Market
Price: $2.15/kg
7-Day Trend: 📈 +2.3%

🔮 14-Day Forecast
Range: $2.08 - $2.28
Best Sale Window: Days 9-11

📦 Inventory
Stock: 50 tons
Hold Duration: 10 days

💡 Recommendation
✋ HOLD
Expected Gain: $6,250
Sell on: Day 10

Model: sarimax_auto_weather_v1
Forecast Date: 2025-01-14
────────────────────────────────────────
```

### JSON Output
```json
{
  "commodity": "Coffee",
  "generated_at": "2025-01-14T10:30:00",
  "current_price": 2.15,
  "7day_trend": 2.3,
  "forecast_range": [2.08, 2.28],
  "best_window": [9, 11],
  "inventory_tons": 50,
  "hold_duration": 10,
  "strategy_decision": {
    "action": "HOLD",
    "expected_gain_per_ton": 125.00,
    "total_expected_gain": 6250.00,
    "optimal_sale_day": 10,
    "reasoning": "Expected to gain $125/ton by selling on day 10"
  },
  "model_version": "sarimax_auto_weather_v1",
  "forecast_date": "2025-01-14"
}
```

## Usage

### Prerequisites
```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/..."
```

### Run Recommendation Generator
```bash
cd trading_agent/whatsapp

# Generate Coffee recommendation
python generate_daily_recommendation.py --commodity Coffee

# Generate with custom inventory
python generate_daily_recommendation.py \
    --commodity Coffee \
    --inventory-tons 75

# Export to JSON
python generate_daily_recommendation.py \
    --commodity Coffee \
    --output-json coffee_recommendation.json
```

### Output Example
```
============================================================
Generating recommendation for Coffee
============================================================

1. Fetching current market price...
   Current price: $2.15 (as of 2025-01-14)

2. Calculating 7-day trend...
   7-day trend: +2.3%

3. Finding available forecast...
Checking arima_111_v1...
  Found forecast dated 2025-01-14
   Using model: arima_111_v1
   Forecast date: 2025-01-14
   Prediction matrix shape: (2000, 14)

4. Calculating 14-day forecast range...
   Range (10th-90th percentile): $2.08 - $2.28

5. Finding best sale window...
   Best 3-day window: Days 9-11

6. Calculating Expected Value strategy recommendation...
   Decision: HOLD
   Expected to gain $125/ton by selling on day 10

============================================================
WHATSAPP MESSAGE
============================================================
[formatted message shown above]
```

## Implementation Status

### ✅ Completed
- [x] Core recommendation engine (ExpectedValue strategy)
- [x] Databricks data integration (market_data, forecast.distributions)
- [x] WhatsApp message formatting
- [x] AWS Lambda webhook handler with real data
- [x] Twilio WhatsApp integration
- [x] Deployment automation script
- [x] QR code onboarding support

### 📝 Implementation Files
- `generate_daily_recommendation.py` - Core recommendation logic
- `lambda_handler_real.py` - AWS Lambda handler with Databricks queries
- `deploy_lambda.sh` - Automated deployment script
- `WHATSAPP_SETUP.md` - Complete setup guide

### 🔄 Next Steps

#### 1. Multi-User Management
- User database (phone numbers, subscriptions)
- Personalized inventory sizes
- Subscription preferences (daily/weekly)

#### 2. Scheduled Execution
- Deploy as Databricks Job (daily 6am ET)
- Store recommendations in Delta table for audit trail
- Alert on failures or missing data

#### 3. Enhanced Features
- Implement Consensus strategy (currently using ExpectedValue fallback)
- Custom alerts (price thresholds)
- Historical recommendation performance tracking
- A/B testing different strategies

#### 4. Production Migration
- Move from Twilio sandbox to WhatsApp Business API
- Configure message templates (requires approval)
- Implement caching layer (DynamoDB)
- Set up monitoring and alerts

## Cost Estimate

### WhatsApp Business API Pricing
- Conversation-based pricing: $0.005 - $0.02 per conversation
- Template messages: Lower cost tier ($0.005)
- User-initiated messages: Free to respond (24hr window)

### For 100 Users
- Daily message: 100 × 30 days × $0.005 = **$15/month**
- Negligible Databricks query cost (<$1/month)

### For 1,000 Users
- Daily message: 1,000 × 30 days × $0.005 = **$150/month**

## References

- **Backtesting Results**: `trading_agent/EXECUTION_RESULTS_SUMMARY.md`
- **Forecast Data Schema**: `forecast_agent/ground_truth/storage/databricks_writer.py:186`
- **Data Loader**: `trading_agent/data_access/forecast_loader.py`
- **WhatsApp Mockup**: `/Users/markgibbons/Downloads/Whatsapp demo.pdf`

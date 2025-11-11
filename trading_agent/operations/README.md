# Operations Module - Daily Trading Recommendations

This module provides operational tools for generating daily trading recommendations using the latest forecasts.

---

## Quick Start

### Single Model
```bash
cd trading_agent
source ../venv/bin/activate
python operations/daily_recommendations.py --commodity coffee --model sarimax_auto_weather_v1
```

### All Models
```bash
python operations/daily_recommendations.py --commodity coffee --all-models
```

---

## What It Does

1. **Queries Unity Catalog** for the latest prediction
   - Finds most recent `forecast_start_date` for specified model
   - Loads prediction matrix (2000 paths × 14 days)

2. **Loads Current State**
   - Inventory level (placeholder: 35.5 tons)
   - Days since harvest (placeholder: 45 days)
   - Current price (from `commodity.prices.daily`)
   - Price history (last 100 days)

3. **Generates Recommendations** for all 9 strategies:
   - **Baselines (4):** ImmediateSale, EqualBatch, PriceThreshold, MovingAverage
   - **Prediction-based (5):** Consensus, ExpectedValue, RiskAdjusted, PriceThresholdPredictive, MovingAveragePredictive

4. **Displays Actionable Guidance**
   - SELL or HOLD for each strategy
   - Quantity to sell (in tons)
   - Reasoning behind decision
   - Summary statistics

---

## Example Output

```
================================================================================
DAILY TRADING RECOMMENDATIONS
================================================================================
Date: 2025-11-10 14:30:15
Commodity: COFFEE
================================================================================

Connecting to Databricks...
✓ Connected

Loading current state...
✓ Current state loaded
  Inventory: 35.5 tons
  Current Price: $105.50
  Days Since Harvest: 45

Processing model: sarimax_auto_weather_v1

================================================================================
MODEL: sarimax_auto_weather_v1
================================================================================
Latest Prediction:
  Forecast Date: 2025-11-10
  Generated: 2025-11-10 06:00:00
  Simulation Paths: 2000
  Forecast Horizon: 14 days

Recommendations:
Strategy                   Action  Quantity (tons)  Reasoning                         Uses Predictions
Immediate Sale             SELL    8.9              sale_frequency_reached           No
Equal Batches              SELL    12.5             scheduled_batch_sale             No
Price Threshold            SELL    8.9              price_5.2%_above_threshold       No
Moving Average             HOLD    0.0              price_below_30d_ma               No
Consensus                  HOLD    0.0              strong_consensus_72%_conf8%      Yes
Expected Value             SELL    12.5             ev_peaks_day_3_net_benefit_125   Yes
Risk-Adjusted              HOLD    0.0              high_confidence_low_uncertainty  Yes
Price Threshold Predictive SELL    8.9              both_indicators_suggest_sell     Yes
Moving Average Predictive  HOLD    0.0              ma+prediction_both_hold          Yes

📊 Summary: 5/9 strategies recommend SELL
   Total recommended: 50.7 tons (142.8% of inventory)

================================================================================
RECOMMENDATIONS COMPLETE
================================================================================
```

---

## Command-Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--commodity` | Commodity to analyze (required) | `--commodity coffee` |
| `--model` | Specific model to use | `--model sarimax_auto_weather_v1` |
| `--all-models` | Process all available models | `--all-models` |

**Note:** Must specify either `--model` or `--all-models`

---

## How It Works

### 1. Strategy Decision Logic

Each strategy has a `decide()` method:

```python
def decide(self, day, inventory, current_price, price_history, predictions=None):
    """
    Make trading decision for current day.

    Returns:
        {'action': 'SELL' | 'HOLD',
         'amount': float (tons),
         'reason': str}
    """
```

**Baseline strategies** (don't use predictions):
- Check price thresholds, moving averages, scheduled sales
- Make decisions based on current price and history

**Prediction-based strategies** (use predictions):
- Analyze prediction matrix (2000 paths × 14 days)
- Calculate consensus, expected value, risk metrics
- Incorporate predictions into decision logic

### 2. Latest Prediction Query

```python
def get_latest_prediction(commodity, model_version, connection):
    # Query: SELECT MAX(forecast_start_date) WHERE ...
    # Returns: prediction_matrix (np.ndarray), forecast_date, generation_timestamp
```

Queries Unity Catalog for the most recent prediction available for the specified model.

### 3. Current State Loading

```python
def get_current_state(commodity, connection):
    # Returns: {inventory, days_since_harvest, current_price, price_history}
```

**Current implementation:** Uses placeholders for inventory and days_since_harvest

**Production TODO:** Integrate with actual inventory management system

### 4. Recommendation Generation

```python
# For each strategy:
decision = strategy.decide(
    day=days_since_harvest,
    inventory=current_inventory,
    current_price=price,
    price_history=history_df,
    predictions=prediction_matrix  # Only for prediction-based strategies
)
```

---

## Integration Points

### TODO: Inventory System Integration

Replace placeholder values in `get_current_state()`:

```python
def get_current_state(commodity, connection):
    # Current: Placeholder
    inventory = 35.5
    days_since_harvest = 45

    # TODO: Query actual inventory system
    # inventory = query_inventory_db(commodity)
    # days_since_harvest = calculate_from_harvest_date(commodity)

    return {...}
```

### TODO: Price Data Source

Currently queries `commodity.prices.daily` table. Verify this is the correct source.

```python
cursor.execute("""
    SELECT date, price
    FROM commodity.prices.daily
    WHERE commodity = %s
    ORDER BY date DESC
    LIMIT 100
""", (commodity.capitalize(),))
```

---

## Deployment Options

### Option 1: Databricks Job (Recommended)

Schedule as daily job in Databricks:

```python
# Databricks job configuration
{
    "name": "Daily Coffee Recommendations",
    "job_clusters": [...],
    "python_wheel_task": {
        "package_name": "trading_agent",
        "entry_point": "daily_recommendations",
        "parameters": ["--commodity", "coffee", "--all-models"]
    },
    "schedule": {
        "quartz_cron_expression": "0 0 7 * * ?",  # 7 AM daily
        "timezone_id": "America/New_York"
    }
}
```

### Option 2: Command-Line (Ad-hoc)

Run manually when needed:

```bash
python operations/daily_recommendations.py --commodity coffee --model sarimax_auto_weather_v1
```

### Option 3: API Endpoint

Wrap in Flask/FastAPI for web service:

```python
@app.get("/recommendations/{commodity}/{model}")
def get_recommendations(commodity: str, model: str):
    # Call daily_recommendations logic
    # Return JSON
```

---

## Output Interpretation

### Action Types

| Action | Meaning | Typical Quantity |
|--------|---------|------------------|
| `SELL` | Sell inventory today | 5-50% of inventory |
| `HOLD` | Do not sell, wait | 0 tons |

### Reasoning Codes

Common reason strings returned by strategies:

**Baseline reasons:**
- `sale_frequency_reached` - Time for scheduled sale
- `price_X%_above_threshold` - Price exceeds threshold
- `price_below_ma` - Price below moving average
- `no_inventory` - No inventory to sell

**Prediction-based reasons:**
- `strong_consensus_72%_conf8%` - 72% of paths predict increase, 8% uncertainty
- `ev_peaks_day_3_net_benefit_125` - Expected value peaks on day 3, $125 benefit
- `weak_consensus_55%_or_high_unc15%` - Low consensus or high uncertainty
- `bearish_consensus_38%_ret-2%` - 38% bullish (bearish signal)
- `both_indicators_suggest_sell` - Multiple indicators agree

### Confidence Indicators

Higher numbers indicate stronger signal:
- Consensus %: 70%+ is strong, 50-60% is weak
- Uncertainty: <5% is low, >15% is high
- Net benefit: >$100 is significant

---

## Use Cases

### 1. Daily Decision Support

**When:** New forecast becomes available (typically morning)

**Workflow:**
1. Run script with `--all-models`
2. Review recommendations across models
3. Check for consensus (multiple strategies agree)
4. Make informed trading decision

### 2. Model Comparison

**When:** Evaluating which model to trust

**Workflow:**
1. Run for multiple models
2. Compare recommendations
3. Identify which models are most aggressive/conservative
4. Cross-reference with backtest performance

### 3. Strategy Validation

**When:** Testing if strategies work as expected

**Workflow:**
1. Run with current live data
2. Compare baseline vs prediction-based
3. Verify matched pairs behave correctly
4. Validate reasoning aligns with strategy logic

### 4. Operational Monitoring

**When:** Regular health check

**Workflow:**
1. Schedule daily runs
2. Log recommendations
3. Track actual decisions vs recommendations
4. Measure strategy accuracy over time

---

## Troubleshooting

### No predictions found

```
Error: No predictions found for coffee - sarimax_auto_weather_v1
```

**Solution:** Check that predictions exist in Unity Catalog
```sql
SELECT MAX(forecast_start_date), COUNT(*)
FROM commodity.forecast.distributions
WHERE commodity = 'Coffee' AND model_version = 'sarimax_auto_weather_v1'
```

### No price data found

```
⚠️  No price data found, using mock data
```

**Solution:** Verify `commodity.prices.daily` table exists and is populated

### Strategy errors

```
Strategy: Consensus | Action: ERROR | Reasoning: 'NoneType' object has no attribute 'shape'
```

**Solution:** Check that prediction matrix is valid numpy array with shape (n_paths, 14)

---

## Future Enhancements

### 1. Inventory State Persistence

Store state between runs to track actual inventory:

```python
# Save state after each run
state_tracker.save(commodity, date, {
    'inventory': inventory_after_sales,
    'sales_made': total_sold,
    'days_since_harvest': days + 1
})

# Load state at next run
state = state_tracker.load(commodity)
```

### 2. Recommendation History

Track recommendations over time:

```python
# Log each recommendation
recommendation_log.append({
    'date': today,
    'model': model,
    'strategy': strategy_name,
    'recommendation': decision,
    'actual_action': actual_action_taken
})

# Analyze accuracy
accuracy = compare_recommendations_vs_actuals(log)
```

### 3. Consensus Aggregation

Aggregate recommendations across models:

```python
# Find consensus across all models
consensus = aggregate_recommendations(all_model_recommendations)

print("🎯 CONSENSUS RECOMMENDATION:")
print(f"   Action: {consensus['action']}")
print(f"   Confidence: {consensus['agreement_pct']}%")
print(f"   Models agreeing: {consensus['n_models']}/{total_models}")
```

### 4. Email/Slack Notifications

Send daily reports:

```python
# Format as email
email = format_recommendations_email(recommendations)
send_email(to="trader@company.com", subject="Daily Recommendations", body=email)

# Or Slack
slack_message = format_recommendations_slack(recommendations)
send_slack(channel="#trading", message=slack_message)
```

---

## Related Documentation

- **Backtest Analysis:** `../commodity_prediction_analysis/trading_prediction_analysis_multi_model.py`
- **Strategy Implementations:** See Notebook 02 in backtest file
- **Data Access:** `../data_access/forecast_loader.py`
- **Accuracy Analysis:** `../commodity_prediction_analysis/ACCURACY_THRESHOLD_ANALYSIS.md`

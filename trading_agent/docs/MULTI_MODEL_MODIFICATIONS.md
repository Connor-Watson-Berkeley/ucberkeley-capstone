# Multi-Model Notebook Modifications

**File:** `trading_prediction_analysis_multi_model.py`
**Created:** 2025-11-10
**Purpose:** Run backtest analysis for all commodity/model combinations

---

## Changes Made

### 1. **Added Unity Catalog Connection Setup** (Line ~3061)

```python
# Setup Databricks connection for Unity Catalog queries
from databricks import sql
import os

# Get connection details from environment or Databricks secrets
try:
    DATABRICKS_HOST = dbutils.secrets.get(scope="default", key="databricks_host")
    DATABRICKS_TOKEN = dbutils.secrets.get(scope="default", key="databricks_token")
    DATABRICKS_HTTP_PATH = dbutils.secrets.get(scope="default", key="databricks_http_path")
except:
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").replace("https://", "")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
    DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")

db_connection = sql.connect(
    server_hostname=DATABRICKS_HOST,
    http_path=DATABRICKS_HTTP_PATH,
    access_token=DATABRICKS_TOKEN
)
```

**Why:** Connects to Unity Catalog to query available models and load forecast data.

---

### 2. **Added Model Loop Inside Commodity Loop** (Line ~3132)

**Old structure:**
```python
for CURRENT_COMMODITY in COMMODITY_CONFIGS.keys():
    # Load prices
    # Load prediction matrices (from local files)
    # Run backtest
    # Store results
```

**New structure:**
```python
for CURRENT_COMMODITY in COMMODITY_CONFIGS.keys():
    # Load prices (once per commodity)

    # Query available models
    available_models = get_available_models(CURRENT_COMMODITY.capitalize(), db_connection)

    # Loop through models
    for CURRENT_MODEL in available_models:
        # Load prediction matrices from Unity Catalog for this model
        prediction_matrices, predictions_source = load_prediction_matrices(
            CURRENT_COMMODITY,
            model_version=CURRENT_MODEL,
            connection=db_connection
        )

        # Run backtest (all existing analysis code)
        # ...

        # Store results
        all_results[CURRENT_COMMODITY][CURRENT_MODEL] = {...}
```

**Why:** Enables analysis of all 15 commodity/model combinations (10 Coffee + 5 Sugar).

---

### 3. **Modified Result Storage Structure** (Line ~3768)

**Old:**
```python
all_commodity_results[CURRENT_COMMODITY] = {
    'commodity': CURRENT_COMMODITY,
    'results_df': results_df,
    ...
}
```

**New:**
```python
all_results[CURRENT_COMMODITY][CURRENT_MODEL] = {
    'commodity': CURRENT_COMMODITY,
    'model_version': CURRENT_MODEL,
    'results_df': results_df,
    'predictions_source': predictions_source,
    ...
}
```

**Why:** Stores results for each (commodity, model) combination separately.

---

### 4. **Added Model Leaderboards** (Line ~3807)

```python
# Create leaderboard for each commodity
for commodity in all_results.keys():
    model_comparison = []
    for model_version, results in all_results[commodity].items():
        model_comparison.append({
            'Model': model_version,
            'Best Strategy': results['best_overall']['strategy'],
            'Net Earnings': results['best_overall']['net_earnings'],
            'Prediction Advantage ($)': results['earnings_diff'],
            ...
        })

    leaderboard_df = pd.DataFrame(model_comparison)
    leaderboard_df = leaderboard_df.sort_values('Net Earnings', ascending=False)
    print(leaderboard_df)
```

**Why:** Allows comparison of model performance within each commodity.

---

### 5. **Updated Cross-Commodity Comparison** (Line ~3840)

**Old:** Compared Coffee vs Sugar (single result per commodity)
**New:** Compares best models per commodity

```python
# Get best model for each commodity
for commodity in all_results.keys():
    best_model_results = max(
        all_results[commodity].values(),
        key=lambda x: x['best_overall']['net_earnings']
    )
    comparison_data.append({
        'Commodity': commodity.upper(),
        'Best Model': best_model_name,
        'Net Earnings': best_model_results['best_overall']['net_earnings'],
        ...
    })
```

**Why:** Provides high-level summary comparing commodities using their best-performing models.

---

## Data Flow

```
Unity Catalog
    └─> commodity.forecast.distributions (1.6M rows)
         ├─> Coffee: 10 models
         └─> Sugar: 5 models

↓ get_available_models()

Available Models List
    ├─> Coffee: [arima_111_v1, prophet_v1, sarimax_auto_weather_v1, ...]
    └─> Sugar: [arima_111_v1, prophet_v1, sarimax_auto_weather_v1, ...]

↓ load_prediction_matrices(commodity, model_version, connection)

Prediction Matrices per Model
    Dict[pd.Timestamp, np.ndarray]
    shape: (n_paths, 14)

↓ Backtest Engine

Results per (Commodity, Model)
    all_results[commodity][model_version] = {
        'results_df': ...,
        'best_overall': {...},
        'earnings_diff': ...,
        ...
    }

↓ Comparison & Visualization

Model Leaderboards + Cross-Commodity Summary
```

---

## Output Structure

### Results Dictionary

```python
all_results = {
    'coffee': {
        'arima_111_v1': {...},
        'prophet_v1': {...},
        'sarimax_auto_weather_v1': {...},
        ... (10 models total)
    },
    'sugar': {
        'arima_111_v1': {...},
        'prophet_v1': {...},
        ... (5 models total)
    }
}
```

### Leaderboard Output (Per Commodity)

```
COFFEE - MODEL LEADERBOARD
================================================================================
Model                      Best Strategy    Net Earnings    Prediction Advantage ($)
sarimax_auto_weather_v1   Consensus        $15,234.50      $2,450.00
prophet_v1                 Aggregate        $14,890.20      $2,100.50
...
```

### Cross-Commodity Summary

```
Best Model Per Commodity:
Commodity  Best Model                Net Earnings  Prediction Advantage ($)
COFFEE     sarimax_auto_weather_v1   $15,234.50    $2,450.00
SUGAR      prophet_v1                 $12,890.30    $1,850.75
```

---

## Usage Instructions

### Running the Notebook

1. **Set up credentials** (choose one):
   - **Option A:** Databricks Secrets
     ```python
     dbutils.secrets.put(scope="default", key="databricks_host", string_value="...")
     dbutils.secrets.put(scope="default", key="databricks_token", string_value="...")
     dbutils.secrets.put(scope="default", key="databricks_http_path", string_value="...")
     ```

   - **Option B:** Environment Variables
     ```bash
     export DATABRICKS_HOST="dbc-5e4780f4-fcec.cloud.databricks.com"
     export DATABRICKS_TOKEN="dapi..."
     export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/..."
     ```

2. **Run the notebook** in Databricks workspace

3. **Execution time:**
   - ~4-5 seconds per model
   - 15 models total = ~60-75 seconds

4. **Review outputs:**
   - Model leaderboards printed per commodity
   - Cross-commodity summary
   - All charts and statistical analysis per (commodity, model)

### Accessing Results Programmatically

```python
# Get results for specific model
coffee_sarimax_results = all_results['coffee']['sarimax_auto_weather_v1']
print(coffee_sarimax_results['best_overall']['net_earnings'])

# Get best model for Coffee
coffee_models = all_results['coffee']
best_coffee_model = max(
    coffee_models.items(),
    key=lambda x: x[1]['best_overall']['net_earnings']
)
print(f"Best Coffee model: {best_coffee_model[0]}")

# Compare two models
model1 = all_results['coffee']['prophet_v1']
model2 = all_results['coffee']['sarimax_auto_weather_v1']
diff = model2['best_overall']['net_earnings'] - model1['best_overall']['net_earnings']
print(f"SARIMAX outperforms Prophet by: ${diff:,.2f}")
```

---

## Key Differences from Original

| Aspect | Original | Multi-Model |
|--------|----------|-------------|
| **Data Source** | Local pickle files | Unity Catalog |
| **Models per Run** | 1 (hardcoded) | 15 (queried dynamically) |
| **Loop Structure** | `for commodity` | `for commodity → for model` |
| **Results Storage** | `all_commodity_results[commodity]` | `all_results[commodity][model]` |
| **Comparisons** | Coffee vs Sugar | Model leaderboards + best per commodity |
| **Execution Time** | ~5 seconds | ~60-75 seconds (15 models) |

---

## Notes

- **Backward Compatibility:** The original notebook remains unchanged in `trading_prediction_analysis.py`
- **All Analysis Preserved:** Every chart, metric, and statistical test from the original is still performed for each model
- **Ready for Dashboard:** Results structure (`all_results`) is designed for easy integration with Plotly Dash dashboard
- **Connection Management:** Database connection is closed at the end to prevent leaks

---

## Next Steps

1. **Run the notebook** to generate results for all 15 models
2. **Review model leaderboards** to identify best performers
3. **Use results** to build interactive dashboard with:
   - Tab 1: Coffee models (leaderboard + detailed analysis)
   - Tab 2: Sugar models (leaderboard + detailed analysis)
   - Tab 3: Coffee vs Sugar comparison

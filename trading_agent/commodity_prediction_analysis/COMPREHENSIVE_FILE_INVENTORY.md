# Comprehensive File Inventory - Commodity Prediction Analysis

**Created:** 2025-11-24
**Purpose:** Complete inventory of all notebooks, scripts, and outputs for integration with diagnostics
**Status:** Current workflow documentation (before diagnostic integration)

---

## 📁 Directory Structure

```
commodity_prediction_analysis/
├── Setup & Configuration
│   └── 00_setup_and_config.ipynb
│
├── Prediction Generation
│   ├── 01_synthetic_predictions.ipynb
│   ├── 01_synthetic_predictions_calibrated.ipynb
│   ├── 01_synthetic_predictions_v6.ipynb
│   ├── 01_synthetic_predictions_v7.ipynb
│   └── 01_synthetic_predictions_v8.ipynb (CURRENT)
│   └── 02_forecast_predictions.ipynb
│
├── Core Trading System
│   ├── 03_strategy_implementations.ipynb
│   └── 04_backtesting_engine.ipynb
│
├── Analysis & Results
│   ├── 05_strategy_comparison.ipynb
│   ├── 06_statistical_validation.ipynb
│   ├── 07_feature_importance.ipynb
│   ├── 08_sensitivity_analysis.ipynb
│   ├── 09_strategy_results_summary.ipynb
│   └── 10_paired_scenario_analysis.ipynb
│
├── Utilities
│   ├── analyze_validation.py
│   └── diagnostic_forecast_coverage.ipynb
│
├── Documentation
│   ├── DATABRICKS_ACCESS_NOTES.md
│   ├── EXECUTIVE_SUMMARY.md
│   ├── RESULTS_ANALYSIS_CRITICAL_FINDINGS.md
│   └── WORKFLOW_ANALYSIS_AND_FINDINGS.md
│
└── diagnostics/
    └── [See diagnostics/MASTER_DIAGNOSTIC_PLAN.md]
```

---

## 📊 Notebook Inventory with Outputs

### 00_setup_and_config.ipynb

**Purpose:** Central configuration for all notebooks

**What it defines:**
- Commodity configurations (coffee, sugar)
- Harvest schedules and windows
- Cost parameters (storage: 0.025%/day, transaction: 0.25%)
- Strategy parameters (baselines and prediction-based)
- Data paths (Delta tables and Volume files)
- Analysis configuration

**Functions provided:**
- `get_data_paths(commodity, model_version)` - Generate all file/table paths
- `get_model_versions(commodity)` - List available models
- `load_forecast_data()` - Load predictions from Unity Catalog
- `load_actual_prices()` - Load actuals
- `get_harvest_schedule()` - Calculate harvest timing

**Key Constants:**
```python
FORECAST_TABLE = "commodity.forecast.distributions"
OUTPUT_SCHEMA = "commodity.trading_agent"
VOLUME_PATH = "/Volumes/commodity/trading_agent/files"
```

**Data Saved:** None (pure configuration)

**Charts Produced:** None

---

### 01_synthetic_predictions_v8.ipynb ⭐ CURRENT

**Purpose:** Generate synthetic predictions at multiple accuracy levels with correct MAPE targeting

**Versions:**
- v8 (CURRENT): Fixed log-normal centering for accurate MAPE
- v7: Saves to volume for download
- v6: Fixed day alignment (100% = 0% MAPE)
- Earlier: calibrated, original

**What it does:**
1. Loads price data from `commodity.bronze.market`
2. Generates predictions for accuracy levels: 100%, 90%, 80%, 70%, 60%
3. For each accuracy level, creates 2000 runs × 14 horizons
4. Validates predictions against actuals
5. Calculates MAPE, MAE, CRPS, directional accuracy, coverage

**Key Function:**
```python
generate_calibrated_predictions(
    prices_df,
    model_version,  # e.g. "synthetic_acc90"
    target_accuracy=0.90,  # 90% accurate = 10% MAPE
    n_runs=2000,
    n_horizons=14
)
```

**v8 Fix Details:**
- Centers log-normal at `±target_mape` (not 0)
- Keeps run_biases for realistic correlation
- Stores actual `future_date` to avoid calendar misalignment

**Data Saved (Delta Tables):**
- `commodity.trading_agent.predictions_coffee`
- `commodity.trading_agent.predictions_sugar`

**Data Saved (Volume Files):**
- `validation_results_v8.pkl` - Validation metrics for all accuracy levels
- Prediction matrices saved by subsequent notebooks

**Charts Produced:** None (validation only)

**Runtime:** ~10-20 minutes per commodity

---

### 02_forecast_predictions.ipynb

**Purpose:** Load real forecast predictions from forecast_agent

**What it does:**
1. Queries `commodity.forecast.distributions` table
2. Filters for real predictions (is_actuals = false)
3. Processes forecasts into prediction matrices format
4. Saves matrices for use by backtesting

**Data Saved (Volume Files):**
- `prediction_matrices_{commodity}_{model_version}_real.pkl`
  - e.g. `prediction_matrices_coffee_xgboost_weather_v1_real.pkl`

**Charts Produced:** None

---

### 03_strategy_implementations.ipynb

**Purpose:** Defines all 9 trading strategies

**Strategies Implemented:**

**Baselines (4):**
1. **ImmediateSaleStrategy** - Sells weekly in batches
2. **EqualBatchStrategy** - Fixed 25% batches every 30 days
3. **PriceThresholdStrategy** - Sells when price > threshold (5%)
4. **MovingAverageStrategy** - Sells when price > 30-day MA

**Prediction-Based (5):**
5. **ConsensusStrategy** - Sells when 70%+ runs predict profit
6. **ExpectedValueStrategy** - Sells when EV improvement < $50
7. **RiskAdjustedStrategy** - Risk-based batch sizing (8%-40%)
8. **PriceThresholdPredictive** - PT baseline + predictions
9. **MovingAveragePredictive** - MA baseline + predictions

**Key Methods:**
- `decide(day, inventory, current_price, price_history, predictions)` - Make trading decision
- `reset()` - Reset strategy state
- `set_harvest_start(day)` - Set harvest window

**Data Saved:** None (pure implementation)

**Charts Produced:** None

---

### 04_backtesting_engine.ipynb

**Purpose:** Backtest engine that runs strategies

**What it does:**
1. Simulates trading over historical period
2. Tracks inventory, revenue, costs daily
3. Applies harvest schedule
4. Enforces max holding period (365 days)
5. Handles forced liquidation

**Key Class:**
```python
class BacktestEngine:
    def __init__(self, prices, prediction_matrices, commodity_config):
        # Initialize with data and configuration

    def run(self, strategy):
        # Run backtest and return results:
        # - trades: list of all trades
        # - daily_state: day-by-day inventory/costs
        # - metrics: summary statistics
```

**Metrics Calculated:**
- Net earnings (revenue - transaction costs - storage costs)
- Total revenue (without costs)
- Total costs breakdown
- Average sale price
- Number of trades
- Final inventory
- Cumulative P&L over time

**Data Saved:** None (used by downstream notebooks)

**Charts Produced:** None

---

### 05_strategy_comparison.ipynb ⭐ MAIN WORKFLOW

**Purpose:** Run all strategies on all commodities and model versions

**What it does:**
1. Auto-discovers all model versions (synthetic + real)
2. Loads prices and prediction matrices
3. Runs all 9 strategies for each commodity-model combination
4. Compares baseline vs prediction performance
5. Analyzes Risk-Adjusted strategy scenarios
6. Generates comprehensive visualizations

**Loop Structure:**
```python
for commodity in ['coffee', 'sugar']:
    for model_version in auto_discovered_models:
        for strategy in all_9_strategies:
            results = engine.run(strategy)
            # Save and visualize
```

**Data Saved (Delta Tables):**
- `commodity.trading_agent.results_{commodity}_{model}`
  - Contains metrics for all 9 strategies
  - Columns: strategy, net_earnings, total_revenue, total_costs, avg_sale_price, n_trades, type (Baseline/Prediction)

**Data Saved (Volume Files):**
- `results_detailed_{commodity}_{model}.pkl` - Full backtest results
  - Contains: trades list, daily_state DataFrame, all metrics
- `detailed_strategy_results.csv` - All strategies, all commodities, all models
- `cross_model_commodity_summary.csv` - Best strategies only

**Charts Produced (PNG files in Volume):**

1. **Net Earnings Bar Chart**
   - `net_earnings_{commodity}_{model}.png`
   - Horizontal bars for all 9 strategies
   - Baselines in blue, predictions in red
   - Shows dollar values

2. **Trading Timeline**
   - `trading_timeline_{commodity}_{model}.png`
   - Price history with trade markers
   - Marker size = trade amount
   - Color-coded by strategy
   - Shows timing differences between strategies

3. **Total Revenue (Without Costs)**
   - `total_revenue_no_costs_{commodity}_{model}.png`
   - Cumulative revenue over time (no costs deducted)
   - Shows gross selling performance
   - Prediction strategies as solid lines, baselines as dashed

4. **Cumulative Net Revenue**
   - `cumulative_returns_{commodity}_{model}.png`
   - Net earnings over time (with all costs)
   - Shows true profitability path
   - Identifies when strategies separate

5. **Inventory Drawdown**
   - `inventory_drawdown_{commodity}_{model}.png`
   - Inventory levels over time
   - Shows liquidation pace
   - Helps diagnose storage cost issues

6. **Cross-Model/Commodity Advantage**
   - `cross_model_commodity_advantage.png`
   - Bar chart comparing prediction advantage across models
   - Grouped by commodity
   - Shows which models benefit most from predictions

7. **Cross-Model/Commodity Earnings**
   - `cross_model_commodity_earnings.png`
   - Grouped bars for baseline vs prediction earnings
   - All commodities and models

**Console Output:**
- Risk-Adjusted scenario distribution (how many trades in each risk bracket)
- Forced liquidation analysis
- Best baseline, best prediction, overall best strategy
- Prediction advantage ($) and (%)

**Runtime:** ~30-60 minutes for all commodities and models

---

### 06_statistical_validation.ipynb

**Purpose:** Statistical significance testing of strategy performance

**What it does:**
1. Bootstrap resampling (1000 iterations)
2. Paired t-tests (prediction vs baseline)
3. Confidence intervals (95%)
4. Effect size calculations
5. Win/loss analysis

**Statistical Tests:**
- **Paired t-test**: Tests if prediction strategies significantly beat baselines
- **Bootstrap confidence intervals**: Quantifies uncertainty in performance difference
- **Effect size (Cohen's d)**: Measures practical significance
- **Win rate**: % of days prediction outperforms baseline

**Data Saved (Volume Files):**
- `statistical_results_{commodity}_{model}.pkl`
  - Contains: bootstrap samples, p-values, confidence intervals, effect sizes
- `statistical_comparisons_{commodity}_{model}.csv`
  - Summary table of all statistical tests
- `bootstrap_summary_{commodity}_{model}.csv`
  - Bootstrap distribution statistics

**Charts Produced:**

1. **Bootstrap Distribution**
   - Histogram of earnings differences
   - Shows 95% confidence interval
   - Indicates if zero is excluded (significance)

2. **Confidence Intervals Plot**
   - Forest plot style
   - Shows CI for each strategy pair
   - Highlights significant differences

3. **Effect Size Visualization**
   - Bar chart of Cohen's d
   - Shows practical significance
   - Categorized as small/medium/large

**Runtime:** ~5-10 minutes per commodity-model

---

### 07_feature_importance.ipynb

**Purpose:** Analyze which prediction features drive strategy decisions

**What it does:**
1. Extracts features from prediction matrices:
   - Mean predicted price
   - Standard deviation (uncertainty)
   - Upside potential (95th percentile)
   - Downside risk (5th percentile)
   - Prediction spread
2. Correlates features with strategy decisions (SELL vs WAIT)
3. Identifies which features matter most for each strategy

**Analysis Methods:**
- **Correlation analysis**: Which features correlate with sell decisions?
- **Feature distribution**: How do features differ between SELL and WAIT decisions?
- **Strategy-specific importance**: What drives each prediction strategy?

**Data Saved (Volume Files):**
- `feature_analysis_{commodity}_{model}.pkl`
  - Contains: feature matrices, correlations, importance scores

**Charts Produced:**

1. **Feature Correlation Heatmap**
   - Correlations between features and decisions
   - Separate for each strategy

2. **Feature Importance Bar Chart**
   - Ranked by correlation with decisions
   - Shows which features drive trading

3. **Feature Distribution Boxplots**
   - SELL vs WAIT distributions
   - Shows how features differ by decision

**Runtime:** ~5-10 minutes per commodity-model

---

### 08_sensitivity_analysis.ipynb

**Purpose:** Test how robust strategies are to parameter changes

**What it does:**
1. Varies key parameters:
   - Storage costs (±50%)
   - Transaction costs (±50%)
   - Strategy-specific thresholds
2. Runs backtests with each parameter variation
3. Measures impact on net earnings
4. Identifies parameter sensitivities

**Parameters Tested:**
- **Storage cost**: 0.0125% to 0.0375% per day (baseline: 0.025%)
- **Transaction cost**: 0.125% to 0.375% (baseline: 0.25%)
- **Strategy thresholds**: ±30% of baseline values

**Data Saved (Volume Files):**
- `sensitivity_results_{commodity}_{model}.pkl`
  - Contains: parameter sweep results, sensitivity metrics

**Charts Produced:**

1. **Parameter Sensitivity Plot**
   - Line chart showing earnings vs parameter value
   - Separate line for each strategy
   - Shows if strategies are robust or fragile

2. **Sensitivity Heatmap**
   - 2D heatmap: storage cost × transaction cost
   - Color = net earnings
   - Shows interaction effects

3. **Tornado Diagram**
   - Shows which parameters have biggest impact
   - Ranked by earnings variance

**Runtime:** ~15-30 minutes per commodity-model (many parameter combinations)

---

### 09_strategy_results_summary.ipynb

**Purpose:** Generate executive summary of all results

**What it does:**
1. Aggregates results across all commodities and models
2. Ranks strategies by performance
3. Identifies best strategies overall
4. Summarizes key findings
5. Generates presentation-ready tables

**Data Saved (Volume Files):**
- `summary_stats_{commodity}_{model}.csv`
  - High-level summary metrics
- `final_summary_{commodity}_{model}.csv`
  - Complete aggregated results

**Charts Produced:**

1. **Strategy Ranking Table**
   - Net earnings for all strategies
   - Sorted by performance
   - Colored by type (baseline/prediction)

2. **Performance Summary Dashboard**
   - Multi-panel figure
   - Earnings, costs, trade counts
   - Comparative view

3. **Key Findings Infographic**
   - Highlights top performers
   - Shows prediction advantage
   - Executive-friendly format

**Runtime:** ~5 minutes (reads existing results)

---

### 10_paired_scenario_analysis.ipynb

**Purpose:** Deep dive into baseline vs predictive strategy pairs

**What it does:**
1. Compares matched pairs:
   - Price Threshold vs Price Threshold Predictive
   - Moving Average vs Moving Average Predictive
2. Trade-by-trade comparison
3. Decision point analysis
4. Cost attribution

**Analysis:**
- **When do predictions help?** (market conditions, timing)
- **When do predictions hurt?** (overfitting, overtrading)
- **Why do strategies differ?** (decision logic, thresholds)

**Data Saved:** None (analysis only)

**Charts Produced:**

1. **Paired Decision Comparison**
   - Shows where baseline and predictive make different decisions
   - Highlights key divergence points

2. **Trade Timing Comparison**
   - When does each strategy trade?
   - How do timings differ?

3. **Cost Attribution Analysis**
   - Breaks down performance difference
   - Revenue vs transaction costs vs storage costs

**Runtime:** ~10-15 minutes per commodity-model

---

## 🗂️ Utility Files

### analyze_validation.py

**Purpose:** Analyze validation_results_v*.pkl files

**What it does:**
- Loads validation pickle
- Extracts MAPE, MAE, coverage metrics
- Checks if 100% accuracy shows 0% MAPE (bug verification)
- Prints summary table

**Usage:**
```bash
python analyze_validation.py
```

**Output:** Console summary only

---

### diagnostic_forecast_coverage.ipynb

**Purpose:** Check forecast data coverage and quality

**What it does:**
- Queries `commodity.forecast.distributions`
- Checks date ranges
- Identifies gaps
- Validates forecast structure

**Output:** Console diagnostics only

---

### trading_prediction_analysis_original_11_11_25.ipynb

**Purpose:** Original monolithic notebook (historical)

**What it is:**
- 4.4MB notebook from November 11, 2025
- Contains full analysis workflow in single file
- Predates the split into notebooks 00-10
- Kept for historical reference

**Status:** ARCHIVED - Use numbered notebooks (00-10) instead

**Why it exists:**
- Original implementation before workflow was modularized
- Useful for understanding evolution of approach
- May contain experimental code or earlier strategy versions

**Should you use it?** NO - Use the numbered notebooks (00-10) which are:
- Better organized
- More maintainable
- Properly documented
- Actively maintained

**Output:** Unknown (historical - not part of current workflow)

---

## 📄 Documentation Files

### DATABRICKS_ACCESS_NOTES.md ⭐

**Purpose:** How to access Databricks and avoid common mistakes

**Key Topics:**
- How notebooks run (IN Databricks, not locally)
- Volume vs local directory (ephemeral risk)
- Using databricks CLI to download files
- Pandas version compatibility

### EXECUTIVE_SUMMARY.md

**Purpose:** High-level summary of findings (historical)

**Status:** May be outdated - created early in project

### RESULTS_ANALYSIS_CRITICAL_FINDINGS.md

**Purpose:** Key findings from analysis (historical)

**Status:** May be outdated - created during initial analysis

### WORKFLOW_ANALYSIS_AND_FINDINGS.md

**Purpose:** Workflow documentation and findings (historical)

**Status:** May be outdated - created during workflow development

---

## 📊 Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 00_setup_and_config                                             │
│   └─► Defines: paths, configs, parameters                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 01_synthetic_predictions_v8 or 02_forecast_predictions          │
│   └─► Generates: prediction matrices (.pkl)                    │
│   └─► Saves: validation results (.pkl)                         │
│   └─► Delta: commodity.trading_agent.predictions_{commodity}   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 03_strategy_implementations                                     │
│   └─► Defines: 9 strategy classes                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 04_backtesting_engine                                           │
│   └─► Defines: BacktestEngine class                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 05_strategy_comparison ⭐ MAIN                                  │
│   └─► Runs: All strategies on all commodities/models           │
│   └─► Saves: results_detailed_{commodity}_{model}.pkl          │
│   └─► Saves: 7 PNG charts per commodity-model                  │
│   └─► Saves: 2 cross-comparison PNGs                           │
│   └─► Delta: commodity.trading_agent.results_{commodity}_{model}│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 06_statistical_validation                                       │
│   └─► Saves: statistical_results_{commodity}_{model}.pkl       │
│   └─► Saves: 3 statistical PNGs                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 07_feature_importance                                           │
│   └─► Saves: feature_analysis_{commodity}_{model}.pkl          │
│   └─► Saves: 3 feature analysis PNGs                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 08_sensitivity_analysis                                         │
│   └─► Saves: sensitivity_results_{commodity}_{model}.pkl       │
│   └─► Saves: 3 sensitivity PNGs                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 09_strategy_results_summary                                     │
│   └─► Saves: summary_stats_{commodity}_{model}.csv             │
│   └─► Saves: 3 summary PNGs                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 10_paired_scenario_analysis                                     │
│   └─► Saves: 3 comparison PNGs                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Complete Output Inventory

### Delta Tables (Unity Catalog)

**Location:** `commodity.trading_agent.*`

| Table Name | Created By | Contains |
|------------|------------|----------|
| `predictions_{commodity}` | 01_synthetic | All prediction runs (timestamp, run_id, day_ahead, predicted_price) |
| `results_{commodity}_{model}` | 05_comparison | Strategy metrics (strategy, net_earnings, revenue, costs, trades) |

**Grain:**
- predictions: (timestamp, run_id, day_ahead, future_date)
- results: (strategy, commodity, model_version)

---

### Volume Files (Binary/Pickle)

**Location:** `/Volumes/commodity/trading_agent/files/`

| File Pattern | Created By | Contains | Size |
|--------------|------------|----------|------|
| `validation_results_v8.pkl` | 01_synthetic | MAPE/MAE validation for all accuracy levels | ~10MB |
| `prediction_matrices_{commodity}_{model}.pkl` | 01_synthetic | Dict of {date: np.array(runs, horizons)} | ~50-200MB |
| `prediction_matrices_{commodity}_{model}_real.pkl` | 02_forecast | Same for real forecasts | ~50-200MB |
| `results_detailed_{commodity}_{model}.pkl` | 05_comparison | Full backtest results (trades, daily_state) | ~5-20MB |
| `statistical_results_{commodity}_{model}.pkl` | 06_statistical | Bootstrap samples, p-values, CIs | ~2-5MB |
| `feature_analysis_{commodity}_{model}.pkl` | 07_feature | Feature correlations, importance | ~1-5MB |
| `sensitivity_results_{commodity}_{model}.pkl` | 08_sensitivity | Parameter sweep results | ~5-10MB |

---

### Volume Files (CSV)

**Location:** `/Volumes/commodity/trading_agent/files/`

| File Pattern | Created By | Contains | Size |
|--------------|------------|----------|------|
| `detailed_strategy_results.csv` | 05_comparison | All strategies, all commodities, all models | ~100KB |
| `cross_model_commodity_summary.csv` | 05_comparison | Best strategies only (summary) | ~10KB |
| `statistical_comparisons_{commodity}_{model}.csv` | 06_statistical | Statistical test results | ~20KB |
| `bootstrap_summary_{commodity}_{model}.csv` | 06_statistical | Bootstrap statistics | ~10KB |
| `summary_stats_{commodity}_{model}.csv` | 09_summary | Aggregated metrics | ~30KB |
| `final_summary_{commodity}_{model}.csv` | 09_summary | Complete summary | ~50KB |

---

### Volume Files (PNG Charts)

**Location:** `/Volumes/commodity/trading_agent/files/`

**Per Commodity-Model (from 05_comparison):**
1. `net_earnings_{commodity}_{model}.png` - Bar chart of all strategy earnings
2. `trading_timeline_{commodity}_{model}.png` - Price + trade markers over time
3. `total_revenue_no_costs_{commodity}_{model}.png` - Cumulative gross revenue
4. `cumulative_returns_{commodity}_{model}.png` - Cumulative net earnings
5. `inventory_drawdown_{commodity}_{model}.png` - Inventory levels over time

**Cross-Comparison (from 05_comparison):**
6. `cross_model_commodity_advantage.png` - Prediction advantage by model/commodity
7. `cross_model_commodity_earnings.png` - Baseline vs prediction earnings comparison

**Per Commodity-Model (from 06_statistical):**
8. `bootstrap_distribution_{commodity}_{model}.png` - Bootstrap histogram + CI
9. `confidence_intervals_{commodity}_{model}.png` - Forest plot of CIs
10. `effect_sizes_{commodity}_{model}.png` - Cohen's d bar chart

**Per Commodity-Model (from 07_feature):**
11. `feature_correlation_heatmap_{commodity}_{model}.png` - Feature correlations
12. `feature_importance_{commodity}_{model}.png` - Ranked feature importance
13. `feature_distributions_{commodity}_{model}.png` - SELL vs WAIT boxplots

**Per Commodity-Model (from 08_sensitivity):**
14. `sensitivity_plot_{commodity}_{model}.png` - Parameter sweep lines
15. `sensitivity_heatmap_{commodity}_{model}.png` - 2D parameter interaction
16. `tornado_diagram_{commodity}_{model}.png` - Parameter impact ranking

**Per Commodity-Model (from 09_summary):**
17. `strategy_ranking_{commodity}_{model}.png` - Summary table visualization
18. `performance_dashboard_{commodity}_{model}.png` - Multi-panel summary
19. `key_findings_{commodity}_{model}.png` - Infographic

**Per Commodity-Model (from 10_paired):**
20. `paired_decisions_{commodity}_{model}.png` - Decision comparison
21. `trade_timing_comparison_{commodity}_{model}.png` - Timing analysis
22. `cost_attribution_{commodity}_{model}.png` - Performance breakdown

**Total:** ~22 PNGs per commodity-model combination

**For 2 commodities × 5 synthetic models = 10 combinations:**
- 220 PNG files minimum
- Plus cross-comparison charts
- Plus real model results (12+ additional combinations)

---

## 🔗 Integration Points for Diagnostics

### Current Issues

1. **No parameter optimization** - Uses hardcoded params from 00_setup
2. **No algorithm validation** - No 100% accuracy test
3. **Limited strategy variants** - Only 9 strategies, no alternatives
4. **No monotonicity testing** - Doesn't verify 60% < 70% < 80% < 90% < 100%

### What Diagnostics Provide

From `diagnostics/`:

**Core Strategy Implementation:**
- **all_strategies_pct.py**: 9 strategies with percentage-based framework
  - 3 baseline strategies (Equal Batches, Price Threshold, Moving Average)
  - 5 prediction strategies (Expected Value, Consensus, Risk-Adjusted, Price Threshold Pred, Moving Average Pred)
  - **UPDATED (2025-11-24)**: Redesigned matched pair strategies with 3-tier confidence system
  - HIGH confidence (CV < 5%): Override baseline completely
  - MEDIUM confidence (5-15%): Blend baseline + predictions
  - LOW confidence (CV > 15%): Follow baseline exactly

**Validation Diagnostics:**
- **diagnostic_100** (`run_diagnostic_100.py`): Algorithm validation with perfect foresight
  - Tests with 100% accurate predictions (synthetic_acc100)
  - **UPDATED (2025-11-24)**: Lowered threshold from 10% to 6% for realistic validation
  - Output: `diagnostic_100_summary.csv`, `diagnostic_100_results.pkl`

- **diagnostic_16** (`run_diagnostic_16.py`): Grid search optimized parameters
  - Optuna-based Bayesian optimization
  - Output: `diagnostic_16_best_params.pkl`, `diagnostic_16_summary.csv`

- **diagnostic_17** (`run_diagnostic_17.py`): Trade-by-trade analysis with optimized params
  - Uses: `diagnostic_16_best_params.pkl`
  - Output: Trade-level analysis

**New Accuracy Analysis Diagnostics (2025-11-24):**
- **check_prediction_models.py**: Explores available prediction models
  - Lists all model_version values in predictions table
  - Calculates CV (Coefficient of Variation) for each model
  - Helps identify confidence tiers for each accuracy level
  - Output: Console only (quick exploration)

- **run_diagnostic_accuracy_threshold.py**: **⭐ COMPREHENSIVE ACCURACY ANALYSIS**
  - **Purpose**: Determine minimum prediction accuracy for statistically significant benefit
  - Tests ALL accuracy levels: 60%, 70%, 80%, 90%, 100%
  - Compares ALL prediction strategies vs ALL baseline strategies
  - **Statistical Methods** (from 06_statistical_validation.ipynb):
    - Paired t-test on daily portfolio value changes
    - Cohen's d effect size calculation
    - Bootstrap confidence intervals (1000 iterations, 95% CI)
    - Statistical significance at p < 0.05
  - **Key Questions Answered**:
    1. What accuracy level provides statistically significant benefit?
    2. How does improvement scale with accuracy?
    3. At what accuracy does each strategy become viable?
    4. What is the confidence-based performance degradation curve?
  - **Outputs**:
    - `diagnostic_accuracy_threshold_results.pkl` - Full results with daily state tracking
    - `diagnostic_accuracy_threshold_summary.csv` - Earnings and improvements by accuracy
    - `diagnostic_accuracy_threshold_stats.csv` - Statistical test results (t-stat, p-value, Cohen's d, CIs)

- **run_diagnostic_confidence_test.py**: Tests 3-tier confidence system
  - Validates that HIGH/MEDIUM/LOW confidence tiers work as expected
  - Tests with multiple accuracy levels (100%, 90%, 80%, 70%)
  - Verifies graceful degradation as accuracy decreases
  - Output: `diagnostic_confidence_test_results.pkl`, `diagnostic_confidence_test_summary.csv`

### Integration Plan (Future Work)

**Step 1:** Port optimized parameters from diagnostics
- Load `diagnostic_16_best_params.pkl`
- Update `BASELINE_PARAMS` and `PREDICTION_PARAMS` in 00_setup

**Step 2:** Integrate diagnostic_100 test
- Add to workflow before 05_strategy_comparison
- Verify algorithms work with 100% accuracy
- Block execution if test fails

**Step 3:** Add monotonicity validation
- After 05_comparison runs all accuracies
- Verify 60% < 70% < 80% < 90% < 100%
- Add to 06_statistical or create new 06_monotonicity notebook

**Step 4:** Consolidate strategy implementations
- Replace 03_strategy_implementations.ipynb with diagnostics/all_strategies_pct.py
- Ensure consistent parameter names
- Preserve decision logging

**Step 5:** Organize outputs into final report
- Create FINAL_REPORT/ directory structure
- Aggregate all CSVs and PNGs
- Generate master summary document

---

## 📋 Execution Checklist (Current Workflow)

### One-Time Setup
- [ ] Configure 00_setup_and_config
- [ ] Set commodity parameters
- [ ] Verify Unity Catalog access

### Per Run
- [ ] Generate predictions (01_synthetic_v8 or 02_forecast)
- [ ] Run 05_strategy_comparison (auto-runs 03, 04)
- [ ] Run 06_statistical_validation
- [ ] Run 07_feature_importance
- [ ] Run 08_sensitivity_analysis
- [ ] Run 09_strategy_results_summary
- [ ] Run 10_paired_scenario_analysis

### Download Results
- [ ] Download all PNGs from volume
- [ ] Download all CSVs from volume
- [ ] Download all pickles for deep analysis
- [ ] Compile into presentation/report

---

## 🎯 Key Findings (What This Workflow Reveals)

From running this workflow with v8 synthetic predictions:

**If prediction strategies beat baselines:**
- Prediction accuracy is sufficient (≥70%)
- Algorithms work correctly
- Parameters are well-tuned

**If prediction strategies lose to baselines:**
- Check diagnostic_100: Do algorithms work with 100% accuracy?
  - YES → Accuracy too low, improve forecasting
  - NO → Algorithms broken, debug decision logic
- Check diagnostic_17: Where do strategies diverge?
  - Revenue? Transaction costs? Storage costs?
- Check monotonicity: Does performance improve with accuracy?
  - YES → Just need better predictions
  - NO → Algorithm not using predictions correctly

---

## 🤖 AUTOMATION MIGRATION PLAN

**Updated:** 2025-11-24
**Status:** Documentation phase - No changes to notebooks yet

### Overview

Successfully implemented automated remote execution pattern for diagnostics (100, 16, 17). This pattern eliminates manual notebook execution and enables fully automated workflows.

### Current State: Interactive Notebooks

**How it works now:**
1. User opens notebook in Databricks UI
2. Manually runs cells one by one
3. Waits for execution to complete
4. Results exist only in ephemeral session outputs
5. Must manually save/export results
6. Repeat for each commodity, model, and configuration

**Problems:**
- ❌ Time-consuming (must babysit executions)
- ❌ Error-prone (easy to skip cells or run out of order)
- ❌ Not reproducible (hard to track what was run when)
- ❌ Doesn't scale (can't run 10 configurations overnight)
- ❌ Results scattered (220+ files in various locations)

### Target State: Automated Execution

**How it will work:**
1. Convert notebooks to executable Python scripts
2. Submit jobs via Databricks CLI
3. Monitor progress automatically
4. Results auto-saved to volumes
5. Chain multiple jobs sequentially
6. Download all results programmatically
7. Generate consolidated reports automatically

**Benefits:**
- ✅ Zero manual intervention ("set and forget")
- ✅ Fully reproducible (scripts in git)
- ✅ Scalable (run 50+ configurations overnight)
- ✅ Robust (Jobs API handles failures)
- ✅ Cost-efficient (uses existing clusters)
- ✅ Audit trail (all runs logged with IDs)

### Migration Path: Notebook → Script Conversion

**Step-by-step for each notebook:**

#### 1. Read and Understand (5 min)
```bash
# Open notebook in Databricks UI
# Understand: inputs, logic, outputs
# Note: dependencies on other notebooks
```

#### 2. Extract to Python Script (15-30 min)
```python
"""
Notebook_NN: Description
Automated execution script for Databricks
"""

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import sys
import os

# Databricks path handling
sys.path.insert(0, '/Workspace/Users/gibbons_tony@berkeley.edu')

def load_data():
    """Load from Delta tables, not pickle files"""
    spark = SparkSession.builder.getOrCreate()
    # Query Delta tables
    return data

def main():
    # Main logic from notebook
    # Save results to volume
    volume_path = "/Volumes/commodity/trading_agent/files"
    output_file = f"{volume_path}/notebook_NN_results.pkl"
    # ...
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

**Key conversions:**
- `%run 00_setup_and_config` → `from config import *`
- `spark.table(...)` → Keep as-is
- `display(df)` → `print(df.head())`
- `dbutils.fs.cp(...)` → `with open(...)`
- Magic commands → Pure Python equivalents

#### 3. Test Locally (if possible) (10 min)
```bash
# If script doesn't need Spark:
python run_notebook_NN.py

# Otherwise, skip to step 4
```

#### 4. Commit and Push (5 min)
```bash
git add trading_agent/commodity_prediction_analysis/run_notebook_NN.py
git commit -m "Convert notebook_NN to automated script"
git push
```

#### 5. Update Databricks Repo (2 min)
```bash
databricks repos update <REPO_ID> --branch main
```

#### 6. Submit Test Job (5 min)
```bash
cat > /tmp/job_notebook_NN.json << 'EOF'
{
  "run_name": "notebook_NN_test",
  "tasks": [{
    "task_key": "run_NN",
    "spark_python_task": {
      "python_file": "file:///Workspace/Repos/.../run_notebook_NN.py"
    },
    "existing_cluster_id": "1111-041828-yeu2ff2q"
  }]
}
EOF

databricks jobs submit --json @/tmp/job_notebook_NN.json
```

#### 7. Monitor and Verify (10 min)
```bash
# Get run ID from submission
# Monitor: databricks jobs get-run <RUN_ID>
# Verify outputs saved to volume
# Check CSV/PKL files created correctly
```

#### 8. Document (5 min)
```markdown
# Add to migration log:
- [x] Notebook NN converted
- Run time: ~X minutes
- Output verified: ✓
- Ready for production
```

**Total time per notebook:** ~1-2 hours (faster after first few)

### Conversion Priority

**Phase 1: Core Workflow (Weeks 1-2)**
High-value, frequently-run notebooks:

1. ✅ **01_synthetic_predictions_v8** → `run_01_synthetic_predictions.py`
   - Most critical (generates all predictions)
   - Runtime: ~20 minutes
   - Priority: URGENT

2. ✅ **05_strategy_comparison** → `run_05_strategy_comparison.py`
   - Main workflow orchestrator
   - Runtime: ~60 minutes
   - Priority: HIGH
   - Dependencies: 01, 03, 04

3. ✅ **06_statistical_validation** → `run_06_statistical_validation.py`
   - Key for significance testing
   - Runtime: ~10 minutes
   - Priority: HIGH
   - Dependencies: 05

**Phase 2: Analysis Notebooks (Weeks 3-4)**
Important but less frequent:

4. **07_feature_importance** → `run_07_feature_importance.py`
5. **08_sensitivity_analysis** → `run_08_sensitivity_analysis.py`
6. **09_strategy_results_summary** → `run_09_strategy_results_summary.py`
7. **10_paired_scenario_analysis** → `run_10_paired_scenario_analysis.py`

**Phase 3: Utilities (Week 5)**
Can defer:

8. **02_forecast_predictions** → `run_02_forecast_predictions.py`
9. **00_setup_and_config** → Extract to `shared_config.py`

**Phase 4: Implementation Notebooks (Optional)**
These define code, don't execute workflows:

- **03_strategy_implementations** → Keep as-is or extract to `strategies.py`
- **04_backtesting_engine** → Keep as-is or extract to `backtest_engine.py`

### Workflow Orchestration

After conversion, create master automation script:

```python
# run_full_workflow.py
"""
Complete automated trading analysis workflow
Runs all notebooks in sequence with proper dependencies
"""

def run_workflow(commodity, model_version):
    # Phase 1: Generate predictions
    run_id_01 = submit_job('run_01_synthetic_predictions.py',
                           params={'commodity': commodity})
    wait_for_completion(run_id_01)

    # Phase 2: Run strategy comparison
    run_id_05 = submit_job('run_05_strategy_comparison.py',
                           params={'commodity': commodity,
                                   'model': model_version})
    wait_for_completion(run_id_05)

    # Phase 3: Analysis (can run in parallel)
    run_ids = []
    for script in ['run_06', 'run_07', 'run_08', 'run_09', 'run_10']:
        run_id = submit_job(f'{script}.py',
                           params={'commodity': commodity,
                                   'model': model_version})
        run_ids.append(run_id)

    wait_for_all(run_ids)

    # Phase 4: Download all results
    download_results(commodity, model_version)

    # Phase 5: Generate consolidated report
    generate_report(commodity, model_version)

# Run for all configurations
for commodity in ['coffee', 'sugar']:
    for model in ['synthetic_acc90', 'synthetic_acc80',
                  'xgboost_weather_v1']:
        run_workflow(commodity, model)
```

**Example overnight run:**
```bash
# Submit at 6pm Friday
python run_full_workflow.py --all-configs

# Returns Monday morning with:
# - 220+ charts generated
# - 30+ CSV files created
# - 25+ pickle files saved
# - Consolidated report ready
# Total cost: ~$50 (cluster runtime)
```

### Integration with Diagnostics

**Before running main workflow:**

```python
# Step 1: Validate algorithms work
run_diagnostic_100()  # Check with 100% accuracy
if not algorithms_valid:
    raise Error("Fix algorithms first!")

# Step 2: Optimize parameters
run_diagnostic_16()  # Grid search best params
best_params = load('diagnostic_16_best_params.pkl')

# Step 3: Update configuration
update_config_with_params(best_params)

# Step 4: Run main workflow with optimized params
run_full_workflow()

# Step 5: Verify monotonicity
run_monotonicity_check()  # 60% < 70% < 80% < 90% < 100%

# Step 6: Deep dive into differences
run_diagnostic_17()  # Trade-by-trade analysis
```

### Monitoring and Logging

**Job monitoring dashboard:**
```python
# monitor_all_jobs.py
jobs = get_running_jobs()
for job in jobs:
    print(f"{job.name}: {job.status} ({job.elapsed_time})")
    if job.status == 'FAILED':
        print(f"  Error: {job.error_message}")
        send_alert(job)
```

**Logging structure:**
```
/Volumes/commodity/trading_agent/logs/
├── run_01_synthetic_predictions_2025-11-24_10-30.log
├── run_05_strategy_comparison_2025-11-24_11-00.log
├── run_06_statistical_validation_2025-11-24_12-00.log
└── ...
```

**Each log contains:**
- Execution start/end timestamps
- Parameters used
- Data loaded (row counts, date ranges)
- Results produced (file paths, metrics)
- Errors/warnings
- Performance stats (runtime, memory)

### Cost Analysis

**Current (manual):**
- Human time: ~4 hours per configuration
- Cluster cost: ~$10 per run (interactive cluster)
- Total: 10 configs × $10 = $100 cluster + 40 hours human time

**Automated:**
- Human time: ~30 minutes (submit jobs, review results)
- Cluster cost: ~$50 (batch jobs, can use cheaper instance types)
- Total: $50 cluster + 0.5 hours human time

**Savings:**
- **Cost:** Same or lower (can optimize cluster selection)
- **Time:** 40 hours → 0.5 hours (**98.75% reduction**)
- **Reproducibility:** Manual → Fully automated
- **Scalability:** 10 configs → 100+ configs (no extra human time)

### Migration Timeline

**Week 1: Infrastructure**
- Set up git repo structure for scripts
- Create shared libraries (config.py, utils.py)
- Test job submission workflow
- Document patterns

**Week 2: Core Conversions**
- Convert notebooks 01, 05, 06
- Test end-to-end workflow
- Verify outputs match notebook versions
- Fix any discrepancies

**Week 3-4: Complete Conversions**
- Convert remaining notebooks (07-10)
- Build orchestration script
- Add monitoring dashboard
- Documentation

**Week 5: Testing and Validation**
- Run full workflow automated
- Compare results to manual runs
- Performance optimization
- User acceptance testing

**Week 6: Production**
- Deprecate manual notebook workflow
- Schedule regular automated runs
- Set up alerts and monitoring
- Train users on new system

### Success Metrics

**Migration successful if:**
- [ ] All notebooks converted to scripts
- [ ] Automated runs produce identical results to manual
- [ ] Runtime < 2x manual execution time
- [ ] Zero manual intervention required
- [ ] Logs capture all important events
- [ ] Failures automatically retry or alert
- [ ] Results downloadable in batch
- [ ] Reports generated automatically

### Rollback Plan

**If automation has issues:**
1. Keep original notebooks in place
2. Run manual workflow as backup
3. Debug automated scripts offline
4. Compare outputs between manual and automated
5. Fix issues before next automated run
6. Gradual migration: start with 1 notebook at a time

**Compatibility period:**
- Maintain both manual and automated for 1 month
- Parallel runs to verify consistency
- Deprecate manual only after confidence builds

---

**Status:** DOCUMENTATION PHASE - No notebooks changed yet
**Next Step:** User approval, then begin Phase 1 conversions
**Owner:** Claude Code
**Reference:** See `diagnostics/DATABRICKS_OUTPUT_ACCESS_GUIDE.md` Section 5

---

**Last Updated:** 2025-11-24
**Owner:** Claude Code
**Purpose:** Complete inventory for diagnostic integration planning
**Next Step:** Begin automation migration after diagnostic results analyzed

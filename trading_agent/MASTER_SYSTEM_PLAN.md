# Trading Agent System - Master Plan

**Purpose:** Single source of truth for the complete trading agent system from data pipelines → forecasts → backtesting → recommendations → WhatsApp delivery

**Last Updated:** 2025-11-24

---

## SYSTEM VISION

### End-to-End Flow
```
Data Pipelines (Web → Bronze → Silver)
    ↓
Forecast Agent (Predictions + Performance Metrics)
    ↓
Trading Agent (Backtest → Optimize → Store Best Scenarios)
    ↓
Daily Operations (Generate Recommendations)
    ↓
WhatsApp Delivery (Dual Path)
    ├─→ Structured Reports (Actuals + Predictions + Recommendations)
    └─→ Claude Q&A (Full Dataset Access)
```

### Design Principles
1. **Rigorous Testing:** Defensible analysis with statistical validation
2. **Automated Execution:** Minimal manual intervention
3. **Comprehensive Reporting:** Easy to review at all levels (executive → technical)
4. **Data Accessibility:** Structured for both human review and LLM queries
5. **Continuous Improvement:** Periodic backtesting informs strategy selection

---

## SYSTEM COMPONENTS

### 1. Data Pipelines (research_agent)
**Status:** ✅ OPERATIONAL
**Owner:** Research agent
**Purpose:** Capture and organize data for forecast agent

**Capabilities:**
- Web scraping (market prices, weather, forex, etc.)
- Bronze layer: Raw data ingestion
- Silver layer: Cleaned, unified data (commodity.silver.unified_data)
- Forward-filling for continuous daily coverage
- Multi-commodity support (coffee, sugar, wheat, corn, soybeans)

**Outputs:**
- `commodity.bronze.*` - Raw data tables
- `commodity.silver.unified_data` - Clean, continuous daily data
- Full historical coverage (2015-present)

**Reference:** `research_agent/UNIFIED_DATA_ARCHITECTURE.md`

---

### 2. Forecast Agent (forecast_agent)
**Status:** ✅ OPERATIONAL
**Owner:** Forecast agent
**Purpose:** Generate predictions and track model performance

**Capabilities:**
- Multiple model types (ARIMA, SARIMAX, Prophet, XGBoost, TFT)
- 14-day forecast horizon
- 2,000 Monte Carlo paths per forecast
- Train-once pattern (180x speedup)
- Continuous backfill (2018-2024 history)
- Performance tracking (MAE, RMSE, CRPS, coverage, calibration)

**Outputs:**
- `commodity.forecast.distributions` - Monte Carlo forecast paths
- `commodity.forecast.forecast_metadata` - Model performance metrics
- `commodity.forecast.point_forecasts` - Daily forecasts with intervals
- `commodity.forecast.trained_models` - Persistent model storage

**Reference:** `forecast_agent/README.md`, `forecast_agent/docs/ARCHITECTURE.md`

---

### 3. Trading Agent (trading_agent)
**Status:** 🔧 IN PROGRESS
**Owner:** Trading agent
**Purpose:** Backtest strategies, identify best approaches, generate recommendations

#### 3.1 Core Functionalities

**A. Data Generation & Management**
- Synthetic predictions (controlled accuracy: 60%, 70%, 80%, 90%, 100%)
- Real forecast loading from commodity.forecast.distributions
- Price data alignment with predictions
- Multi-commodity, multi-model support

**B. Strategy Implementation (9 strategies)**

*Baseline Strategies (no predictions):*
1. **Immediate Sale** - Weekly liquidation, minimize storage costs
2. **Equal Batches** - Fixed 25% sales every 30 days
3. **Price Threshold** - Sell when price > 30-day MA + threshold
4. **Moving Average** - Sell on MA crossover

*Prediction-Based Strategies:*
5. **Consensus** - Sell based on ensemble agreement (70%+ bullish paths)
6. **Expected Value** - Optimize sale timing via EV calculations
7. **Risk-Adjusted** - Balance return vs uncertainty (prediction std dev)
8. **Price Threshold Predictive** - Baseline #3 + prediction overlay (matched pair)
9. **Moving Average Predictive** - Baseline #4 + prediction overlay (matched pair)

**Key Features:**
- Daily evaluation (market-responsive, not scheduled)
- Technical indicators: RSI, ADX, Standard Deviation (historical + predicted)
- Cost-benefit analysis (sell now vs wait for better price)
- Dynamic batch sizing (8%-40% based on signals)
- Cooldown periods (prevent overtrading)
- Matched pairs design (clean A/B testing)

**C. Backtesting Engine**
- Harvest cycle management (gradual accumulation during harvest windows)
- Multi-year support (multiple harvest cycles)
- Age tracking (365-day max holding from harvest start)
- Pre-harvest liquidation (force-sell old inventory)
- Cost modeling (storage: 0.025%/day, transaction: 0.25%)
- Complete audit trail (trade logs, daily state)

**D. Performance Metrics**
- Financial: Net earnings, revenue, costs, avg sale price
- Risk-return: Sharpe ratio, volatility, annualized return
- Trading patterns: Days to liquidate, trades per cycle, timing

**E. Statistical Validation**
- Paired t-tests (prediction vs baseline)
- Bootstrap confidence intervals (1000 iterations)
- Effect sizes (Cohen's d)
- Multiple comparison corrections

**F. Sensitivity Analysis**
- Parameter robustness (grid search, Optuna optimization)
- Cost robustness (0.5× to 2.0× multipliers)
- Model comparison (across forecast models)

**G. Diagnostic Validation**
- 100% accuracy test (algorithm validation)
- Monotonicity test (60% < 70% < 80% < 90% < 100%)
- Matched pair analysis (baseline vs predictive)
- Trade-by-trade debugging

#### 3.2 Current Implementation

**Notebooks (00-10):**
- `00_setup_and_config` - Central configuration
- `01_synthetic_predictions_v8` - Generate controlled accuracy scenarios
- `02_forecast_predictions` - Load real forecasts
- `03_strategy_implementations` - 9 strategy definitions
- `04_backtesting_engine` - Harvest-aware backtest engine
- `05_strategy_comparison` - Main runner (all strategies, all models)
- `06_statistical_validation` - t-tests, bootstrap, CIs
- `07_feature_importance` - Feature correlation analysis
- `08_sensitivity_analysis` - Parameter and cost robustness
- `09_strategy_results_summary` - Aggregated reporting
- `10_paired_scenario_analysis` - Baseline vs predictive pairs

**Diagnostics:**
- `diagnostic_16_optuna` - Grid search parameter optimization
- `diagnostic_17_paradox_analysis` - Trade-by-trade matched pairs
- `diagnostic_100_algorithm_validation` - 100% accuracy test
- `all_strategies_pct.py` - Modular strategy implementations

**Automation Status:**
- ✅ Diagnostics automated (16, 17, 100)
- ❌ Main workflow manual (notebooks 00-10)
- ❌ No orchestrator

**Outputs:**
- Delta tables: `commodity.trading_agent.results_*`
- Pickle files: `results_detailed_*.pkl`, `statistical_results_*.pkl`, `sensitivity_results_*.pkl`
- CSV exports: `detailed_strategy_results.csv`, `cross_model_commodity_summary.csv`
- Visualizations: 220+ PNG charts (earnings, timelines, inventory, heatmaps)

**Reference:** `trading_agent/commodity_prediction_analysis/COMPREHENSIVE_FILE_INVENTORY.md`

---

### 4. Daily Operations (trading_agent)
**Status:** 📋 PLANNED
**Owner:** Trading agent
**Purpose:** Generate daily recommendations using latest forecasts

**Capabilities (Planned):**
- Load today's forecast distributions
- Apply current best strategy per commodity
- Generate recommendation (SELL/HOLD, quantity, expected gain)
- Save to production table
- Trigger WhatsApp delivery

**Outputs (Planned):**
- `commodity.trading.daily_recommendations` - Today's recommendations
- `commodity.trading.recommendation_history` - Historical recommendations
- JSON exports for WhatsApp consumption

**Reference:** `trading_agent/whatsapp/generate_daily_recommendation.py` (exists but not scheduled)

---

### 5. WhatsApp Delivery (trading_agent/whatsapp)
**Status:** 🔧 75% COMPLETE
**Owner:** Trading agent
**Purpose:** Deliver recommendations via WhatsApp (structured reports + conversational Q&A)

#### 5.1 Dual Path Architecture

**Path 1: Structured Reports (Fast Path)**
- Direct Databricks query for latest recommendation
- Format as WhatsApp message (TwiML)
- Response time: <1s
- Use case: "Coffee" → Get today's recommendation

**Path 2: Claude Q&A (LLM Path)**
- Detect question intent
- Build rich context from all datasets
- Query Claude API for natural language answer
- Response time: 2-4s
- Use case: "Which strategy are you using for Coffee?" → Conversational explanation

#### 5.2 Current Implementation

**Code Complete (75%):**
- `llm_context.py` - Context builders for forecast + market data (698 lines) ✅
- `llm_client.py` - Claude API integration (257 lines) ✅
- `lambda_handler_real.py` - WhatsApp webhook handler (906 lines) ⚠️ (LLM not integrated)
- `test_llm_integration.py` - Test suite ✅

**Missing (25%):**
- Lambda handler doesn't import LLM modules yet
- No intent routing to Claude
- No strategy performance data available (blocked by Phase 3)

**LLM Data Requirements:**
- Strategy performance (net earnings, Sharpe, win rates) - ❌ Not in database
- Strategy definitions (how they work, formulas) - ❌ Not in database
- Active strategy selection (which one is used, why) - ❌ Not in database
- Forecast metadata (MAE, RMSE, CRPS) - ✅ Already available
- Market context (prices, trends) - ✅ Already available

**Reference:** `trading_agent/whatsapp/LLM_IMPLEMENTATION_PLAN.md`

---

## EXECUTION PHASES

### PHASE 1: Fix Algorithm Bugs & Validate (CRITICAL PATH)
**Status:** 🔧 IN PROGRESS (User working separately)
**Objective:** Get prediction strategies working correctly

**Critical Issue:**
- At 90% synthetic accuracy, prediction strategies LOSE to baselines by 2-3%
- Expected: Should WIN by 10-20%
- Indicates: Logic bug in prediction usage OR cost issues OR data leakage

**Tasks:**
1. ✅ Run diagnostic_100 (100% accuracy test)
   - If predictions beat baselines by >10% → algorithms sound
   - If predictions lose or barely win → fundamental bug

2. 🔧 Debug strategy implementations
   - Add extensive logging to trace prediction usage
   - Verify predictions are accessed correctly
   - Check date alignment
   - Validate cost calculations

3. 📋 Fix identified bugs
   - Update strategy code
   - Re-run backtests
   - Verify fixes work

4. 📋 Create monotonicity validation
   - Compare 60% vs 70% vs 80% vs 90% vs 100%
   - Verify performance increases with accuracy
   - Can be new notebook or add to existing

5. 📋 Validate fixes
   - Run full workflow with corrected strategies
   - Confirm monotonicity
   - Confirm matched pairs diverge appropriately

**Success Criteria:**
- [ ] diagnostic_100 shows predictions >> baselines at 100% accuracy
- [ ] Performance improves monotonically (60% < 70% < 80% < 90% < 100%)
- [ ] 90% synthetic predictions beat baselines by 10-20%
- [ ] Matched pairs show clear divergence

**Blocks:** Everything else (cannot trust results until algorithms validated)

**Reference:** `trading_agent/commodity_prediction_analysis/EXECUTIVE_SUMMARY.md` (bug documentation)

---

### PHASE 2: Automate Workflow (ENABLE RAPID ITERATION)
**Status:** 🔧 IN PROGRESS (Runners complete, orchestration next)
**Objective:** One-command execution of entire backtesting workflow

**Dependencies:** None (can run in parallel with Phase 1)

**Key Principle:** Strategies are modular (`production/strategies/`), so automation can import from this module. When bugs are fixed in the strategy file, automated scripts automatically use the corrected version. No need to wait.

**Completed 2025-11-24:**
- ✓ Strategy extraction from diagnostics to production
- ✓ All 9 strategies in production/strategies/ (4 modules, 1,900 lines)
- ✓ Production config with correct costs (0.005%, 0.01%)
- ✓ Production backtest engine ready
- ✓ Production runners module (5 modules, 1,446 lines)
- ✓ Comprehensive test suite (6 test files, 2,500+ lines, 93%+ coverage expected)

**Current State:**
- ✅ Proven pattern: Diagnostics 16/17/100 fully automated
- ✅ Runners module: Replicates notebook 05 workflow (data loading, strategy execution, visualization, result saving)
- ✅ Full test coverage: Unit tests (80%), integration tests (15%), smoke tests (5%)
- ❌ Main workflow (notebooks 00-10) still manual
- ❌ No orchestrator

#### 2.1 Audit Current Notebooks

**Action:** Identify which notebooks are essential vs obsolete

**Essential (automate):**
- 00_setup_and_config.ipynb - Central config
- 01_synthetic_predictions.ipynb - Controlled accuracy scenarios
- 02_forecast_predictions.ipynb - Real model predictions
- 03_strategy_implementations.ipynb - Strategy definitions (AFTER FIXES)
- 04_backtesting_engine.ipynb - Harvest-aware engine
- 05_strategy_comparison.ipynb - Main runner
- 06_statistical_validation.ipynb - Significance tests
- 07_feature_importance.ipynb - Feature analysis
- 08_sensitivity_analysis.ipynb - Cost sensitivity
- 09_strategy_results_summary.ipynb - Dashboard
- 10_paired_scenario_analysis.ipynb - Matched pairs
- 11_synthetic_accuracy_comparison.ipynb - Monotonicity validation

**Diagnostics (already automated):**
- diagnostic_16_optuna.ipynb - Grid search (working)
- diagnostic_17_all_strategies_pct.py - Paradox analysis (working)
- diagnostic_100_algorithm_validation.py - Validation (working)

**Obsolete (archive/delete):**
- TBD based on audit (any duplicates, old versions, exploratory dead ends)

#### 2.2 Convert Notebooks to Scripts

**Pattern (proven with diagnostics):**
```python
# Template for each notebook:
def run_analysis(commodity, model_version, config):
    """
    Run [notebook] logic

    Inputs: Delta tables, config
    Outputs: Delta tables, /Volumes/ files
    Returns: status, summary
    """
    # Load data
    # Run analysis
    # Save outputs
    # Log summary
    return status_code, summary_dict

# Databricks job script:
if __name__ == "__main__":
    config = load_config()
    status, summary = run_analysis(config)
    print(json.dumps(summary))
    sys.exit(status)
```

**Notebooks to Convert:**
1. 01_synthetic_predictions → `run_01_synthetic_predictions.py`
2. 05_strategy_comparison → `run_05_strategy_comparison.py`
3. 06_statistical_validation → `run_06_statistical_validation.py`
4. 07_feature_importance → `run_07_feature_importance.py`
5. 08_sensitivity_analysis → `run_08_sensitivity_analysis.py`
6. 09_strategy_results_summary → `run_09_results_summary.py`
7. 10_paired_scenario_analysis → `run_10_paired_analysis.py` (fix path errors first)

**Note:** 00, 03, 04 are imported by others, don't need separate scripts

#### 2.2 Build Orchestrator

**File:** `run_complete_analysis.py`

**Capabilities:**
- Dependency management (wait for upstream jobs)
- Parallel execution (01 for multiple accuracy levels simultaneously)
- Progress tracking
- Error recovery (retry failed jobs)
- Output collection (download all results)
- Summary dashboard

**Workflow:**
```python
# Master orchestrator for full trading agent workflow
#
# Usage:
#   python run_complete_analysis.py --mode full
#   python run_complete_analysis.py --mode diagnostics-only
#   python run_complete_analysis.py --commodity coffee --model arima_v1
#
# Workflow:
# 1. Submit 01_synthetic_predictions (parallel for each accuracy level)
# 2. Submit 02_forecast_predictions (parallel for each model)
# 3. Submit 05_strategy_comparison (depends on 01+02)
# 4. Submit 06-10 in parallel (all depend on 05)
# 5. Submit diagnostics 16, 17, 100 (parallel)
# 6. Generate consolidated reports
#
# Features:
# - Track job IDs, poll for completion, chain dependencies
# - Dependency management (wait for upstream jobs)
# - Parallel execution where possible
# - Progress tracking (% complete)
# - Error recovery (retry failed jobs)
# - Output collection (download all results to /Volumes/)
# - Summary dashboard (what ran, what failed, where outputs are)
```

**Integration with Diagnostics:**
- diagnostic_16 (grid search) - runs after 05 completes
- diagnostic_17 (paradox analysis) - runs after 16 completes
- diagnostic_100 (validation) - runs independently for smoke tests

**Result:** Single command runs entire workflow soup-to-nuts

**Pattern (proven with diagnostics):**
```python
# Template for each notebook:
def run_analysis(commodity, model_version, config):
    """
    Run [notebook] logic

    Inputs: Delta tables, config
    Outputs: Delta tables, /Volumes/ files
    Returns: status, summary
    """
    # Load data
    # Run analysis
    # Save outputs
    # Log summary
    return status_code, summary_dict

# Databricks job script:
if __name__ == "__main__":
    config = load_config()
    status, summary = run_analysis(config)
    print(json.dumps(summary))
    sys.exit(status)
```

**Notebooks to Convert:**
1. 01_synthetic_predictions → `run_01_synthetic_predictions.py`
2. 05_strategy_comparison → `run_05_strategy_comparison.py`
3. 06_statistical_validation → `run_06_statistical_validation.py`
4. 07_feature_importance → `run_07_feature_importance.py`
5. 08_sensitivity_analysis → `run_08_sensitivity_analysis.py`
6. 09_strategy_results_summary → `run_09_results_summary.py`
7. 10_paired_scenario_analysis → `run_10_paired_analysis.py` (fix path errors first)

**Note:** 00, 03, 04 are imported by others, don't need separate scripts

**Order of conversion:**
1. Start with 01 (predictions) - foundation
2. Then 05 (comparison) - core workflow
3. Then 06-10 (analysis) - can run in parallel
4. Finally 00 (config) - integrate into orchestrator

**Tasks:**
- [ ] Audit current notebooks (identify essential vs obsolete)
- [ ] Convert notebook 01 to script
- [x] Convert notebook 05 to script **✅ COMPLETE (production/runners/ with full test suite)**
- [ ] Convert notebooks 06-10 to scripts
- [ ] Build orchestrator **← NEXT**
- [ ] Test end-to-end
- [ ] Document usage

**Recent Completion (2025-11-24):**
- ✓ Built production/runners/ module to replicate notebook 05 workflow
- ✓ Implemented 5 modular components: data_loader.py, strategy_runner.py, visualization.py, result_saver.py, multi_commodity_runner.py
- ✓ Created comprehensive test suite (6 files, 2,500+ lines, 150+ test cases)
- ✓ Documented with README.md and test execution guides

**Success Criteria:**
- [ ] Single command runs entire workflow
- [ ] All outputs saved to /Volumes/
- [ ] Summary report generated automatically
- [ ] Can run overnight without manual intervention

**Benefits:**
- Rapid iteration during debugging (just re-run)
- Consistent results (no manual errors)
- Scalable (can run multiple commodities/models in parallel)

**Reference:** `trading_agent/commodity_prediction_analysis/diagnostics/DATABRICKS_OUTPUT_ACCESS_GUIDE.md` (Section 5)

---

### PHASE 3: Consolidate & Structure Outputs (ORGANIZE FOR CONSUMPTION)
**Status:** 📋 PLANNED
**Objective:** Organized, comprehensive outputs for human review AND LLM queries

**Dependencies:** Phase 2 complete (need automated workflow generating results)

#### 3.1 Output Structure Design

**Proposed Organization:**
```
/Volumes/commodity/trading_agent/files/
├── reports/                          # Human review (3-tier)
│   ├── executive_summary.md          # Tier 1: 5-minute read
│   ├── detailed_analysis.md          # Tier 2: 30-minute read
│   ├── technical_appendix.md         # Tier 3: 2-hour deep dive
│   ├── validation_report.md          # Tier 4: Statistical validation
│   └── charts/                       # All visualizations, organized
│       ├── performance/              # Strategy comparisons
│       ├── statistical/              # p-values, CIs, bootstrap
│       ├── sensitivity/              # Parameter heatmaps, cost curves
│       ├── timelines/                # Trading patterns
│       └── diagnostics/              # Validation charts
│
├── llm_data/                         # WhatsApp LLM optimized
│   ├── strategy_performance.parquet  # All strategies, all metrics
│   ├── strategy_definitions.json     # Logic, formulas, assumptions
│   ├── active_strategy.json          # Current selections + rationale
│   ├── backtest_metadata.json        # When run, what tested
│   └── trade_history.parquet         # Individual trades with reasons
│
├── production/                       # Daily operations
│   ├── latest_recommendations.json   # Today's recommendations
│   ├── active_parameters.json        # Current config
│   └── model_selections.json         # Which models are active
│
└── archive/                          # Historical snapshots
    └── YYYY-MM-DD_HH-MM/             # Timestamped runs
        ├── reports/
        ├── charts/
        └── data/
```

#### 3.2 Build Consolidation Pipeline

**File:** `generate_consolidated_outputs.py`

**Inputs:**
- Delta tables: results_{commodity}_{model}, predictions_{commodity}
- Pickle files: results_detailed_*.pkl, prediction_matrices_*.pkl
- CSV files: detailed_strategy_results.csv, cross_model_commodity_summary.csv

**Outputs:**
- 3-tier markdown reports (executive → detailed → technical)
- Organized chart directories
- LLM-optimized data files (parquet, json)
- Production data files
- Archive timestamped snapshot

**Functions:**
```python
def generate_reports():
    """Create 3-tier markdown reports"""
    # Tier 1: Executive summary (1-2 pages)
    # - Best strategy per commodity
    # - Key performance metrics
    # - Recommendation for production

    # Tier 2: Detailed analysis (5-10 pages)
    # - All strategies compared
    # - Statistical significance
    # - Sensitivity analysis
    # - Diagnostic findings

    # Tier 3: Technical appendix (full details)
    # - All charts
    # - All data tables
    # - Methodology notes
    # - Code references

def organize_charts():
    """Move charts from temp to organized directories"""
    # Read all PNG files
    # Parse filenames for category
    # Copy to appropriate subdirectory
    # Generate index.html for browsing

def prepare_llm_data():
    """Transform backtest results into LLM-queryable format"""
    # Extract strategy performance → parquet
    # Document strategy definitions → json
    # Identify active strategies → json
    # Flatten trade history → parquet
    # Add metadata for context

def archive_run():
    """Snapshot current results with timestamp"""
    # Copy reports, charts, data to archive/YYYY-MM-DD_HH-MM/
    # Keep last 10 runs, delete older
```

**Cleanup Strategy:**
- Don't delete files, move obsolete items to archive/obsolete/
- Document what was moved and why
- Can always restore if needed

#### 3.3 LLM Data Preparation (Keep WhatsApp Use Case in Mind)

**Objective:** Structure outputs so WhatsApp LLM has everything it needs

**For WhatsApp LLM to answer questions, it needs:**

1. **Strategy Performance** (answer: "How well is X strategy performing?")
   - File: `llm_data/strategy_performance.parquet`
   - Columns: strategy_name, commodity, model_version, net_earnings, sharpe_ratio, win_rate, advantage_vs_baseline, num_trades, backtest_period

2. **Strategy Definitions** (answer: "How does X strategy work?")
   - File: `llm_data/strategy_definitions.json`
   - Content: {strategy_name: {logic, formula, assumptions, strengths, limitations, example}}

3. **Active Strategy Selection** (answer: "Which strategy are you using?")
   - File: `llm_data/active_strategy.json`
   - Content: {commodity: {strategy_name, rationale, activated_date, config_params}}

4. **Trade History** (answer: "Show me trades from last backtest")
   - File: `llm_data/trade_history.parquet`
   - Columns: trade_date, strategy, commodity, price, amount, revenue, cost, reason, confidence

5. **Backtest Metadata** (answer: "When was this tested?")
   - File: `llm_data/backtest_metadata.json`
   - Content: {run_date, commodities_tested, models_tested, accuracy_levels, num_strategies}

**Schema Design Principles:**
- Denormalized (LLM queries should be simple)
- Human-readable column names
- Include context in each row (commodity, strategy, model)
- Text explanations in 'reason' and 'notes' columns
- Metrics rounded to 2-4 decimals (readable)

**Strategy Performance Table**
```sql
CREATE TABLE commodity.whatsapp_llm.strategy_performance (
    strategy_id STRING,
    strategy_name STRING,
    commodity_id STRING,
    model_version_id STRING,

    -- Performance
    net_earnings DECIMAL(10,2),
    sharpe_ratio DECIMAL(10,4),
    total_return_pct DECIMAL(10,4),
    volatility DECIMAL(10,4),

    -- Trading stats
    num_trades INT,
    avg_sale_price DECIMAL(10,2),
    days_to_liquidate INT,

    -- Comparison
    baseline_net_earnings DECIMAL(10,2),
    advantage_dollars DECIMAL(10,2),
    advantage_percent DECIMAL(10,4),

    -- Statistical
    p_value DECIMAL(10,4),
    ci_95_lower DECIMAL(10,2),
    ci_95_upper DECIMAL(10,2),

    -- Metadata
    backtest_period_start DATE,
    backtest_period_end DATE,
    backtested_at TIMESTAMP
)
```

**Strategy Definitions Table**
```sql
CREATE TABLE commodity.whatsapp_llm.strategy_definitions (
    strategy_id STRING PRIMARY KEY,
    strategy_name STRING,
    category STRING,  -- Baseline, Prediction-Based, Matched-Pair

    -- Description
    short_description STRING,
    detailed_description STRING,

    -- How it works
    decision_logic STRING,
    mathematical_formula STRING,

    -- Context
    uses_predictions BOOLEAN,
    uses_technical_indicators BOOLEAN,
    technical_indicators STRING,  -- JSON array

    -- Parameters
    configurable_parameters STRING,  -- JSON
    default_parameters STRING,  -- JSON

    -- Analysis
    best_suited_for STRING,
    limitations STRING,
    assumptions STRING,
    example_scenario STRING
)
```

**Active Strategy Table**
```sql
CREATE TABLE commodity.whatsapp_llm.active_strategy (
    commodity_id STRING PRIMARY KEY,
    strategy_id STRING,
    strategy_name STRING,

    -- Selection
    activated_date DATE,
    selection_rationale STRING,
    selected_by STRING,  -- Manual, Automated, Backtest

    -- Configuration
    model_version STRING,
    config_parameters STRING,  -- JSON

    -- Review schedule
    next_review_date DATE,
    review_frequency STRING,

    last_updated TIMESTAMP
)
```

#### 3.4 Report Generation (4-Tier Structure)

**Tier 1: Executive Summary (5 minutes)**

*Purpose:* High-level decision support

*Contents:*
- Do predictions help? (Yes/No + $ advantage)
- Best strategy recommendation per commodity
- Statistical significance (p-value)
- Robustness confirmation (persists under cost variations)
- Key chart: Best prediction vs best baseline

*Format:* 1-2 pages, markdown

**Tier 2: Detailed Analysis (30 minutes)**

*Purpose:* Analyst deep dive

*Contents:*
- All 9 strategies compared (sorted by net earnings)
- Statistical validation (t-tests, CIs, effect sizes)
- Risk-return metrics (Sharpe, volatility)
- Sensitivity analysis (parameter heatmaps, cost curves)
- Cross-model/commodity comparisons
- Key charts: Strategy rankings, statistical tests, sensitivity

*Format:* 5-10 pages, markdown

**Tier 3: Technical Appendix (2 hours)**

*Purpose:* Technical validation and methodology

*Contents:*
- Strategy definitions (how each works, formulas)
- Trade-by-trade logs (every sale with reason)
- Daily state tracking (inventory, costs, portfolio value)
- Timeline visualizations (when strategies trade)
- Cost attribution (transaction vs storage breakdown)
- Matched pair analysis (baseline vs predictive side-by-side)
- All charts and data tables

*Format:* Full documentation, markdown + charts

**Tier 4: Validation Report**

*Purpose:* Statistical rigor verification

*Contents:*
- 100% accuracy test results
- Monotonicity validation (60% to 100% comparison)
- Statistical test details (full t-test outputs, bootstrap distributions)
- Parameter optimization results (grid search winners)
- Matched pair validation (proper isolation)

*Format:* Technical report

#### 3.5 Implementation

**Tasks:**
- [ ] Create `generate_consolidated_reports.py`
  - Read backtest outputs (Delta + pickle + CSV)
  - Generate 4-tier markdown reports
  - Organize charts into directories

- [ ] Create `prepare_llm_data.py`
  - Extract strategy performance → parquet
  - Document strategy definitions → json
  - Identify active strategies → json
  - Flatten trade history → parquet

- [ ] Create `archive_results.py`
  - Snapshot current run with timestamp
  - Keep last 10 runs
  - Clean up old archives

- [ ] Create `12_llm_data_export.ipynb`
  - Load llm_data/ files
  - Write to commodity.whatsapp_llm.* tables
  - Validate data quality

**Success Criteria:**
- [ ] 4-tier reports generated automatically
- [ ] Charts organized in logical directories
- [ ] LLM data files created (parquet, json)
- [ ] LLM tables populated in Databricks
- [ ] Production data updated
- [ ] Archive snapshot created
- [ ] No functionality lost from original workflow

**Benefits:**
- Easy human review at all levels
- WhatsApp LLM has all data it needs
- Historical tracking (archived snapshots)
- Organized, not scattered

**Reference:** `trading_agent/commodity_prediction_analysis/CONSOLIDATED_REVIEW_PROPOSAL.md`

---

## PARALLEL WORKSTREAM COORDINATION

**Phase 1 (Algorithm Fixes) - MUST COMPLETE FIRST**
- Diagnostic 100 → Debug → Fix → Validate
- Blocks everything else

**Phase 2 (Automation) - START AFTER PHASE 1**
- Can begin converting notebooks while waiting for final validation
- But don't run automated workflow until strategies confirmed working
- Parallel work: Convert notebooks 01, 05, 06-10 simultaneously

**Phase 3 (Consolidation) - PARALLEL WITH PHASE 2**
- Can design output structure while automation is being built
- Can implement consolidation scripts in parallel
- But don't generate final reports until strategies confirmed working

**Phase 4 (WhatsApp) - AFTER PHASES 1-3**
- LLM code already 75% complete
- Data preparation happens in Phase 3
- Deploy after validated results available

---

### PHASE 4: Deploy WhatsApp LLM (CONVERSATIONAL Q&A)
**Status:** 📋 PLANNED
**Objective:** Conversational WhatsApp bot that answers questions about strategies

**Dependencies:** Phase 3 complete (need LLM data in Databricks tables)

#### 4.1 Complete Lambda Handler Integration

**Current State:**
- llm_context.py has forecast queries ✅
- llm_client.py has Claude integration ✅
- lambda_handler_real.py doesn't use them yet ❌

**Changes Needed:**

```python
# lambda_handler_real.py modifications:

# 1. Add imports
from llm_context import detect_intent, build_llm_context
from llm_client import query_claude, format_llm_response

# 2. Add intent routing
def lambda_handler(event, context):
    intent = detect_intent(message_body)

    if intent == 'commodity_lookup':
        # Fast path (existing code)
        return generate_structured_report()

    elif intent == 'question':
        # LLM path (NEW)
        context = build_llm_context(message_body, commodity, intent)
        response = query_claude(message_body, context, commodity)
        return format_llm_response(response)

    elif intent == 'help':
        return help_message()
```

#### 4.2 Enhance LLM Context Builder

**Add Strategy Query Functions:**

```python
# llm_context.py additions:

def get_strategy_performance_context(commodity, strategy_name):
    """Query strategy_performance table"""
    # Returns: net_earnings, sharpe, advantage, p_value, etc.

def get_active_strategy_context(commodity):
    """Query active_strategy table"""
    # Returns: strategy_name, rationale, activated_date, config

def get_strategy_definition_context(strategy_name):
    """Query strategy_definitions table"""
    # Returns: how it works, formula, assumptions, limitations

def get_strategy_comparison_context(commodity):
    """Compare all strategies"""
    # Returns: rankings, top 3, performance gaps
```

#### 4.3 Deployment

**Tasks:**
- [ ] Add strategy queries to llm_context.py
- [ ] Integrate LLM routing in lambda_handler_real.py
- [ ] Get Anthropic API key
- [ ] Build deployment package (requests + anthropic)
- [ ] Update Lambda config (60s timeout, 512MB memory)
- [ ] Add ANTHROPIC_API_KEY environment variable
- [ ] Deploy to Lambda
- [ ] Test via WhatsApp

**Success Criteria:**
- [ ] Fast path still works (commodity lookups <1s)
- [ ] LLM path works (questions answered 2-4s)
- [ ] Answers reference actual data (strategy performance, definitions)
- [ ] Costs monitored (<$5/day)

**Example Interactions:**

*Fast Path:*
- User: "Coffee"
- Bot: "Coffee - HOLD for 7 days. Current price $105.50, expected gain $142/ton..."

*LLM Path:*
- User: "Which strategy are you using for Coffee?"
- Bot: "I'm using the Expected Value Strategy for Coffee. It was selected based on backtest performance showing +3.4% returns vs +2.1% for the best baseline strategy..."

- User: "How does Expected Value Strategy work?"
- Bot: "The Expected Value Strategy calculates the expected price at each future day (1-14 days), subtracts cumulative storage costs, and identifies the optimal sell day. The formula is: NetValue(t) = E[Price(t)] - (Price(0) × storage_rate × t)..."

- User: "How well is it performing?"
- Bot: "In backtesting over 42 harvest windows, the Expected Value Strategy achieved net earnings of $X/ton with a Sharpe ratio of 1.23 and 62% win rate. It beat the baseline by $2,340 (+3.4%)..."

**Reference:** `trading_agent/whatsapp/LLM_IMPLEMENTATION_PLAN.md`

#### 4.4 Future Enhancement: RAG for Documentation (Optional)

**Status:** Not planned for initial deployment

**Use Case:** Semantic search over large document corpus
- Analysis reports (EXECUTIVE_SUMMARY.md, WORKFLOW_ANALYSIS.md)
- Strategy documentation (detailed methodology)
- Historical analyses (past backtest reports)
- Trade reasoning (natural language explanations)

**Implementation:**
- Embed documents with Anthropic's Contextual Retrieval
- Store in vector DB (Pinecone, Weaviate, or pgvector)
- Hybrid approach: SQL for metrics + RAG for explanations

**Benefits:**
- Handle exploratory questions ("What have we learned about volatility?")
- Search across historical analyses
- Find relevant documentation semantically

**Costs:**
- Embedding API calls (~$0.10 per 1M tokens)
- Vector DB storage (~$50-100/month)
- Increased complexity

**Decision:** Defer until we have >100 documents and users request exploratory search

#### 4.5 Productionization: FastAPI + EC2 Migration (If Scaling)

**Status:** Not needed for capstone demo

**Context:** Current Lambda approach is demo-appropriate (free, auto-scaling, zero ops). For production deployment at scale, consider migration.

**When to migrate:**
- Volume exceeds 10,000 requests/day (Lambda costs become significant)
- Need consistent sub-500ms latency (cold starts problematic)
- Team prefers container-based deployments
- Need longer-running tasks or stateful connections

**Migration effort:**
- Initial setup: ~8 hours (FastAPI app, containerization, EC2, SSL, monitoring)
- Ongoing maintenance: ~1 hour/month (OS updates, container updates, health checks)
- Cost: ~$7.50/month (EC2 t3.micro minimum)

**Trade-off:**
- Lambda (demo): $0/month, zero maintenance, acceptable latency for chat
- FastAPI (production): $7.50-50/month, 1 hr/month maintenance, no cold starts

**Decision:** Use Lambda for capstone demo, document FastAPI migration path for future production scaling.

---

## KEY FILES TO CREATE/MODIFY

### Phase 1:
**Modify:**
- `03_strategy_implementations.ipynb` - Add logging, fix bugs
- `diagnostics/all_strategies_pct.py` - Fix bugs

**Run:**
- `diagnostic_100_algorithm_validation.py` - Validate algorithms
- `11_synthetic_accuracy_comparison.ipynb` - Test monotonicity

### Phase 2:
**Create:**
- `scripts/01_synthetic_predictions.py` - Synthetic data generation
- `scripts/02_forecast_predictions.py` - Real forecast loading
- `scripts/05_strategy_comparison.py` - Core strategy comparison
- `scripts/06_statistical_validation.py` - Statistical testing
- `scripts/07_feature_importance.py` - Feature analysis
- `scripts/08_sensitivity_analysis.py` - Sensitivity testing
- `scripts/09_strategy_results_summary.py` - Results aggregation
- `scripts/10_paired_scenario_analysis.py` - Paired analysis
- `run_complete_analysis.py` - Master orchestrator

**Migrate:**
- Integrate diagnostics 16, 17, 100 into orchestration workflow

### Phase 3:
**Create:**
- `generate_consolidated_reports.py` - Generate 4-tier markdown reports
- `organize_charts.py` - Organize 220+ charts into directories
- `prepare_llm_data.py` - Prepare strategy_performance.parquet, strategy_definitions.json, etc.
- `archive_results.py` - Snapshot results with timestamp
- `12_llm_data_export.ipynb` - Load LLM data into Databricks tables

**Create Tables:**
- `commodity.whatsapp_llm.strategy_performance` - All metrics
- `commodity.whatsapp_llm.strategy_definitions` - How strategies work
- `commodity.whatsapp_llm.active_strategy` - Current selections
- `commodity.whatsapp_llm.trade_history` (optional) - Individual trades

### Phase 4:
**Modify:**
- `whatsapp/llm_context.py` - Add strategy query functions
- `whatsapp/lambda_handler_real.py` - Add LLM routing and integration

**Deploy:**
- Update Lambda function with LLM modules
- Configure environment variables (ANTHROPIC_API_KEY)
- Test in production WhatsApp

---

## SUCCESS CRITERIA BY PHASE

### Phase 1: Algorithm Validation
- [ ] diagnostic_100: Predictions beat baselines by >10% at 100% accuracy
- [ ] Monotonicity: 60% < 70% < 80% < 90% < 100% performance
- [ ] Synthetic 90%: Predictions beat baselines by 10-20%
- [ ] Matched pairs: Clear divergence between baseline and predictive versions
- [ ] Statistical significance: p < 0.05 for prediction advantage
- [ ] All strategies fixed and validated

### Phase 2: Automation
- [ ] Single command runs entire workflow
- [ ] Audit complete (identify essential vs obsolete notebooks)
- [ ] All notebooks (01, 05, 06-10) converted to automated scripts
- [ ] Orchestrator chains dependencies correctly
- [ ] All outputs saved to /Volumes/ automatically
- [ ] Diagnostics 16, 17, 100 integrated into workflow
- [ ] Can run overnight without manual intervention
- [ ] Summary report generated automatically
- [ ] End-to-end test passes

### Phase 3: Consolidation
- [ ] 4-tier reports generated (executive, detailed, technical, validation)
- [ ] Charts organized in logical directories (performance, statistical, sensitivity, timelines, diagnostics)
- [ ] LLM data files created (strategy_performance.parquet, strategy_definitions.json, active_strategy.json, etc.)
- [ ] LLM tables populated in Databricks (strategy_performance, strategy_definitions, active_strategy)
- [ ] Production data files updated (latest_recommendations.json, model_selections.json)
- [ ] Archive snapshots created with timestamps (keep last 10 runs)
- [ ] No functionality lost from original workflow
- [ ] Obsolete files cleaned up/archived

### Phase 4: WhatsApp LLM
- [ ] LLM context builder has strategy queries (get_strategy_performance_context, get_active_strategy_context, etc.)
- [ ] Lambda handler routes to LLM for questions
- [ ] Fast path preserved for commodity lookups (<1s response)
- [ ] WhatsApp bot answers strategy questions accurately
- [ ] Answers reference actual backtest data
- [ ] Costs <$5/day
- [ ] Response times: Fast path <1s, LLM path 2-4s
- [ ] Deployment package includes anthropic + requests libraries
- [ ] ANTHROPIC_API_KEY configured in Lambda environment

---

## CURRENT STATUS TRACKING

### Component Status Legend
- ✅ OPERATIONAL: Working in production
- 🔧 IN PROGRESS: Actively being worked on
- 📋 PLANNED: Designed but not started
- ❌ BLOCKED: Cannot proceed due to dependencies
- ⚠️ PARTIAL: Some functionality works, some doesn't

### Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Data Pipelines | ✅ OPERATIONAL | research_agent managing |
| Forecast Agent | ✅ OPERATIONAL | Producing predictions daily |
| Trading Agent - Algorithms | 🔧 IN PROGRESS | Debugging prediction underperformance |
| Trading Agent - Automation | 🔧 IN PROGRESS | Runners complete, orchestration next |
| Trading Agent - Consolidation | 📋 PLANNED | Design complete, implementation pending |
| Daily Operations | 📋 PLANNED | Script exists, not scheduled |
| WhatsApp - Fast Path | ✅ OPERATIONAL | Structured reports working |
| WhatsApp - LLM Path | ⚠️ PARTIAL | Code 75% complete, no data yet |

### Phase Progress

| Phase | Status | Progress | Blockers |
|-------|--------|----------|----------|
| Phase 1: Algorithm Fix | 🔧 IN PROGRESS | 30% | Debugging prediction logic |
| Phase 2: Automation | 🔧 IN PROGRESS | 45% | Strategies + Runners complete, Orchestration next |
| Phase 3: Consolidation | 📋 PLANNED | 0% | Blocked by Phase 1 + Phase 2 |
| Phase 4: WhatsApp LLM | 📋 PLANNED | 0% | Blocked by Phase 3 |

### Next Immediate Actions

**Phase 1 (User working separately):**
1. Run diagnostic_100
2. Debug strategy implementations in `all_strategies_pct.py`
3. Validate fixes with monotonicity test
4. Confirm 90% synthetic beats baselines

**Phase 2 (IN PROGRESS - 2025-11-24):**
1. ✓ Extract strategies to production/strategies/ (COMPLETE)
   - Import strategies from `production.strategies`
   - When Phase 1 fixes bugs, automation automatically uses fixed version
2. ✓ Build production runners (COMPLETE)
   - Built production/runners/ to replicate notebook 05
   - 5 modular components: data_loader, strategy_runner, visualization, result_saver, multi_commodity_runner
   - Comprehensive test suite: 6 test files, 2,500+ lines, 150+ test cases
3. 📋 Convert notebook 01 to script (synthetic predictions) - NEXT
4. 📋 Build orchestrator (chain 01 → 05)
5. 📋 Test end-to-end
6. 📋 Expand to remaining notebooks (06-10)

**Phase 3 (After Phase 2):**
1. Design final output structure
2. Build consolidation scripts
3. Generate 4-tier reports
4. Prepare LLM data files
5. Populate Databricks tables

**Phase 4 (After Phase 3):**
1. Add strategy queries to llm_context.py
2. Integrate LLM routing in lambda_handler
3. Deploy to Lambda
4. Test in production

---

## DEPENDENCIES & CRITICAL PATH

```
Phase 1 (Algorithm Fix) ─────────────┐
    ↓                                │
Daily Operations                     │
(needs validated strategies)         │
                                     │
Phase 2 (Automation) ────────────────┤ (PARALLEL - strategies are modular)
    ↓                                │
Phase 3 (Consolidation) ─────────────┘
(needs automated workflow + working algorithms)
    ↓
Phase 4 (WhatsApp LLM)
(needs consolidated data)
```

**Critical Path:**
1. Phase 1 + Phase 2 **can run in parallel** (strategies are modular)
   - Phase 1 fixes bugs in `all_strategies_pct.py`
   - Phase 2 builds automation that imports from `all_strategies_pct.py`
   - When bugs fixed, automation automatically uses corrected version
2. Phase 3 requires BOTH Phase 1 and Phase 2 complete
   - Need working algorithms (Phase 1)
   - Need automated workflow (Phase 2)
3. Phase 4 requires Phase 3 complete (need consolidated data in tables)

**Why This Order:**
- Phase 1 & 2 parallel: Modularity allows independent work, fixes propagate automatically
- Phase 3 waits for both: Can't consolidate broken results, can't consolidate without automation
- Phase 4 waits for Phase 3: Can't deploy LLM until data is structured (no tables to query)
- Daily Operations waits for Phase 1: Can't deploy broken strategies to production

---

## KEY DESIGN DECISIONS

### 1. Automation Pattern
**Decision:** Replicate diagnostic automation pattern for main workflow
**Rationale:** Proven approach (diagnostics 16/17/100 work), minimal risk
**Alternative Considered:** Airflow/Prefect orchestration (too complex)

### 2. Output Structure
**Decision:** 3-tier reports + LLM data + production + archive
**Rationale:** Serves all audiences (executive, analyst, technical, LLM)
**Alternative Considered:** Single comprehensive report (too dense)

### 3. LLM Data Format
**Decision:** Denormalized parquet + json files → Databricks tables
**Rationale:** Simple queries, human-readable, LLM-friendly
**Alternative Considered:** Normalized relational (too complex for LLM)

### 4. WhatsApp Dual Path
**Decision:** Fast path (structured) + LLM path (conversational)
**Rationale:** Best of both worlds (speed + flexibility)
**Alternative Considered:** LLM-only (too slow, too expensive)

### 6. Lambda vs FastAPI for WhatsApp Webhook
**Decision:** Stick with AWS Lambda (current implementation)
**Rationale:**
- **Serverless:** No infrastructure to manage, auto-scales
- **Cost-effective:** Pay per request (~$0.20 per 1M requests), not always-on
- **Right fit:** Request pattern is sporadic/bursty (users send messages occasionally)
- **Already working:** Fast path deployed and operational
- **Cold starts acceptable:** ~1-2s cold start is fine for chat (users tolerate 2-5s response)
- **Simple deployment:** Zip upload, no container orchestration
- **Twilio integration:** Well-documented, proven pattern

**Alternative Considered: FastAPI on EC2/ECS/Cloud Run**
- Pros: No cold starts, easier local dev, more control
- Cons: Always-on cost ($20-50/month), infrastructure to manage, overkill for webhook
- Would make sense if: High volume (>1000 req/min), need WebSockets, complex background tasks

**Conclusion:** Lambda is ideal for webhook handlers - serverless, scalable, cost-effective

### 5. Strategy Implementation
**Decision:** Keep strategies in separate module (`all_strategies_pct.py`)
**Rationale:**
- Modularity allows Phase 1 (debugging) and Phase 2 (automation) to work in parallel
- Automation scripts import from module - when bugs fixed, automatically use corrected version
- Proven pattern: Diagnostics already work this way
- Notebook version exists for integrated workflow, but automation uses module
**Alternative Considered:** Strategies embedded in notebooks (blocks parallelization)

---

## REFERENCE DOCUMENTATION

### By Component
- **Data Pipelines:** `research_agent/UNIFIED_DATA_ARCHITECTURE.md`
- **Forecast Agent:** `forecast_agent/README.md`, `forecast_agent/docs/ARCHITECTURE.md`
- **Trading Agent:** `trading_agent/commodity_prediction_analysis/COMPREHENSIVE_FILE_INVENTORY.md`
- **Diagnostics:** `trading_agent/commodity_prediction_analysis/diagnostics/MASTER_DIAGNOSTIC_PLAN.md`
- **WhatsApp LLM:** `trading_agent/whatsapp/LLM_IMPLEMENTATION_PLAN.md`

### By Phase
- **Phase 1:** `trading_agent/commodity_prediction_analysis/EXECUTIVE_SUMMARY.md` (bug docs)
- **Phase 2:** `trading_agent/commodity_prediction_analysis/diagnostics/DATABRICKS_OUTPUT_ACCESS_GUIDE.md` (automation pattern)
- **Phase 3:** `trading_agent/commodity_prediction_analysis/CONSOLIDATED_REVIEW_PROPOSAL.md` (report structure)
- **Phase 4:** `trading_agent/whatsapp/LLM_IMPLEMENTATION_PLAN.md` (deployment details)

### Cross-Cutting
- **Documentation Strategy:** `docs/DOCUMENTATION_STRATEGY.md`
- **Workflow Instructions:** `CLAUDE.md` (root level)
- **Refactoring Plan:** `trading_agent/REFACTORING_PLAN.md` (deferred)

---

## PRINCIPLES & CONSTRAINTS

### Guiding Principles
1. **Fix first, automate second** - Don't automate broken code
2. **Validate rigorously** - Statistical significance required
3. **Document comprehensively** - Easy to review at all levels
4. **Structure for queries** - Both human and LLM consumption
5. **Preserve functionality** - Don't lose capabilities during reorganization

### Technical Constraints
- Databricks execution environment (PySpark, Unity Catalog)
- AWS Lambda limits (60s timeout, 512MB memory, 50MB package size)
- WhatsApp message size limits (~4KB)
- Claude API rate limits (not a concern at current volume)
- Cost constraints (<$5/day for LLM)

### Business Constraints
- Daily forecast updates (predictions change daily)
- Periodic strategy review (quarterly or as-needed)
- Defensible recommendations (statistical validation required)
- User-friendly delivery (WhatsApp, multiple detail levels)

---

## RISK MITIGATION

### Risk: Algorithm bugs persist after Phase 1
**Impact:** Cannot proceed to Phase 2
**Mitigation:** Extensive logging, multiple validation tests, matched pair analysis
**Fallback:** Manual review of trade-by-trade logs to isolate bug

### Risk: Automation introduces new bugs
**Impact:** Results differ from manual execution
**Mitigation:** Compare automated vs manual results on same input
**Fallback:** Keep manual notebooks working alongside automation

### Risk: LLM provides incorrect information
**Impact:** User mistrust, bad recommendations
**Mitigation:** Validate LLM answers against source data, add disclaimers
**Fallback:** Disable LLM path, keep fast path only

### Risk: Cost overruns from LLM usage
**Impact:** Budget exceeded
**Mitigation:** CloudWatch alerts at $5/day, use cheap Haiku model, 500 token limit
**Fallback:** Disable LLM, revert to structured reports only

---

## OPEN QUESTIONS & DECISIONS NEEDED

### Phase 1
- [ ] Has diagnostic_100 been run yet? What were results?
- [ ] Has root cause of prediction underperformance been identified?
- [ ] Should monotonicity validation be separate notebook or integrated?

### Phase 2
- [ ] Which notebooks have priority for automation? (Recommend: 01, 05 first)
- [ ] Should we parallelize analysis notebooks (06-10) or run sequentially?
- [ ] What frequency for automated runs? (Daily, weekly, on-demand?)

### Phase 3
- [ ] What level of detail in executive summary? (1 page vs 2 pages?)
- [ ] Should archive keep all runs or just last N? (Recommend: last 10)
- [ ] How to handle obsolete files? (Archive vs delete?)

### Phase 4
- [ ] Deploy LLM immediately after Phase 3 or wait for user testing?
- [ ] What cost threshold triggers alert? (Currently $5/day)
- [ ] Should we A/B test different Claude models? (Haiku vs Sonnet?)

---

## CHANGELOG

**2025-11-24:** Initial comprehensive master plan created
- Cataloged all system components
- Defined 4 execution phases
- Documented current status
- Identified dependencies and critical path

**2025-11-24 (Update 1):** Corrected Phase 1/2 dependency
- Phase 1 (Algorithm Fix) and Phase 2 (Automation) can run **in parallel**
- Modularity of `all_strategies_pct.py` allows independent work
- Bug fixes automatically propagate to automation when complete
- Phase 3 still requires both Phase 1 and Phase 2 complete

**2025-11-24 (Update 2):** Added future enhancements
- RAG for documentation (Section 4.4) - defer until >100 documents
- FastAPI + EC2 migration path (Section 4.5) - productionization option
- Lambda vs FastAPI comparison (Key Design Decision #6)
- Rationale: Lambda appropriate for demo, document alternatives for production scaling

---

**Document Owner:** System Integration
**Status:** Living Document (update as phases progress)
**Review Frequency:** After each phase completion

# Trading Agent - Modern Analysis Suite

**Purpose:** Modern analysis tools for strategy evaluation and tuning

**Status:** ✅ Phase 1 Complete (Core Framework)

---

## Overview

This directory contains the **NEW** analysis framework using the theoretical maximum benchmark approach. This is separate from the older `diagnostics/` directory which used paired t-tests.

### Distinction from Other Directories:

**`diagnostics/`** (OLD - Keep for Reference)
- Uses paired t-tests on daily portfolio changes
- Bootstrap confidence intervals
- Historical approach - keep for reference but not actively used

**`production/`** (Operational)
- `run_backtest_workflow.py` - Strategy selection for production use
- Runs monthly/quarterly to identify best strategies
- Pure operational focus

**`analysis/`** (NEW - This Directory)
- Theoretical maximum benchmark
- Efficiency ratio analysis
- Strategy effectiveness evaluation
- Run as needed for strategy research and tuning

---

## Components

### theoretical_max/ ✅
Calculates the best possible performance with perfect foresight using dynamic programming.

**Key Concept:**
- What's the BEST we could do if we knew future prices perfectly?
- Provides upper bound for strategy performance

**Implementation:**
- `TheoreticalMaxCalculator` class with dynamic programming algorithm
- Discretized inventory levels for computational efficiency
- Backward induction from last day to first day
- Considers storage costs, transaction costs, and future value

### efficiency/ ✅
Analyzes how efficiently strategies exploit available information.

**Key Metrics:**
- **Efficiency Ratio** = Actual Earnings / Theoretical Max Earnings
- Reveals how much value is left on the table
- Decision-by-decision breakdown
- Efficiency categories: EXCELLENT (≥80%), GOOD (70-80%), MODERATE (60-70%), POOR (<60%)

**Implementation:**
- `EfficiencyAnalyzer` class with comparative analysis
- Summary reports and interpretations
- Critical decision identification

### run_strategy_analysis.py ✅
Main orchestrator for running comprehensive strategy analysis.

**Features:**
- Loads price data and predictions
- Calculates theoretical maximum
- Compares actual strategy results
- Generates efficiency reports
- Saves results to Unity Catalog volumes

---

## Usage

### Basic Analysis

```bash
cd /Users/markgibbons/capstone/ucberkeley-capstone/trading_agent/commodity_prediction_analysis

# Analyze specific commodity and model
python analysis/run_strategy_analysis.py --commodity coffee --model arima_v1

# Compare all available models for a commodity
python analysis/run_strategy_analysis.py --commodity coffee --compare-all

# Use custom results table
python analysis/run_strategy_analysis.py --commodity coffee --model arima_v1 \
    --results-table commodity.trading_agent.custom_results
```

### Output Files

Results are saved to `/Volumes/commodity/trading_agent/files/analysis/`:

```
analysis/
├── theoretical_max_decisions_{commodity}_{model}.csv    # Optimal decision path
├── efficiency_analysis_{commodity}_{model}.csv          # Efficiency ratios by strategy
└── analysis_summary_{commodity}_{model}.pkl             # Complete analysis (pickle)
```

### Example Output

```
================================================================================
THEORETICAL MAXIMUM (Perfect Foresight + Optimal Policy)
================================================================================
Net Earnings:        $45,234.56
Total Revenue:       $50,000.00
Transaction Costs:   $500.00
Storage Costs:       $4,265.44
Number of Trades:    12

================================================================================
EFFICIENCY ANALYSIS
================================================================================
                       Strategy  Actual Earnings  Theoretical Max  Efficiency %  Opportunity Gap  Category
    Risk-Adjusted (Prediction)       $38,456.78       $45,234.56          85.0%        $6,777.78  EXCELLENT
         Consensus (Prediction)       $36,789.01       $45,234.56          81.3%        $8,445.55  EXCELLENT
  Expected Value (Prediction)         $34,123.45       $45,234.56          75.4%       $11,111.11  GOOD
...

✅ EXCELLENT: Best strategy achieves >80% efficiency
   Our algorithms are effectively exploiting available predictions.
```

---

## Why Separate from Diagnostics?

**diagnostics/** used the OLD approach:
- Paired t-tests on daily changes
- Problem: Daily variance >> signal, no statistical significance
- See `diagnostics/ALGORITHM_PERFORMANCE_ANALYSIS.md` for details

**analysis/** uses the NEW approach:
- Theoretical maximum benchmark (dynamic programming)
- Efficiency ratios show if strategies can exploit predictions
- Much clearer signal about strategy effectiveness

---

## Why Separate from Production?

**production/** is for operational decisions:
- "Which strategy should we use this month?"
- Runs on a schedule (monthly/quarterly)
- Output: Best strategy selection

**analysis/** is for research and tuning:
- "How good could our strategies be?"
- "Are we exploiting predictions effectively?"
- Runs on-demand when designing/tuning strategies
- Output: Insights for improvement

---

## Development Plan

### Phase 1: Core Framework ✅ COMPLETE
- [✅] Extract theoretical max calculation from diagnostics
- [✅] Create clean efficiency analysis module
- [✅] Build analysis orchestrator

**Delivered:**
- `theoretical_max/calculator.py` - 280+ lines, clean DP implementation
- `efficiency/analyzer.py` - 240+ lines, comprehensive efficiency analysis
- `run_strategy_analysis.py` - 360+ lines, full orchestrator

### Phase 2: Visualization (Future)
- [ ] Efficiency ratio charts (bar charts, heatmaps)
- [ ] Decision-by-decision comparison (line plots)
- [ ] Strategy heatmaps (efficiency matrix)
- [ ] Opportunity gap analysis (waterfall charts)

### Phase 3: Parameter Optimization (Future Migration)
- [ ] Extract Optuna optimization from diagnostics/
- [ ] Create `analysis/optimization/` module
- [ ] Modernize diagnostic_16 → `run_parameter_optimization.py`
- [ ] Integrate with efficiency analysis (optimize for efficiency ratio, not raw earnings)
- [ ] Support multi-objective optimization (earnings vs risk vs trade frequency)

**Current state:** Parameter optimization exists in `diagnostics/run_diagnostic_16.py` using Optuna.
This should be migrated to `analysis/` and integrated with the theoretical max benchmark.

### Phase 4: Integration (Future)
- [ ] Add to Databricks job scheduling
- [ ] Create example analyses for coffee/sugar
- [ ] Document best practices and use cases
- [ ] Link to production workflow documentation

---

## Relationship to Diagnostics

### What's in diagnostics/ (To Be Migrated)

**diagnostics/** contains several analysis tools that should eventually move to `analysis/`:

1. **Parameter Optimization** (Diagnostic 16)
   - `run_diagnostic_16.py` - Optuna optimization with 200 trials per strategy
   - Finds best parameters for each of the 9 strategies
   - Saves `diagnostic_16_best_params.pkl`
   - **Should migrate to:** `analysis/optimization/`

2. **Theoretical Maximum** (Already Migrated!)
   - `run_diagnostic_theoretical_max.py` - DP-based optimal policy
   - **Migrated to:** `analysis/theoretical_max/calculator.py` ✅

3. **Paradox Analysis** (Diagnostic 17)
   - `run_diagnostic_17.py` - Trade-by-trade comparison
   - Uses optimized parameters from diagnostic 16
   - **Should integrate with:** `analysis/efficiency/analyzer.py`

### Migration Priority

1. ✅ **DONE:** Theoretical max → `analysis/theoretical_max/`
2. 🔄 **NEXT:** Parameter optimization → `analysis/optimization/`
3. 🔄 **THEN:** Paradox analysis → integrate with efficiency analysis

---

## Related Documentation

- **OLD Diagnostics:** `../diagnostics/MASTER_DIAGNOSTIC_PLAN.md`
- **Parameter Optimization:** `../diagnostics/run_diagnostic_16.py` (to be migrated)
- **Theoretical Max (OLD):** `../diagnostics/run_diagnostic_theoretical_max.py` (reference only)
- **Production System:** `../production/README.md`
- **Master Plan:** `../../MASTER_SYSTEM_PLAN.md`

---

**Created:** 2025-11-24
**Status:** Phase 1 Complete (Core Framework)
**Owner:** Trading Agent Team

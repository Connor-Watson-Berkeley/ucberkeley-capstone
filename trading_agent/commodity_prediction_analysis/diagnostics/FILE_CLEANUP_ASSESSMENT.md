# Diagnostics Folder - File Cleanup Assessment

**Date:** 2025-11-24
**Purpose:** Document which files to keep, archive, or delete

---

## ✅ KEEP - Active Files

### Strategy Implementations
- **`all_strategies_pct.py`** (54K) - Current version with all 9 strategies
  - Used by: diagnostic_16, diagnostic_17, diagnostic_100
  - Status: ✅ ACTIVE

### Diagnostic Notebooks
- **`diagnostic_16_optuna_with_params.ipynb`** (20K) - Grid search optimization
  - Saves: `diagnostic_16_best_params.pkl`
  - Status: ✅ ACTIVE

- **`diagnostic_17_paradox_analysis.ipynb`** (28K) - Trade analysis
  - Loads: `diagnostic_16_best_params.pkl`
  - Status: ✅ ACTIVE (newer version, keep this one)

### Diagnostic Scripts
- **`diagnostic_100_algorithm_validation.py`** (13K) - 100% accuracy test
  - Status: ✅ ACTIVE (new)

### Support Files
- **`cost_config_small_farmer.py`** (1.4K) - Cost configurations
  - Used by: strategies
  - Status: ✅ ACTIVE

- **`test_all_strategies.py`** (4.2K) - Smoke tests
  - Status: ✅ ACTIVE (useful for quick validation)

### Master Documentation
- **`MASTER_DIAGNOSTIC_PLAN.md`** (17K) - Complete workflow guide
  - Status: ✅ ACTIVE (just created, single source of truth)

- **`ACCURACY_LEVEL_TESTING_GUIDE.md`** (8.9K) - 100% testing guide
  - Status: ✅ ACTIVE (just created)

---

## 🗑️ DELETE - Obsolete/Duplicate Files

### Duplicate Notebook
- **`diagnostic_17_prediction_paradox.ipynb`** (28K)
  - Reason: Exact duplicate of `diagnostic_17_paradox_analysis.ipynb`
  - Older timestamp: Nov 23 22:30 (vs 22:31)
  - Action: **DELETE**

### Obsolete Strategy Versions
- **`fixed_strategies.py`** (21K)
  - Reason: Earlier bug fix attempt, superseded by `all_strategies_pct.py`
  - Historical only
  - Action: **DELETE**

- **`corrected_strategies.py`** (26K)
  - Reason: Earlier iteration, superseded by `all_strategies_pct.py`
  - Historical only
  - Action: **DELETE**

### Obsolete Documentation
- **`SYNTHETIC_PREDICTION_TEST_PLAN.md`** (15K)
  - Reason: References non-existent diagnostic_01-20
  - Superseded by: MASTER_DIAGNOSTIC_PLAN.md
  - Action: **DELETE**

- **`DIAGNOSTIC_EXECUTION_GUIDE.md`** (13K)
  - Reason: References non-existent diagnostic_01-07
  - Superseded by: MASTER_DIAGNOSTIC_PLAN.md
  - Action: **DELETE**

- **`COMPREHENSIVE_GRID_SEARCH_PLAN.md`** (10K)
  - Reason: Superseded by actual implementation (diagnostic_16)
  - Historical planning doc
  - Action: **DELETE**

- **`DEBUGGING_PLAN.md`** (10K)
  - Reason: Historical debugging strategy
  - Superseded by: MASTER_DIAGNOSTIC_PLAN.md
  - Action: **DELETE**

- **`CONSOLIDATED_ANALYSIS.md`** (6.9K)
  - Reason: Historical analysis
  - Superseded by: actual diagnostic notebooks
  - Action: **DELETE**

- **`BUG_FIX_SUMMARY.md`** (6.7K)
  - Reason: Historical bug fix record
  - Could keep as historical reference, but...
  - Action: **DELETE** (info captured in git history)

---

## ❓ REVIEW - Utility Documentation

These are utility guides that may still be useful but should be reviewed:

- **`DATABRICKS_OUTPUT_ACCESS_GUIDE.md`** (8.8K)
  - Purpose: How to access Databricks outputs
  - Recommendation: **REVIEW** - May have useful info to merge into MASTER plan
  - Decision: TBD

- **`DATABRICKS_QUERY_GUIDE.md`** (2.6K)
  - Purpose: Quick reference for Databricks queries
  - Recommendation: **KEEP** (small, useful reference)
  - Decision: KEEP

- **`HOW_TO_GET_NOTEBOOK_RESULTS.md`** (3.8K)
  - Purpose: Guide to downloading results from Databricks
  - Recommendation: **REVIEW** - Merge useful parts into MASTER or parent DATABRICKS_ACCESS_NOTES.md
  - Decision: TBD

- **`RESULTS_ACCESS_ANALYSIS.md`** (5.9K)
  - Purpose: Analysis of how to access results
  - Recommendation: **DELETE** (likely superseded)
  - Decision: DELETE

- **`rebuild_notebook.py`** (3.1K)
  - Purpose: Utility script (unclear what it does)
  - Recommendation: **REVIEW** - Check if still needed
  - Decision: TBD

---

## Summary

### Files to Delete (10 files)

1. `diagnostic_17_prediction_paradox.ipynb` - Duplicate
2. `fixed_strategies.py` - Obsolete strategy version
3. `corrected_strategies.py` - Obsolete strategy version
4. `SYNTHETIC_PREDICTION_TEST_PLAN.md` - References non-existent files
5. `DIAGNOSTIC_EXECUTION_GUIDE.md` - References non-existent files
6. `COMPREHENSIVE_GRID_SEARCH_PLAN.md` - Superseded by diagnostic_16
7. `DEBUGGING_PLAN.md` - Superseded by MASTER_DIAGNOSTIC_PLAN
8. `CONSOLIDATED_ANALYSIS.md` - Historical
9. `BUG_FIX_SUMMARY.md` - Historical
10. `RESULTS_ACCESS_ANALYSIS.md` - Likely obsolete

### Files to Keep (9 files)

1. `all_strategies_pct.py` - Active strategy implementations
2. `diagnostic_16_optuna_with_params.ipynb` - Active
3. `diagnostic_17_paradox_analysis.ipynb` - Active
4. `diagnostic_100_algorithm_validation.py` - Active
5. `cost_config_small_farmer.py` - Active
6. `test_all_strategies.py` - Active
7. `MASTER_DIAGNOSTIC_PLAN.md` - Active
8. `ACCURACY_LEVEL_TESTING_GUIDE.md` - Active
9. `DATABRICKS_QUERY_GUIDE.md` - Utility reference

### Files to Review (3 files)

1. `DATABRICKS_OUTPUT_ACCESS_GUIDE.md` - May have useful content
2. `HOW_TO_GET_NOTEBOOK_RESULTS.md` - May have useful content
3. `rebuild_notebook.py` - Unclear if needed

---

## Cleanup Actions

### Immediate Actions (Safe to delete)

```bash
cd /Users/markgibbons/capstone/ucberkeley-capstone/trading_agent/commodity_prediction_analysis/diagnostics/

# Delete duplicate
rm diagnostic_17_prediction_paradox.ipynb

# Delete obsolete strategy versions
rm fixed_strategies.py
rm corrected_strategies.py

# Delete obsolete documentation
rm SYNTHETIC_PREDICTION_TEST_PLAN.md
rm DIAGNOSTIC_EXECUTION_GUIDE.md
rm COMPREHENSIVE_GRID_SEARCH_PLAN.md
rm DEBUGGING_PLAN.md
rm CONSOLIDATED_ANALYSIS.md
rm BUG_FIX_SUMMARY.md
rm RESULTS_ACCESS_ANALYSIS.md
```

### After Cleanup - Directory Structure

```
diagnostics/
├── all_strategies_pct.py                        ← Core strategies
├── cost_config_small_farmer.py                  ← Config
├── test_all_strategies.py                       ← Testing
│
├── diagnostic_16_optuna_with_params.ipynb       ← Grid search
├── diagnostic_17_paradox_analysis.ipynb         ← Analysis
├── diagnostic_100_algorithm_validation.py       ← 100% test
│
├── MASTER_DIAGNOSTIC_PLAN.md                    ← Main guide
├── ACCURACY_LEVEL_TESTING_GUIDE.md             ← Testing guide
├── DATABRICKS_QUERY_GUIDE.md                    ← Utility
│
└── [Files to review]
    ├── DATABRICKS_OUTPUT_ACCESS_GUIDE.md
    ├── HOW_TO_GET_NOTEBOOK_RESULTS.md
    └── rebuild_notebook.py
```

---

## Before/After Comparison

**Before:** 22 files (many obsolete, duplicates, references to non-existent files)
**After:** 9-12 files (clean, focused, all active or useful)

**Space saved:** ~120K of obsolete documentation
**Clarity gained:** Single source of truth (MASTER_DIAGNOSTIC_PLAN.md)

---

**Status:** Ready to execute cleanup
**Risk:** Low (all deletions are obsolete/duplicate files)
**Backup:** Git history preserves everything

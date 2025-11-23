# Results Access Analysis

**Created:** 2025-11-22
**Purpose:** Document why production results are accessible but diagnostic_12/13 results are not

---

## What Works: Production Results ✅

**File:** `results_detailed_coffee_synthetic_acc90.pkl`
**Location:** `/dbfs/Volumes/commodity/trading_agent/files/`
**Created by:** Notebook `05_strategy_comparison.ipynb`

**How it saves (from 05_strategy_comparison.ipynb:450):**
```python
# Save detailed results
with open(MODEL_DATA_PATHS['results_detailed'], 'wb') as f:
    pickle.dump(results_dict, f)
print(f"  ✓ Saved: {MODEL_DATA_PATHS['results_detailed']}")
```

**Where MODEL_DATA_PATHS comes from:**
```python
MODEL_DATA_PATHS = get_data_paths(CURRENT_COMMODITY, MODEL_VERSION)
```

This resolves to:
```
/dbfs/Volumes/commodity/trading_agent/files/results_detailed_coffee_synthetic_acc90.pkl
```

**Why this works:**
- Production notebook runs in Databricks
- Saves directly to `/dbfs/Volumes/` path (Unity Catalog Volume)
- File persists after notebook execution
- Accessible via `databricks fs cp` command
- Can be downloaded and analyzed locally

**Current status:**
- ✅ File exists
- ✅ Downloaded successfully
- ✅ Contains BUGGY strategy results (baseline + prediction strategies with defer bug)
- Last modified: 2025-11-22T16:49:39Z

---

## What Doesn't Work: Diagnostic Results ❌

**Expected files:**
- `diagnostic_12_results.pkl`
- `diagnostic_12_summary.json`
- `diagnostic_13_results.pkl`
- `diagnostic_13_summary.json`

**Expected location:** `/dbfs/Volumes/commodity/trading_agent/files/`

**How diagnostics should save (from updated diagnostic_12:cell-18):**
```python
# Save to accessible location (Volume)
volume_path = '/dbfs/Volumes/commodity/trading_agent/files/diagnostic_12_results.pkl'
with open(volume_path, 'wb') as f:
    pickle.dump(optimal_params, f)
```

**Why this doesn't work - Root Causes:**

### Issue 1: Notebooks Might Have Run Before Git Pull
- User pushed changes at commit `425a731`
- User ran diagnostics
- User pulled git (got commit `7f5ada8`) AFTER running
- **Result:** Diagnostics ran with OLD code that didn't have save logic

### Issue 2: Databricks Repos Cache
Even if git was pulled:
- Databricks may cache notebook content in browser
- Need to refresh/reload notebook after git pull
- Or restart kernel
- **Result:** Old code runs even after git pull

### Issue 3: Execution Never Reached Save Cell
- Diagnostic notebooks are long
- If any cell errors before the final save cell
- Results are computed but never saved
- **Result:** Results exist in memory but not persisted

### Issue 4: Path Issues
- Typo in path
- Permission issues writing to Volume
- Volume not mounted correctly
- **Result:** Save silently fails or errors

---

## Comparison: What's Different

| Aspect | Production (Works) | Diagnostics (Doesn't Work) |
|--------|-------------------|---------------------------|
| **Save location** | `/dbfs/Volumes/.../results_detailed_...` | `/dbfs/Volumes/.../diagnostic_12_results.pkl` |
| **Save pattern** | Always saves (part of standard workflow) | Only saves in final cell (easy to miss) |
| **Code maturity** | Battle-tested, runs for all models | Just added today |
| **Git status** | Committed weeks ago | Committed hours ago |
| **Execution history** | Ran many times successfully | First run after code addition |

---

## Key Discovery

**The pattern that WORKS:**
```python
with open(MODEL_DATA_PATHS['results_detailed'], 'wb') as f:
    pickle.dump(results_dict, f)
```

This uses `MODEL_DATA_PATHS` from `00_setup_and_config.ipynb` which provides standardized paths.

**The pattern diagnostic_12 uses:**
```python
volume_path = '/dbfs/Volumes/commodity/trading_agent/files/diagnostic_12_results.pkl'
with open(volume_path, 'wb') as f:
    pickle.dump(optimal_params, f)
```

This hardcodes the path - should work but might have issues.

---

## Solution Options

### Option A: Re-run Diagnostics (Recommended)
1. User pulls latest git in Databricks Repos
2. User refreshes notebook in browser (Cmd+R or Ctrl+R)
3. User re-runs diagnostic_12 from start
4. Final cell saves results to Volume
5. Claude can then download and analyze

### Option B: Add Save Cell to Existing Execution
1. User adds new cell at end of already-run notebook
2. Runs this cell to save existing variables:
```python
import pickle
results_to_save = {
    'optimal_params': optimal_params,
    'fixed_results': fixed_results,
    'best_baseline_earnings': best_baseline_earnings
}
with open('/dbfs/Volumes/commodity/trading_agent/files/diagnostic_12_results.pkl', 'wb') as f:
    pickle.dump(results_to_save, f)
```

### Option C: Manual Output Sharing
1. User copies final output from notebook
2. Pastes in chat
3. Claude analyzes manually

### Option D: Use Databricks Repos Commit
1. User runs notebook with "Run All"
2. Outputs are embedded in notebook
3. User commits executed notebook via Databricks Repos UI
4. Claude pulls git and reads notebook outputs
5. (This worked for diagnostic_11!)

---

## Recommendation

**Try Option D first** - it's how we got diagnostic_11 results:
1. In Databricks, use Repos to commit the EXECUTED notebook with outputs
2. Push to git
3. I pull and read the embedded outputs

This is better because:
- No re-execution needed
- Captures exactly what user saw
- No risk of different results on re-run
- Works for both diagnostic_12 and diagnostic_13

---

## Files Status

### Available Now ✅
- `results_detailed_coffee_synthetic_acc90.pkl` - BUGGY results from production
- Downloaded to: `/tmp/results_detailed_fresh.pkl`
- Contains 9 strategies with Expected Value LOSING by $19,020

### Missing ❌
- `diagnostic_12_results.pkl` - Fixed strategy results + grid search
- `diagnostic_12_summary.json` - Human-readable summary
- `diagnostic_13_results.pkl` - Comprehensive grid search all 9 strategies
- `diagnostic_13_summary.json` - Summary with matched pairs

---

**Next Action:** User should commit executed notebooks via Databricks Repos, or re-run after git pull + refresh.

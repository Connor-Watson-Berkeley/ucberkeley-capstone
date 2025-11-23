# Databricks Notebook Output Access Guide

**Created:** 2025-11-22
**Purpose:** Systematic approach to accessing notebook execution results

---

## The Problem

Databricks notebooks have **two states**:
1. **Source** - The code/markdown cells (persisted in git/workspace)
2. **Execution outputs** - Results from running cells (ephemeral, session-based)

**When you export via Workspace API:**
- ✅ Gets source code
- ❌ Does NOT get execution outputs (even with HTML/IPYNB format)

**When you commit via Repos:**
- ✅ Source code is saved
- ❌ Outputs are stripped from .ipynb files

---

## Solutions (In Order of Preference)

### Solution 1: Auto-Save Results to Files ⭐ RECOMMENDED

**How it works:**
- Final cell in notebook saves results to `/dbfs/Volumes/`
- Results persist after notebook execution
- Accessible via `databricks fs cp` command
- Can be downloaded and analyzed programmatically

**Implementation pattern:**
```python
# Last cell of any diagnostic notebook
import pickle
import json
from datetime import datetime

# Collect all important results
results = {
    'timestamp': datetime.now().isoformat(),
    'summary': {
        'best_strategy': best_strategy_name,
        'net_earnings': float(best_earnings),
        # ... other key metrics
    },
    'detailed_results': all_results_dict
}

# Save pickle for programmatic access
pkl_path = f'/dbfs/Volumes/commodity/trading_agent/files/{notebook_name}_results.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(results, f)

# Save JSON for human readability
json_path = pkl_path.replace('.pkl', '.json')
with open(json_path, 'w') as f:
    json.dump(results['summary'], f, indent=2)

print(f"✅ Results saved to:")
print(f"   PKL: {pkl_path}")
print(f"   JSON: {json_path}")
print(f"\n📥 Download with:")
print(f"   databricks fs cp dbfs:{pkl_path.replace('/dbfs', '')} /tmp/")
```

**Advantages:**
- ✅ Fully automated
- ✅ Works for any notebook type
- ✅ Programmatically accessible
- ✅ Persists indefinitely
- ✅ Can version/timestamp results

**Status:**
- ✅ Implemented in updated diagnostic_12 and diagnostic_13
- ❌ User ran notebooks before pulling updates

---

### Solution 2: Run as Databricks Job

**How it works:**
- Create a Job that runs the notebook
- Use Jobs Runs Export API to get executed notebook with outputs
- Outputs are preserved in job run metadata

**Implementation:**
```bash
# Create job
databricks jobs create --json '{
  "name": "diagnostic_12_runner",
  "tasks": [{
    "task_key": "run_diagnostic",
    "notebook_task": {
      "notebook_path": "/Workspace/Repos/.../diagnostic_12_fixed_strategy_validation",
      "base_parameters": {}
    },
    "new_cluster": { ... }
  }]
}'

# Run job
databricks jobs run-now --job-id JOB_ID

# Export run with outputs
databricks runs export --run-id RUN_ID --file /tmp/diagnostic_12_with_outputs.html
```

**Advantages:**
- ✅ Outputs are preserved
- ✅ Can be exported with Runs API
- ✅ Automated execution

**Disadvantages:**
- ❌ Requires job setup
- ❌ More complex for ad-hoc analysis
- ❌ Outputs still in HTML (need parsing)

---

### Solution 3: Save to Spark Tables

**How it works:**
- Write results to Delta tables
- Query tables remotely

**Implementation:**
```python
# In notebook
results_df = pd.DataFrame([{
    'notebook': 'diagnostic_12',
    'timestamp': datetime.now(),
    'strategy': 'Expected Value Fixed',
    'net_earnings': 755000.0,
    'vs_baseline_pct': 3.8
}])

spark.createDataFrame(results_df).write.mode('append').saveAsTable(
    'commodity.trading_agent.diagnostic_results'
)
```

**Query remotely:**
```bash
databricks sql query "SELECT * FROM commodity.trading_agent.diagnostic_results WHERE notebook = 'diagnostic_12' ORDER BY timestamp DESC LIMIT 1"
```

**Advantages:**
- ✅ Structured data
- ✅ Queryable remotely
- ✅ Time-series tracking

**Disadvantages:**
- ❌ Only works for tabular results
- ❌ Requires table schema design
- ❌ Doesn't capture detailed trade-by-trade data well

---

### Solution 4: Manual Export with "Include Outputs"

**How it works:**
- In Databricks UI: File → Export → Notebook
- Check "Include cell outputs"
- Export as .ipynb or HTML
- Upload to git or cloud storage

**Advantages:**
- ✅ Gets ALL outputs
- ✅ Works for interactive runs

**Disadvantages:**
- ❌ Manual process
- ❌ Not automated
- ❌ Requires user action

**Note:** User cannot/will not do this per their requirements.

---

## Recommended Workflow

### For Diagnostic Notebooks

**Every diagnostic should:**

1. **Have a results save cell at the end:**
```python
# Standard pattern for all diagnostics
save_diagnostic_results(
    notebook_name='diagnostic_12',
    results_dict=optimal_params,
    summary_dict=summary_for_json
)
```

2. **Print download instructions:**
```python
print(f"\n{'='*80}")
print(f"RESULTS SAVED - READY FOR DOWNLOAD")
print(f"{'='*80}")
print(f"\nTo download and analyze:")
print(f"  databricks fs cp dbfs:/Volumes/.../diagnostic_12_results.pkl /tmp/")
print(f"  python analyze_results.py /tmp/diagnostic_12_results.pkl")
```

3. **Include timestamp and metadata:**
```python
results = {
    'timestamp': datetime.now().isoformat(),
    'notebook': 'diagnostic_12',
    'git_commit': os.environ.get('GIT_COMMIT', 'unknown'),
    'cluster_id': spark.conf.get('spark.databricks.clusterUsageTags.clusterId'),
    # ... actual results
}
```

### For Production Notebooks

Production notebooks (01-10) already follow this pattern:
- Save to `/dbfs/Volumes/commodity/trading_agent/files/`
- Use standardized naming: `{notebook_name}_{commodity}_{model_version}.pkl`
- Accessible via `databricks fs cp`

---

## Why This Matters

**Current situation:**
1. User runs diagnostic_12 in Databricks ✅
2. Notebook completes successfully ✅
3. Results exist in memory/outputs ✅
4. User tries to access results remotely ❌
5. Workspace export gives source only ❌
6. Outputs are ephemeral, lost when cluster stops ❌

**With auto-save pattern:**
1. User runs diagnostic_12 in Databricks ✅
2. Notebook completes successfully ✅
3. Final cell saves to `/dbfs/Volumes/` ✅
4. Results persist indefinitely ✅
5. Remote download via `databricks fs cp` ✅
6. Programmatic analysis possible ✅

---

## Implementation Status

### ✅ Completed
- diagnostic_12: Updated with auto-save (commit 425a731)
- diagnostic_13: Updated with auto-save (commit 425a731)
- Both save to `/dbfs/Volumes/commodity/trading_agent/files/`
- Both create .pkl and .json files

### ❌ Issue
- User ran notebooks BEFORE pulling git updates
- Notebooks executed with old code (no auto-save)
- Results exist only in ephemeral cluster session
- Cannot be accessed remotely

### 🔄 Solution
**User must re-run diagnostic_12 and diagnostic_13** after:
1. Pulling latest git in Databricks Repos ✅ (already done)
2. Refreshing notebook in browser (Cmd+R / Ctrl+R)
3. Running all cells from start

**Then:**
4. Final cell saves results to Volume
5. Claude downloads via `databricks fs cp`
6. Analysis proceeds

---

## Template: Diagnostic Results Save Function

```python
def save_diagnostic_results(notebook_name, results_dict, summary_dict=None):
    """
    Standard function for saving diagnostic results.

    Args:
        notebook_name: Name of the diagnostic (e.g., 'diagnostic_12')
        results_dict: Full results dictionary (saved as pickle)
        summary_dict: Optional human-readable summary (saved as JSON)
    """
    import pickle
    import json
    from datetime import datetime

    # Base path
    base_path = f'/dbfs/Volumes/commodity/trading_agent/files/{notebook_name}_results'

    # Add timestamp and metadata
    results_with_meta = {
        'timestamp': datetime.now().isoformat(),
        'notebook': notebook_name,
        'results': results_dict
    }

    # Save pickle
    pkl_path = f'{base_path}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(results_with_meta, f)
    print(f"✅ Saved pickle: {pkl_path}")

    # Save JSON summary if provided
    if summary_dict:
        json_path = f'{base_path}.json'
        with open(json_path, 'w') as f:
            json.dump({
                'timestamp': results_with_meta['timestamp'],
                'notebook': notebook_name,
                'summary': summary_dict
            }, f, indent=2)
        print(f"✅ Saved JSON: {json_path}")

    # Print download instructions
    print(f"\n{'='*80}")
    print(f"RESULTS READY FOR DOWNLOAD")
    print(f"{'='*80}")
    print(f"\nDownload command:")
    print(f"  databricks fs cp dbfs:{pkl_path.replace('/dbfs', '')} /tmp/")
    if summary_dict:
        print(f"  databricks fs cp dbfs:{json_path.replace('/dbfs', '')} /tmp/")

    return pkl_path
```

---

## Next Steps

1. **Immediate:** User re-runs diagnostics with updated code
2. **Short-term:** Add `save_diagnostic_results()` to `00_setup_and_config.ipynb`
3. **Long-term:** Consider Job-based execution for full automation

---

**Status:** Documented systematic solution. Waiting for user to re-run diagnostics.

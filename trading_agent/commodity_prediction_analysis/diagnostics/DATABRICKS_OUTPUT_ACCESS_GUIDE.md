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

---

## Solution 5: Automated Remote Execution Pattern ⭐ NEW RECOMMENDED

**Created:** 2025-11-24
**Purpose:** Fully automated execution from local machine without manual intervention

### Overview

This pattern converts Jupyter notebooks to executable Python scripts that can be submitted and monitored remotely via the Databricks Jobs API. **No manual execution required.**

### When to Use

- ✅ Long-running diagnostics (> 30 minutes)
- ✅ Computationally intensive tasks (Optuna optimization, etc.)
- ✅ Sequential workflows (diagnostic_16 → diagnostic_17)
- ✅ When you want to "set it and forget it"
- ✅ Production-grade automation

### Implementation Pattern

#### Step 1: Convert Notebook to Executable Python Script

**Template structure:**
```python
"""
Diagnostic N: Description
Databricks execution script with result saving
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path

# Databricks imports
from pyspark.sql import SparkSession
import sys
import os

# Add user workspace to Python path for imports
sys.path.insert(0, '/Workspace/Users/gibbons_tony@berkeley.edu')

# Import strategies from the workspace
try:
    import all_strategies_pct as strat
except ImportError:
    # Fall back to reading from same directory
    import importlib.util
    strategies_path = '/Workspace/Users/gibbons_tony@berkeley.edu/all_strategies_pct.py'
    if not os.path.exists(strategies_path):
        strategies_path = 'all_strategies_pct.py'
    spec = importlib.util.spec_from_file_location('all_strategies_pct', strategies_path)
    strat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strat)


def load_data_from_delta():
    """Load data from Delta tables (not pickle files)"""
    spark = SparkSession.builder.getOrCreate()

    # Load prices
    market_df = spark.table("commodity.bronze.market").filter(
        f"lower(commodity) = '{commodity}'"
    ).toPandas()

    market_df['date'] = pd.to_datetime(market_df['date'])
    market_df['price'] = market_df['close']
    prices_df = market_df[['date', 'price']].sort_values('date').reset_index(drop=True)

    # Load predictions
    pred_df = spark.table(f"commodity.trading_agent.predictions_{commodity}").filter(
        "model_version = 'synthetic_acc100'"
    ).toPandas()

    # Convert to matrices
    prediction_matrices = convert_to_matrices(pred_df)

    return prices_df, prediction_matrices


def main():
    print("="*80)
    print("DIAGNOSTIC N: Title")
    print("="*80)
    print(f"Execution time: {datetime.now()}")

    # Load data
    prices, predictions = load_data_from_delta()

    # Run analysis
    results = run_analysis(prices, predictions)

    # Save results to volume
    volume_path = "/Volumes/commodity/trading_agent/files"
    output_file = f"{volume_path}/diagnostic_N_results.pkl"

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"✓ Saved results to: {output_file}")

    # Also save CSV summary
    csv_file = f"{volume_path}/diagnostic_N_summary.csv"
    summary_df = create_summary(results)
    summary_df.to_csv(csv_file, index=False)
    print(f"✓ Saved summary to: {csv_file}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

**Key differences from notebooks:**
1. **No magic commands** (`%run`, `%pip`, etc.) - use pure Python
2. **Load from Delta tables**, not pickle files
3. **Handle `__file__` not being defined** in Databricks jobs
4. **Save results to volume** automatically
5. **Exit with proper status code** for job monitoring

#### Step 2: Commit and Push to Git

```bash
cd /path/to/repo
git add diagnostics/run_diagnostic_N.py
git commit -m "Add executable diagnostic_N script"
git push
```

#### Step 3: Update Databricks Repo

```bash
# Update repo to pull latest changes
databricks repos update <REPO_ID> --branch main

# Find repo ID with:
# databricks repos list --output json | grep ucberkeley-capstone
```

#### Step 4: Submit Job via Databricks CLI

```bash
# Create job config
cat > /tmp/diagnostic_N_job.json << 'EOF'
{
  "run_name": "diagnostic_N_description",
  "tasks": [{
    "task_key": "diagnostic_N",
    "spark_python_task": {
      "python_file": "file:///Workspace/Repos/Project_Git/ucberkeley-capstone/trading_agent/commodity_prediction_analysis/diagnostics/run_diagnostic_N.py"
    },
    "existing_cluster_id": "1111-041828-yeu2ff2q",
    "libraries": [],
    "timeout_seconds": 14400
  }]
}
EOF

# Submit job
databricks jobs submit --json @/tmp/diagnostic_N_job.json
```

**This returns a run_id for monitoring.**

#### Step 5: Monitor Execution

```bash
# Check status
databricks jobs get-run <RUN_ID> --output json | jq '.state'

# Get run URL
databricks jobs get-run <RUN_ID> --output json | jq -r '.run_page_url'
```

**Or use automated monitoring script:**
```python
import subprocess
import json
import time

def monitor_job(run_id):
    while True:
        result = subprocess.run(
            ["databricks", "jobs", "get-run", run_id, "--output", "json"],
            capture_output=True, text=True
        )
        status = json.loads(result.stdout)
        state = status.get('state', {})

        if state.get('life_cycle_state') == 'TERMINATED':
            return state.get('result_state') == 'SUCCESS'

        time.sleep(60)  # Check every minute
```

#### Step 6: Download Results

```bash
# After job completes
databricks fs cp dbfs:/Volumes/commodity/trading_agent/files/diagnostic_N_results.pkl /tmp/
databricks fs cp dbfs:/Volumes/commodity/trading_agent/files/diagnostic_N_summary.csv /tmp/
```

### Real-World Example: Diagnostic Workflow

**Implemented for diagnostics 100, 16, 17 (Nov 2024):**

```python
# monitor_diagnostics.py - Automated pipeline
def main():
    # Phase 1: Monitor diagnostic_16 (Optuna optimization)
    print("Monitoring diagnostic_16 (1800 trials)...")
    while not is_complete(diagnostic_16_run_id):
        time.sleep(60)

    # Phase 2: Auto-submit diagnostic_17
    print("Submitting diagnostic_17...")
    d17_run_id = submit_diagnostic_17()

    # Phase 3: Monitor diagnostic_17
    while not is_complete(d17_run_id):
        time.sleep(30)

    # Phase 4: Download all results
    download_results(['diagnostic_100', 'diagnostic_16', 'diagnostic_17'])

    # Phase 5: Analyze
    analyze_complete_results()
```

**Result:** Fully automated 2-hour workflow with zero manual intervention.

### Advantages

✅ **Zero manual intervention** - Submit and forget
✅ **Sequential automation** - Chain multiple jobs
✅ **Robust error handling** - Jobs API handles failures
✅ **Cost efficient** - Uses existing clusters
✅ **Reproducible** - Scripts in git, versioned
✅ **Scalable** - Can run on large clusters
✅ **Monitoring built-in** - Jobs API provides status
✅ **Results automatically saved** - No manual export needed

### Disadvantages

❌ Initial setup time (convert notebook → script)
❌ Requires understanding of Databricks Jobs API
❌ Debugging is harder (no interactive REPL)
❌ Import path issues need careful handling

### Best Practices

1. **Always load from Delta tables**, never pickle files
2. **Handle `__file__` being undefined** with try/except
3. **Add workspace paths to sys.path** for imports
4. **Save both .pkl and .csv** results
5. **Use descriptive run names** for easy identification
6. **Set appropriate timeouts** (default: 2 hours)
7. **Monitor via background script** for long-running jobs
8. **Chain jobs sequentially** when one depends on another

### Comparison: Notebook vs. Script Execution

| Aspect | Interactive Notebook | Remote Script Execution |
|--------|---------------------|------------------------|
| Execution | Manual, cell-by-cell | Automated, start-to-finish |
| Monitoring | Must watch manually | Background monitoring |
| Results | Must manually save | Auto-saved to volume |
| Reproducibility | Depends on manual steps | Fully reproducible |
| Time investment | Low (for simple tasks) | High (initial setup) |
| Long-term value | Low | High (reusable) |
| Debugging | Easy (interactive) | Harder (logs only) |
| Scaling | Limited | Excellent |

### When to Use Each Pattern

**Use Interactive Notebooks when:**
- Quick exploratory analysis
- Developing new code
- Debugging issues
- One-off experiments
- Short execution time (< 10 min)

**Use Remote Script Execution when:**
- Production workflows
- Long-running jobs (> 30 min)
- Reproducibility required
- Chaining multiple steps
- Want to "set and forget"

### Migration Path: Notebook → Script

**For converting existing trading_analysis notebooks:**

1. **Read the notebook** to understand logic
2. **Extract setup code** (imports, configs)
3. **Convert data loading** to use Delta tables
4. **Extract main logic** into functions
5. **Add result saving** to volume
6. **Test locally** if possible
7. **Commit to git** and push
8. **Submit test job** on small data
9. **Verify results** downloaded correctly
10. **Scale up** to full dataset

**Template checklist:**
```python
# ✅ Imports (no magic commands)
# ✅ Databricks path handling
# ✅ Load from Delta tables
# ✅ Main logic in functions
# ✅ Save results to volume
# ✅ Exit with status code
# ✅ Print execution summary
```

### Implementation Status

**✅ Completed (Nov 2024):**
- `run_diagnostic_100.py` - Algorithm validation with 100% accuracy
- `run_diagnostic_16.py` - Optuna parameter optimization (1800 trials)
- `run_diagnostic_17.py` - Trade-by-trade paradox analysis
- `monitor_diagnostics.py` - Automated monitoring and chaining

**🔄 Planned:**
- Convert remaining diagnostics (1-15) to this pattern
- Convert production notebooks (01-10) for automated backtesting
- Create reusable templates for common patterns

---

**Recommendation:** For any new diagnostic or production workflow, start with this automated execution pattern rather than interactive notebooks. The upfront investment pays off in reproducibility and automation.

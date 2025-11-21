# Databricks Cluster Setup Guide

Complete guide for setting up and configuring Databricks clusters for forecast agent training and backfilling.

## Quick Reference

**Current ML Runtime Cluster ID**: `1121-061338-wj2kqadu`

**Required Cluster Libraries** (already installed on current cluster):
- `databricks-sql-connector` - Unity Catalog SQL access
- `pmdarima` - Auto-ARIMA model selection for SARIMAX

## Option 1: Installing Libraries via Python API (Recommended for New Clusters)

If you create a new cluster or need to install libraries programmatically:

### Step 1: Load Environment Variables

```bash
cd /path/to/forecast_agent
set -a && source ../infra/.env && set +a
```

### Step 2: Update Cluster ID in Script

Create `/tmp/install_cluster_libraries.py` and update the `cluster_id` variable:

```python
#!/usr/bin/env python3
"""
Install packages as cluster libraries (persistent across sessions)
"""
import urllib.request
import json
import os

host = os.environ['DATABRICKS_HOST']
token = os.environ['DATABRICKS_TOKEN']
cluster_id = 'YOUR_CLUSTER_ID_HERE'  # ← Update this!

# Libraries to install
libraries = [
    {"pypi": {"package": "databricks-sql-connector"}},
    {"pypi": {"package": "pmdarima"}}
]

print("=" * 80)
print("Installing Cluster Libraries")
print("=" * 80)
print(f"Cluster ID: {cluster_id}")
print(f"\nInstalling packages:")
for lib in libraries:
    print(f"  - {lib['pypi']['package']}")

# Install libraries
url = f"{host}/api/2.0/libraries/install"
data = {
    "cluster_id": cluster_id,
    "libraries": libraries
}

req = urllib.request.Request(
    url,
    data=json.dumps(data).encode(),
    headers={
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
)

try:
    with urllib.request.urlopen(req) as response:
        print(f"\n✅ Libraries installation initiated!")

        # Check library status
        import time
        print("\nWaiting for installation to complete...")

        for i in range(30):  # Poll for up to 2.5 minutes
            time.sleep(5)

            # Get cluster libraries status
            status_url = f"{host}/api/2.0/libraries/cluster-status?cluster_id={cluster_id}"
            status_req = urllib.request.Request(status_url)
            status_req.add_header('Authorization', f'Bearer {token}')

            with urllib.request.urlopen(status_req) as status_response:
                result = json.loads(status_response.read().decode())
                library_statuses = result.get('library_statuses', [])

                all_installed = True
                pending = False

                for lib_status in library_statuses:
                    lib_name = lib_status.get('library', {}).get('pypi', {}).get('package', 'unknown')
                    status = lib_status.get('status', 'UNKNOWN')

                    if status in ['PENDING', 'INSTALLING']:
                        pending = True
                        all_installed = False
                    elif status != 'INSTALLED':
                        all_installed = False

                if all_installed and len(library_statuses) == len(libraries):
                    print(f"\n✅✅✅ ALL LIBRARIES INSTALLED! ✅✅✅")
                    print("\n" + "=" * 80)
                    print("Installed Libraries:")
                    print("=" * 80)
                    for lib_status in library_statuses:
                        lib_name = lib_status.get('library', {}).get('pypi', {}).get('package', 'unknown')
                        status = lib_status.get('status', 'UNKNOWN')
                        print(f"  ✓ {lib_name}: {status}")

                    print("\n" + "=" * 80)
                    print("Next Steps:")
                    print("=" * 80)
                    print("1. Cluster libraries are now persistent (no pip install needed)")
                    print("2. Restart your cluster to activate libraries")
                    print("3. Run your notebook - packages are pre-loaded!")
                    break
                elif pending:
                    dots = '.' * (i % 4)
                    print(f"\r  [{i*5}s] Installing{dots:<3}", end='', flush=True)
        else:
            print(f"\n⚠️  Libraries still installing after 2.5 minutes")
            print("Check status at: Clusters → [Your Cluster] → Libraries")

except urllib.error.HTTPError as e:
    error_body = e.read().decode()
    print(f"\n❌ Failed to install libraries")
    print(f"Error: {e.code} {e.reason}")
    print(f"Details: {error_body}")
```

### Step 3: Run Installation Script

```bash
python /tmp/install_cluster_libraries.py
```

**Installation takes 30-60 seconds**. Libraries persist across all cluster sessions.

## Option 2: Installing Libraries via Databricks UI

If you prefer the UI method:

1. Navigate to: **Compute** → **[Your Cluster Name]** → **Libraries** tab
2. Click **Install New**
3. Select **PyPI** as the library source
4. Install each package:
   - Package name: `databricks-sql-connector` → Install
   - Package name: `pmdarima` → Install
5. Wait for status to show **INSTALLED** (green checkmark)
6. Restart cluster to activate libraries

## Verifying Library Installation

Run this in a Databricks notebook cell:

```python
# Verify libraries are installed
import databricks.sql
import pmdarima
print(f"✓ databricks-sql-connector: {databricks.sql.__version__}")
print(f"✓ pmdarima: {pmdarima.__version__}")
```

Expected output:
```
✓ databricks-sql-connector: 3.x.x
✓ pmdarima: 2.x.x
```

## Removing Pip Install from Notebooks

Once cluster libraries are installed, you can remove this line from your notebooks:

```python
# BEFORE (with pip install):
# MAGIC %pip install databricks-sql-connector pmdarima

# AFTER (with cluster libraries - REMOVE this line):
# (nothing needed - libraries are pre-loaded)
```

**Benefits**:
- Faster notebook startup (no package download/install)
- No dependency version conflicts between sessions
- Libraries persist across cluster restarts

## Why These Libraries?

### databricks-sql-connector
- **Purpose**: Unity Catalog SQL access for reading/writing forecast data
- **Alternative**: Native Spark SQL (not yet implemented)
- **Pre-installed in ML Runtime?**: No

### pmdarima
- **Purpose**: `auto_arima()` function for automatic SARIMAX order selection
- **Alternative**: Manual order specification like `order=(1,1,1)` using statsmodels only
- **Pre-installed in ML Runtime?**: No
- **Dependency**: Wraps `statsmodels.tsa.statespace.sarimax.SARIMAX` (which IS pre-installed)

**Note**: `statsmodels` is pre-installed in ML Runtime 14.3.x and includes SARIMAX. We only need `pmdarima` for automatic order finding. If you manually specify ARIMA orders, you don't need pmdarima.

## Troubleshooting

### Library Installation Fails

**Error**: `Library installation failed: Could not find package pmdarima`
- **Solution**: Check spelling and package availability on PyPI
- **Workaround**: Use `%pip install` in notebook (slower, but works)

### Import Errors After Installation

**Error**: `ModuleNotFoundError: No module named 'databricks.sql'`
- **Solution**: Restart cluster to activate newly installed libraries
- **Check**: Verify library status shows **INSTALLED** (green) not **PENDING** (yellow)

### Version Conflicts

**Error**: `ImportError: cannot import name 'sql' from 'databricks'`
- **Cause**: Namespace conflict with pre-installed `databricks` package
- **Solution**: Use `import databricks.sql as sql` instead of `from databricks import sql`
- **Fix in code**: Already applied in `databricks_train_fresh_models.py:54`

### Library Status Stuck at PENDING

**Symptoms**: Library shows PENDING for >5 minutes
- **Solution 1**: Restart cluster
- **Solution 2**: Uninstall and reinstall library
- **Solution 3**: Check cluster event log for errors

## Creating a New ML Runtime Cluster

If you need to create a new cluster with all required configurations:

### Step 1: Save Cluster Creation Script

Create `/tmp/create_ml_cluster.py`:

```python
#!/usr/bin/env python3
"""
Create a Databricks ML Runtime cluster via API
"""
import urllib.request
import json
import os
import time

host = os.environ['DATABRICKS_HOST']
token = os.environ['DATABRICKS_TOKEN']

# Cluster configuration
cluster_config = {
    "cluster_name": "ML Runtime Cluster - Forecast Agent",
    "spark_version": "14.3.x-cpu-ml-scala2.12",  # ML Runtime
    "node_type_id": "i3.xlarge",
    "num_workers": 4,  # Scale to 4 workers for parallel training
    "autotermination_minutes": 120,  # 2 hours idle timeout
    "data_security_mode": "SINGLE_USER",  # Unity Catalog compatible
    "runtime_engine": "STANDARD",
    "custom_tags": {
        "Project": "ForecastAgent",
        "Purpose": "Training"
    }
}

# For single-node cluster (within quota), use this instead:
# "num_workers": 0,
# "spark_conf": {
#     "spark.databricks.cluster.profile": "singleNode",
#     "spark.master": "local[*]"
# },

print("=" * 80)
print("Creating ML Runtime Cluster")
print("=" * 80)
print(f"\nCluster Name: {cluster_config['cluster_name']}")
print(f"Runtime: {cluster_config['spark_version']}")
print(f"Node Type: {cluster_config['node_type_id']}")
print(f"Workers: {cluster_config['num_workers']}")

# Create cluster
url = f"{host}/api/2.0/clusters/create"
req = urllib.request.Request(
    url,
    data=json.dumps(cluster_config).encode(),
    headers={
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
)

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        cluster_id = result['cluster_id']

        print(f"\n✅ Cluster created successfully!")
        print(f"\nCluster ID: {cluster_id}")
        print(f"View at: {host}/#setting/clusters/{cluster_id}/configuration")

        # Save cluster ID for library installation
        with open('/tmp/new_cluster_id.txt', 'w') as f:
            f.write(cluster_id)

        print(f"\n{'='*80}")
        print("Next Steps:")
        print(f"{'='*80}")
        print(f"1. Update cluster_id in /tmp/install_cluster_libraries.py to: {cluster_id}")
        print(f"2. Run: python /tmp/install_cluster_libraries.py")
        print(f"3. Wait for cluster to start (2-3 minutes)")
        print(f"4. Open your training notebook and select this cluster")

except urllib.error.HTTPError as e:
    error_body = e.read().decode()
    print(f"\n❌ Failed to create cluster")
    print(f"Error: {e.code} {e.reason}")
    print(f"Details: {error_body}")
```

### Step 2: Create and Configure Cluster

```bash
# Load credentials
set -a && source ../infra/.env && set +a

# Create cluster
python /tmp/create_ml_cluster.py

# Wait for cluster ID output, then install libraries
# (Update cluster_id in install script first)
python /tmp/install_cluster_libraries.py
```

### Step 3: Verify Setup

```bash
# Check cluster status
curl -X GET "${DATABRICKS_HOST}/api/2.0/clusters/get?cluster_id=YOUR_CLUSTER_ID" \
  -H "Authorization: Bearer ${DATABRICKS_TOKEN}"
```

## ML Runtime Pre-Installed Packages

ML Runtime 14.3.x includes 50+ ML packages (no installation needed):

**Data Science Essentials**:
- pandas, numpy, scipy
- scikit-learn, xgboost
- statsmodels (includes SARIMAX!)
- matplotlib, seaborn, plotly

**Deep Learning**:
- TensorFlow, PyTorch, Keras
- transformers, datasets

**MLOps**:
- MLflow, hyperopt, optuna
- shap, lime

**Full list**: [Databricks ML Runtime Release Notes](https://docs.databricks.com/en/release-notes/runtime/14.3ml.html)

## Summary

**For Current Cluster** (`1121-061338-wj2kqadu`):
- Libraries already installed
- No action needed
- Use `%pip install` line in notebooks (fast since libraries exist)

**For New Clusters**:
1. Create cluster via API or UI
2. Install libraries via Python script or UI
3. Restart cluster
4. Remove `%pip install` lines from notebooks
5. Enjoy faster startups!

**For Future Reference**:
- Save `/tmp/install_cluster_libraries.py` with your cluster ID
- Re-run anytime you create a new cluster
- Libraries persist until cluster is deleted

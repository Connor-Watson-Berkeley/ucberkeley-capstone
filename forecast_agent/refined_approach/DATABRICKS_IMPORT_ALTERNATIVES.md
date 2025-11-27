# Databricks Import Alternatives - AWS Credentials Error Fix

## The Problem

When trying to pull the GitHub repo into Databricks, you're getting:
```
Error creating Git folder
Failed to clone repo. Repo may be incomplete. 
Failure reason: Missing credentials to access AWS bucket
```

This is likely a Databricks workspace/Unity Catalog validation issue, not a repo problem.

---

## ✅ Solution 1: Upload to Workspace (Recommended - Fastest)

**Skip Repos entirely and upload files directly:**

### Step 1: Create Folder Structure

1. In Databricks, go to **Workspace** (left sidebar)
2. Navigate to your user folder (or create a project folder)
3. Create folder structure:
   ```
   forecast_agent/
     └── refined_approach/
         ├── notebooks/
         └── [other files]
   ```

### Step 2: Upload Notebooks

1. Right-click `forecast_agent/refined_approach/notebooks/` folder
2. Click **Import**
3. Choose files from your local machine:
   - `00_setup_and_imports.py`
   - `01_train_models.py`
   - `00_daily_production.py`
4. Upload them as **Python** files (not notebooks - they're `.py` files)

### Step 3: Upload Python Modules

1. Create a folder: `forecast_agent/refined_approach/`
2. Upload all `.py` files:
   - `data_loader.py`
   - `evaluator.py`
   - `cross_validator.py`
   - `model_pipeline.py`
   - `model_persistence.py`
   - `distributions_writer.py`
   - `daily_production.py`

**Method A: Upload via UI (one at a time)**
- Right-click folder → Import → Select file

**Method B: Upload via Databricks CLI (faster)**
```bash
# Install Databricks CLI
pip install databricks-cli

# Configure
databricks configure --token
# Host: https://your-workspace.cloud.databricks.com
# Token: <your-token>

# Upload files
databricks workspace import_dir \
  forecast_agent/refined_approach \
  /Users/YourUsername/forecast_agent/refined_approach \
  -o
```

### Step 4: Update Import Path in Notebooks

In `01_train_models.py`, change the path detection to use Workspace path:

```python
# Change this section to use Workspace path
import sys
from pathlib import Path

# Workspace path (adjust to your username)
workspace_path = Path('/Workspace/Users/your-email@domain.com/forecast_agent/refined_approach')
if workspace_path.exists():
    sys.path.insert(0, str(workspace_path))
    print(f"✅ Added {workspace_path} to path")
else:
    # Fallback to Repos path
    workspace_path = Path('/Workspace/Repos')
    repo_path = None
    for repo in workspace_path.iterdir():
        if repo.is_dir():
            refined_path = repo / 'forecast_agent' / 'refined_approach'
            if refined_path.exists():
                repo_path = refined_path
                break
    if repo_path:
        sys.path.insert(0, str(repo_path))
    else:
        sys.path.insert(0, str(Path.cwd() / 'forecast_agent' / 'refined_approach'))
```

---

## ✅ Solution 2: Use DBFS (Databricks File System)

Upload files to DBFS, then reference them:

### Step 1: Upload to DBFS

```python
# Run in a Databricks notebook or via CLI
import os
import shutil

# Upload via Databricks CLI
databricks fs cp -r forecast_agent/refined_approach dbfs:/FileStore/forecast_agent/refined_approach
```

Or use Python:
```python
# In a Databricks notebook
dbutils.fs.mkdirs("dbfs:/FileStore/forecast_agent/refined_approach")

# Copy files (one by one)
dbutils.fs.cp("file:/path/to/local/file.py", "dbfs:/FileStore/forecast_agent/refined_approach/")
```

### Step 2: Add to Path in Notebook

```python
import sys
sys.path.insert(0, '/dbfs/FileStore/forecast_agent/refined_approach')
```

---

## ✅ Solution 3: Fix Repos Configuration (If Repos is Required)

### Option A: Disable Unity Catalog Validation

1. Go to **Workspace Admin** → **Repos**
2. Check if there's a validation setting for AWS
3. Temporarily disable Unity Catalog integration for repos

### Option B: Configure Storage Credentials First

If you need Unity Catalog, set up storage credentials first:

1. Go to **SQL Editor** in Databricks
2. Run (if you have AWS IAM role):
```sql
CREATE STORAGE CREDENTIAL IF NOT EXISTS s3_groundtruth_capstone
USING AWS_IAM_ROLE
WITH (
  role_arn = 'arn:aws:iam::534150427458:role/databricks-s3-access-role'
);
```

Then try cloning repo again.

### Option C: Clone Without Unity Catalog Context

1. Create a new workspace folder (not in Repos)
2. Clone repo there
3. Or use a different workspace without Unity Catalog enabled

---

## ✅ Solution 4: Use GitHub Raw Files (Simplest for Testing)

For quick testing, copy-paste code directly:

1. Open GitHub repo in browser
2. Navigate to file (e.g., `forecast_agent/refined_approach/notebooks/01_train_models.py`)
3. Click **Raw** button
4. Copy all code
5. Paste into new Databricks notebook
6. Save

Repeat for other files.

**Pros:**
- Fast
- No git/aws issues
- Good for testing

**Cons:**
- Manual process
- Harder to keep in sync

---

## 🎯 Recommended Approach

**For Quick Testing (Right Now):**
1. Use **Solution 1** (Workspace Upload) - fastest, no AWS issues
2. Upload `01_train_models.py` as a notebook
3. Upload Python modules to a folder
4. Update path in notebook
5. Run!

**For Long-Term:**
1. Fix Repos configuration (Solution 3)
2. Or use Workspace + Git sync manually
3. Or use DBFS (Solution 2) for persistent storage

---

## Quick Fix Script

If you have Databricks CLI set up, run this locally:

```bash
#!/bin/bash
# Upload refined_approach to Databricks Workspace

DATABRICKS_USER="your-email@domain.com"  # Change this
WORKSPACE_PATH="/Users/$DATABRICKS_USER/forecast_agent"

# Create directory structure
databricks workspace mkdirs "$WORKSPACE_PATH/refined_approach"
databricks workspace mkdirs "$WORKSPACE_PATH/refined_approach/notebooks"

# Upload Python files
for file in forecast_agent/refined_approach/*.py; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        databricks workspace import \
            "$file" \
            "$WORKSPACE_PATH/refined_approach/$filename" \
            --language PYTHON
        echo "Uploaded $filename"
    fi
done

# Upload notebooks
for file in forecast_agent/refined_approach/notebooks/*.py; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        databricks workspace import \
            "$file" \
            "$WORKSPACE_PATH/refined_approach/notebooks/$filename" \
            --language PYTHON
        echo "Uploaded notebook $filename"
    fi
done

echo "✅ Upload complete!"
```

---

## Need Help?

If none of these work:
1. Check Databricks workspace permissions
2. Verify you're not in a Unity Catalog context that requires AWS
3. Try creating a simple test repo first
4. Contact Databricks support about the AWS bucket error

The error is likely a workspace configuration issue, not your code! ✅


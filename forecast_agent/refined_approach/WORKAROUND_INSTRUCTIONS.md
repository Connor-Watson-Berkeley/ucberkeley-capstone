# Workaround for AWS Credential Issues

## The Problem

Databricks is blocking file uploads due to AWS bucket validation, even though this is just code files (not data).

## ✅ Solution: Download from GitHub Inside Databricks

Since we can't upload via CLI, let's have Databricks pull the code directly from GitHub!

### Step 1: Create a New Notebook in Databricks

1. Go to **Workspace** in Databricks
2. Navigate to your user folder or create a new folder
3. Click **Create** → **Notebook**
4. Name it: `pull_refined_approach`
5. Set language to **Python**

### Step 2: Copy the Download Code

1. Open the file: `forecast_agent/refined_approach/notebooks/00_pull_from_github.py`
2. Copy all the code
3. Paste into your new Databricks notebook
4. Save the notebook

### Step 3: Run the Notebook

1. Attach notebook to a cluster
2. Run all cells
3. The notebook will download all files from GitHub to your Workspace

### Step 4: Use the Downloaded Code

1. Navigate to: `Workspace → Users → [Your Name] → forecast_agent → refined_approach`
2. Open `notebooks/01_train_models.py`
3. Update the import path in the notebook (it will auto-detect, but you may need to adjust)

## Alternative: Manual Copy-Paste

If the download notebook doesn't work:

1. Go to GitHub: https://github.com/Connor-Watson-Berkeley/ucberkeley-capstone/tree/main/forecast_agent/refined_approach
2. For each `.py` file:
   - Click on the file
   - Click **Raw** button
   - Copy all code
   - Create new file in Databricks Workspace
   - Paste code
   - Save

## Why This Works

- GitHub downloads work fine (no AWS validation)
- Once files are in Workspace, they can be used normally
- Bypasses the workspace/Unity Catalog AWS checks

## Files You Need

**Python Modules:**
- `data_loader.py`
- `evaluator.py`
- `model_pipeline.py`
- `model_persistence.py`
- `distributions_writer.py`
- `daily_production.py`
- `cross_validator.py`

**Notebooks:**
- `notebooks/01_train_models.py` (most important!)
- `notebooks/00_daily_production.py`
- `notebooks/00_setup_and_imports.py`

**Optional (docs):**
- `README.md`
- `MODEL_RECOMMENDATIONS.md`
- Other `.md` files

## Quick Test

After downloading, test that imports work:

```python
import sys
sys.path.insert(0, '/Workspace/Users/[your-email]/forecast_agent/refined_approach')

try:
    from data_loader import TimeSeriesDataLoader
    print("✅ Import successful!")
except Exception as e:
    print(f"❌ Import failed: {e}")
```

If imports work, you're ready to run the training notebook! 🚀


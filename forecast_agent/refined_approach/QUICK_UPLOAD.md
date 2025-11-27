# Quick Upload Guide

## Step 1: Refresh Your Databricks Token

Your CLI token appears to be expired. Refresh it:

```bash
databricks configure --token
```

**You'll need:**
- **Host**: Your Databricks workspace URL (looks like `https://dbc-xxxxx-xxxxx.cloud.databricks.com`)
- **Token**: Generate a new one at:
  - Databricks → Settings → User Settings → Access Tokens → Generate New Token

## Step 2: Run the Upload Script

Once your token is refreshed, run:

```bash
cd /Users/connorwatson/Documents/Data\ Science/ucberkeley-capstone
./forecast_agent/refined_approach/upload_to_databricks.sh
```

## Step 3: Verify Upload

The script will:
- ✅ Check authentication
- ✅ Verify repo exists
- ✅ Create directory structure
- ✅ Upload all Python files
- ✅ Upload all notebooks
- ✅ Upload documentation

## Alternative: Manual Upload (If Script Fails)

If you prefer to upload manually:

```bash
# Set your paths
REPO_BASE="/Repos/Project_Git/ucberkeley-capstone"
TARGET="$REPO_BASE/forecast_agent/refined_approach"
LOCAL="forecast_agent/refined_approach"

# Create directories
databricks workspace mkdirs "$TARGET/notebooks"

# Upload main Python files
databricks workspace import "$LOCAL/data_loader.py" "$TARGET/data_loader.py" --language PYTHON
databricks workspace import "$LOCAL/evaluator.py" "$TARGET/evaluator.py" --language PYTHON
databricks workspace import "$LOCAL/model_pipeline.py" "$TARGET/model_pipeline.py" --language PYTHON
# ... (repeat for all .py files)

# Upload notebooks
databricks workspace import "$LOCAL/notebooks/01_train_models.py" "$TARGET/notebooks/01_train_models.py" --language PYTHON
# ... (repeat for all notebooks)
```

## Troubleshooting

**"Authorization failed"**
- Token expired → Refresh with `databricks configure --token`

**"Repo not found"**
- Check repo name in Databricks
- Update `REPO_BASE` path in script
- Or create repo first: Databricks → Repos → Add Repo

**"File not found"**
- Make sure you're in the project root directory
- Check that `forecast_agent/refined_approach/` exists

## After Upload

1. Go to Databricks → Repos → Project_Git → ucberkeley-capstone
2. Navigate to `forecast_agent/refined_approach/notebooks/`
3. Open `01_train_models.py`
4. Attach to a cluster
5. Run all cells!


"""
Databricks Notebook: Pull refined_approach from GitHub

This notebook downloads the refined_approach code directly from GitHub.
Use this if Repos cloning fails due to AWS credential issues.

Run this notebook first, then use the downloaded code.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC # Pull refined_approach from GitHub
# MAGIC 
# MAGIC This notebook downloads the code directly from GitHub to bypass AWS validation issues.

# COMMAND ----------

import os
import requests
from pathlib import Path
import json

# GitHub repository details
GITHUB_REPO = "Connor-Watson-Berkeley/ucberkeley-capstone"
GITHUB_BRANCH = "main"
BASE_PATH = "forecast_agent/refined_approach"

# Target location in Databricks
TARGET_DIR = "/Workspace/Users/" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get() + "/forecast_agent/refined_approach"

print(f"📥 Downloading from: https://github.com/{GITHUB_REPO}/tree/{GITHUB_BRANCH}/{BASE_PATH}")
print(f"📁 Target directory: {TARGET_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Get file list from GitHub API

# COMMAND ----------

def get_github_file_url(repo, branch, file_path):
    """Get raw GitHub URL for a file."""
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{file_path}"

def download_file(url, target_path):
    """Download a file from URL to target path."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()
    with open(target_path, 'wb') as f:
        f.write(response.content)
    return len(response.content)

# Files to download (Python modules)
PYTHON_FILES = [
    "__init__.py",
    "cross_validator.py",
    "daily_production.py",
    "data_loader.py",
    "distributions_writer.py",
    "evaluator.py",
    "model_persistence.py",
    "model_pipeline.py",
]

# Notebooks
NOTEBOOK_FILES = [
    "00_daily_production.py",
    "00_setup_and_imports.py",
    "01_train_models.py",
]

# Create target directories
os.makedirs(f"{TARGET_DIR}/notebooks", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/docs", exist_ok=True)

print(f"✅ Created directories")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Download Python Modules

# COMMAND ----------

print("📄 Downloading Python modules...")
downloaded = 0

for filename in PYTHON_FILES:
    url = get_github_file_url(GITHUB_REPO, GITHUB_BRANCH, f"{BASE_PATH}/{filename}")
    target = f"{TARGET_DIR}/{filename}"
    try:
        size = download_file(url, target)
        print(f"  ✅ {filename} ({size:,} bytes)")
        downloaded += 1
    except Exception as e:
        print(f"  ❌ {filename}: {str(e)[:100]}")

print(f"\n✅ Downloaded {downloaded}/{len(PYTHON_FILES)} Python files")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Download Notebooks

# COMMAND ----------

print("📓 Downloading notebooks...")
downloaded = 0

for filename in NOTEBOOK_FILES:
    url = get_github_file_url(GITHUB_REPO, GITHUB_BRANCH, f"{BASE_PATH}/notebooks/{filename}")
    target = f"{TARGET_DIR}/notebooks/{filename}"
    try:
        size = download_file(url, target)
        print(f"  ✅ {filename} ({size:,} bytes)")
        downloaded += 1
    except Exception as e:
        print(f"  ❌ {filename}: {str(e)[:100]}")

print(f"\n✅ Downloaded {downloaded}/{len(NOTEBOOK_FILES)} notebooks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Verify Downloads

# COMMAND ----------

print("\n📋 Verification:")
print(f"Target directory: {TARGET_DIR}")
print("\nFiles downloaded:")

for root, dirs, files in os.walk(TARGET_DIR):
    level = root.replace(TARGET_DIR, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath)
        print(f"{subindent}{file} ({size:,} bytes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC ✅ Files downloaded successfully!
# MAGIC 
# MAGIC **Now you can:**
# MAGIC 1. Navigate to: `Workspace → Users → [Your Name] → forecast_agent → refined_approach`
# MAGIC 2. Open `notebooks/01_train_models.py`
# MAGIC 3. Update the path in the notebook to use:
# MAGIC    ```python
# MAGIC    sys.path.insert(0, '/Workspace/Users/[your-email]/forecast_agent/refined_approach')
# MAGIC    ```
# MAGIC 4. Run the training notebook!

# COMMAND ----------


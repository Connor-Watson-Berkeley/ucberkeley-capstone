# Files Are in Databricks - How to Access Them

## ✅ Good News!

The `refined_approach` folder **does exist** in your Databricks repo at:
```
/Repos/Project_Git/ucberkeley-capstone/forecast_agent/refined_approach
```

However, the files might not be visible or synced. Here's how to access them:

---

## Method 1: Check if Files Are There (UI)

1. **In Databricks, navigate to:**
   - Repos → Project_Git → ucberkeley-capstone → forecast_agent → refined_approach

2. **Look for:**
   - Python files (`.py`) in the root
   - `notebooks/` folder
   - `docs/` folder

3. **If you see folders but no files:**
   - The repo might need to sync
   - Try refreshing the page (F5)
   - Or use Method 2 below

---

## Method 2: Sync Files from GitHub (Recommended)

Since the repo might be out of sync, let's download the files directly:

### Step 1: Create a New Notebook

1. Go to **Workspace** (not Repos)
2. Create folder: `setup` (or any name)
3. Create new **Python** notebook: `sync_refined_approach`

### Step 2: Run This Code

Paste and run this in the notebook:

```python
# Download refined_approach files from GitHub
import requests
import os

GITHUB_REPO = "Connor-Watson-Berkeley/ucberkeley-capstone"
BRANCH = "main"
BASE_PATH = "forecast_agent/refined_approach"

# Get your username
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
target_base = f"/Workspace/Users/{username}/forecast_agent/refined_approach"

# Create directories
os.makedirs(f"{target_base}/notebooks", exist_ok=True)
os.makedirs(f"{target_base}/docs", exist_ok=True)

def download_file(file_path):
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{BRANCH}/{file_path}"
    target = file_path.replace(BASE_PATH, target_base)
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'wb') as f:
            f.write(response.content)
        print(f"✅ {os.path.basename(file_path)}")
        return True
    except Exception as e:
        print(f"❌ {os.path.basename(file_path)}: {str(e)[:50]}")
        return False

# Download Python files
print("📄 Downloading Python modules...")
files = [
    "forecast_agent/refined_approach/__init__.py",
    "forecast_agent/refined_approach/data_loader.py",
    "forecast_agent/refined_approach/evaluator.py",
    "forecast_agent/refined_approach/cross_validator.py",
    "forecast_agent/refined_approach/model_pipeline.py",
    "forecast_agent/refined_approach/model_persistence.py",
    "forecast_agent/refined_approach/distributions_writer.py",
    "forecast_agent/refined_approach/daily_production.py",
]

# Download notebooks
notebooks = [
    "forecast_agent/refined_approach/notebooks/00_setup_and_imports.py",
    "forecast_agent/refined_approach/notebooks/00_daily_production.py",
    "forecast_agent/refined_approach/notebooks/01_train_models.py",
    "forecast_agent/refined_approach/notebooks/00_pull_from_github.py",
]

for file_path in files:
    download_file(file_path)

print("\n📓 Downloading notebooks...")
for file_path in notebooks:
    download_file(file_path)

print(f"\n✅ Files downloaded to: {target_base}")
print("\nNext: Navigate to Workspace → Users → [Your Name] → forecast_agent → refined_approach")
```

### Step 3: Use the Downloaded Files

1. Navigate to: **Workspace → Users → [Your Name] → forecast_agent → refined_approach**
2. Open: `notebooks/01_train_models.py`
3. Update the path if needed (should auto-detect)
4. Run!

---

## Method 3: Use Repos Folder (If Files Are There)

If the files ARE in the Repos folder:

1. Navigate to: **Repos → Project_Git → ucberkeley-capstone → forecast_agent → refined_approach**
2. Open: `notebooks/01_train_models.py`
3. The path detection in the notebook should find it automatically

---

## Method 4: Copy Files from Repos to Workspace

If files exist in Repos but you want them in Workspace:

```python
# Run in a Databricks notebook
import shutil

source = "/Repos/Project_Git/ucberkeley-capstone/forecast_agent/refined_approach"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
target = f"/Workspace/Users/{username}/forecast_agent/refined_approach"

# Copy files (this might not work if files aren't synced)
# Better to use Method 2 above
```

---

## Quick Check: Are Files Really There?

Run this in a Databricks notebook to check:

```python
import os

repo_path = "/Repos/Project_Git/ucberkeley-capstone/forecast_agent/refined_approach"

if os.path.exists(repo_path):
    print("✅ Folder exists!")
    files = os.listdir(repo_path)
    print(f"Files/folders found: {len(files)}")
    for item in files[:20]:  # Show first 20
        item_path = os.path.join(repo_path, item)
        if os.path.isfile(item_path):
            size = os.path.getsize(item_path)
            print(f"  📄 {item} ({size:,} bytes)")
        else:
            print(f"  📁 {item}/")
else:
    print("❌ Folder not found")
```

---

## Recommended Next Steps

1. **Try Method 2** (download from GitHub) - most reliable
2. Files will be in your Workspace folder
3. Easy to access and modify
4. No AWS credential issues

Once files are downloaded, you can start training immediately! 🚀


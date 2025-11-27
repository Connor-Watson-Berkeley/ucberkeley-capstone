# Simple Download Notebook - Copy This Into Databricks

## Quick Solution: Download Files Directly

Since the repo files aren't visible, here's a ready-to-use notebook code you can copy directly into Databricks:

---

## Step 1: Create New Notebook

1. In Databricks, go to **Workspace**
2. Create folder: `setup` (or use your user folder)
3. Click **Create** → **Notebook**
4. Name it: `download_refined_approach`
5. Language: **Python**

## Step 2: Copy This Entire Code Block

Paste this into the notebook and run it:

```python
# COMMAND ----------

# Download refined_approach from GitHub
import requests
import os

GITHUB_REPO = "Connor-Watson-Berkeley/ucberkeley-capstone"
BRANCH = "main"
BASE_PATH = "forecast_agent/refined_approach"

# Get your username
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
target_base = f"/Workspace/Users/{username}/forecast_agent/refined_approach"

print(f"📥 Downloading from GitHub...")
print(f"📁 Saving to: {target_base}\n")

# Create directories
os.makedirs(f"{target_base}/notebooks", exist_ok=True)
os.makedirs(f"{target_base}/docs", exist_ok=True)

def download_file(file_path):
    """Download a single file from GitHub."""
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{BRANCH}/{file_path}"
    # Convert repo path to target path
    target = file_path.replace(BASE_PATH, target_base)
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'wb') as f:
            f.write(response.content)
        size = len(response.content)
        print(f"✅ {os.path.basename(file_path)} ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"❌ {os.path.basename(file_path)}: {str(e)[:60]}")
        return False

# COMMAND ----------

# Download Python modules
print("📄 Downloading Python modules...\n")
files = [
    f"{BASE_PATH}/__init__.py",
    f"{BASE_PATH}/data_loader.py",
    f"{BASE_PATH}/evaluator.py",
    f"{BASE_PATH}/cross_validator.py",
    f"{BASE_PATH}/model_pipeline.py",
    f"{BASE_PATH}/model_persistence.py",
    f"{BASE_PATH}/distributions_writer.py",
    f"{BASE_PATH}/daily_production.py",
]

downloaded = 0
for file_path in files:
    if download_file(file_path):
        downloaded += 1

print(f"\n✅ Downloaded {downloaded}/{len(files)} Python files")

# COMMAND ----------

# Download notebooks
print("\n📓 Downloading notebooks...\n")
notebooks = [
    f"{BASE_PATH}/notebooks/00_setup_and_imports.py",
    f"{BASE_PATH}/notebooks/00_daily_production.py",
    f"{BASE_PATH}/notebooks/01_train_models.py",
]

downloaded_nb = 0
for file_path in notebooks:
    if download_file(file_path):
        downloaded_nb += 1

print(f"\n✅ Downloaded {downloaded_nb}/{len(notebooks)} notebooks")

# COMMAND ----------

# Verify files
print(f"\n📋 Files saved to: {target_base}")
print("\nListing files:")

if os.path.exists(target_base):
    for root, dirs, files in os.walk(target_base):
        level = root.replace(target_base, '').count(os.sep)
        indent = ' ' * 2 * level
        folder_name = os.path.basename(root) if root != target_base else "refined_approach"
        if folder_name:
            print(f"{indent}📁 {folder_name}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in sorted(files):
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                size = os.path.getsize(filepath)
                print(f"{subindent}📄 {file} ({size:,} bytes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Download Complete!
# MAGIC 
# MAGIC **Next steps:**
# MAGIC 1. Navigate to: **Workspace → Users → [Your Name] → forecast_agent → refined_approach**
# MAGIC 2. Open: `notebooks/01_train_models.py`
# MAGIC 3. Update path in notebook if needed (should auto-detect)
# MAGIC 4. Run the training notebook!

```

---

## Step 3: Run All Cells

1. Attach notebook to a cluster
2. Click **Run All**
3. Wait for files to download (should take 10-30 seconds)

## Step 4: Access Your Files

After download completes:
1. Go to **Workspace** (left sidebar)
2. Navigate to: **Users → [Your Email] → forecast_agent → refined_approach**
3. You should see all the Python files and notebooks folder!

## Step 5: Open Training Notebook

1. Navigate to: `notebooks/01_train_models.py`
2. Open it in Databricks
3. Attach to a cluster
4. Set parameters and run!

---

## That's It!

This bypasses all the AWS credential issues and gets your files directly from GitHub. Once downloaded, everything will work normally! 🚀


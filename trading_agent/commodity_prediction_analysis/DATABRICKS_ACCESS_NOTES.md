# Databricks Access Pattern for trading_agent

## How Notebooks Run

**These are DATABRICKS notebooks** - they run IN Databricks, not locally:
- Notebooks use `%run ./00_setup_and_config` to load config
- The `spark` object is pre-configured in Databricks environment
- NO local SparkSession setup needed - it's already there

## File Storage Locations

### Unity Catalog Volume (Persistent Storage)
```python
VOLUME_PATH = "/Volumes/commodity/trading_agent/files"
```

**Use this for:**
- Pickle files: `prediction_matrices_*.pkl`, `results_detailed_*.pkl`, etc.
- Images: `*.png` files
- CSV exports: `*.csv` files

**Access via databricks CLI:**
```bash
export DATABRICKS_HOST=https://dbc-5e4780f4-fcec.cloud.databricks.com
export DATABRICKS_TOKEN=dapi8f0886905a2b080bc5456595a8746b89

# List files
databricks fs ls dbfs:/Volumes/commodity/trading_agent/files/

# Download a file
databricks fs cp dbfs:/Volumes/commodity/trading_agent/files/myfile.pkl ./myfile.pkl

# Upload a file
databricks fs cp ./myfile.pkl dbfs:/Volumes/commodity/trading_agent/files/myfile.pkl
```

### Delta Tables (Structured Data)
```python
OUTPUT_SCHEMA = "commodity.trading_agent"
```

**Tables created:**
- `commodity.trading_agent.predictions_{commodity}`
- `commodity.trading_agent.predictions_prepared_{commodity}_{model}`
- `commodity.trading_agent.results_{commodity}_{model}`
- etc.

**Access via spark (in notebook):**
```python
df = spark.table("commodity.trading_agent.predictions_coffee")
```

### Notebook Local Directory (EPHEMERAL - Lost when cluster stops!)
When you save with `open('file.pkl', 'wb')` without a path, it goes to:
- `/databricks/driver/` on the cluster
- **This is TEMPORARY** - lost when cluster terminates
- **Always save to VOLUME_PATH instead!**

## Correct Pattern for Saving Files

### ❌ WRONG (ephemeral):
```python
with open('validation_results.pkl', 'wb') as f:
    pickle.dump(data, f)
```

### ✅ CORRECT (persistent):
```python
VOLUME_PATH = "/Volumes/commodity/trading_agent/files"
output_path = f"{VOLUME_PATH}/validation_results.pkl"

with open(output_path, 'wb') as f:
    pickle.dump(data, f)
```

## Reading Files from Volume

```python
VOLUME_PATH = "/Volumes/commodity/trading_agent/files"
input_path = f"{VOLUME_PATH}/validation_results.pkl"

with open(input_path, 'rb') as f:
    data = pickle.load(f)
```

## Downloading Files to Local Machine

```bash
# Set credentials
export DATABRICKS_HOST=https://dbc-5e4780f4-fcec.cloud.databricks.com
export DATABRICKS_TOKEN=dapi8f0886905a2b080bc5456595a8746b89

# Download from volume
databricks fs cp dbfs:/Volumes/commodity/trading_agent/files/validation_results_full.pkl ./validation_results_full.pkl

# Check file size
databricks fs ls dbfs:/Volumes/commodity/trading_agent/files/ | grep validation
```

## User Workspace

Primary user: `ground.truth.datascience@gmail.com`

Notebooks located at:
- `/Workspace/Users/ground.truth.datascience@gmail.com/`

## Common Commodities

```python
COMMODITY_CONFIGS = {
    'coffee': {...},
    'sugar': {...}
}
```

## Remember

1. **These notebooks run IN Databricks** - not locally
2. **Always save to VOLUME_PATH** - not local directory
3. **Use databricks CLI** to download files to local machine
4. **Delta tables** for structured data, **Volume** for binary/images
5. **spark object** is pre-configured - don't try to create it

## Analyzing Validation Results Locally

If you get pandas version errors when loading pickles:

```bash
# Downgrade to pandas 1.5.3 which is compatible
pip install 'pandas==1.5.3'
```

Then load normally:
```python
import pickle
with open('validation_results_full.pkl', 'rb') as f:
    data = pickle.load(f)
```

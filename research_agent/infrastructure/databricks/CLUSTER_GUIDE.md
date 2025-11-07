# Databricks Cluster Guide

## Overview

This project uses a **two-cluster architecture** to separate Unity Catalog queries from S3 ingestion jobs. This design solves the incompatibility between Unity Catalog's security model and S3 instance profile access.

## Quick Reference

| Cluster | Purpose | When to Use | Access Mode | S3 Access Method |
|---------|---------|-------------|-------------|------------------|
| **unity-catalog-cluster** | Queries, analytics, model training | ✅ Reading Delta tables<br>✅ SQL queries<br>✅ Model training<br>✅ Data analysis | Single User | Storage Credentials |
| **s3-ingestion-cluster** | Data ingestion from S3 | ✅ Auto Loader jobs<br>✅ Reading raw S3 files<br>✅ ETL pipelines | No Isolation Shared | Instance Profile |

## Cluster Details

### Unity Catalog Cluster

**Name**: `unity-catalog-cluster`

**Purpose**: All Unity Catalog operations including querying Bronze/Silver/Forecast tables

**Configuration**:
- Access Mode: `SINGLE_USER`
- Runtime: 13.3.x LTS
- Unity Catalog: Enabled
- S3 Access: Via storage credentials (NO instance profile)
- Auto-termination: 30 minutes
- Autoscaling: 1-2 workers

**Use this cluster for**:
- ✅ Querying `commodity.bronze.*` tables
- ✅ Querying `commodity.silver.*` tables
- ✅ Querying `commodity.forecast.*` tables
- ✅ Training SARIMAX models
- ✅ Data analysis and exploration
- ✅ Creating visualizations
- ✅ Running SQL queries in notebooks

**Example notebook code**:
```python
# Use Unity Catalog
spark.sql("USE CATALOG commodity")

# Query bronze tables
df = spark.sql("""
    SELECT * FROM commodity.bronze.weather
    WHERE region = 'Sul_de_Minas'
    LIMIT 100
""")

# Query silver tables
unified = spark.sql("""
    SELECT * FROM commodity.silver.unified_data
    WHERE commodity = 'coffee'
""")
```

**Cannot be used for**:
- ❌ Direct S3 reads without external locations
- ❌ Auto Loader streaming ingestion (use s3-ingestion-cluster)

---

### S3 Ingestion Cluster

**Name**: `s3-ingestion-cluster`

**Purpose**: Running Auto Loader jobs to ingest raw data from S3 landing zone

**Configuration**:
- Access Mode: `NONE` (No Isolation Shared)
- Runtime: 13.3.x LTS
- Unity Catalog: Disabled
- S3 Access: Via instance profile
- Instance Profile: `arn:aws:iam::534150427458:instance-profile/databricks-s3-access`
- Auto-termination: 20 minutes
- Autoscaling: 1-3 workers

**Use this cluster for**:
- ✅ Auto Loader streaming ingestion jobs
- ✅ Reading raw S3 files directly
- ✅ ETL pipelines from S3 → Delta
- ✅ Lambda-triggered data ingestion

**Example notebook code**:
```python
# Auto Loader ingestion from S3
df = (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", checkpoint_path)
    .load("s3://groundtruth-capstone/landing/weather_v2/")
)

# Write to Delta (NOT Unity Catalog managed table)
df.writeStream \
    .format("delta") \
    .option("checkpointLocation", checkpoint_path) \
    .trigger(availableNow=True) \
    .start("s3://groundtruth-capstone/delta/bronze/weather_v2/")
```

**Cannot be used for**:
- ❌ Querying Unity Catalog tables
- ❌ Running `USE CATALOG commodity`
- ❌ Accessing Unity Catalog managed tables

---

## Setup Instructions

### 1. Prerequisites

```bash
# Install Databricks CLI
pip install databricks-cli

# Configure with personal access token
databricks configure --token
# Host: https://dbc-fd7b00f3-7a6d.cloud.databricks.com
# Token: <your personal access token>
```

### 2. Create Unity Catalog Storage Setup (Admin Only)

**Run in Databricks SQL Editor as admin**:

1. Open SQL Editor in Databricks workspace
2. Open file: `research_agent/infrastructure/databricks/databricks_unity_catalog_storage_setup.sql`
3. Run the entire SQL script
4. Verify creation:
   ```sql
   SHOW EXTERNAL LOCATIONS;
   DESCRIBE STORAGE CREDENTIAL s3_groundtruth_capstone;
   ```

This creates:
- Storage credential using IAM role
- External locations for landing, bronze, silver, forecast, config, weather_v2
- Read permissions for users

### 3. Create Both Clusters

```bash
# Option 1: Create both clusters at once (recommended)
python research_agent/infrastructure/databricks/create_databricks_clusters.py --cluster both

# Option 2: Create individually
python research_agent/infrastructure/databricks/create_databricks_clusters.py --cluster unity
python research_agent/infrastructure/databricks/create_databricks_clusters.py --cluster s3-ingestion
```

The script will:
- Check if clusters already exist
- Create clusters from JSON configs
- Wait for clusters to reach RUNNING state
- Provide cluster IDs and test commands

### 4. Verify Unity Catalog Access

**Attach a notebook to `unity-catalog-cluster` and run**:

```python
# Test catalog access
spark.sql("USE CATALOG commodity")
display(spark.sql("SELECT current_catalog(), current_schema()"))

# Test bronze table access
display(spark.sql("SELECT COUNT(*) FROM commodity.bronze.weather"))

# Test silver table access
display(spark.sql("SELECT COUNT(*) FROM commodity.silver.unified_data"))

# Test forecast table access
display(spark.sql("SELECT COUNT(*) FROM commodity.forecast.point_forecasts"))
```

**Expected result**: All queries should return results without hanging.

### 5. Verify S3 Ingestion Access

**Attach a notebook to `s3-ingestion-cluster` and run**:

```python
# Test S3 direct access
df = spark.read.json("s3://groundtruth-capstone/landing/market_data/year=2024/month=11/day=01/*.json")
display(df.limit(5))

# Test Auto Loader
df = (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .load("s3://groundtruth-capstone/landing/weather_v2/")
    .limit(10))
display(df)
```

**Expected result**: S3 reads should work without authentication errors.

---

## How to Use the Right Cluster

### Scenario 1: Running SQL Queries on Bronze/Silver/Forecast Tables

**Use**: `unity-catalog-cluster`

**Why**: Unity Catalog tables require Single User access mode and storage credentials.

**How to attach**:
1. Open your notebook
2. Click cluster dropdown at top
3. Select "unity-catalog-cluster"
4. Run your queries

### Scenario 2: Running Auto Loader Ingestion Jobs

**Use**: `s3-ingestion-cluster`

**Why**: Auto Loader requires direct S3 access via instance profile.

**How to attach**:
1. Open notebook with Auto Loader code (e.g., `databricks_etl_setup.py`)
2. Click cluster dropdown at top
3. Select "s3-ingestion-cluster"
4. Run ingestion job

### Scenario 3: Training Models on Unity Catalog Data

**Use**: `unity-catalog-cluster`

**Why**: Models need to query Bronze/Silver tables for training data.

**Example**:
```python
# Attach to unity-catalog-cluster
spark.sql("USE CATALOG commodity")

# Load training data from Unity Catalog
train_df = spark.sql("""
    SELECT date, commodity, price, temperature_max, precipitation_sum
    FROM commodity.silver.unified_data
    WHERE date >= '2020-01-01'
""").toPandas()

# Train SARIMAX model
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(train_df['price'], exog=train_df[['temperature_max', 'precipitation_sum']])
results = model.fit()
```

### Scenario 4: Creating New Bronze Tables from S3

**Use**: `s3-ingestion-cluster` for ingestion, then `unity-catalog-cluster` for queries

**Workflow**:
1. **On s3-ingestion-cluster**: Run Auto Loader to create Delta files
   ```python
   # Write to S3 Delta location (NOT Unity Catalog table yet)
   df.writeStream \
       .format("delta") \
       .start("s3://groundtruth-capstone/delta/bronze/weather_v2/")
   ```

2. **In SQL Editor**: Register Delta location as Unity Catalog table
   ```sql
   CREATE TABLE IF NOT EXISTS commodity.bronze.weather_v2
   USING DELTA
   LOCATION 's3://groundtruth-capstone/delta/bronze/weather_v2/';
   ```

3. **On unity-catalog-cluster**: Query the new table
   ```python
   spark.sql("SELECT COUNT(*) FROM commodity.bronze.weather_v2").show()
   ```

---

## Common Issues and Troubleshooting

### Issue 1: "USE CATALOG commodity" Hangs in Notebook

**Symptom**: Query spins infinitely, never completes

**Cause**: Wrong cluster - likely using s3-ingestion-cluster or a cluster with instance profile

**Solution**: Switch to `unity-catalog-cluster`

**How to fix**:
1. Stop notebook execution
2. Change cluster to "unity-catalog-cluster"
3. Restart notebook
4. Rerun queries

---

### Issue 2: "Access Denied" When Reading S3 on Unity Catalog Cluster

**Symptom**: `java.nio.file.AccessDeniedException` or S3 403 errors

**Cause**: Unity Catalog cluster uses storage credentials, not instance profile

**Solution**: Use external locations or switch to s3-ingestion-cluster

**Options**:

A. Use Unity Catalog external locations (preferred):
```python
# Instead of direct S3 path
df = spark.read.json("s3://groundtruth-capstone/landing/weather_v2/*.json")

# Use Delta table registered in Unity Catalog
df = spark.sql("SELECT * FROM commodity.bronze.weather_v2")
```

B. Switch to s3-ingestion-cluster for direct S3 reads:
```python
# On s3-ingestion-cluster
df = spark.read.json("s3://groundtruth-capstone/landing/weather_v2/*.json")
```

---

### Issue 3: Auto Loader Fails on Unity Catalog Cluster

**Symptom**: Auto Loader streaming job fails with permission errors

**Cause**: Unity Catalog cluster cannot use Auto Loader with instance profile

**Solution**: Always run Auto Loader on `s3-ingestion-cluster`

**Correct approach**:
```python
# Switch notebook to s3-ingestion-cluster
# Then run Auto Loader
df = (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .load("s3://groundtruth-capstone/landing/weather_v2/"))
```

---

### Issue 4: Unity Catalog Tables Not Found

**Symptom**: `Table or view not found: commodity.bronze.weather`

**Cause**: Storage credential or external locations not set up

**Solution**: Run Unity Catalog storage setup SQL

1. Open SQL Editor
2. Run: `research_agent/infrastructure/databricks/databricks_unity_catalog_storage_setup.sql`
3. Verify: `SHOW EXTERNAL LOCATIONS;`
4. Verify: `DESCRIBE STORAGE CREDENTIAL s3_groundtruth_capstone;`

---

### Issue 5: Cluster Not Found in Dropdown

**Symptom**: Cluster doesn't appear in notebook cluster dropdown

**Cause**: Cluster not created yet or terminated

**Solution**: Create cluster using automation script

```bash
python research_agent/infrastructure/databricks/create_databricks_clusters.py --cluster both
```

Or manually create in Databricks UI using the JSON configs.

---

## Daily Operations

### Morning Startup Checklist

1. **Check if Lambda functions ran successfully**:
   ```bash
   aws lambda list-functions | grep -E "(market|weather|vix|cftc)"
   aws logs tail /aws/lambda/market-data-fetcher --follow
   ```

2. **Start s3-ingestion-cluster for daily ETL**:
   ```bash
   databricks clusters start --cluster-id <s3-ingestion-cluster-id>
   ```

3. **Run Auto Loader ingestion on s3-ingestion-cluster**:
   - Attach `databricks_etl_setup.py` notebook to s3-ingestion-cluster
   - Run all cells to ingest latest data

4. **Verify data loaded**:
   - Switch to unity-catalog-cluster
   - Run: `SELECT MAX(date) FROM commodity.silver.unified_data`
   - Should show yesterday's date

5. **Run forecasts on unity-catalog-cluster**:
   - Attach forecast notebook to unity-catalog-cluster
   - Run SARIMAX models
   - Save predictions to commodity.forecast.point_forecasts

### When to Use SQL Warehouse Instead

**SQL Warehouse** (serverless) is more expensive but offers:
- Zero startup time
- Auto-scaling without management
- Built-in Unity Catalog support
- Good for: Ad-hoc queries, dashboards, quick analysis

**Clusters** (this guide) are more cost-effective but require:
- Manual startup/shutdown
- Cluster management
- Good for: Regular ETL jobs, model training, development

**Recommendation**: Use clusters for scheduled jobs and development. Use SQL Warehouse for executive dashboards and ad-hoc analysis.

---

## Cost Optimization Tips

1. **Use auto-termination**:
   - Unity Catalog cluster: 30 minutes (already configured)
   - S3 ingestion cluster: 20 minutes (already configured)

2. **Stop clusters when not in use**:
   ```bash
   databricks clusters delete --cluster-id <cluster-id>
   ```

3. **Use SPOT instances** (already configured):
   - Both clusters use SPOT_WITH_FALLBACK
   - Saves ~70% on compute costs
   - Automatically falls back to on-demand if spot unavailable

4. **Minimize cluster size**:
   - Start with 1 worker, scale up only if needed
   - Both clusters already configured for minimal size

5. **Schedule jobs during off-peak hours**:
   - Auto Loader: Run at 6 AM daily
   - Model training: Run weekly on Sundays

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     AWS Lambda Functions                     │
│  (market-data, weather-data, vix-data, cftc-data, gdelt)    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
           ┌─────────────────────┐
           │   S3 Landing Zone   │
           │  landing/*/...json  │
           └─────────┬───────────┘
                     │
                     │ Auto Loader (cloudFiles)
                     │
                     ▼
           ┌─────────────────────┐
           │ s3-ingestion-cluster│◄── Use this for Auto Loader
           │  (Instance Profile) │
           └─────────┬───────────┘
                     │
                     │ Write Delta
                     │
                     ▼
           ┌─────────────────────┐
           │   S3 Delta Tables   │
           │ delta/bronze/...    │
           │ delta/silver/...    │
           │ delta/forecast/...  │
           └─────────┬───────────┘
                     │
                     │ Unity Catalog External Locations
                     │
                     ▼
           ┌─────────────────────┐
           │unity-catalog-cluster│◄── Use this for queries
           │ (Storage Credential)│
           └─────────┬───────────┘
                     │
                     ▼
           ┌─────────────────────┐
           │  Unity Catalog      │
           │  commodity.bronze.* │
           │  commodity.silver.* │
           │  commodity.forecast.*│
           └─────────────────────┘
```

---

## Summary

**Two clusters solve two problems**:

1. **s3-ingestion-cluster**: Direct S3 access for Lambda output ingestion
   - Uses instance profile for unfettered S3 access
   - Runs Auto Loader streaming jobs
   - Cannot access Unity Catalog

2. **unity-catalog-cluster**: Governed data access for analytics
   - Uses storage credentials for secure S3 access
   - Queries Bronze/Silver/Forecast tables
   - Cannot run Auto Loader with instance profile

**Key principle**: Ingest raw → s3-ingestion-cluster, Query curated → unity-catalog-cluster

---

## Related Files

- **Cluster Configs**:
  - `research_agent/infrastructure/databricks/databricks_unity_catalog_cluster.json`
  - `research_agent/infrastructure/databricks/databricks_s3_ingestion_cluster.json`

- **Setup Scripts**:
  - `research_agent/infrastructure/databricks/create_databricks_clusters.py`
  - `research_agent/infrastructure/databricks/databricks_unity_catalog_storage_setup.sql`

- **ETL Notebooks**:
  - `lambda_migration/databricks_etl_setup.py`

- **Lambda Functions**:
  - `lambda_migration/market_data_fetcher.py`
  - `lambda_migration/weather_data_fetcher.py`
  - `lambda_migration/vix_data_fetcher.py`

---

## Questions?

If you encounter issues not covered here:

1. Check Databricks cluster logs (Cluster → Event Log)
2. Check Unity Catalog permissions (SQL Editor → Data Explorer)
3. Verify storage credential: `DESCRIBE STORAGE CREDENTIAL s3_groundtruth_capstone`
4. Verify external locations: `SHOW EXTERNAL LOCATIONS`
5. Check S3 bucket access: `aws s3 ls s3://groundtruth-capstone/`

**Last updated**: 2025-11-06

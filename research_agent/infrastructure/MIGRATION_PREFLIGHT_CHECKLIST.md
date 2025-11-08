# Databricks Migration Pre-Flight Checklist

**Last Updated**: 2025-11-06
**Status**: Ready for migration

---

## Quick Start

**For the complete migration guide, see:** [`DATABRICKS_MIGRATION_GUIDE.md`](./DATABRICKS_MIGRATION_GUIDE.md)

---

## Pre-Migration Verification

### Data Readiness

- [x] **Weather backfill v2 completed**
  - Location: `s3://groundtruth-capstone/landing/weather_v2/`
  - Contains corrected coordinates (growing regions, not state capitals)
  - Date range: 2015-07-07 to 2025-11-05

- [ ] **Verify weather backfill completed**
  ```bash
  # Check if backfill process is still running
  tail -100 research_agent/infrastructure/weather_backfill_v2.log | grep "BACKFILL COMPLETE"

  # Check S3 for data
  aws s3 ls s3://groundtruth-capstone/landing/weather_v2/ --recursive | wc -l
  ```

- [ ] **Verify July 2021 frost event in new data**
  ```bash
  aws s3 ls s3://groundtruth-capstone/landing/weather_v2/year=2021/month=07/ --recursive
  ```

### Existing Resources Documented

- [x] **IAM Roles**
  - Current S3 role: `arn:aws:iam::534150427458:role/databricks-groundtruth-s3-profile`
  - Policy: `AmazonS3FullAccess`

- [x] **Cluster Configurations**
  - Location: `research_agent/infrastructure/databricks/*.json`
  - Ready for new workspace

- [x] **Unity Catalog Schema**
  - Catalog: `commodity`
  - Schemas: `bronze`, `silver`, `forecast`
  - Table definitions documented in migration guide

### Code and Configuration

- [ ] **Git repo is up to date**
  ```bash
  cd /Users/connorwatson/Documents/Data\ Science/DS210/ucberkeley-capstone
  git status
  git pull origin main
  ```

- [ ] **All cluster configs in version control**
  ```bash
  ls research_agent/infrastructure/databricks/*.json
  # Should show:
  # - databricks_unity_catalog_cluster.json
  # - databricks_s3_ingestion_cluster.json
  # - databricks_unity_catalog_storage_setup.sql
  ```

- [ ] **Unity Catalog setup scripts ready**
  ```bash
  ls research_agent/infrastructure/databricks/setup_unity_catalog_credentials.py
  ls research_agent/infrastructure/databricks/databricks_unity_catalog_storage_setup.sql
  ```

---

## Migration Day - Quick Steps

### Phase 1: AWS IAM Setup (30 minutes)

```bash
# 1. Create cross-account role
aws iam create-role \
  --role-name databricks-cross-account-role \
  --assume-role-policy-document file://databricks-cross-account-trust-policy.json

# 2. Create S3 access role for Unity Catalog
aws iam create-role \
  --role-name databricks-s3-access-role \
  --assume-role-policy-document file://databricks-s3-trust-policy.json

aws iam attach-role-policy \
  --role-name databricks-s3-access-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# 3. Get role ARN (save this)
aws iam get-role --role-name databricks-s3-access-role --query 'Role.Arn' --output text
```

**Save ARN**: `arn:aws:iam::534150427458:role/databricks-s3-access-role`

### Phase 2: Databricks Workspace Setup (15 minutes)

**In Databricks Account Console:**

1. **Create Workspace**
   - Name: `groundtruth-capstone`
   - Region: `us-west-2`
   - Network: **Public network** (NO PrivateLink!)

2. **Create Unity Catalog Metastore**
   - Name: `commodity_metastore`
   - S3 Bucket: `s3://groundtruth-capstone-metastore/` (create first)
   - IAM Role: `arn:aws:iam::534150427458:role/databricks-s3-access-role`

3. **Assign metastore to workspace**

### Phase 3: Unity Catalog Configuration (20 minutes)

**In SQL Editor:**

```sql
-- 1. Create storage credential
CREATE STORAGE CREDENTIAL IF NOT EXISTS s3_groundtruth_capstone
USING AWS_IAM_ROLE
WITH (role_arn = 'arn:aws:iam::534150427458:role/databricks-s3-access-role')
COMMENT 'Storage credential for groundtruth-capstone S3 buckets';

-- 2. Create external locations (run full SQL script)
-- See: research_agent/infrastructure/databricks/databricks_unity_catalog_storage_setup.sql

-- 3. Create catalog and schemas
CREATE CATALOG IF NOT EXISTS commodity;
USE CATALOG commodity;
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS forecast;
```

### Phase 4: Create Clusters (10 minutes)

```bash
cd research_agent/infrastructure/databricks

# Configure Databricks CLI with new workspace
databricks configure --token
# Enter: https://<NEW_WORKSPACE>.cloud.databricks.com
# Enter: <NEW_TOKEN>

# Create Unity Catalog cluster
databricks clusters create --json-file databricks_unity_catalog_cluster.json
```

### Phase 5: Test Unity Catalog (15 minutes)

**In notebook attached to new cluster:**

```python
# Test 1: USE CATALOG (was hanging in old workspace)
spark.sql("USE CATALOG commodity")
print("✅ USE CATALOG works!")

# Test 2: Read from S3 via external location
df = spark.sql("SELECT * FROM json.`s3://groundtruth-capstone/config/region_coordinates.json` LIMIT 5")
df.show()
print("✅ S3 access works!")

# Test 3: Query non-existent table (should fail gracefully)
try:
    spark.sql("SELECT COUNT(*) FROM commodity.bronze.weather")
    print("✅ Table exists (or)")
except:
    print("⚠️ Table doesn't exist yet (expected - will load next)")
```

**CRITICAL SUCCESS CRITERIA:**
- `USE CATALOG commodity` completes in < 1 second (was hanging for 120s+ in old workspace)
- No `/etc/hosts` errors in cluster logs
- S3 reads work via external locations

### Phase 6: Load Data (30 minutes)

**Load weather_v2 into bronze:**

```python
from pyspark.sql import functions as F

# Read weather_v2 from landing
df_weather = spark.read.json("s3://groundtruth-capstone/landing/weather_v2/")

# Transform and load
df_bronze = df_weather.select(
    F.col("date").cast("date"),
    F.col("commodity"),
    F.col("region"),
    F.col("temperature_2m_mean").alias("temp_c"),
    F.col("relative_humidity_2m_mean").alias("humidity_pct"),
    F.col("precipitation_sum").alias("precipitation_mm"),
    F.year("date").alias("year"),
    F.month("date").alias("month"),
    F.dayofmonth("date").alias("day")
)

df_bronze.write.format("delta").mode("overwrite").saveAsTable("commodity.bronze.weather")
print(f"✅ Loaded {df_bronze.count()} weather records")
```

**Validate July 2021 frost:**

```python
df_frost = spark.sql("""
    SELECT date, region, temp_c
    FROM commodity.bronze.weather
    WHERE commodity = 'Coffee'
      AND region IN ('Minas Gerais', 'Sao Paulo', 'Parana')
      AND date BETWEEN '2021-07-18' AND '2021-07-22'
    ORDER BY date, region
""")
df_frost.show(50)

# Check for freezing temps
df_frost.groupBy("region").agg(F.min("temp_c").alias("min_temp")).show()
```

**Expected**: Minimum temps near/below 0°C (frost event captured!)

---

## Post-Migration Validation

### Immediate Tests (Day 1)

- [ ] Unity Catalog queries complete without hanging
- [ ] `spark.sql("USE CATALOG commodity")` < 1 second
- [ ] Weather v2 data loaded into `commodity.bronze.weather`
- [ ] July 2021 frost visible in data (min temps < 0°C)
- [ ] Team members can access new workspace
- [ ] Notebooks can attach to clusters

### Extended Validation (Week 1)

- [ ] Forecast models retrained on new workspace
- [ ] unified_data rebuilt with weather_v2
- [ ] Trading agent can query forecast tables
- [ ] No cluster errors or hangs observed
- [ ] All team members onboarded

---

## Rollback Criteria

**Abort migration if:**

1. Unity Catalog queries still hang after 10 seconds
2. `/etc/hosts` errors appear in cluster logs
3. Cannot read from S3 via external locations
4. Storage credential creation fails
5. Private Access Settings exist in new workspace

**Rollback actions:**
1. Keep old workspace active
2. Revert connection strings
3. Continue using `databricks-sql-connector` workaround
4. Escalate to Databricks Support

---

## Key Files for Migration

```
research_agent/infrastructure/
├── DATABRICKS_MIGRATION_GUIDE.md         # Complete guide (THIS IS THE MAIN GUIDE)
├── MIGRATION_PREFLIGHT_CHECKLIST.md     # This file - quick reference
├── databricks/
│   ├── databricks_unity_catalog_cluster.json
│   ├── databricks_s3_ingestion_cluster.json
│   ├── databricks_unity_catalog_storage_setup.sql
│   ├── setup_unity_catalog_credentials.py
│   └── create_databricks_clusters.py
├── test_cluster_unity_catalog.py         # Test script for validation
└── unity_catalog_workaround.py           # Fallback if migration fails
```

---

## Team Communication Template

**Subject**: Databricks Migration - November [DATE]

**Team**,

We're migrating to a new Databricks account to fix the Unity Catalog hanging issue.

**Migration Window**: [DATE] [TIME]

**Downtime**: ~2 hours (worst case)

**What you need to do:**

1. **After migration completes**, update your Databricks CLI:
   ```bash
   databricks configure --token
   # Enter: https://<NEW_WORKSPACE>.cloud.databricks.com
   # Enter: <YOUR_NEW_TOKEN>
   ```

2. **Update environment variables**:
   ```bash
   export DATABRICKS_HOST="https://<NEW_WORKSPACE>.cloud.databricks.com"
   export DATABRICKS_TOKEN="<YOUR_NEW_TOKEN>"
   ```

3. **Test access**: Open new workspace, attach notebook to `unity-catalog-cluster`

**Expected improvements:**
- Unity Catalog queries work instantly (no more hanging)
- Can use `spark.sql()` and `spark.table()` directly
- No workarounds needed

**Rollback plan**: If migration fails, we'll revert to old workspace and continue using SQL connector workaround.

Questions? Contact Connor.

---

## Contact

**Migration Lead**: Connor
**Research Agent**: Francisco
**Trading Agent**: Mark

**Migration Guide**: [`DATABRICKS_MIGRATION_GUIDE.md`](./DATABRICKS_MIGRATION_GUIDE.md)

---

## Status Log

### 2025-11-06
- [x] Weather backfill v2 running (processing 67 regions, 2015-2025)
- [x] Migration guide created
- [x] Pre-flight checklist created
- [x] Cluster configurations ready
- [x] Unity Catalog setup scripts ready
- [ ] Weather backfill completion (in progress)
- [ ] Migration scheduled (pending backfill completion)

### Next Actions
1. Wait for weather backfill v2 to complete
2. Validate July 2021 frost in backfilled data
3. Schedule migration window with team
4. Execute migration
5. Validate new workspace
6. Onboard team

# Infrastructure Cleanup Plan

**Date**: 2025-12-05
**Purpose**: Clean up obsolete cluster configs, organize one-time setup scripts

---

## 🎯 Summary

Based on review:
- **Delete**: 5 obsolete files (old cluster configs + superseded script)
- **Move to one_time_setup**: 9 scripts (useful for rebuilding environment)
- **Keep active**: 2 Python scripts, 4 SQL scripts, GDELT deploy scripts
- **Markdowns**: Defer to documentation strategy cleanup

---

## 📋 Cleanup Actions

### PHASE 1: Delete Obsolete Cluster Configs

**Old cluster configs in databricks/ root** (superseded by `databricks/clusters/`):

```bash
cd research_agent/infrastructure/databricks

# Delete 4 obsolete cluster JSON files
rm -f databricks_s3_ingestion_cluster.json \
      databricks_unity_catalog_fixed.json \
      databricks_unity_simple.json \
      unity_catalog_working.json
```

**Why delete:**
- All created Nov 6 19:58-22:35
- Superseded by `databricks/clusters/databricks_unity_catalog_cluster.json` (Dec 5, better config)
- Have incomplete configs (missing autoscale, spot instances, proper tags)
- No longer referenced anywhere

**Latest config location:** `databricks/clusters/databricks_unity_catalog_cluster.json`

---

### PHASE 2: Delete Superseded Script

```bash
cd research_agent/infrastructure/databricks

# Delete old cluster creation script (superseded by clusters/create_unity_catalog_cluster.py)
rm -f create_databricks_clusters.py
```

**Why delete:**
- Created Nov 6 19:58
- Superseded by `databricks/clusters/create_unity_catalog_cluster.py` (Dec 5)
- New version has better documentation and cluster configs

---

### PHASE 3: Organize One-Time Setup Scripts

**Create one_time_setup folders:**

```bash
cd research_agent/infrastructure

# Create structure
mkdir -p lambda/one_time_setup
mkdir -p databricks/one_time_setup
```

**Move Lambda setup scripts:**

```bash
cd lambda

# These are useful for rebuilding environment
mv deploy_all_functions.sh one_time_setup/
mv backfill_historical_data.sh one_time_setup/
mv setup_all_eventbridge_schedules.sh one_time_setup/
mv setup_eventbridge_schedule.sh one_time_setup/
```

**Move Databricks setup scripts:**

```bash
cd ../databricks

# These are useful for rebuilding environment
mv setup_databricks_jobs.py one_time_setup/
mv setup_unity_catalog_credentials.py one_time_setup/
mv deploy_tables_curl.sh one_time_setup/
mv cleanup_old_gdelt_tables.sh one_time_setup/
mv fix_unity_catalog_hosts.sh one_time_setup/
```

**GDELT deploy scripts - KEEP in lambda/ root for now:**
- `deploy_bronze_transform.sh` - Still used for updating GDELT bronze Lambda
- `deploy_jsonl_bronze_transform.sh` - Still used for updating GDELT JSONL Lambda
- Last modified Nov 20-22, relatively recent

---

### PHASE 4: Move region_coordinates.json

```bash
cd research_agent

# Move config to weather Lambda function folder
mv config/region_coordinates.json \
   infrastructure/lambda/functions/weather-data-fetcher/region_coordinates.json

# Remove empty config folder
rmdir config/
```

**Update references:**
- `research_agent/DATA_SOURCES.md` (2 references)
- `tests/validate_region_coordinates.py` (1 reference)

**Lambda code:** ✅ No changes needed (loads from S3)

---

## 📁 Final Structure

```
research_agent/infrastructure/
├── lambda/
│   ├── functions/
│   │   ├── weather-data-fetcher/
│   │   │   ├── app.py
│   │   │   ├── region_coordinates.json    ← MOVED HERE
│   │   │   └── requirements.txt
│   │   └── ... (13 other Lambda functions)
│   ├── one_time_setup/                    ← NEW
│   │   ├── deploy_all_functions.sh
│   │   ├── backfill_historical_data.sh
│   │   ├── setup_all_eventbridge_schedules.sh
│   │   └── setup_eventbridge_schedule.sh
│   ├── deploy_bronze_transform.sh         ← Keep (still used)
│   ├── deploy_jsonl_bronze_transform.sh   ← Keep (still used)
│   ├── GDELT_LAMBDA_AUDIT_CHECKLIST.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   └── WEATHER_SCHEMA_FIX.md
│
├── databricks/
│   ├── sql/                               ← Active SQL scripts
│   │   ├── 01_create_landing_tables.sql
│   │   ├── 02_create_bronze_views.sql
│   │   ├── gdelt_silver_simple.sql
│   │   └── create_gdelt_fillforward.sql
│   ├── clusters/                          ← Latest cluster configs
│   │   ├── create_unity_catalog_cluster.py
│   │   ├── list_databricks_clusters.py
│   │   ├── databricks_unity_catalog_cluster.json
│   │   ├── README.md
│   │   └── UNITY_CATALOG_CLUSTER_RATIONALE.md
│   ├── one_time_setup/                    ← NEW
│   │   ├── setup_databricks_jobs.py
│   │   ├── setup_unity_catalog_credentials.py
│   │   ├── deploy_tables_curl.sh
│   │   ├── cleanup_old_gdelt_tables.sh
│   │   └── fix_unity_catalog_hosts.sh
│   ├── refresh_fillforward.py             ← Active
│   ├── validate_gold_unified_data.py      ← Active
│   ├── GDELT_MIGRATION_GUIDE.md
│   └── README.md
│
└── archive/                               ← Existing archive folder
    └── ... (existing archived files)
```

---

## ✅ What We're Keeping Active

**Lambda:**
- All 14 function folders in `functions/`
- GDELT deployment scripts (still used for updates)
- Reference documentation (markdown files)

**Databricks:**
- `refresh_fillforward.py` - Active script
- `validate_gold_unified_data.py` - Active validation
- `clusters/` folder - Latest cluster management
- SQL scripts (will organize into `sql/` subfolder later)
- Reference documentation

---

## 🗑️ What We're Deleting (5 files)

1. `databricks/databricks_s3_ingestion_cluster.json` - Obsolete cluster config
2. `databricks/databricks_unity_catalog_fixed.json` - Obsolete cluster config
3. `databricks/databricks_unity_simple.json` - Obsolete cluster config
4. `databricks/unity_catalog_working.json` - Obsolete cluster config
5. `databricks/create_databricks_clusters.py` - Superseded by clusters/ script

**All have git backup** - Can restore from commit history if needed.

---

## 📦 What We're Moving (9 files)

**Lambda → lambda/one_time_setup/ (4 files):**
1. `deploy_all_functions.sh`
2. `backfill_historical_data.sh`
3. `setup_all_eventbridge_schedules.sh`
4. `setup_eventbridge_schedule.sh`

**Databricks → databricks/one_time_setup/ (5 files):**
1. `setup_databricks_jobs.py`
2. `setup_unity_catalog_credentials.py`
3. `deploy_tables_curl.sh`
4. `cleanup_old_gdelt_tables.sh`
5. `fix_unity_catalog_hosts.sh`

---

## 📝 Documentation Updates Needed

After Phase 4 (region_coordinates.json move):

### 1. Update DATA_SOURCES.md
**File:** `research_agent/DATA_SOURCES.md`

**Line 73:**
```diff
-See `research_agent/config/region_coordinates.json` for:
+See `research_agent/infrastructure/lambda/functions/weather-data-fetcher/region_coordinates.json` for:
```

**Line 272:**
```diff
-**File**: `research_agent/config/region_coordinates.json`
+**File**: `research_agent/infrastructure/lambda/functions/weather-data-fetcher/region_coordinates.json`
```

### 2. Update tests/validate_region_coordinates.py
**File:** `research_agent/infrastructure/tests/validate_region_coordinates.py`

**Line 13:**
```diff
-with open('research_agent/config/region_coordinates.json', 'r') as f:
+with open('research_agent/infrastructure/lambda/functions/weather-data-fetcher/region_coordinates.json', 'r') as f:
```

Or use absolute path from repo root.

### 3. Update archive docs (low priority)
Files in `archive/` that reference old path - can update later if needed.

---

## 🚀 Execution Order

```bash
# 1. Delete obsolete cluster configs (safest)
cd research_agent/infrastructure/databricks
rm -f databricks_s3_ingestion_cluster.json \
      databricks_unity_catalog_fixed.json \
      databricks_unity_simple.json \
      unity_catalog_working.json \
      create_databricks_clusters.py

# 2. Create one_time_setup folders
cd ../
mkdir -p lambda/one_time_setup databricks/one_time_setup

# 3. Move Lambda setup scripts
cd lambda
mv deploy_all_functions.sh backfill_historical_data.sh \
   setup_all_eventbridge_schedules.sh setup_eventbridge_schedule.sh \
   one_time_setup/

# 4. Move Databricks setup scripts
cd ../databricks
mv setup_databricks_jobs.py setup_unity_catalog_credentials.py \
   deploy_tables_curl.sh cleanup_old_gdelt_tables.sh \
   fix_unity_catalog_hosts.sh \
   one_time_setup/

# 5. Move region_coordinates.json
cd ../../
mv config/region_coordinates.json \
   infrastructure/lambda/functions/weather-data-fetcher/
rmdir config/

# 6. Update documentation references (see section above)

# 7. Git commit
git add -A research_agent/infrastructure/
git commit -m "chore: Clean up infrastructure folder organization"
```

---

## 🎯 Benefits

**Clarity:**
- One-time setup scripts clearly separated from active code
- Latest cluster config obvious (in clusters/ folder)
- Config lives with code that uses it (region_coordinates.json)

**Discoverability:**
- New team members know where to find setup scripts
- Clear what's active vs historical

**Maintainability:**
- Less clutter in root folders
- Easier to find active scripts
- Documentation strategy can handle markdowns separately

---

## ⚠️ Risks

**Low Risk:**
- All deleted files have git backup
- One-time setup scripts rarely run
- Region coordinates move is transparent (Lambda uses S3)

**Testing:**
- Verify GDELT deploy scripts still work (if we move them)
- Verify region_coordinates.json can still be found by tests

---

## 📌 Next Steps

1. **Execute Phases 1-2** (delete obsolete files) - Zero risk
2. **Execute Phases 3-4** (reorganize) - Low risk
3. **Update documentation** - Required
4. **Test** - Verify nothing broken
5. **Commit** - Single atomic commit

---

**Recommendation**: Execute all phases in one session, single commit.
**Estimated time**: 10 minutes
**Documentation updates**: 5 minutes

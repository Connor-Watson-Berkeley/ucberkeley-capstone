# Infrastructure Reorganization Proposal

**Date**: 2025-12-05
**Purpose**: Clean up cluttered lambda/ and databricks/ folders, improve discoverability

---

## 🎯 Goals

1. **Move region_coordinates.json closer to the Lambda that uses it**
2. **Establish consistent naming patterns**
3. **Archive obsolete files**
4. **Make active vs inactive components obvious**

---

## 📊 Current State Analysis

### Active Lambda Functions (from AWS)
```
✅ market-data-fetcher          (EventBridge: groundtruth-market-data-daily)
✅ vix-data-fetcher             (EventBridge: groundtruth-vix-data-daily)
✅ fx-calculator-fetcher        (EventBridge: groundtruth-fx-data-daily)
✅ weather-data-fetcher         (EventBridge: groundtruth-weather-data-daily)
✅ cftc-data-fetcher            (EventBridge: groundtruth-cftc-data-daily)
✅ gdelt-daily-discovery        (EventBridge: gdelt-daily-discovery-schedule)
✅ gdelt-bronze-transform
✅ gdelt-silver-transform
✅ gdelt-silver-discovery
✅ gdelt-silver-backfill
✅ gdelt-processor
✅ gdelt-sqs-loader
✅ gdelt-queue-monitor
✅ gdelt-sqs-trigger-manager
✅ gdelt-generate-date-batches
✅ gdelt-jsonl-bronze-transform
```

### Active SQL Scripts (from dependency graph)
```
✅ 01_create_landing_tables.sql     - Creates commodity.landing.* tables
✅ 02_create_bronze_views.sql       - Creates commodity.bronze.* views
✅ gdelt_silver_simple.sql          - Refreshes GDELT silver external table
✅ create_gdelt_fillforward.sql     - Alternative GDELT approach (keep for reference)
✅ create_gold_unified_data.sql     - Main gold layer (lives in research_agent/sql/)
```

### Files in lambda/ Root (Needs Cleanup)
```
❓ backfill_historical_data.sh         - One-time historical backfill?
✅ deploy_all_functions.sh             - Active deployment script
❓ deploy_bronze_transform.sh          - GDELT-specific deployment
❓ deploy_jsonl_bronze_transform.sh    - GDELT-specific deployment
✅ GDELT_LAMBDA_AUDIT_CHECKLIST.md     - Reference doc
✅ IMPLEMENTATION_SUMMARY.md           - Reference doc
❓ setup_all_eventbridge_schedules.sh  - One-time setup?
❓ setup_eventbridge_schedule.sh       - Utility script?
🗑️ weather_forecast_fetcher.py         - OLD VERSION (archive/ has newer)
✅ WEATHER_SCHEMA_FIX.md               - Reference doc
```

### Files in databricks/ Root (Needs Cleanup)
```
✅ 01_create_landing_tables.sql
✅ 02_create_bronze_views.sql
❓ cleanup_old_gdelt_tables.sh
✅ clusters/                           - Cluster configs (keep)
❓ create_databricks_clusters.py
✅ create_gdelt_fillforward.sql
❓ databricks_s3_ingestion_cluster.json
❓ databricks_unity_catalog_fixed.json
❓ databricks_unity_catalog_storage_setup.sql  - One-time setup?
❓ databricks_unity_simple.json
❓ deploy_tables_curl.sh
❓ fix_unity_catalog_hosts.sh
❓ gdelt_bronze_simple.sql
✅ GDELT_MIGRATION_GUIDE.md            - Reference doc
❓ gdelt_silver_simple.sql
✅ README.md
✅ refresh_fillforward.py
❓ setup_databricks_jobs.py
❓ setup_unity_catalog_credentials.py
❓ unity_catalog_working.json
✅ validate_gold_unified_data.py
```

---

## 🎨 Proposed New Structure

### Option A: Minimal Reorganization (Safer)
```
research_agent/infrastructure/
├── lambda/
│   ├── functions/
│   │   ├── weather-data-fetcher/
│   │   │   ├── app.py
│   │   │   ├── region_coordinates.json  ← MOVED HERE (from research_agent/config/)
│   │   │   └── requirements.txt
│   │   ├── market-data-fetcher/
│   │   ├── vix-data-fetcher/
│   │   └── ...
│   ├── deployment/                      ← NEW: Group deployment scripts
│   │   ├── deploy_all_functions.sh
│   │   ├── deploy_bronze_transform.sh
│   │   └── deploy_jsonl_bronze_transform.sh
│   ├── docs/                            ← NEW: Consolidate reference docs
│   │   ├── GDELT_LAMBDA_AUDIT_CHECKLIST.md
│   │   ├── IMPLEMENTATION_SUMMARY.md
│   │   └── WEATHER_SCHEMA_FIX.md
│   └── scripts/                         ← NEW: One-time setup scripts
│       ├── backfill_historical_data.sh
│       ├── setup_all_eventbridge_schedules.sh
│       └── setup_eventbridge_schedule.sh
│
├── databricks/
│   ├── sql/                             ← NEW: Active SQL scripts
│   │   ├── 01_create_landing_tables.sql
│   │   ├── 02_create_bronze_views.sql
│   │   ├── gdelt_silver_simple.sql
│   │   └── create_gdelt_fillforward.sql
│   ├── python/                          ← NEW: Active Python scripts
│   │   ├── refresh_fillforward.py
│   │   └── validate_gold_unified_data.py
│   ├── clusters/                        ← Keep as-is
│   │   └── ...
│   ├── docs/                            ← NEW: Reference docs
│   │   ├── GDELT_MIGRATION_GUIDE.md
│   │   └── README.md
│   └── setup/                           ← NEW: One-time setup scripts
│       ├── cleanup_old_gdelt_tables.sh
│       ├── create_databricks_clusters.py
│       ├── databricks_unity_catalog_storage_setup.sql
│       ├── deploy_tables_curl.sh
│       ├── fix_unity_catalog_hosts.sh
│       ├── setup_databricks_jobs.py
│       └── setup_unity_catalog_credentials.py
│
└── archive/
    └── lambda/
        └── weather_forecast_fetcher.py  ← Move obsolete version here
```

### Option B: Aggressive Cleanup (More Disruptive)
```
research_agent/infrastructure/
├── lambda/
│   └── functions/                       ← ONLY function code
│       ├── weather-data-fetcher/
│       │   ├── app.py
│       │   ├── region_coordinates.json  ← Config lives with code
│       │   └── requirements.txt
│       └── ...
│
├── databricks/
│   ├── 01_create_landing_tables.sql     ← Flat structure
│   ├── 02_create_bronze_views.sql
│   ├── gdelt_silver_simple.sql
│   ├── create_gdelt_fillforward.sql
│   ├── refresh_fillforward.py
│   ├── validate_gold_unified_data.py
│   ├── clusters/
│   └── README.md
│
├── deployment/                          ← All deployment scripts
│   ├── deploy_lambda_functions.sh
│   └── ...
│
└── archive/
    ├── lambda/
    │   ├── weather_forecast_fetcher.py
    │   └── docs/                        ← Archive all markdown
    │       ├── GDELT_LAMBDA_AUDIT_CHECKLIST.md
    │       └── ...
    └── databricks/
        └── setup/                       ← Archive all one-time setup
            ├── create_databricks_clusters.py
            └── ...
```

---

## 🎯 Recommended Approach: Option A (Minimal)

**Why Option A:**
- Less disruptive (keeps existing structure mostly intact)
- Groups related files into clear subfolders
- Preserves reference documentation
- Easy to find active vs setup/archive files

**Key Moves:**
1. ✅ `region_coordinates.json` → `lambda/functions/weather-data-fetcher/`
2. ✅ Deployment scripts → `lambda/deployment/`
3. ✅ Reference docs → `lambda/docs/` and `databricks/docs/`
4. ✅ One-time setup → `lambda/scripts/` and `databricks/setup/`
5. ✅ Active SQL → `databricks/sql/`
6. ✅ Active Python → `databricks/python/`
7. ✅ Obsolete files → `archive/lambda/`

---

## 📋 Migration Plan

### Phase 1: Move region_coordinates.json (Immediate)
```bash
# Move file
mv research_agent/config/region_coordinates.json \
   research_agent/infrastructure/lambda/functions/weather-data-fetcher/

# Update Lambda code (already uses S3, no change needed)
# The Lambda loads from S3: s3://groundtruth-capstone/config/region_coordinates.json

# Update documentation references:
# - research_agent/DATA_SOURCES.md (2 references)
# - tests/validate_region_coordinates.py (1 reference)
```

### Phase 2: Reorganize lambda/ (Low Risk)
```bash
# Create new folders
mkdir -p lambda/deployment lambda/docs lambda/scripts

# Move deployment scripts
mv lambda/deploy_*.sh lambda/deployment/
mv lambda/setup_*.sh lambda/scripts/
mv lambda/backfill_historical_data.sh lambda/scripts/

# Move docs
mv lambda/*.md lambda/docs/

# Move obsolete code to archive
mv lambda/weather_forecast_fetcher.py archive/lambda/
```

### Phase 3: Reorganize databricks/ (Medium Risk)
```bash
# Create new folders
mkdir -p databricks/sql databricks/python databricks/setup databricks/docs

# Move active SQL
mv databricks/*.sql databricks/sql/
mv databricks/sql/databricks_unity_catalog_storage_setup.sql databricks/setup/

# Move active Python
mv databricks/refresh_fillforward.py databricks/python/
mv databricks/validate_gold_unified_data.py databricks/python/

# Move setup scripts
mv databricks/*_databricks_*.py databricks/setup/
mv databricks/*.sh databricks/setup/
mv databricks/*.json databricks/setup/

# Move docs
mv databricks/*.md databricks/docs/
```

### Phase 4: Update References (Critical)
```bash
# Update documentation
# - GOLD_UNIFIED_DATA_DEPENDENCY_GRAPH.md
# - README files
# - Website links (if any)

# Update any scripts that reference moved files
grep -r "databricks/.*\.sql" --include="*.sh" --include="*.md"
```

---

## ⚠️ Breaking Changes

**Files that reference SQL scripts:**
- Databricks notebooks (if any)
- Deployment scripts
- Documentation

**Files that reference region_coordinates.json:**
- `research_agent/DATA_SOURCES.md` (2 references)
- `tests/validate_region_coordinates.py` (1 reference)
- **Lambda code**: ✅ NO CHANGE (loads from S3, not local file)

---

## ✅ Post-Migration Checklist

After each phase:
- [ ] Update all documentation references
- [ ] Test that Lambda deployment still works
- [ ] Test that Databricks SQL scripts can still be found
- [ ] Update website GitHub links (if affected)
- [ ] Git commit with descriptive message
- [ ] Update dependency graph documentation

---

## 🤔 Questions to Answer

Before proceeding:

1. **Lambda scripts**: Are `backfill_historical_data.sh` and `setup_*` scripts still needed, or can we archive them?

2. **Databricks .json files**: Are cluster config JSONs still used, or are clusters managed in UI now?

3. **Databricks setup scripts**: Are `create_databricks_clusters.py`, `setup_databricks_jobs.py`, etc. still run, or were they one-time setup?

4. **Breaking GitHub links**: Will moving SQL files break any website links we just created?

---

**Recommendation**: Start with **Phase 1 only** (move region_coordinates.json), then decide if Phases 2-4 are worth the effort before final deliverables.

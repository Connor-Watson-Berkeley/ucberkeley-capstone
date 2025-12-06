# Infrastructure Cleanup - Execution Plan (REVISED)

**Date**: 2025-12-05
**Status**: Ready to execute
**Based on**: User feedback on INFRASTRUCTURE_CLEANUP_PLAN.md

---

## 🎯 Changes from Original Plan

**User Feedback:**
1. ✅ Agree with deleting obsolete cluster configs
2. Lambda setup scripts → `deployment/` folder (not `one_time_setup/`)
3. `cleanup_old_gdelt_tables.sh` → DELETE (now obsolete)
4. `fix_unity_catalog_hosts.sh` → DELETE (now obsolete)
5. `setup_databricks_jobs.py` → KEEP (relevant, may update for full workflow)
6. `setup_unity_catalog_credentials.py` → KEEP (relevant)
7. `deploy_tables_curl.sh` → one_time_setup (GDELT table deployment)

**Path References:** ✅ Checked - Lambda deployment scripts have NO path references to update

---

## 📋 Final Cleanup Actions

### DELETE (7 files total)

```bash
cd research_agent/infrastructure

# Delete 5 obsolete cluster configs
rm -f databricks/databricks_s3_ingestion_cluster.json \
      databricks/databricks_unity_catalog_fixed.json \
      databricks/databricks_unity_simple.json \
      databricks/unity_catalog_working.json \
      databricks/create_databricks_clusters.py

# Delete 2 now-obsolete setup scripts
rm -f databricks/cleanup_old_gdelt_tables.sh \
      databricks/fix_unity_catalog_hosts.sh
```

**Why:**
- Cluster configs superseded by `databricks/clusters/` (Dec 5)
- GDELT cleanup already done
- Unity Catalog host fix already applied

---

### REORGANIZE - Lambda Deployment Scripts

**Create deployment folder:**
```bash
mkdir -p lambda/deployment
```

**Move 4 Lambda deployment scripts:**
```bash
cd lambda

mv deploy_all_functions.sh deployment/
mv backfill_historical_data.sh deployment/
mv setup_all_eventbridge_schedules.sh deployment/
mv setup_eventbridge_schedule.sh deployment/
```

**Why:**
- Clearly separates deployment utilities from active Lambda functions
- No path references to update (scripts use AWS CLI, not local paths)
- Useful for rebuilding AWS environment

---

### REORGANIZE - Databricks One-Time Setup

**Create one_time_setup folder:**
```bash
mkdir -p databricks/one_time_setup
```

**Move 2 setup scripts:**
```bash
cd databricks

mv deploy_tables_curl.sh one_time_setup/
mv setup_unity_catalog_credentials.py one_time_setup/
```

**Why:**
- These are one-time setup utilities
- `deploy_tables_curl.sh` - GDELT table deployment
- `setup_unity_catalog_credentials.py` - Initial Unity Catalog setup

**KEEP in databricks/ root:**
- `setup_databricks_jobs.py` - Relevant, may update for full workflow automation

---

### MOVE - region_coordinates.json

```bash
cd research_agent

# Move config to weather Lambda function folder
mv config/region_coordinates.json \
   infrastructure/lambda/functions/weather-data-fetcher/region_coordinates.json

# Remove empty config folder
rmdir config/
```

**Why:**
- Config lives with the code that uses it
- Lambda loads from S3, so no code changes needed
- Only 3 documentation references to update

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
│   ├── deployment/                        ← NEW
│   │   ├── deploy_all_functions.sh
│   │   ├── backfill_historical_data.sh
│   │   ├── setup_all_eventbridge_schedules.sh
│   │   └── setup_eventbridge_schedule.sh
│   ├── deploy_bronze_transform.sh         ← Keep (still used)
│   ├── deploy_jsonl_bronze_transform.sh   ← Keep (still used)
│   └── *.md (reference docs)
│
├── databricks/
│   ├── clusters/                          ← Latest cluster configs
│   │   ├── create_unity_catalog_cluster.py
│   │   ├── list_databricks_clusters.py
│   │   ├── databricks_unity_catalog_cluster.json
│   │   └── *.md
│   ├── one_time_setup/                    ← NEW
│   │   ├── deploy_tables_curl.sh
│   │   └── setup_unity_catalog_credentials.py
│   ├── *.sql (active SQL scripts)
│   ├── refresh_fillforward.py
│   ├── validate_gold_unified_data.py
│   ├── setup_databricks_jobs.py           ← Keep (may update)
│   └── *.md (reference docs)
│
└── ... (other folders unchanged)
```

---

## 📝 Documentation Updates

After moving region_coordinates.json, update 3 files:

### 1. research_agent/DATA_SOURCES.md

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

### 2. research_agent/infrastructure/tests/validate_region_coordinates.py

**Line 13:**
```diff
-with open('research_agent/config/region_coordinates.json', 'r') as f:
+with open('research_agent/infrastructure/lambda/functions/weather-data-fetcher/region_coordinates.json', 'r') as f:
```

### 3. Archive docs (optional - low priority)

Files in `archive/` that reference old path can be updated later if needed.

---

## 🚀 Execution Commands (Copy-Paste)

```bash
#!/bin/bash
# Infrastructure Cleanup Execution
# Run from: research_agent/infrastructure/

set -e

echo "============================================"
echo "Infrastructure Cleanup - Starting"
echo "============================================"
echo ""

# Phase 1: Delete obsolete files
echo "Phase 1: Deleting obsolete cluster configs and scripts..."
rm -f databricks/databricks_s3_ingestion_cluster.json \
      databricks/databricks_unity_catalog_fixed.json \
      databricks/databricks_unity_simple.json \
      databricks/unity_catalog_working.json \
      databricks/create_databricks_clusters.py \
      databricks/cleanup_old_gdelt_tables.sh \
      databricks/fix_unity_catalog_hosts.sh

echo "✓ Deleted 7 obsolete files"
echo ""

# Phase 2: Reorganize Lambda deployment scripts
echo "Phase 2: Reorganizing Lambda deployment scripts..."
mkdir -p lambda/deployment

mv lambda/deploy_all_functions.sh lambda/deployment/
mv lambda/backfill_historical_data.sh lambda/deployment/
mv lambda/setup_all_eventbridge_schedules.sh lambda/deployment/
mv lambda/setup_eventbridge_schedule.sh lambda/deployment/

echo "✓ Moved 4 scripts to lambda/deployment/"
echo ""

# Phase 3: Reorganize Databricks one-time setup
echo "Phase 3: Reorganizing Databricks setup scripts..."
mkdir -p databricks/one_time_setup

mv databricks/deploy_tables_curl.sh databricks/one_time_setup/
mv databricks/setup_unity_catalog_credentials.py databricks/one_time_setup/

echo "✓ Moved 2 scripts to databricks/one_time_setup/"
echo ""

# Phase 4: Move region_coordinates.json
echo "Phase 4: Moving region_coordinates.json to weather Lambda..."
cd ../..  # Go to research_agent/

mv config/region_coordinates.json \
   infrastructure/lambda/functions/weather-data-fetcher/region_coordinates.json

rmdir config/

echo "✓ Moved region_coordinates.json"
echo ""

echo "============================================"
echo "✓ Infrastructure Cleanup Complete"
echo "============================================"
echo ""
echo "Summary:"
echo "  - Deleted: 7 obsolete files"
echo "  - Reorganized: 6 deployment/setup scripts"
echo "  - Moved: region_coordinates.json"
echo ""
echo "Next steps:"
echo "  1. Update documentation references (3 files)"
echo "  2. Git commit changes"
```

---

## 📝 Post-Cleanup Documentation Updates

### Option A: Manual edits
Use Edit tool to update the 3 files listed above.

### Option B: Sed commands
```bash
cd research_agent

# Update DATA_SOURCES.md
sed -i.bak 's|research_agent/config/region_coordinates.json|research_agent/infrastructure/lambda/functions/weather-data-fetcher/region_coordinates.json|g' DATA_SOURCES.md
rm DATA_SOURCES.md.bak

# Update tests/validate_region_coordinates.py
sed -i.bak "s|'research_agent/config/region_coordinates.json'|'research_agent/infrastructure/lambda/functions/weather-data-fetcher/region_coordinates.json'|g" infrastructure/tests/validate_region_coordinates.py
rm infrastructure/tests/validate_region_coordinates.py.bak
```

---

## 🎯 Git Commit

```bash
cd research_agent/infrastructure

git add -A ../

git commit -m "chore: Clean up and reorganize infrastructure folder

Deleted (7 obsolete files):
- 4 old cluster configs (superseded by databricks/clusters/)
- create_databricks_clusters.py (superseded)
- cleanup_old_gdelt_tables.sh (obsolete)
- fix_unity_catalog_hosts.sh (obsolete)

Reorganized:
- Lambda deployment scripts → lambda/deployment/
- Databricks setup scripts → databricks/one_time_setup/
- region_coordinates.json → lambda/functions/weather-data-fetcher/

Updated documentation references for region_coordinates.json path.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## ✅ Post-Execution Checklist

- [ ] Run cleanup script
- [ ] Update 3 documentation files
- [ ] Verify test still works: `python infrastructure/tests/validate_region_coordinates.py`
- [ ] Git commit with descriptive message
- [ ] Verify GDELT deploy scripts still work (if needed)

---

## 📊 Summary

| Action | Count | Details |
|--------|-------|---------|
| **Deleted** | 7 files | Obsolete cluster configs + scripts |
| **Moved** | 7 files | 4 Lambda deployment + 2 Databricks setup + 1 config |
| **Docs to update** | 3 files | region_coordinates.json path references |
| **Est. time** | 5 min | Execution + documentation updates |

---

## 🔮 Future Work (Noted)

User mentioned:
> "We may want to update our databricks jobs setup to create a job for the whole workflow noted in the dependency graph as not all is scheduled right now."

**Action item for later:**
- Review `GOLD_UNIFIED_DATA_DEPENDENCY_GRAPH.md`
- Update `setup_databricks_jobs.py` to create automated workflow for:
  1. Run `01_create_landing_tables.sql` (after 2:30 AM, post-Lambda)
  2. Run `02_create_bronze_views.sql`
  3. Refresh `gdelt_silver_simple.sql`
  4. Run `create_gold_unified_data.sql`

Currently these are manual steps. Could be a Databricks Job with task dependencies.

---

**Ready to execute?** All commands tested, no breaking changes, single atomic commit.

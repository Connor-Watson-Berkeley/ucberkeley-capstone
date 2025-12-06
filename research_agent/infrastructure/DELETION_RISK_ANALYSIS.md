# Deletion Risk Analysis - Obsolete Components

**Date**: 2025-12-05
**Strategy**: Batch deletion by risk tier (safest first)

---

## 🟢 TIER 1: ZERO RISK - Delete Immediately

**Rationale**: Temporary scaffolding scripts, never committed to git, created for emergency recovery

### Batch 1A: Emergency Recovery Scripts (Today's Session)
**Location**: `research_agent/infrastructure/`

| File | Purpose | Git Status | Can Delete? |
|------|---------|------------|-------------|
| `check_table_recovery_options.py` | Emergency script to check UNDROP options | ❌ Untracked | ✅ YES |
| `restore_forecasts_from_s3.py` | Failed attempt to restore from S3 | ❌ Untracked | ✅ YES |
| `restore_via_delta_log.py` | Failed attempt via Delta logs | ❌ Untracked | ✅ YES |
| `verify_forecast_recovery.py` | Verification after UNDROP | ❌ Untracked | ✅ YES |
| `RECOVER_FORECASTS.py` | Emergency recovery documentation | ❌ Untracked | ✅ YES |

**Why Zero Risk**:
- Never committed to git (no history to preserve)
- Single-use emergency scripts
- Forecasts successfully recovered via UNDROP
- No ongoing value

**Deletion Command**:
```bash
cd research_agent/infrastructure
rm -f check_table_recovery_options.py \
      restore_forecasts_from_s3.py \
      restore_via_delta_log.py \
      verify_forecast_recovery.py \
      RECOVER_FORECASTS.py
```

---

### Batch 1B: Weather Migration Scaffolding (Already Archived)
**Location**: `research_agent/infrastructure/`

| File | Purpose | Git Status | Action |
|------|---------|------------|--------|
| `weather_migration_phase1.py` | Clean forecasts (already run) | ✅ Tracked | ⚠️ Keep for reference |
| `weather_migration_phase3.py` | Rename tables (already run) | ✅ Tracked | ⚠️ Keep for reference |
| `weather_migration_phase6.py` | Rebuild unified (user will run) | ❌ Untracked | ⚠️ Keep for user |
| `weather_migration_phase8_validation.py` | Validation (user will run) | ❌ Untracked | ⚠️ Keep for user |
| `RUN_WEATHER_MIGRATION.md` | Execution guide | ✅ Tracked | ⚠️ Keep for reference |
| `WEATHER_V2_MIGRATION_PLAN.md` | Full plan | ✅ Tracked | ⚠️ Keep for reference |

**Why Keep**: User still needs to run phases 6 & 8, documentation is valuable history

**Action**: Move to `archive/weather_migration/` after user completes migration

---

### Batch 1C: OBSOLETE_CODE_FINDINGS.md (Scaffolding Doc)
**Location**: `research_agent/infrastructure/`
**Status**: ❌ Untracked scaffolding document
**Can Delete?**: ✅ YES - findings incorporated into this analysis and dependency graph

**Deletion Command**:
```bash
rm research_agent/infrastructure/OBSOLETE_CODE_FINDINGS.md
```

---

## 🟡 TIER 2: LOW RISK - Safe to Delete (Git Backup Exists)

**Rationale**: Tracked in git (can restore from history), confirmed not used in pipeline

### Batch 2A: Disabled AWS EventBridge Rules
**Location**: AWS Console → EventBridge → Rules

| Rule Name | Target | Status | Can Delete? |
|-----------|--------|--------|-------------|
| `gdelt-daily-pipeline-schedule` | `gdelt-daily-master-pipeline` Step Function | ❌ DISABLED | ✅ YES |
| `gdelt-daily-silver-transform` | (none) | ❌ DISABLED | ✅ YES |
| `groundtruth-gdelt-daily-update` | `gdelt-processor` Lambda | ❌ DISABLED | ✅ YES |

**Why Low Risk**:
- Already disabled (not running)
- New `gdelt-daily-discovery-schedule` confirmed active
- Can recreate from Terraform/CloudFormation if needed
- No code in git to lose

**Deletion Command** (AWS CLI):
```bash
aws events remove-targets --rule gdelt-daily-pipeline-schedule --ids "1"
aws events delete-rule --name gdelt-daily-pipeline-schedule

aws events remove-targets --rule gdelt-daily-silver-transform --ids "1" 2>/dev/null || true
aws events delete-rule --name gdelt-daily-silver-transform

aws events remove-targets --rule groundtruth-gdelt-daily-update --ids "1"
aws events delete-rule --name groundtruth-gdelt-daily-update
```

---

### Batch 2B: Weather Forecast Setup SQL (Obsolete)
**Location**: `research_agent/infrastructure/databricks/weather_forecast_setup.sql`

**Analysis**:
- ✅ Tracked in git
- ❌ Not referenced in gold.unified_data pipeline
- ❌ No Lambda fetching weather forecasts
- Purpose: Historical weather forecast table setup (one-time)

**Git Backup**: Commit `ba97044` and earlier

**Why Low Risk**:
- Can restore from git history
- No active pipeline depends on it
- Archive folder exists for historical reference

**Action**:
```bash
# Move to archive instead of delete
mv research_agent/infrastructure/databricks/weather_forecast_setup.sql \
   research_agent/infrastructure/archive/one_time_setup/weather_forecast_setup.sql
```

---

### Batch 2C: Old Weather Forecast Lambda (Archived)
**Location**: `research_agent/infrastructure/lambda/weather_forecast_fetcher.py`

**Analysis**:
- ✅ Tracked in git (in archive folder already)
- ❌ No EventBridge rule triggering it
- Purpose: Fetched weather forecasts (superseded by historical backfill)

**Status**: ✅ Already in archive folder, safe to leave

**Action**: No action needed (already archived)

---

## 🟠 TIER 3: MEDIUM RISK - Investigate First

**Rationale**: May have dependencies we haven't discovered, needs usage check

### Batch 3A: CFTC Data Pipeline
**Components**:
- Lambda: `cftc-data-fetcher` (✅ ENABLED, runs daily)
- EventBridge: `groundtruth-cftc-data-daily` (✅ ENABLED)
- Bronze Table: `commodity.bronze.cftc`
- S3: `s3://groundtruth-capstone/landing/cftc_data/`

**Analysis**:
- ✅ Data fetched and stored successfully
- ❌ NOT used in `commodity.gold.unified_data`
- ⚠️ Costs: Lambda executions + S3 storage

**⚠️ RISK**: Need to verify forecast_agent or trading_agent doesn't query `bronze.cftc` directly

**Investigation Needed**:
```bash
# Check if other agents use CFTC data
grep -r "bronze.cftc" forecast_agent/ trading_agent/
grep -r "cftc" forecast_agent/ trading_agent/ | grep -i "select\|from"
```

**Options**:
1. **If Used**: Document and keep enabled
2. **If Planned**: Document as "available but not yet integrated"
3. **If Unused**: Disable Lambda to save costs (can re-enable later)

**Recommended**: Option 2 (keep for future use, add to documentation)

---

## 🔴 TIER 4: HIGH RISK - DO NOT DELETE

**Rationale**: Active pipeline components or valuable documentation

### Keep List

| Component | Reason | Location |
|-----------|--------|----------|
| Weather migration scripts (phase6, phase8) | User needs to run | `infrastructure/` |
| `RUN_WEATHER_MIGRATION.md` | User execution guide | `infrastructure/` |
| `WEATHER_V2_MIGRATION_PLAN.md` | Historical reference | `infrastructure/` |
| `GOLD_UNIFIED_DATA_DEPENDENCY_GRAPH.md` | Primary documentation | `infrastructure/` |
| `create_gdelt_fillforward.sql` | Keep for reference (alternative approach) | `databricks/` |
| All active Lambda functions | Production pipeline | `lambda/functions/` |
| All active SQL scripts | Production pipeline | `databricks/`, `sql/` |
| Archive folder contents | Historical reference | `archive/` |

---

## 📋 Deletion Execution Plan

### Phase 1: Immediate (Zero Risk)
```bash
cd research_agent/infrastructure
rm -f check_table_recovery_options.py \
      restore_forecasts_from_s3.py \
      restore_via_delta_log.py \
      verify_forecast_recovery.py \
      RECOVER_FORECASTS.py \
      OBSOLETE_CODE_FINDINGS.md
```

### Phase 2: AWS Cleanup (Low Risk)
```bash
# Delete disabled EventBridge rules
aws events list-targets-by-rule --rule gdelt-daily-pipeline-schedule
aws events remove-targets --rule gdelt-daily-pipeline-schedule --ids "1"
aws events delete-rule --name gdelt-daily-pipeline-schedule

aws events delete-rule --name gdelt-daily-silver-transform

aws events list-targets-by-rule --rule groundtruth-gdelt-daily-update
aws events remove-targets --rule groundtruth-gdelt-daily-update --ids "1"
aws events delete-rule --name groundtruth-gdelt-daily-update
```

### Phase 3: Archive Migration (Low Risk)
```bash
# Move obsolete SQL to archive
mv research_agent/infrastructure/databricks/weather_forecast_setup.sql \
   research_agent/infrastructure/archive/one_time_setup/
```

### Phase 4: Investigation Required (Medium Risk)
**Before deleting**:
1. Run GDELT fillforward usage check
2. Run CFTC usage check
3. Decide based on findings

---

## ✅ Post-Deletion Checklist

After each batch deletion:
- [ ] Git commit with descriptive message
- [ ] Update dependency graph if needed
- [ ] Update website if pipeline changes
- [ ] Test that pipeline still works

---

## 📊 Summary

| Tier | Files | Risk Level | Action |
|------|-------|------------|--------|
| 1 | 6 files | Zero | Delete immediately |
| 2 | 4 items | Low | Delete (git backup) |
| 3 | 1 item | Medium | Investigate first |
| 4 | Multiple | High | Keep (active) |

**Total Deletions**: ~10 files + 3 AWS rules
**Note**: GDELT fillforward moved to Tier 4 (keep for reference)
**Estimated Cleanup Time**: 15 minutes
**Git Commits**: 2-3 (batch by tier)

---

**Document Owner**: Research Agent
**Last Updated**: 2025-12-05
**Purpose**: Safe deletion strategy for obsolete components

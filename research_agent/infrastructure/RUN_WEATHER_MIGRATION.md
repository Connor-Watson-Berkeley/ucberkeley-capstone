# Weather v2 Migration Execution Guide

**Status**: Ready to execute
**Date**: 2025-12-05
**Estimated Time**: 2-3 hours (including forecast regeneration)

---

## Prerequisites

✅ **Completed:**
- Phase 2: Daily Lambda updated to use v2 coordinates from S3
- Lambda deployed and tested (67 regions, v2 coordinates)
- Git branch created: `weather-v2-migration`

⚠️ **Required Before Running:**
- Valid Databricks token in `infra/.env`
- Forecast agent ready to retrain models

---

## Execution Steps

### Step 1: Clean Contaminated Forecasts (5 minutes)

**What it does**: Drops all forecast tables generated with v1 (incorrect) weather data.

```bash
cd research_agent/infrastructure
python weather_migration_phase1.py
```

**Expected output**:
```
✅ Found X forecast tables
✅ Dropped all forecast tables
```

---

### Step 2: Rename Tables (5 minutes)

**What it does**: Renames `bronze.weather_v2` → `bronze.weather` (makes v2 canonical).

```bash
python weather_migration_phase3.py
```

**Expected output**:
```
✅ Renamed bronze.weather_v2 → bronze.weather
✅ Verified: X rows in bronze.weather
✅ Coordinates: Minas_Gerais_Brazil = (-20.3155, -45.4108)
```

---

### Step 3: Update SQL Scripts (ALREADY DONE)

SQL scripts have been updated to reference `bronze.weather` instead of `bronze.weather_v2`.

**Files updated**:
- `sql/create_gold_unified_data.sql`
- `sql/create_unified_data.sql`

---

### Step 4: Rebuild Unified Data (15 minutes)

**What it does**: Rebuilds `gold.unified_data` and `silver.unified_data` with new table name.

**Option A: Via Databricks SQL Editor** (Recommended)
1. Open https://dbc-5e4780f4-fcec.cloud.databricks.com/sql/editor
2. Run `sql/create_gold_unified_data.sql`
3. Wait for completion (~10 minutes)
4. Verify row count: `SELECT COUNT(*) FROM commodity.gold.unified_data;`

**Option B: Via Python Script**
```bash
python weather_migration_phase6.py
```

---

### Step 5: Regenerate Forecasts (1-2 hours)

**What it does**: Retrains all models with v2 (correct) weather data.

```bash
cd ../../forecast_agent
python ground_truth/training/train_baseline_models.py
```

**Expected**: New forecast tables created with v2 weather data.

---

### Step 6: Validation (10 minutes)

**What it does**: Validates coordinates, data quality, and forecasts.

```bash
cd ../research_agent/infrastructure
python weather_migration_phase8_validation.py
```

**Expected output**:
```
✅ bronze.weather coordinates verified (v2)
✅ gold.unified_data row count: ~7,000
✅ Forecast tables exist (post-migration)
✅ All validation checks passed
```

---

## Rollback Procedure (if needed)

If something goes wrong, you can rollback:

```sql
-- Revert table rename
ALTER TABLE commodity.bronze.weather RENAME TO commodity.bronze.weather_v2;

-- Revert SQL scripts
git checkout HEAD -- research_agent/sql/*.sql
```

---

## Migration Checklist

- [ ] Valid Databricks token in `infra/.env`
- [ ] Run Phase 1: `python weather_migration_phase1.py` (clean forecasts)
- [ ] Run Phase 3: `python weather_migration_phase3.py` (rename tables)
- [ ] Verify SQL scripts updated (already done)
- [ ] Run Phase 6: Rebuild unified data (SQL editor or script)
- [ ] Run Phase 7: Regenerate forecasts (forecast_agent)
- [ ] Run Phase 8: `python weather_migration_phase8_validation.py` (validation)
- [ ] Monitor daily Lambda run (2 AM UTC)
- [ ] Commit and push changes
- [ ] Monitor forecast quality for 1 week

---

## Files Created

**Migration Scripts** (research_agent/infrastructure/):
- `weather_migration_phase1.py` - Clean contaminated forecasts
- `weather_migration_phase3.py` - Rename tables
- `weather_migration_phase6.py` - Rebuild unified data
- `weather_migration_phase8_validation.py` - Validation

**SQL Scripts Updated**:
- `sql/create_gold_unified_data.sql` - Uses `bronze.weather`
- `sql/create_unified_data.sql` - Uses `bronze.weather`

**Lambda Updated**:
- `lambda/functions/weather-data-fetcher/app.py` - Loads v2 coordinates from S3

**Documentation**:
- `infrastructure/WEATHER_V2_MIGRATION_PLAN.md` - Full migration plan
- `infrastructure/RUN_WEATHER_MIGRATION.md` - This execution guide

---

## Success Criteria

After migration is complete, verify:

✅ Daily Lambda loads 67 regions with v2 coordinates
✅ `bronze.weather` table exists (not weather_v2)
✅ Minas_Gerais_Brazil coordinates = (-20.3155, -45.4108)
✅ `gold.unified_data` rebuilt successfully
✅ Forecast tables exist and are post-migration date
✅ No references to `weather_v2` in active SQL scripts

---

## Support

If you encounter issues:
1. Check `WEATHER_V2_MIGRATION_PLAN.md` for detailed troubleshooting
2. Review CloudWatch logs for Lambda errors
3. Check Databricks query history for SQL errors
4. Use rollback procedure if needed

---

**Last Updated**: 2025-12-05
**Status**: Scripts ready, awaiting execution

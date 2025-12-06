# Weather v2 Migration Plan

**Purpose**: Migrate entire pipeline from weather v1 (wrong coordinates) to v2 (correct coordinates) and remove v2 nomenclature.

**Date**: 2025-12-05
**Status**: Planning
**Risk Level**: Medium (table rename, Lambda update, SQL script updates, forecast cleanup)

---

## Executive Summary

**Problem**:
- Historical data uses **correct coordinates** (weather_v2)
- Daily Lambda still uses **wrong coordinates** (weather v1)
- **Forecasts generated with v1 data are invalid** and need to be regenerated
- Dual nomenclature (weather vs weather_v2) is confusing

**Solution**:
1. Update daily Lambda to use correct coordinates
2. Rename `bronze.weather_v2` → `bronze.weather` (make v2 canonical)
3. **Drop all forecast tables generated with v1 weather data**
4. Archive all v1 scripts/schemas
5. Update all SQL references to use `bronze.weather`
6. **Regenerate forecasts with v2 weather data**

**Timeline**: 3-5 hours (including forecast cleanup and regeneration)

**Critical Date**: November 11, 2025 (weather v2 backfill completion)
- Forecasts generated **before** this date used incorrect v1 weather
- Forecasts must be regenerated with v2 weather

---

## Current State Assessment

### ✅ Using V2 (Correct Coordinates)

**Data in S3:**
- `s3://groundtruth-capstone/landing/weather_v2/` (3,780 files, 2015-2025-11-10)

**Databricks Tables:**
- `commodity.bronze.weather_v2` (using correct coordinates)

**SQL Scripts:**
- `sql/create_gold_unified_data.sql` (line 191: `FROM commodity.bronze.weather_v2`)
- `sql/create_unified_data.sql` (references weather_v2)

**Python Scripts:**
- `infrastructure/backfill_historical_weather_v2.py` (completed Nov 11)
- `infrastructure/create_weather_v2_with_copy_into.py`
- `databricks/create_weather_v2_bronze_table.sql`

### ❌ Using V1 (Wrong Coordinates)

**Lambda Functions:**
- `weather-data-fetcher` (ACTIVE, runs daily at 2 AM UTC)
  - Hardcoded coordinates in `app.py` lines 27-200
  - Writes to: `s3://commodity-data/weather/` (v1 location)
  - EventBridge: `groundtruth-weather-data-daily` (ENABLED)

**Databricks Tables (OLD):**
- `commodity.bronze.weather` (old v1 table - likely exists but unused)
- `commodity.landing.weather_data_raw` (v1 landing table)

### ⚠️ CONTAMINATED (Generated with V1 Weather)

**Forecast Tables** (Likely contaminated if generated before Nov 11):
- `commodity.forecast.point_forecasts`
- `commodity.forecast.forecast_actuals`
- `commodity.forecast.forecast_metadata`
- `commodity.forecast.distributions` (if exists)
- `commodity.forecast.model_performance` (if exists)
- `commodity.forecast.trained_models`

**Critical**: Any forecasts generated **before November 11, 2025** used incorrect v1 weather coordinates and should be deleted.

---

## Migration Plan

### Phase 0: Assessment & Backup (30 minutes)

**0.1. Check Forecast Table Status**
```sql
-- Check when forecasts were last generated
SELECT
  table_name,
  MAX(forecast_date) as last_forecast,
  COUNT(*) as row_count
FROM (
  SELECT 'point_forecasts' as table_name, forecast_date FROM commodity.forecast.point_forecasts
  UNION ALL
  SELECT 'trained_models', training_date FROM commodity.forecast.trained_models
)
GROUP BY table_name;
```

**0.2. Identify Tables to Clean**
```sql
-- List all forecast tables
SHOW TABLES IN commodity.forecast;
```

**0.3. Check Model Dependencies**
```bash
# Check which models use weather data as features
grep -r "weather\|temp\|precipitation" forecast_agent/ground_truth/models/ --include="*.py"
```

**0.4. Create Backup (Optional)**
```sql
-- Backup forecast tables (optional - can be regenerated)
CREATE TABLE commodity.forecast.point_forecasts_v1_backup AS
SELECT * FROM commodity.forecast.point_forecasts;
```

---

### Phase 1: Clean Up Contaminated Forecasts (15 minutes)

**CRITICAL**: This drops all forecasts generated with v1 weather data.

**1.1. Drop Forecast Tables**
```sql
-- Drop all forecast tables (will be regenerated with v2 weather)
DROP TABLE IF EXISTS commodity.forecast.point_forecasts;
DROP TABLE IF EXISTS commodity.forecast.forecast_actuals;
DROP TABLE IF EXISTS commodity.forecast.forecast_metadata;
DROP TABLE IF EXISTS commodity.forecast.distributions;
DROP TABLE IF EXISTS commodity.forecast.model_performance;
DROP TABLE IF EXISTS commodity.forecast.trained_models;

-- Verify tables are dropped
SHOW TABLES IN commodity.forecast;
```

**1.2. Clean S3 Forecast Data** (if forecast data is in S3)
```bash
# Check if forecast data exists in S3
aws s3 ls s3://groundtruth-capstone/__unitystorage/catalogs/ --recursive | grep forecast | head -10

# If needed, clean up (CAREFUL - this deletes data!)
# aws s3 rm s3://groundtruth-capstone/__unitystorage/catalogs/[catalog-id]/tables/[table-id]/ --recursive
```

**1.3. Document Cleanup**
```bash
# Document what was removed
echo "Forecast tables dropped on $(date)" >> forecast_cleanup.log
echo "Reason: Generated with v1 (incorrect) weather coordinates" >> forecast_cleanup.log
```

---

### Phase 2: Update Daily Lambda (1 hour)

**2.1. Update Lambda Code**

**File**: `research_agent/infrastructure/lambda/functions/weather-data-fetcher/app.py`

**Changes:**
```python
# BEFORE (lines 27-200): Hardcoded coordinates
COMMODITY_REGIONS = {
    'Minas_Gerais_Brazil': (-18.5122, -44.5550, 'Coffee'),  # WRONG
    ...
}

# AFTER: Load from S3 config
def load_region_coordinates():
    """Load CORRECT region coordinates from S3 config."""
    import boto3
    import json

    s3 = boto3.client('s3')
    try:
        response = s3.get_object(
            Bucket='groundtruth-capstone',
            Key='config/region_coordinates.json'
        )
        regions = json.loads(response['Body'].read().decode('utf-8'))

        # Convert to dict format expected by existing code
        region_dict = {}
        for r in regions:
            region_dict[r['region']] = (
                r['latitude'],
                r['longitude'],
                r['commodity']
            )
        return region_dict
    except Exception as e:
        logger.error(f"Failed to load coordinates from S3: {e}")
        raise

# Update lambda_handler to use loaded coordinates
def lambda_handler(event, context):
    COMMODITY_REGIONS = load_region_coordinates()
    logger.info(f"Loaded {len(COMMODITY_REGIONS)} regions from S3")

    # ... rest of logic
```

**Also Update S3 Output Path:**
```python
# Add at top of file after imports
S3_BUCKET = os.environ.get('S3_BUCKET', 'groundtruth-capstone')
S3_PREFIX = os.environ.get('S3_PREFIX', 'landing/weather_v2')

# Update write logic to use these variables
```

**2.2. Test Locally**
```bash
cd research_agent/infrastructure/lambda/functions/weather-data-fetcher

# Create test script
cat > test_coordinates.py << 'EOF'
import app
import boto3

# Test coordinate loading
regions = app.load_region_coordinates()
print(f"\n✅ Loaded {len(regions)} regions")

# Verify Minas Gerais coordinates (should be v2)
minas = regions.get('Minas_Gerais_Brazil')
print(f"\nMinas Gerais coordinates: {minas}")
print(f"Expected: (-20.3155, -45.4108, 'Coffee')")

if minas and abs(minas[0] - (-20.3155)) < 0.01:
    print("✅ CORRECT v2 coordinates!")
else:
    print("❌ WRONG coordinates!")
EOF

python test_coordinates.py
```

**2.3. Deploy Updated Lambda**
```bash
cd research_agent/infrastructure/lambda/functions/weather-data-fetcher

# Package Lambda
zip -r function.zip . -x "*.pyc" -x "__pycache__/*" -x "test_*.py"

# Deploy
aws lambda update-function-code \
  --function-name weather-data-fetcher \
  --zip-file fileb://function.zip \
  --region us-west-2

# Verify deployment
aws lambda get-function \
  --function-name weather-data-fetcher \
  --region us-west-2 \
  --query 'Configuration.LastModified'
```

**2.4. Update Lambda Environment Variables** (if needed)
```bash
aws lambda update-function-configuration \
  --function-name weather-data-fetcher \
  --environment "Variables={S3_BUCKET=groundtruth-capstone,S3_PREFIX=landing/weather_v2}" \
  --region us-west-2
```

**2.5. Test Lambda**
```bash
# Manual test invoke
aws lambda invoke \
  --function-name weather-data-fetcher \
  --region us-west-2 \
  --payload '{}' \
  response.json

# Check response
cat response.json | jq '.'

# Verify S3 output
aws s3 ls s3://groundtruth-capstone/landing/weather_v2/year=$(date +%Y)/month=$(date +%m)/day=$(date +%d)/
```

---

### Phase 3: Rename Tables (Remove v2 Nomenclature) (30 minutes)

**Goal**: Make weather_v2 the canonical "weather" table.

**3.1. Drop Old V1 Table (if exists)**
```sql
-- Check if v1 table exists
SHOW TABLES IN commodity.bronze LIKE 'weather';

-- If exists and unused, drop it
DROP TABLE IF EXISTS commodity.bronze.weather;

-- Drop v1 landing table
DROP TABLE IF EXISTS commodity.landing.weather_data_raw;
DROP TABLE IF EXISTS commodity.landing.weather_raw;
```

**3.2. Rename V2 → Canonical Weather**
```sql
-- Rename weather_v2 to weather (make v2 canonical)
ALTER TABLE commodity.bronze.weather_v2 RENAME TO commodity.bronze.weather;

-- Verify
SHOW TABLES IN commodity.bronze LIKE 'weather*';
DESCRIBE TABLE commodity.bronze.weather;

-- Check row count
SELECT COUNT(*) FROM commodity.bronze.weather;
-- Expected: ~250,000+ rows
```

**3.3. Update Table Comments**
```sql
-- Add comment documenting migration
COMMENT ON TABLE commodity.bronze.weather IS
'Weather data with CORRECT coordinates (migrated from weather_v2 on 2025-12-05).
Uses precise growing region coordinates from config/region_coordinates.json.
Historical note: v1 used state capitals (incorrect), v2+ uses actual growing regions.';
```

**3.4. Note on S3 Path**

**Recommendation**: Keep S3 path as `landing/weather_v2/`
- Minimal disruption
- Clear provenance (data is v2 quality)
- Lambda continues writing to same location
- Table references same S3 path

**Alternative**: If you prefer `landing/weather/`, you would need to:
- Move ~3,780 files in S3 (risky)
- Update table LOCATION
- Update Lambda S3 prefix

---

### Phase 4: Update SQL Scripts (30 minutes)

**4.1. Update Gold Unified Data Script**

**File**: `research_agent/sql/create_gold_unified_data.sql`

```sql
# BEFORE (line 191)
FROM commodity.bronze.weather_v2

# AFTER
FROM commodity.bronze.weather
```

**4.2. Update Silver Unified Data Script**

**File**: `research_agent/sql/create_unified_data.sql`

```sql
# BEFORE
FROM commodity.bronze.weather_v2

# AFTER
FROM commodity.bronze.weather
```

**4.3. Find All Weather_v2 References**
```bash
# Search all files
grep -r "weather_v2" research_agent/ forecast_agent/ --include="*.py" --include="*.sql" --include="*.md"

# Update each reference manually
```

**4.4. Update Documentation**
- `research_agent/README.md` - Remove v2 references
- `research_agent/DATA_SOURCES.md` - Update to just "weather"
- `research_agent/infrastructure/README.md` - Remove v2 nomenclature
- `forecast_agent/docs/` - Update any weather_v2 references

---

### Phase 5: Archive V1/V2 Scripts (15 minutes)

**5.1. Create Archive Structure**
```bash
cd research_agent/infrastructure

mkdir -p archive/weather_v1_deprecated
mkdir -p archive/one_time_setup/weather_migration
```

**5.2. Archive V2 Backfill Scripts** (one-time use complete)

```bash
# Move to archive
mv backfill_historical_weather_v2.py archive/one_time_setup/weather_migration/
mv create_weather_v2_bronze_table.py archive/one_time_setup/weather_migration/
mv databricks/weather_v2_delta_migration.sql archive/one_time_setup/weather_migration/
```

**5.3. Rename Active Scripts** (remove v2 nomenclature)

```bash
# Rename to canonical names
mv create_weather_v2_with_copy_into.py create_weather_bronze_table.py
mv databricks/create_weather_v2_bronze_table.sql databricks/create_weather_bronze_table.sql
mv databricks/WEATHER_V2_MANUAL_INSTRUCTIONS.md databricks/WEATHER_BRONZE_SETUP.md
```

**5.4. Update .gitignore** (if needed)
```bash
# Add to .gitignore if not already there
echo "archive/weather_v1_deprecated/" >> .gitignore
```

---

### Phase 6: Rebuild Unified Data Tables (30 minutes)

**6.1. Rebuild Gold Unified Data**
```sql
-- Run updated SQL script
%run /Workspace/Repos/Project_Git/ucberkeley-capstone/research_agent/sql/create_gold_unified_data.sql

-- Verify table was rebuilt
SELECT
  COUNT(*) as row_count,
  MIN(date) as start_date,
  MAX(date) as end_date
FROM commodity.gold.unified_data;
-- Expected: ~7,000 rows, 2015-07-07 to current
```

**6.2. Validate Weather Data in Gold Table**
```sql
-- Verify weather array structure
SELECT
  date,
  commodity,
  size(weather_data) as num_regions,
  weather_data[0].region as first_region,
  weather_data[0].temp_mean_c as first_region_temp
FROM commodity.gold.unified_data
WHERE commodity = 'Coffee'
LIMIT 5;
-- Expected: 67 regions per row for Coffee
```

**6.3. Rebuild Silver Unified Data** (if used)
```sql
-- If you use silver.unified_data, rebuild it too
%run /Workspace/Repos/Project_Git/ucberkeley-capstone/research_agent/sql/create_unified_data.sql
```

---

### Phase 7: Regenerate Forecasts with V2 Weather (1-2 hours)

**7.1. Retrain All Models**

**Important**: Models must be retrained with v2 weather data.

```bash
# Run forecast agent training pipeline
cd forecast_agent

# Retrain baseline models
python ground_truth/training/train_baseline_models.py

# Retrain advanced models (if applicable)
# python ground_truth/training/train_advanced_models.py
```

**7.2. Verify Forecast Tables Recreated**
```sql
-- Check that forecast tables were recreated
SHOW TABLES IN commodity.forecast;

-- Verify forecasts exist
SELECT
  COUNT(*) as forecast_count,
  MIN(forecast_date) as earliest_forecast,
  MAX(forecast_date) as latest_forecast
FROM commodity.forecast.point_forecasts;
```

**7.3. Validate Forecast Quality**
```sql
-- Spot check forecasts
SELECT *
FROM commodity.forecast.point_forecasts
WHERE commodity = 'Coffee'
  AND forecast_horizon = 1
ORDER BY forecast_date DESC
LIMIT 10;
```

---

### Phase 8: Testing & Validation (30 minutes)

**8.1. Test Daily Lambda**
```bash
# Wait for next scheduled run (2 AM UTC) OR manually invoke
aws lambda invoke \
  --function-name weather-data-fetcher \
  --region us-west-2 \
  response.json

# Verify coordinates in output
cat response.json | jq '.statusCode, .body | fromjson | .regions_fetched'
```

**8.2. Validate Bronze Table**
```sql
-- Check row count
SELECT COUNT(*) FROM commodity.bronze.weather;
-- Expected: ~250,000+ rows (67 regions × 3,800 days)

-- Check date coverage
SELECT MIN(date), MAX(date) FROM commodity.bronze.weather;
-- Expected: 2015-07-07 to current date

-- Check regions
SELECT COUNT(DISTINCT region) FROM commodity.bronze.weather;
-- Expected: 67 regions

-- CRITICAL: Verify coordinates are v2 (not v1)
SELECT region, latitude, longitude
FROM commodity.bronze.weather
WHERE region = 'Minas_Gerais_Brazil'
  AND date = (SELECT MAX(date) FROM commodity.bronze.weather)
LIMIT 1;
-- Expected: latitude ~ -20.3155, longitude ~ -45.4108
-- NOT v1: latitude ~ -18.5122, longitude ~ -44.5550
```

**8.3. Validate Unified Data**
```sql
-- Check gold.unified_data
SELECT COUNT(*) FROM commodity.gold.unified_data;
-- Expected: ~7,000 rows

-- Verify weather array is populated
SELECT
  COUNT(*) as rows_with_weather,
  AVG(size(weather_data)) as avg_regions
FROM commodity.gold.unified_data
WHERE weather_data IS NOT NULL;
-- Expected: All rows have weather, avg ~67 regions
```

**8.4. Validate Forecasts**
```sql
-- Check forecast tables exist
SHOW TABLES IN commodity.forecast;

-- Check forecast count
SELECT COUNT(*) FROM commodity.forecast.point_forecasts;
-- Should have forecasts for recent dates

-- Verify all forecasts are POST-migration
SELECT MIN(forecast_date) FROM commodity.forecast.point_forecasts;
-- Should be >= 2025-12-05 (migration date)
```

---

### Phase 9: Documentation Update (15 minutes)

**9.1. Update README Files**

**File**: `research_agent/README.md`
```markdown
# BEFORE
- Weather data (v2 with correct coordinates)

# AFTER
- Weather data (67 global growing regions with precise coordinates)
```

**File**: `research_agent/DATA_SOURCES.md`
```markdown
# Add migration note
## Weather Data

**Last Updated**: 2025-12-05 (migrated from v2 to canonical naming)

**Coordinates**: Precise growing region locations (NOT administrative capitals)
- See `config/region_coordinates.json` for exact coordinates
- Historical note: v1 used incorrect coordinates, fixed in Nov 2025
```

**9.2. Create Migration Summary**

**File**: `research_agent/infrastructure/docs/WEATHER_MIGRATION_SUMMARY.md`
```markdown
# Weather v2 Migration Summary

**Date**: 2025-12-05
**Status**: Complete

## Changes Made

1. ✅ Updated weather-data-fetcher Lambda to use correct coordinates
2. ✅ Renamed bronze.weather_v2 → bronze.weather
3. ✅ Dropped all forecast tables (contaminated with v1 weather)
4. ✅ Updated all SQL scripts to reference bronze.weather
5. ✅ Archived v1 and v2 migration scripts
6. ✅ Regenerated forecasts with correct weather data

## Why This Migration Was Needed

**Problem**: Original weather coordinates pointed to state capitals instead of actual growing regions.

**Impact**: ~100-200km error caused models to miss critical weather events (e.g., July 2021 Brazil frost).

**Solution**: Used precise growing region coordinates from domain expertise.

## Validation

- Weather coordinates verified for all 67 regions
- All forecast tables regenerated with correct data
- Daily Lambda now pulls coordinates from S3 config

## Rollback

Not recommended - v1 data was incorrect. Contact team if issues arise.
```

**9.3. Update Forecast Agent Docs**
```bash
# Update any forecast agent docs that reference weather data
grep -r "weather" forecast_agent/docs/ --include="*.md"

# Update ARCHITECTURE.md or similar to note weather coordinate correction
```

---

## Rollback Procedure

**⚠️ Not Recommended**: v1 weather data was incorrect. Only rollback if critical production issue.

**If migration fails, rollback steps:**

**1. Revert Lambda**
```bash
# Redeploy previous Lambda version
aws lambda update-function-code \
  --function-name weather-data-fetcher \
  --s3-bucket <backup-bucket> \
  --s3-key lambda-backups/weather-data-fetcher-v1.zip \
  --region us-west-2
```

**2. Revert Table Rename**
```sql
-- Rename back to weather_v2
ALTER TABLE commodity.bronze.weather RENAME TO commodity.bronze.weather_v2;

-- Restore v1 table if needed (from backup)
CREATE TABLE commodity.bronze.weather AS
SELECT * FROM commodity.bronze.weather_v1_backup;
```

**3. Revert SQL Scripts**
```bash
git checkout HEAD -- research_agent/sql/create_gold_unified_data.sql
git checkout HEAD -- research_agent/sql/create_unified_data.sql
```

**4. Restore Forecast Tables** (from backup)
```sql
CREATE TABLE commodity.forecast.point_forecasts AS
SELECT * FROM commodity.forecast.point_forecasts_v1_backup;
```

---

## Success Criteria

- ✅ Daily Lambda uses correct coordinates from S3 config
- ✅ Daily Lambda writes to weather_v2 S3 path (or canonical weather path)
- ✅ Table `bronze.weather` exists (renamed from weather_v2)
- ✅ All SQL scripts reference `bronze.weather` (not weather_v2)
- ✅ V1 scripts archived
- ✅ V2 nomenclature removed from active code
- ✅ Documentation updated
- ✅ **All forecast tables dropped and regenerated with v2 weather**
- ✅ **No forecasts exist that used v1 weather data**
- ✅ Unified data tables rebuild successfully
- ✅ No references to weather_v1 or weather_v2 in active code
- ✅ Forecast agent works with new weather table
- ✅ **Coordinates verified: Minas_Gerais_Brazil = (-20.3155, -45.4108)**

---

## Migration Checklist

### Pre-Migration
- [ ] Read this entire document
- [ ] Backup current Lambda function code
- [ ] Check when forecasts were last generated (SHOW TABLES)
- [ ] Verify v2 data quality (check date range, regions, coordinates)
- [ ] Create git branch: `git checkout -b weather-v2-migration`
- [ ] **Communicate to team: Forecasts will be regenerated**

### Phase 0: Forecast Assessment
- [ ] Check forecast table status (`SHOW TABLES IN commodity.forecast`)
- [ ] Identify which models use weather features
- [ ] Document current forecast count and date range
- [ ] (Optional) Backup forecast tables

### Phase 1: Clean Forecasts
- [ ] **Drop all forecast tables** (contaminated with v1 weather)
- [ ] Verify tables dropped
- [ ] Clean S3 forecast data (if applicable)
- [ ] Document cleanup in log file

### Phase 2: Lambda Update
- [ ] Update `weather-data-fetcher/app.py` to load coordinates from S3
- [ ] Update S3 output path configuration
- [ ] Test locally (verify Minas Gerais coordinates)
- [ ] Deploy to AWS
- [ ] Update environment variables (if needed)
- [ ] Test deployed Lambda
- [ ] Verify S3 output

### Phase 3: Table Rename
- [ ] Drop old `bronze.weather` (v1) if exists
- [ ] Rename `bronze.weather_v2` → `bronze.weather`
- [ ] Verify rename successful
- [ ] Update table comments
- [ ] Verify row count unchanged

### Phase 4: SQL Scripts Update
- [ ] Update `sql/create_gold_unified_data.sql`
- [ ] Update `sql/create_unified_data.sql`
- [ ] Search for other weather_v2 references
- [ ] Update all found references

### Phase 5: Archive Scripts
- [ ] Archive v2 backfill scripts (one-time use complete)
- [ ] Rename active scripts (remove v2)
- [ ] Update .gitignore if needed

### Phase 6: Rebuild Unified Data
- [ ] Rebuild gold.unified_data
- [ ] Validate weather array structure
- [ ] Rebuild silver.unified_data (if used)
- [ ] Verify row counts

### Phase 7: Regenerate Forecasts
- [ ] **Retrain all models with v2 weather data**
- [ ] Verify forecast tables recreated
- [ ] Validate forecast quality
- [ ] **Confirm NO forecasts from before migration date**

### Phase 8: Testing
- [ ] Test daily Lambda (manual invoke)
- [ ] Validate bronze.weather table
- [ ] Verify coordinates are correct (v2)
- [ ] Validate gold.unified_data
- [ ] **Validate forecast tables (all post-migration)**
- [ ] Test forecast agent end-to-end

### Phase 9: Documentation
- [ ] Update README.md files
- [ ] Update DATA_SOURCES.md
- [ ] Create migration summary document
- [ ] Update forecast agent docs

### Finalization
- [ ] Commit changes: `git commit -m "feat: Migrate to weather v2, regenerate forecasts with correct coordinates"`
- [ ] Push branch: `git push origin weather-v2-migration`
- [ ] Create PR (if workflow requires)
- [ ] **Monitor forecast quality for 1 week**
- [ ] Monitor next scheduled Lambda run (2 AM UTC)
- [ ] Verify 24 hours of stable operation

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Pre-Migration Assessment | 30 min | None |
| Clean Contaminated Forecasts | 15 min | Assessment complete |
| Update Daily Lambda | 1 hour | Forecasts cleaned |
| Rename Tables | 30 min | Lambda updated |
| Update SQL Scripts | 30 min | Tables renamed |
| Archive V1/V2 Scripts | 15 min | SQL scripts updated |
| Rebuild Unified Data | 30 min | SQL scripts updated |
| Regenerate Forecasts | 1-2 hours | Unified data rebuilt |
| Testing & Validation | 30 min | Forecasts regenerated |
| Documentation Update | 15 min | Testing complete |
| **Total** | **~5-6 hours** | (includes forecast regeneration) |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Forecast regeneration takes too long | Medium | Medium | Start with subset of models, parallelize if possible |
| Lambda fails to load S3 config | Low | High | Test locally first, add error handling |
| Table rename breaks downstream queries | Medium | Medium | Search all code for references first |
| Data loss during migration | Low | High | Keep weather_v2 data in S3, don't delete |
| Incorrect forecasts generated | Low | High | **Validate coordinates before regenerating** |
| Trading team uses old forecasts | Medium | High | **Communicate migration to trading team** |

---

## Communication Plan

**Before Migration**:
- Notify forecast agent team
- Notify trading agent team (if using forecasts)
- Estimate downtime for forecast regeneration

**During Migration**:
- Post status updates in team channel
- Document any issues encountered

**After Migration**:
- Confirm forecasts regenerated successfully
- Share validation results
- Update team on new coordinate accuracy

---

## Next Steps

**Ready to begin migration?**

1. ✅ Review this document thoroughly
2. ✅ **Communicate to team** (especially if trading agent uses forecasts)
3. ✅ Create git branch: `git checkout -b weather-v2-migration`
4. ✅ Start with Phase 0: Forecast Assessment
5. ✅ Work through phases sequentially
6. ✅ Check off items in Migration Checklist
7. ✅ **Monitor forecast quality for 1 week post-migration**

**Questions to Answer Before Starting**:

1. **Timing**: When should we perform this migration?
   - Recommendation: Off-hours to avoid disrupting daily runs
   - Consider: Forecast regeneration may take 1-2 hours

2. **Forecast Regeneration**: Which models should we retrain?
   - All models or just production models?
   - Sequential or parallel training?

3. **Communication**: Who needs to be notified?
   - Trading team?
   - Other stakeholders using forecasts?

4. **Validation**: What forecast quality checks should we run?
   - Backtesting?
   - Comparison with v1 forecasts (for documentation)?

---

**Document Version**: 1.0
**Last Updated**: 2025-12-05
**Author**: Claude Code (AI Assistant)
**Status**: Ready for Review and Execution

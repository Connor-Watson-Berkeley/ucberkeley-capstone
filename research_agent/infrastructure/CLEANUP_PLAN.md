# Infrastructure Cleanup Plan

## ✅ KEEP (Production - Root Level)

### Active Scripts
- `backfill_historical_weather_v2.py` - **PRODUCTION** - Current weather backfill
- `create_weather_v2_bronze_table.py` - **PRODUCTION** - Creates bronze table
- `create_unified_data.py` - **PRODUCTION?** - Check if still used for pipeline
- `rebuild_all_layers.py` - **PRODUCTION?** - Check if still used
- `unity_catalog_workaround.py` - **FALLBACK** - SQL connector workaround
- `databricks/setup_unity_catalog_credentials.py` - **PRODUCTION** - Unity Catalog setup
- `.env` / `.env.example` - **PRODUCTION** - Secrets management

### Documentation (Keep)
- `README.md` - Main docs
- `DATABRICKS_MIGRATION_GUIDE.md` - Current migration guide
- `MIGRATION_PREFLIGHT_CHECKLIST.md` - Current checklist

### Validation (Keep - Useful)
- `validate_july2021_frost.py` - Data quality check
- `validate_region_coordinates.py` - Coordinate validation

---

## 📦 ARCHIVE (Move to archive/)

### Old Backfill Versions
- `backfill_sugar_data.py` → archive/ (replaced by v2)
- `backfill_weather_chunked.py` → archive/ (replaced by v2)
- `backfill_weather_data.py` → archive/ (replaced by v2)
- `backfill_weather_forecasts.py` → archive/ (replaced by v2)
- `backfill_synthetic_weather_forecasts.py` → archive/ (replaced by v2)

### One-Time Setup Scripts
- `migrate_catalog.py` → archive/ (one-time migration, completed)
- `drop_old_views.py` → archive/ (one-time cleanup, completed)
- `consolidate_to_forecast.py` → archive/ (one-time migration, completed)
- `setup_databricks_pipeline.py` → archive/ (one-time setup)
- `diagnose_cluster_issue.py` → archive/ (old workspace debugging)
- `wait_for_cluster.py` → archive/ (cluster startup helper)

### Old Documentation
- `COORDINATE_VALIDATION.md` → archive/ (one-time fix, completed)
- `DATABRICKS_UNITY_CATALOG_ISSUE.md` → archive/ (old workspace issue, resolved)
- `WEATHER_COORDINATES_ISSUE.md` → archive/ (one-time issue, fixed)
- `WEATHER_FORECAST_LIMITATION.md` → archive/ (analysis document)
- `WEATHER_FORECAST_PROPOSAL.md` → archive/ (proposal, implemented)
- `EVALUATION_AND_TESTING.md` → archive/ (review first)
- `HISTORICAL_FORECAST_OPTIONS.md` → archive/ (review first)

### Shell Scripts
- `automate_weather_v2_pipeline.sh` → archive/ (automation helper)
- `monitor_and_automate_v2_migration.sh` → archive/ (migration helper)

---

## 🧪 CONSOLIDATE → tests/

Create `tests/` folder with organized test suites:

### tests/test_data_quality.py (Consolidate):
- `validate_data_quality.py`
- `validate_pipeline.py`
- `validate_unified_data_inputs.py`
- `validate_weather_pipeline.py`
- `check_databricks_tables.py`

### tests/test_catalog.py (Consolidate):
- `check_catalog_structure.py`
- `check_forecast_schemas.py`
- `test_forecast_schema.py`

### tests/test_pipeline.py (Consolidate):
- `test_pipeline.py`
- `test_full_pipeline.py`
- `test_unified_data.py`

### tests/test_sources.py (Consolidate):
- `check_gdelt_data.py`
- `check_landing_sugar.py`
- `check_sugar_weather.py`

### DELETE (One-Time Tests):
- `test_cluster_unity_catalog.py` → DELETE (old workspace testing)
- `test_new_workspace_unity_catalog.py` → DELETE (already tested, passed)
- `test_open_meteo_previous_runs.py` → DELETE (one-time experiment)
- `test_weather_forecast_api.py` → DELETE (one-time experiment)

### KEEP for now (Utilities):
- `dashboard_pipeline_health.py` - Keep if actively monitoring
- `list_databricks_repos.py` - Keep if needed for setup
- `pull_databricks_repo.py` - Keep if needed for setup  
- `load_historical_to_databricks.py` - Keep if still loading data

---

## 🗑️ DELETE COMPLETELY

- `test_cluster_unity_catalog.py` - Old workspace, no longer relevant
- `test_new_workspace_unity_catalog.py` - One-time test, already passed
- `test_open_meteo_previous_runs.py` - Experiment, not production
- `test_weather_forecast_api.py` - Experiment, not production

# Weather v2 Migration Scripts (ARCHIVED)

**Status**: One-time use complete
**Migration Date**: November 11, 2025
**Archive Date**: December 5, 2025

---

## Purpose

These scripts were used for the **one-time migration** from weather v1 (incorrect coordinates) to weather v2 (correct growing region coordinates). The migration is now complete and these scripts are no longer needed.

---

## What Was Done

### The Problem
- **v1 coordinates**: Used state capital coordinates (~100-200km off from actual growing regions)
- **v2 coordinates**: Used precise growing region coordinates from `config/region_coordinates.json`

### The Migration (Nov 11, 2025)
1. Created `bronze.weather_v2` table in Databricks
2. Backfilled historical weather data (2015-present) using v2 coordinates
3. Updated daily Lambda to use v2 coordinates from S3 config
4. Renamed `bronze.weather_v2` → `bronze.weather` (v2 became canonical)

---

## Archived Files

### Python Scripts
- **`backfill_historical_weather_v2.py`** - Backfilled 2015-2025 weather data with v2 coordinates (3,780 files to S3)
- **`create_weather_v2_bronze_table.py`** - Created `bronze.weather_v2` table in Databricks
- **`create_weather_v2_with_copy_into.py`** - Alternative approach using COPY INTO

### SQL Scripts
- **`weather_v2_delta_migration.sql`** - Databricks SQL for creating v2 table
- **`create_weather_v2_bronze_table.sql`** - Table DDL with v2 schema

### Automation Scripts
- **`automate_weather_v2_pipeline.sh`** - Automated backfill orchestration
- **`monitor_and_automate_v2_migration.sh`** - Migration monitoring and automation

### Documentation
- **`WEATHER_V2_MANUAL_INSTRUCTIONS.md`** - Manual instructions for v2 migration

---

## Migration Results

- **3,780 weather files** backfilled to S3 (`landing/weather_v2/`)
- **67 regions** with correct coordinates
- **Daily Lambda** updated to load coordinates from S3
- **All SQL scripts** updated to reference `bronze.weather` (not `weather_v2`)

---

## Why Archived?

These scripts served their purpose and are no longer needed because:
1. Historical backfill is complete
2. Daily Lambda now uses v2 coordinates automatically
3. `bronze.weather_v2` has been renamed to `bronze.weather`
4. All downstream pipelines updated

---

## See Also

- Active migration plan: `../../WEATHER_V2_MIGRATION_PLAN.md`
- Execution guide: `../../RUN_WEATHER_MIGRATION.md`
- Data source docs: `../../DATA_SOURCES.md`

---

**Archive Reason**: One-time migration complete
**Safe to Delete**: Yes (after git commit for historical record)
**Last Used**: November 11, 2025

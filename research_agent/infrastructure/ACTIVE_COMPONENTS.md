# GDELT Pipeline - Active Components

**Last Updated:** 2025-11-22 20:50 UTC

This document identifies which components are actively used in production vs. legacy/experimental.

---

## ✅ ACTIVE LAMBDA FUNCTIONS (Currently Used in Production)

### Daily Incremental Pipeline (Production)

1. **gdelt-daily-discovery** ✅ IN USE
   - **Location:** `lambda/functions/gdelt-daily-discovery/`
   - **Deployed Name:** `gdelt-daily-discovery`
   - **Purpose:** Discovers new GDELT files daily by streaming master list
   - **Trigger:** EventBridge schedule (2 AM UTC daily)
   - **Status:** ✅ PRODUCTION - Fixed Nov 22 (streaming optimization)
   - **Code:** `lambda_function.py`

2. **gdelt-csv-bronze-direct** ✅ IN USE
   - **Location:** `lambda/functions/gdelt-csv-bronze-direct/`
   - **Deployed Name:** `gdelt-bronze-transform`
   - **Purpose:** Downloads CSV from GDELT, filters, writes Bronze Parquet, queues dates to silver
   - **Trigger:** SQS queue `groundtruth-gdelt-backfill-queue`
   - **Status:** ✅ PRODUCTION - Updated Nov 22 (added silver queue loading)
   - **Code:** `lambda_function.py`
   - **Note:** Deployed to AWS function name `gdelt-bronze-transform`

3. **gdelt-silver-transform** ✅ IN USE
   - **Location:** `lambda/functions/gdelt-silver-transform/`
   - **Deployed Name:** `gdelt-silver-transform`
   - **Purpose:** Daily silver aggregation (Bronze → Silver wide format)
   - **Trigger:** EventBridge schedule (3 AM UTC daily)
   - **Status:** ✅ PRODUCTION
   - **Code:** `lambda_function.py`
   - **Note:** For daily incremental dates (usually <50 files/day)

4. **gdelt-silver-discovery** ✅ IN USE (NEW)
   - **Location:** `lambda/functions/gdelt-silver-discovery/`
   - **Deployed Name:** `gdelt-silver-discovery` (needs deployment)
   - **Purpose:** Scans DynamoDB for bronze files without silver status, queues dates for processing
   - **Trigger:** Manual or scheduled
   - **Status:** ✅ CODE READY - Needs deployment
   - **Code:** `lambda_function.py`
   - **Note:** Two paths for silver queue loading: (1) Auto from bronze, (2) Manual discovery

### ⚠️ NEEDS FIXING - Silver Backfill Lambda

5. **gdelt-silver-backfill** ⚠️ HAS ISSUES
   - **Deployed Name:** `gdelt-silver-backfill`
   - **Purpose:** Historical backfill - Bronze → Silver with chunked processing
   - **Trigger:** SQS queue `groundtruth-gdelt-silver-backfill-queue` (DISABLED)
   - **Status:** ⚠️  NEEDS FIX - Chunked processing threshold too high (>50 files)
   - **Code:** Deployed from `/tmp/gdelt_silver_backfill_lambda.py`
   - **Issue:** Chunked processing only triggers for >50 files, but 49-file dates still cause OOM
   - **Fix needed:** Lower threshold to ~30 files or use record count instead
   - **Note:** Event source mapping currently DISABLED

### ⏸️ COMPLETED - Historical Backfill

6. **gdelt-bronze-transform** (JSONL mode) ⏸️ COMPLETE
   - **Location:** `lambda/functions/gdelt-bronze-transform/`
   - **Deployed Name:** `gdelt-jsonl-bronze-transform`
   - **Purpose:** Historical backfill - JSONL → Bronze Parquet
   - **Status:** ⏸️  COMPLETE - Backfill finished (168,704 files processed)
   - **Code:** `lambda_function.py`
   - **Note:** Can be re-enabled if historical backfill needed

---

## ACTIVE EVENTBRIDGE SCHEDULES

| Rule Name | Schedule | Target Lambda | Status |
|-----------|----------|---------------|--------|
| `gdelt-daily-discovery-schedule` | `cron(0 2 * * ? *)` | gdelt-daily-discovery | ✅ ENABLED |
| `gdelt-daily-silver-transform` | `cron(0 3 * * ? *)` | gdelt-silver-transform | ✅ ENABLED |

---

## ACTIVE SQS QUEUES & TRIGGERS

| Queue Name | Triggered Lambda | Status | Purpose |
|------------|-----------------|--------|---------|
| `groundtruth-gdelt-backfill-queue` | gdelt-bronze-transform | ✅ ENABLED | Daily CSV→Bronze |
| `groundtruth-gdelt-silver-backfill-queue` | gdelt-silver-backfill | ✅ ENABLED | Backfill Bronze→Silver |

---

## LEGACY/EXPERIMENTAL COMPONENTS (Not in Production)

Moved to: `infrastructure/legacy/`

### Legacy Lambda Functions

1. **berkeley-datasci210-capstone-processor**
   - Old GDELT processor (replaced by modular pipeline)
   - Location: `legacy/lambda_functions/berkeley-datasci210-capstone-processor/`

2. **gdelt-csv-sqs-loader**
   - Experimental SQS loader (not used)
   - Location: `legacy/lambda_functions/gdelt-csv-sqs-loader/`

3. **gdelt-generate-date-batches**
   - Experimental batch date generator (not used)
   - Location: `legacy/lambda_functions/gdelt-generate-date-batches/`
   - Deployed in AWS but not actively used

4. **gdelt-jsonl-to-silver**
   - Old JSONL→Silver direct converter (replaced by gdelt-silver-transform)
   - Location: `legacy/lambda_functions/gdelt-jsonl-to-silver/`

5. **gdelt-queue-monitor**
   - Monitoring utility (not actively used)
   - Location: `legacy/lambda_functions/gdelt-queue-monitor/`
   - Deployed in AWS but not actively used

### Legacy Step Functions

All Step Functions are **DISABLED** in favor of EventBridge scheduled Lambdas:

1. **gdelt_bronze_silver_pipeline.json**
   - Old orchestration approach
   - Location: `legacy/step_functions/gdelt_bronze_silver_pipeline.json`

2. **gdelt_daily_incremental_pipeline.json**
   - Experimental daily orchestration
   - Location: `legacy/step_functions/gdelt_daily_incremental_pipeline.json`

3. **gdelt_daily_master_pipeline.json**
   - Experimental master pipeline
   - Location: `legacy/step_functions/gdelt_daily_master_pipeline.json`

4. **groundtruth_gdelt_backfill_sqs.json**
   - Old SQS backfill approach
   - Location: `legacy/step_functions/groundtruth_gdelt_backfill_sqs.json`

5. **groundtruth_gdelt_backfill_with_bronze_silver.json**
   - Old backfill orchestration
   - Location: `legacy/step_functions/groundtruth_gdelt_backfill_with_bronze_silver.json`

---

## DEPLOYMENT SCRIPTS (Active)

| Script | Purpose | Status |
|--------|---------|--------|
| `lambda/deploy_bronze_transform.sh` | Deploy CSV→Bronze Lambda | ✅ ACTIVE |
| `lambda/deploy_jsonl_bronze_transform.sh` | Deploy JSONL→Bronze Lambda | ⏸️  Used for backfill |
| `step_functions/deploy_gdelt_pipeline.sh` | Deploy Step Function | ❌ NOT USED |
| `step_functions/deploy_daily_master.sh` | Deploy daily master SF | ❌ NOT USED |

---

## CURRENT ARCHITECTURE (Active Production System)

```
┌─────────────────────────────────────────────────────────────────┐
│ DAILY INCREMENTAL PIPELINE (Active)                            │
└─────────────────────────────────────────────────────────────────┘

EventBridge (2 AM UTC)
    ↓
gdelt-daily-discovery
    ↓ (SQS: groundtruth-gdelt-backfill-queue)
gdelt-bronze-transform (CSV→Bronze)
    ↓ (1 hour gap)
EventBridge (3 AM UTC)
    ↓
gdelt-silver-transform (Bronze→Silver)


┌─────────────────────────────────────────────────────────────────┐
│ HISTORICAL BACKFILL PIPELINE (One-time, 98.9% complete)        │
└─────────────────────────────────────────────────────────────────┘

SQS Queue (groundtruth-gdelt-silver-backfill-queue)
    ↓
gdelt-silver-backfill (Bronze→Silver for 1,767 dates)
```

---

## FILES TO IGNORE

These files are in the repo but not actively used:

- `infrastructure/legacy/` - All legacy Lambda functions and Step Functions
- `lambda/functions/berkeley-datasci210-capstone-processor/` - Old processor
- Any Step Function JSON files (using EventBridge instead)

---

## FILES TO MAINTAIN

These are the core active components:

**Lambda Functions:**
- `lambda/functions/gdelt-daily-discovery/lambda_function.py`
- `lambda/functions/gdelt-csv-bronze-direct/lambda_function.py`
- `lambda/functions/gdelt-silver-transform/lambda_function.py`
- `lambda/functions/gdelt-bronze-transform/lambda_function.py` (JSONL mode for backfill)

**Deployment Scripts:**
- `lambda/deploy_bronze_transform.sh` (for CSV→Bronze)
- `lambda/deploy_jsonl_bronze_transform.sh` (for historical backfill)

**Documentation:**
- `/tmp/GDELT_PROJECT_STATUS.md` - Master status file
- `/tmp/GDELT_CURRENT_STATUS_SUMMARY.md` - Current operational status
- `/tmp/DAILY_PIPELINE_STATUS.md` - Daily pipeline documentation

---

**Next Action:** Monitor tonight's daily pipeline run (2-3 AM UTC) to verify end-to-end operation

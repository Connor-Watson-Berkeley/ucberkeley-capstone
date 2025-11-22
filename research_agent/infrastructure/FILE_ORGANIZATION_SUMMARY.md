# GDELT Pipeline - File Organization Summary

**Date:** 2025-11-22  
**Action:** Separated active production code from legacy/experimental components

---

## What Was Done

Reorganized the `research_agent/infrastructure/` directory to clearly separate:
- **Active production components** (currently in use)
- **Legacy/experimental components** (deprecated, for reference only)

---

## Directory Structure

### BEFORE Organization
```
infrastructure/
├── lambda/
│   └── functions/
│       ├── berkeley-datasci210-capstone-processor/  (OLD - not used)
│       ├── gdelt-bronze-transform/                  (ACTIVE)
│       ├── gdelt-csv-bronze-direct/                 (ACTIVE)
│       ├── gdelt-csv-sqs-loader/                    (OLD - not used)
│       ├── gdelt-daily-discovery/                   (ACTIVE)
│       ├── gdelt-generate-date-batches/             (OLD - not used)
│       ├── gdelt-jsonl-to-silver/                   (OLD - not used)
│       ├── gdelt-queue-monitor/                     (OLD - not used)
│       └── gdelt-silver-transform/                  (ACTIVE)
└── step_functions/
    ├── gdelt_bronze_silver_pipeline.json            (OLD - not used)
    ├── gdelt_daily_incremental_pipeline.json        (OLD - not used)
    ├── gdelt_daily_master_pipeline.json             (OLD - not used)
    ├── groundtruth_gdelt_backfill_sqs.json          (OLD - not used)
    └── groundtruth_gdelt_backfill_with_bronze_silver.json (OLD - not used)
```

### AFTER Organization
```
infrastructure/
├── lambda/
│   └── functions/
│       ├── gdelt-bronze-transform/          ✅ ACTIVE (backfill JSONL→Bronze)
│       ├── gdelt-csv-bronze-direct/         ✅ ACTIVE (daily CSV→Bronze)
│       ├── gdelt-daily-discovery/           ✅ ACTIVE (daily discovery)
│       ├── gdelt-silver-transform/          ✅ ACTIVE (daily Bronze→Silver)
│       └── [other data fetchers...]         (non-GDELT components)
│
├── legacy/
│   ├── README.md                            📖 Explains legacy components
│   ├── lambda_functions/
│   │   ├── berkeley-datasci210-capstone-processor/
│   │   ├── gdelt-csv-sqs-loader/
│   │   ├── gdelt-generate-date-batches/
│   │   ├── gdelt-jsonl-to-silver/
│   │   └── gdelt-queue-monitor/
│   └── step_functions/
│       ├── gdelt_bronze_silver_pipeline.json
│       ├── gdelt_daily_incremental_pipeline.json
│       ├── gdelt_daily_master_pipeline.json
│       ├── groundtruth_gdelt_backfill_sqs.json
│       └── groundtruth_gdelt_backfill_with_bronze_silver.json
│
├── ACTIVE_COMPONENTS.md                     📖 Active architecture guide
└── FILE_ORGANIZATION_SUMMARY.md             📖 This document
```

---

## Active Components (Production)

### Lambda Functions

| Function | Location | Purpose | Trigger | Status |
|----------|----------|---------|---------|--------|
| `gdelt-daily-discovery` | `lambda/functions/gdelt-daily-discovery/` | Discover new GDELT files | EventBridge (2 AM UTC) | ✅ ACTIVE |
| `gdelt-csv-bronze-direct` | `lambda/functions/gdelt-csv-bronze-direct/` | CSV→Bronze Parquet | SQS queue | ✅ ACTIVE |
| `gdelt-silver-transform` | `lambda/functions/gdelt-silver-transform/` | Bronze→Silver aggregation | EventBridge (3 AM UTC) | ✅ ACTIVE |
| `gdelt-bronze-transform` | `lambda/functions/gdelt-bronze-transform/` | JSONL→Bronze (backfill) | SQS queue (disabled) | ⏸️  COMPLETE |

**Note:** `gdelt-csv-bronze-direct` deploys to AWS function name `gdelt-bronze-transform`

### EventBridge Schedules

| Schedule | Time (UTC) | Target | Status |
|----------|------------|--------|--------|
| `gdelt-daily-discovery-schedule` | 2:00 AM | gdelt-daily-discovery | ✅ ENABLED |
| `gdelt-daily-silver-transform` | 3:00 AM | gdelt-silver-transform | ✅ ENABLED |

### SQS Queues

| Queue | Triggered Lambda | Status |
|-------|-----------------|--------|
| `groundtruth-gdelt-backfill-queue` | gdelt-bronze-transform | ✅ ENABLED |
| `groundtruth-gdelt-silver-backfill-queue` | gdelt-silver-backfill | ✅ ENABLED |

---

## Legacy Components (Archived)

All moved to `infrastructure/legacy/`

### Why Legacy?

1. **Architecture Evolution:**
   - Started with monolithic processor
   - Evolved to modular Discovery → Bronze → Silver pipeline
   - Step Functions experiments were too complex for linear workflow

2. **Simpler = Better:**
   - EventBridge schedules are simpler than Step Functions
   - Direct SQS triggers are more reliable
   - Each Lambda can be tested independently

3. **Cost Optimization:**
   - No Step Function execution charges
   - Simpler infrastructure = lower operational costs

### What's in Legacy?

**Lambda Functions:**
- `berkeley-datasci210-capstone-processor` - Original monolithic processor
- `gdelt-csv-sqs-loader` - Experimental SQS loader
- `gdelt-generate-date-batches` - Experimental batch generator
- `gdelt-jsonl-to-silver` - Old direct JSONL→Silver (skipped Bronze)
- `gdelt-queue-monitor` - Monitoring utility

**Step Functions:**
- All Step Function definitions (5 files)
- Replaced by EventBridge scheduled Lambdas
- Kept for historical reference

**Status:** Some are still deployed in AWS but not actively used. Can be deleted after 30-day observation period.

---

## Documentation Files

### Main Documentation

| File | Location | Purpose |
|------|----------|---------|
| `ACTIVE_COMPONENTS.md` | `infrastructure/` | Lists active vs legacy components |
| `FILE_ORGANIZATION_SUMMARY.md` | `infrastructure/` | This document - organization summary |
| `legacy/README.md` | `infrastructure/legacy/` | Explains legacy components |

### Status Files (in /tmp)

| File | Purpose |
|------|---------|
| `/tmp/GDELT_PROJECT_STATUS.md` | Master status from Nov 21 (historical) |
| `/tmp/GDELT_CURRENT_STATUS_SUMMARY.md` | Current operational status (Nov 22) |
| `/tmp/DAILY_PIPELINE_STATUS.md` | Daily pipeline details |

---

## What Changed Today (Nov 22)

1. ✅ **Fixed Discovery Lambda** - Streaming optimization for OOM error
2. ✅ **Organized Files** - Moved 5 Lambda functions + 5 Step Functions to legacy/
3. ✅ **Created Documentation** - ACTIVE_COMPONENTS.md, legacy/README.md, this file
4. ✅ **Verified Active Components** - All production components clearly identified

---

## How to Use This Organization

### For Development

**Working on active components?**
→ Look in `lambda/functions/gdelt-*` (not in legacy/)

**Need to reference old implementations?**
→ Check `legacy/lambda_functions/` or `legacy/step_functions/`

**Want to understand architecture?**
→ Read `ACTIVE_COMPONENTS.md`

### For Deployment

**Deploy daily pipeline components:**
```bash
# Discovery Lambda
cd lambda/functions/gdelt-daily-discovery
./deploy.sh  # (if exists, or use zip + AWS update)

# Bronze Lambda (CSV→Bronze)
cd lambda/functions/gdelt-csv-bronze-direct
./deploy.sh

# Silver Lambda
cd lambda/functions/gdelt-silver-transform
# (deploy script location TBD)
```

**DO NOT deploy anything from legacy/ folder**

### For Cleanup

**Can delete legacy/ folder?**
- Not yet - keep for reference until daily pipeline runs smoothly for 1-2 months
- After validation: Safe to delete from main branch (will remain in git history)

**Can delete Lambda functions from AWS?**
- Yes, after 30-day observation period
- Deployed but unused: gdelt-processor, gdelt-sqs-loader, gdelt-queue-monitor, gdelt-generate-date-batches
- Confirm no errors/dependencies before deleting

---

## Next Steps

1. ✅ Organization complete
2. ✅ Documentation created
3. ⏳ Monitor tonight's daily pipeline run (2-3 AM UTC Nov 23)
4. ⏳ After 30 days: Review and delete unused AWS Lambda functions
5. ⏳ After 60 days: Consider removing legacy/ folder from main branch

---

**Status:** File organization complete. All active components clearly identified and separated from legacy code.

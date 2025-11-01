# Research Agent Evaluation and Testing

**Last Updated**: 2025-10-31
**Owner**: Research Agent Team (Francisco - data pipelines, Connor - validation)

## Overview

This document describes the evaluation and testing framework for the Research Agent data pipelines. The framework consists of:
1. **One-time historical validation** - Deep analysis after major changes
2. **Continuous health checks** - Daily monitoring with email alerts
3. **Schema validation** - Ensure DATA_CONTRACTS.md compliance

## Table of Contents

- [Validation Infrastructure](#validation-infrastructure)
- [Latest Validation Results](#latest-validation-results)
- [Continuous Testing](#continuous-testing)
- [Known Issues](#known-issues)
- [Runbook](#runbook)

---

## Validation Infrastructure

### Directory Structure

```
research_agent/infrastructure/validation/
├── one_time/
│   └── validate_historical_data.py      # Deep historical analysis
├── continuous/
│   ├── health_checks.py                 # Daily SQL unit tests
│   └── run_health_checks_job.py         # Databricks job wrapper
└── EVALUATION_AND_TESTING.md            # This document
```

### Scripts

#### 1. Historical Validation (`one_time/validate_historical_data.py`)

**Purpose**: Comprehensive data quality analysis after pipeline rebuilds

**Checks**:
- ✓ Row counts across all layers (landing, bronze, silver)
- ✓ Null patterns in raw data BEFORE forward-fill
- ✓ Date gaps and completeness by commodity/region
- ✓ Anomalous values (impossible prices, extreme outliers)
- ✓ Schema compliance with DATA_CONTRACTS.md
- ✓ Regional coverage
- ✓ Data lineage (landing → bronze → silver)

**Usage**:
```bash
export DATABRICKS_TOKEN=<your-token>
cd research_agent/infrastructure/validation/one_time
python validate_historical_data.py > validation_report_$(date +%Y%m%d).txt
```

**When to Run**:
- After major pipeline changes
- After schema updates
- After backfilling historical data
- When investigating data quality issues

#### 2. Continuous Health Checks (`continuous/health_checks.py`)

**Purpose**: Fast daily checks with pass/fail status

**Tests** (10 total):
1. Row count minimum (Coffee >= 70k rows)
2. No nulls in OHLC fields
3. Data freshness (<5 days old)
4. Regional coverage (Coffee >= 15 regions)
5. Weather null rate (<2%)
6. VIX data completeness (>95%)
7. No impossible prices
8. No OHLC violations
9. Bronze-Silver consistency
10. Schema stability (45-55 columns)

**Usage**:
```bash
export DATABRICKS_TOKEN=<your-token>
cd research_agent/infrastructure/validation/continuous
python health_checks.py
echo $?  # 0 = pass, 1 = fail
```

**When to Run**:
- **Daily at 8am** (automated via Databricks job)
- Before deploying forecast models
- After data ingestion runs

---

## Latest Validation Results

**Execution Date**: 2025-10-31 00:56 UTC
**Data Version**: Post-pipeline rebuild with 15 weather fields

### Summary

| Data Source | Status | Details |
|-------------|--------|---------|
| Coffee Market Data | ✅ PASS | 75,472 rows, 3,770 dates (2015-07-07 to 2025-10-31) |
| Coffee Weather | ✅ PASS | 20 regions, 376 nulls per field (0.5%) |
| Sugar Market Data | ✅ PASS | 2,598 dates in bronze (2015-07-07 to 2025-10-30) |
| Sugar Weather | ❌ **CRITICAL** | Only 8 days (2025-10-23 to 2025-10-31) - missing historical data |
| VIX Data | ✅ PASS | Complete coverage, values within normal range |
| Macro/FX Data | ✅ PASS | 24 currencies, reasonable null rates (<15%) |
| Schema | ✅ PASS | 49 fields match enhanced contract |

### Detailed Findings

#### ✅ Coffee Data - HEALTHY

**Market Data (OHLCV)**:
- Rows: 75,472 across 20 regions
- Dates: 3,770 unique dates
- Date Range: 2015-07-07 to 2025-10-31
- Nulls: 0 in all OHLCV fields ✓

**Weather Data (15 fields)**:
- Total Rows: ~75k
- Null Count: 376 per field (temp_max_c, temp_min_c, humidity, wind, solar, ET0)
- Null Rate: 0.5% (acceptable - forward-fill handles this)
- Precipitation: 0 nulls ✓

**Regions Covered**: 20 major coffee-producing regions including:
- Brazil: Sao_Paulo, Minas_Gerais, Espirito_Santo, Bahia
- Colombia: Huila, Eje_Cafetero
- Vietnam: Central_Highlands
- Indonesia: Sumatra, Java
- Ethiopia: Sidamo, Yirgacheffe
- And 10 more regions

#### ❌ Sugar Data - CRITICAL ISSUE

**Problem**: Missing historical weather data for Sugar regions

**Impact**:
- Silver `unified_data` table has only 304 Sugar rows (should be ~75k like Coffee)
- Only 8 dates: 2025-10-23 to 2025-10-31
- Market data EXISTS in bronze (2,598 dates)
- Weather data MISSING in bronze (only 8 days)

**Root Cause**:
- Weather Lambda recently started fetching Sugar regions
- Historical backfill not yet completed for 38 Sugar regions
- Unified_data SQL uses INNER JOIN on weather, dropping all Sugar records without weather

**Action Required**:
1. **URGENT**: Backfill 10 years of weather data for 38 Sugar regions
2. Rebuild landing.weather_data_inc table
3. Rebuild bronze.weather_data view
4. Rebuild silver.unified_data table
5. Validate Sugar data completeness

**Sugar Regions Affected** (38 total):
- Argentina, Belarus, Belgium, Cuba, France, Germany
- Escuintla_Guatemala, Guangxi_China, Hokkaido_Japan
- Iran, Jalisco_Mexico, Java_Indonesia_Sugar
- Khon_Kaen_Thailand, KwaZulu_Natal_South_Africa
- Louisiana_USA, Maharashtra_India, Nakhon_Sawan_Thailand
- Negros_Occidental_Philippines, Netherlands, Nile_Delta_Egypt
- North_China_Beet, Poland, Punjab_Pakistan, Qena_Egypt
- Queensland_Australia, Red_River_Valley_USA
- Sao_Paulo_Brazil_Sugar, Sindh_Pakistan, South_Florida_USA
- Tambov_Russia, Turkey, UK, Ukraine
- Uttar_Pradesh_India, Valle_del_Cauca_Colombia
- Veracruz_Mexico_Sugar, Voronezh_Russia, Yunnan_China_Sugar

#### ✅ VIX & Macro Data - HEALTHY

**VIX**:
- Coverage: 2,598 dates (2015-07-07 to 2025-10-30)
- Nulls: Minimal (<5%)
- Value Range: Within normal bounds (8-90)

**Macro/FX** (24 currencies):
- Coverage: 2,598 dates
- Nulls: <15% per currency (weekends/holidays)
- Forward-fill handles gaps effectively

#### ✅ Schema Compliance

**Expected Fields** (per DATA_CONTRACTS.md):
- Core: date, is_trading_day, commodity, region
- Market: open, high, low, close, volume (5 fields)
- Weather: 15 enhanced fields (temp, precip, humidity, wind, solar, ET0)
- Macro: 24 FX rates
- VIX: vix

**Actual Schema**: 49 columns ✓

**Enhancements Over Contract**:
- Added 12 additional weather fields beyond original 3
- All original contract fields preserved
- No fields dropped (per user preference)

---

## Continuous Testing

### Databricks Job Setup

**Job Name**: `research_agent_health_checks`

**Schedule**: Daily at 8:00 AM UTC

**Configuration**:
```json
{
  "name": "research_agent_health_checks",
  "tasks": [
    {
      "task_key": "health_checks",
      "python_script_task": {
        "source": "GIT",
        "file": "research_agent/infrastructure/validation/continuous/run_health_checks_job.py"
      },
      "existing_cluster_id": "your-cluster-id"
    }
  ],
  "email_notifications": {
    "on_failure": ["team@example.com"]
  },
  "schedule": {
    "quartz_cron_expression": "0 0 8 * * ?",
    "timezone_id": "UTC"
  }
}
```

**Email Alert Configuration**:
- Trigger: Job exit code 1 (test failure)
- Recipients: Research Agent team, Forecast Agent team
- Content: Test failure details from stdout

### Test Criteria

| Test ID | Description | Threshold | Impact if Failed |
|---------|-------------|-----------|------------------|
| 1 | Row count minimum | ≥70k | Data loss detected |
| 2 | OHLC nulls | = 0 | Invalid market data |
| 3 | Data freshness | ≤5 days | Stale data |
| 4 | Regional coverage | ≥15 regions | Missing geography |
| 5 | Weather null rate | <2% | Incomplete weather |
| 6 | VIX completeness | >95% | Missing volatility |
| 7 | Price validity | No invalid | Data corruption |
| 8 | OHLC consistency | No violations | Data corruption |
| 9 | Layer consistency | <10 date diff | Pipeline issue |
| 10 | Schema stability | 45-55 columns | Schema drift |

### Response Procedure

**When health checks fail**:

1. **Acknowledge** (within 1 hour)
   - Check email alert
   - Review test failure details
   - Assess severity (critical vs warning)

2. **Investigate** (within 4 hours)
   - Run historical validation: `python validate_historical_data.py`
   - Check Databricks job logs
   - Query affected tables directly

3. **Remediate** (within 24 hours)
   - Fix data pipeline issues
   - Backfill missing data
   - Rebuild affected tables
   - Re-run health checks

4. **Document** (after resolution)
   - Update this document with root cause
   - Add to Known Issues section
   - Update runbook if new scenario

---

## Known Issues

### Issue #1: Sugar Historical Weather Data Missing

**Status**: 🔴 CRITICAL - In Progress
**Discovered**: 2025-10-31
**Impact**: Sugar forecasting not possible (only 8 days of data)

**Root Cause**:
Weather Lambda only recently started fetching 38 Sugar regions. Historical backfill (2015-01-01 to 2025-10-22) never executed.

**Fix**:
1. ✅ Identified 38 Sugar regions
2. ⏳ Trigger weather Lambda backfill (3,957 days × 38 regions = ~150k rows)
3. ⏳ Rebuild Databricks landing/bronze/silver layers
4. ⏳ Validate Sugar data completeness

**Timeline**: 2-4 hours for backfill + table rebuild

**Workaround**: None - Coffee-only forecasts until fixed

---

### Issue #2: Small Cluster (8GB/2-core) Fails on Unified Data Job

**Status**: 🟡 KNOWN
**Discovered**: 2025-10-31
**Impact**: Databricks job intermittently fails with "Failed to reach the driver"

**Root Cause**:
- Unified data SQL processes 10+ years, 75k+ rows, complex window functions
- 8GB/2-core cluster too small for query planning + execution
- Driver becomes unresponsive under memory pressure

**Fix**:
Upgrade cluster OR switch to SQL Warehouse (serverless):
- Minimum: 4 cores / 16GB RAM
- Recommended: SQL Warehouse (auto-scaling, no management)

**Workaround**:
- Run unified_data script directly via SQL API (bypasses cluster issues)
- Retry job (sometimes succeeds)

**Timeline**: Can upgrade cluster immediately

---

## Runbook

### Scenario 1: Health Check Failure Email

```bash
# 1. Check which test failed
cd research_agent/infrastructure/validation/continuous
python health_checks.py

# 2. Run full historical validation
cd ../one_time
python validate_historical_data.py > report_$(date +%Y%m%d).txt
less report_*.txt

# 3. Check Databricks tables directly
databricks sql query "SELECT COUNT(*) FROM commodity.silver.unified_data"

# 4. Review recent data ingestion logs
# Check AWS Lambda logs for weather-data-fetcher, market-data-fetcher, etc.
```

### Scenario 2: Missing Historical Data

```bash
# 1. Identify affected commodity/region
cd research_agent/infrastructure
python -c "
from create_unified_data import execute_sql
sql = '''SELECT commodity, region, COUNT(*), MIN(date), MAX(date)
         FROM commodity.silver.unified_data
         GROUP BY commodity, region
         HAVING COUNT(*) < 1000
         ORDER BY COUNT(*)'''
success, result = execute_sql(sql)
# Review results
"

# 2. Check if data exists in bronze
# If yes -> rebuild silver
# If no -> check landing
# If no -> backfill from source

# 3. Trigger backfill
aws lambda invoke --function-name weather-data-fetcher \
  --payload '{"days_to_fetch": [3957, 0]}' \
  /tmp/weather-backfill.json

# 4. Rebuild layers
python create_unified_data.py
```

### Scenario 3: Schema Drift Detected

```bash
# 1. Compare actual vs expected schema
databricks sql query "DESCRIBE commodity.silver.unified_data"

# 2. Check DATA_CONTRACTS.md for expected schema

# 3. If contract changed:
#    - Update SQL files
#    - Coordinate with Forecast Agent team
#    - Update this document

# 4. Rebuild table with correct schema
python create_unified_data.py
```

### Scenario 4: Anomalous Values Detected

```bash
# 1. Run anomaly detection
cd research_agent/infrastructure/validation/one_time
python validate_historical_data.py | grep "ANOMALY"

# 2. Check source data
# Query bronze layer for suspicious values

# 3. Investigate Lambda function
# Check if data source API changed

# 4. If bad data:
#    - Fix source
#    - Re-ingest
#    - Rebuild silver

# 5. If outlier is legitimate:
#    - Document in Known Issues
#    - Adjust anomaly thresholds
```

---

## Validation Schedule

| Frequency | Script | Purpose |
|-----------|--------|---------|
| **Daily** | `continuous/health_checks.py` | Catch issues quickly |
| **Weekly** | `one_time/validate_historical_data.py` | Trend analysis |
| **After Changes** | `one_time/validate_historical_data.py` | Comprehensive validation |
| **Before Release** | Both | Gate for production deploy |

---

## Metrics & SLAs

**Data Quality SLAs**:
- Null Rate: <2% in critical fields
- Freshness: Data ≤5 days old
- Completeness: ≥95% expected date coverage
- Accuracy: 0 impossible values
- Availability: 99.5% daily health check pass rate

**Response Time SLAs**:
- Acknowledge: <1 hour
- Investigate: <4 hours
- Fix critical: <24 hours
- Fix warning: <1 week

---

## Change Log

| Date | Change | Owner |
|------|--------|-------|
| 2025-10-31 | Initial validation framework created | Connor |
| 2025-10-31 | Discovered Sugar data issue | Connor |
| 2025-10-31 | Added 15 enhanced weather fields to pipeline | Francisco |
| 2025-10-31 | Created continuous health check job | Connor |

---

## Contact

**Questions or Issues?**
- Data Pipeline: Francisco (@francisco)
- Validation: Connor (@connor)
- Forecast Agent: Connor (@connor)
- Trading Agent: Team

**Documentation**:
- [DATA_CONTRACTS.md](../../project_overview/DATA_CONTRACTS.md) - Schema specifications
- [Weather Lambda](../lambda/functions/weather-data-fetcher/) - Weather data ingestion
- [Unified Data SQL](../sql/create_unified_data.sql) - Silver layer transformation

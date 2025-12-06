# Research Agent Testing & Validation

This folder contains scripts for testing, validation, and monitoring of the research agent's data pipelines and tables.

## Structure

```
tests/
├── validation/          # One-time validation scripts
│   └── validate_gold_tables.py - Comprehensive 6-test validation of gold layer tables
├── health_checks/       # Ongoing health check scripts (placeholder)
└── monitoring/          # Continuous monitoring scripts (placeholder)
```

## Script Categories

### `validation/` - One-Time Validation

Scripts that validate a specific implementation or migration. Run once after major changes to verify correctness.

**Current scripts:**
- **`validate_gold_tables.py`** - Validates both gold layer tables after build
  - Checks row counts, NULL rates, missingness flags, GDELT capitalization
  - Run after building `commodity.gold.unified_data` and `commodity.gold.unified_data_raw`
  - Usage: `python research_agent/tests/validation/validate_gold_tables.py`

### `health_checks/` - Periodic Health Checks

Scripts that check ongoing data quality and pipeline health. Run periodically (daily/weekly) to catch issues.

**Planned:**
- Table freshness checks (data recency)
- NULL rate drift detection
- Row count anomaly detection

### `monitoring/` - Continuous Monitoring

Scripts for production monitoring and alerting. Designed for automation (cron, EventBridge, etc.).

**Planned:**
- Lambda function execution monitoring
- Databricks table build failures
- Data quality alerts (schema changes, unexpected patterns)

---

## Usage

### Running Validation Scripts

```bash
# Validate gold tables after rebuild
python research_agent/tests/validation/validate_gold_tables.py

# Expected output:
# ✅ All 6 tests pass
# - Row counts match (7,612)
# - Production NULL rates correct (0% for market data)
# - Raw NULL rates correct (~30% market, ~73% GDELT)
# - Missingness flags work correctly
# - GDELT commodities capitalized
```

### Adding New Scripts

1. **Determine category**: Validation (one-time), health check (periodic), or monitoring (continuous)
2. **Create script** in appropriate folder
3. **Document** in this README under the relevant section
4. **Add to automation** if health check or monitoring script

---

## Related Documentation

- **Build Instructions**: `research_agent/docs/BUILD_INSTRUCTIONS.md` - How to build gold tables
- **Data Contracts**: `docs/DATA_CONTRACTS.md` - Schema definitions
- **Gold Migration Guide**: `research_agent/docs/GOLD_MIGRATION_GUIDE.md` - Table selection and usage

---

**Last Updated**: December 5, 2024

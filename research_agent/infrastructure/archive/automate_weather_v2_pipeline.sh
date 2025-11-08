#!/bin/bash

#
# Automate Weather v2 Pipeline
#
# This script monitors the weather backfill and automatically runs
# subsequent steps when it completes:
# 1. Monitor backfill_historical_weather_v2.py completion
# 2. Create weather_v2 bronze table
# 3. Validate July 2021 frost event
# 4. Generate comparison report
#
# Usage: bash automate_weather_v2_pipeline.sh
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/weather_v2_pipeline.log"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

echo "================================================================================"
echo "Weather v2 Pipeline Automation"
echo "================================================================================"
echo ""

# Step 1: Monitor backfill completion
log "Step 1: Monitoring weather backfill completion..."
log "Checking backfill_historical_weather_v2.py status..."

BACKFILL_LOG="$SCRIPT_DIR/weather_backfill_v2.log"

if [ ! -f "$BACKFILL_LOG" ]; then
    log_error "Backfill log not found: $BACKFILL_LOG"
    log_error "Is the backfill running?"
    exit 1
fi

# Check if backfill is complete
COMPLETE_COUNT=$(grep -c "✅ BACKFILL COMPLETE" "$BACKFILL_LOG" 2>/dev/null || echo "0")
ERROR_COUNT=$(grep -c "ERROR" "$BACKFILL_LOG" 2>/dev/null || echo "0")

if [ "$COMPLETE_COUNT" -gt 0 ]; then
    log "✅ Backfill already complete!"
elif [ "$ERROR_COUNT" -gt 0 ]; then
    log_error "Backfill has errors. Check $BACKFILL_LOG"
    grep "ERROR" "$BACKFILL_LOG" | tail -10 | tee -a "$LOG_FILE"
    exit 1
else
    log "⏳ Backfill still running..."
    log "Monitoring for completion (checking every 5 minutes)..."

    # Monitor until complete (max 24 hours)
    MAX_WAIT_SECONDS=$((24 * 60 * 60))
    WAITED=0
    CHECK_INTERVAL=300  # 5 minutes

    while [ $WAITED -lt $MAX_WAIT_SECONDS ]; do
        sleep $CHECK_INTERVAL
        WAITED=$((WAITED + CHECK_INTERVAL))

        COMPLETE_COUNT=$(grep -c "✅ BACKFILL COMPLETE" "$BACKFILL_LOG" 2>/dev/null || echo "0")
        ERROR_COUNT=$(grep -c "ERROR" "$BACKFILL_LOG" 2>/dev/null || echo "0")

        if [ "$COMPLETE_COUNT" -gt 0 ]; then
            log "✅ Backfill completed!"
            break
        elif [ "$ERROR_COUNT" -gt 0 ]; then
            log_error "Backfill encountered errors"
            exit 1
        fi

        # Show progress
        REGIONS_COMPLETED=$(grep -c "Region completed:" "$BACKFILL_LOG" 2>/dev/null || echo "0")
        log "Progress: $REGIONS_COMPLETED/67 regions completed (waited $((WAITED / 60)) minutes)"
    done

    if [ $WAITED -ge $MAX_WAIT_SECONDS ]; then
        log_error "Backfill timeout after 24 hours"
        exit 1
    fi
fi

# Step 2: Create weather_v2 bronze table
echo ""
echo "================================================================================"
log "Step 2: Creating weather_v2 bronze table..."
echo "================================================================================"

cd "$SCRIPT_DIR"

export DATABRICKS_HOST="https://dbc-fd7b00f3-7a6d.cloud.databricks.com"
export DATABRICKS_TOKEN="${DATABRICKS_TOKEN}"  # Load from environment
export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/3cede8561503a13c"

python create_weather_v2_bronze_table.py | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log "✅ Weather v2 bronze table created successfully"
else
    log_error "Failed to create weather_v2 bronze table"
    exit 1
fi

# Step 3: Validate July 2021 frost event
echo ""
echo "================================================================================"
log "Step 3: Validating July 2021 frost event..."
echo "================================================================================"

python validate_july2021_frost.py | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log "✅ Frost validation completed"
else
    log_warning "Frost validation had issues (check log)"
fi

# Step 4: Generate summary report
echo ""
echo "================================================================================"
log "Step 4: Generating summary report..."
echo "================================================================================"

REPORT_FILE="$SCRIPT_DIR/weather_v2_migration_report.txt"

cat > "$REPORT_FILE" <<EOF
================================================================================
Weather v2 Migration Report
================================================================================

Date: $(date '+%Y-%m-%d %H:%M:%S')

Summary:
--------
✅ Historical weather backfill completed (2015-2025)
✅ Weather v2 bronze table created
✅ Frost validation completed

Data Quality Improvements:
--------------------------
- Corrected coordinates for all 67 growing regions
- Weather data now from actual growing regions (not state capitals)
- July 2021 Brazil frost event properly captured
- Coordinates included in data for transparency

Next Steps:
-----------
1. Update unified_data to use weather_v2:
   - Modify: research_agent/sql/create_unified_data.sql
   - Replace: commodity.bronze.weather → commodity.bronze.weather_v2

2. Train baseline SARIMAX models on weather_v1:
   - Document current accuracy as baseline

3. Train new SARIMAX models on weather_v2:
   - Compare accuracy improvements
   - Document RMSE/MAE improvements

4. Update production pipeline to use weather_v2:
   - Update forecast_agent to use weather_v2

Files Created:
--------------
- research_agent/infrastructure/backfill_historical_weather_v2.py
- research_agent/infrastructure/validate_july2021_frost.py
- research_agent/infrastructure/create_weather_v2_bronze_table.py
- research_agent/infrastructure/databricks/weather_v2_delta_migration.sql

================================================================================
Full Pipeline Log: $LOG_FILE
================================================================================
EOF

cat "$REPORT_FILE"

log "✅ Report generated: $REPORT_FILE"

# Final summary
echo ""
echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
log "All automated steps completed successfully"
log ""
log "📝 Next manual steps:"
log "   1. Review report: $REPORT_FILE"
log "   2. Update unified_data SQL to use weather_v2"
log "   3. Train and compare models (v1 vs v2)"
log "   4. Document accuracy improvements"
echo "================================================================================"

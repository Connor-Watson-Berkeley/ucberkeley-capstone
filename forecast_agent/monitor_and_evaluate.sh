#!/bin/bash

set -a && source ../infra/.env && set +a

echo "=========================================="
echo "Monitoring Backfills Until Completion"
echo "=========================================="
echo ""

# Function to check if process is still running
check_running() {
    local pid=$1
    ps -p $pid > /dev/null 2>&1
    return $?
}

# Monitor both backfills
echo "Waiting for backfills to complete..."
echo ""

# Get PIDs of running backfills (ed6498 and e8b4a0)
COFFEE_RUNNING=true
SUGAR_RUNNING=true

while $COFFEE_RUNNING || $SUGAR_RUNNING; do
    sleep 300  # Check every 5 minutes
    
    # Check Coffee progress
    if $COFFEE_RUNNING; then
        COFFEE_COUNT=$(python -c "
from databricks import sql
import os
conn = sql.connect(
    server_hostname=os.environ['DATABRICKS_HOST'].replace('https://', ''),
    http_path=os.environ['DATABRICKS_HTTP_PATH'],
    access_token=os.environ['DATABRICKS_TOKEN']
)
cursor = conn.cursor()
cursor.execute(\"SELECT COUNT(DISTINCT forecast_start_date) FROM commodity.forecast.distributions WHERE commodity='Coffee' AND model_version IN ('naive', 'sarimax_auto_weather', 'xgboost')\")
print(cursor.fetchone()[0])
conn.close()
" 2>&1 | tail -1)
        echo "$(date): Coffee forecasts: $COFFEE_COUNT"
    fi
    
    # Check Sugar progress
    if $SUGAR_RUNNING; then
        SUGAR_COUNT=$(python -c "
from databricks import sql
import os
conn = sql.connect(
    server_hostname=os.environ['DATABRICKS_HOST'].replace('https://', ''),
    http_path=os.environ['DATABRICKS_HTTP_PATH'],
    access_token=os.environ['DATABRICKS_TOKEN']
)
cursor = conn.cursor()
cursor.execute(\"SELECT COUNT(DISTINCT forecast_start_date) FROM commodity.forecast.distributions WHERE commodity='Sugar' AND model_version IN ('naive', 'sarimax_auto_weather', 'xgboost')\")
print(cursor.fetchone()[0])
conn.close()
" 2>&1 | tail -1)
        echo "$(date): Sugar forecasts: $SUGAR_COUNT"
    fi
done

echo ""
echo "=========================================="
echo "Backfills Complete! Running Evaluation..."
echo "=========================================="
echo ""

# Check final coverage
python check_backfill_coverage.py --commodity Coffee --models naive sarimax_auto_weather xgboost
python check_backfill_coverage.py --commodity Sugar --models naive sarimax_auto_weather xgboost

# Run evaluation
python evaluate_historical_forecasts.py --commodity Coffee --models naive sarimax_auto_weather xgboost
python evaluate_historical_forecasts.py --commodity Sugar --models naive sarimax_auto_weather xgboost

echo ""
echo "✅ All tasks complete!"

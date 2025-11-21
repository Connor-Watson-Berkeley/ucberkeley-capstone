#!/bin/bash

set -a && source ../infra/.env && set +a

LOG="autonomous_monitor.log"
echo "$(date): ========================================" >> $LOG
echo "$(date): Autonomous Monitor Started" >> $LOG
echo "$(date): ========================================" >> $LOG

while true; do
    echo "$(date): Checking backfill status..." >> $LOG
    
    # Check if backfills are still running
    COFFEE_RUNNING=$(ps aux | grep "ed6498\|Coffee.*backfill_rolling_window" | grep -v grep | wc -l)
    SUGAR_RUNNING=$(ps aux | grep "e8b4a0\|Sugar.*backfill_rolling_window" | grep -v grep | wc -l)
    
    echo "$(date): Coffee running: $COFFEE_RUNNING, Sugar running: $SUGAR_RUNNING" >> $LOG
    
    # Check current counts
    COFFEE_COUNT=$(python -c "
from databricks import sql
import os
try:
    conn = sql.connect(
        server_hostname=os.environ['DATABRICKS_HOST'].replace('https://', ''),
        http_path=os.environ['DATABRICKS_HTTP_PATH'],
        access_token=os.environ['DATABRICKS_TOKEN']
    )
    cursor = conn.cursor()
    cursor.execute(\"SELECT COUNT(DISTINCT forecast_start_date) FROM commodity.forecast.distributions WHERE commodity='Coffee' AND model_version IN ('naive', 'sarimax_auto_weather', 'xgboost') AND is_actuals=FALSE\")
    count = cursor.fetchone()[0]
    conn.close()
    print(count)
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1 | tail -1)
    
    SUGAR_COUNT=$(python -c "
from databricks import sql
import os
try:
    conn = sql.connect(
        server_hostname=os.environ['DATABRICKS_HOST'].replace('https://', ''),
        http_path=os.environ['DATABRICKS_HTTP_PATH'],
        access_token=os.environ['DATABRICKS_TOKEN']
    )
    cursor = conn.cursor()
    cursor.execute(\"SELECT COUNT(DISTINCT forecast_start_date) FROM commodity.forecast.distributions WHERE commodity='Sugar' AND model_version IN ('naive', 'sarimax_auto_weather', 'xgboost') AND is_actuals=FALSE\")
    count = cursor.fetchone()[0]
    conn.close()
    print(count)
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1 | tail -1)
    
    echo "$(date): Coffee forecasts: $COFFEE_COUNT, Sugar forecasts: $SUGAR_COUNT" >> $LOG
    
    # Check if both completed (target: ~2871 dates * 3 models = ~8613 per commodity)
    if [ "$COFFEE_RUNNING" -eq 0 ] && [ "$SUGAR_RUNNING" -eq 0 ]; then
        echo "$(date): Both backfills completed! Running evaluation..." >> $LOG
        
        # Run coverage check
        python check_backfill_coverage.py --commodity Coffee --models naive sarimax_auto_weather xgboost >> $LOG 2>&1
        python check_backfill_coverage.py --commodity Sugar --models naive sarimax_auto_weather xgboost >> $LOG 2>&1
        
        # Run evaluation
        echo "$(date): Evaluating Coffee..." >> $LOG
        python evaluate_historical_forecasts.py --commodity Coffee --models naive sarimax_auto_weather xgboost >> $LOG 2>&1
        
        echo "$(date): Evaluating Sugar..." >> $LOG
        python evaluate_historical_forecasts.py --commodity Sugar --models naive sarimax_auto_weather xgboost >> $LOG 2>&1
        
        # Check for errors in forecasts
        echo "$(date): Checking for forecast errors..." >> $LOG
        python -c "
from databricks import sql
import os

conn = sql.connect(
    server_hostname=os.environ['DATABRICKS_HOST'].replace('https://', ''),
    http_path=os.environ['DATABRICKS_HTTP_PATH'],
    access_token=os.environ['DATABRICKS_TOKEN']
)

cursor = conn.cursor()

# Check for NULL forecasts
cursor.execute('''
SELECT commodity, model_version, COUNT(*) as null_count
FROM commodity.forecast.distributions
WHERE is_actuals = FALSE
  AND (day_1 IS NULL OR day_7 IS NULL OR day_14 IS NULL)
GROUP BY commodity, model_version
''')

null_results = cursor.fetchall()
if null_results:
    print('⚠️  Found NULL forecasts:')
    for row in null_results:
        print(f'  {row[0]} {row[1]}: {row[2]} nulls')
else:
    print('✅ No NULL forecasts found')

# Check for unrealistic values (>$1000 or <$0)
cursor.execute('''
SELECT commodity, model_version, COUNT(*) as bad_count
FROM commodity.forecast.distributions  
WHERE is_actuals = FALSE
  AND (day_1 > 1000 OR day_1 < 0 OR day_14 > 1000 OR day_14 < 0)
GROUP BY commodity, model_version
''')

bad_results = cursor.fetchall()
if bad_results:
    print('⚠️  Found unrealistic forecasts:')
    for row in bad_results:
        print(f'  {row[0]} {row[1]}: {row[2]} bad values')
else:
    print('✅ No unrealistic forecasts found')

conn.close()
" >> $LOG 2>&1
        
        echo "$(date): ========================================" >> $LOG
        echo "$(date): ✅ ALL TASKS COMPLETE!" >> $LOG
        echo "$(date): ========================================" >> $LOG
        break
    fi
    
    # If backfills crashed, restart them
    if [ "$COFFEE_RUNNING" -eq 0 ] && [ "$COFFEE_COUNT" -lt 8000 ]; then
        echo "$(date): ⚠️  Coffee backfill crashed! Restarting..." >> $LOG
        nohup python -u backfill_rolling_window.py --commodity Coffee --models naive sarimax_auto_weather xgboost --train-frequency semiannually --start-date 2018-01-01 --end-date 2025-11-10 --model-version-tag v1.0 >> coffee_backfill.log 2>&1 &
        echo "$(date): Coffee backfill restarted" >> $LOG
    fi
    
    if [ "$SUGAR_RUNNING" -eq 0 ] && [ "$SUGAR_COUNT" -lt 8000 ]; then
        echo "$(date): ⚠️  Sugar backfill crashed! Restarting..." >> $LOG
        nohup python -u backfill_rolling_window.py --commodity Sugar --models naive sarimax_auto_weather xgboost --train-frequency semiannually --start-date 2018-01-01 --end-date 2025-11-10 --model-version-tag v1.0 >> sugar_backfill.log 2>&1 &
        echo "$(date): Sugar backfill restarted" >> $LOG
    fi
    
    # Sleep for 10 minutes before next check
    echo "$(date): Sleeping for 10 minutes..." >> $LOG
    sleep 600
done

echo "$(date): Autonomous monitor exiting" >> $LOG

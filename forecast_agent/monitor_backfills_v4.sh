#!/bin/bash
set -a && source ../infra/.env && set +a

while true; do
    echo ""
    echo "========== $(date) =========="
    
    # Check if processes are running
    coffee_running=$(ps aux | grep "17532" | grep -v grep | wc -l)
    sugar_running=$(ps aux | grep "17544" | grep -v grep | wc -l)
    
    echo "Coffee PID 17532: $coffee_running process(es)"
    echo "Sugar PID 17544: $sugar_running process(es)"
    
    # Check database counts with timeout
    python3 << 'PYEOF'
from databricks import sql
import os
import sys
import signal

def timeout_handler(signum, frame):
    print("\n⚠️  Database query timed out after 90 seconds")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(90)

try:
    connection = sql.connect(
        server_hostname=os.environ['DATABRICKS_HOST'].replace('https://', ''),
        http_path=os.environ['DATABRICKS_HTTP_PATH'],
        access_token=os.environ['DATABRICKS_TOKEN']
    )
    
    cursor = connection.cursor()
    
    print("\nCoffee Progress:")
    for model in ['naive', 'xgboost', 'sarimax_auto_weather']:
        query = f"""
        SELECT COUNT(DISTINCT forecast_start_date)
        FROM commodity.forecast.distributions
        WHERE commodity = 'Coffee'
          AND model_version = '{model}'
          AND forecast_start_date BETWEEN '2018-01-01' AND '2025-11-10'
        """
        cursor.execute(query)
        count = cursor.fetchone()[0]
        pct = (count/2871)*100
        print(f"  {model:25s}: {count:4d}/2871 ({pct:5.1f}%)")
    
    print("\nSugar Progress:")
    for model in ['naive', 'xgboost', 'sarimax_auto_weather']:
        query = f"""
        SELECT COUNT(DISTINCT forecast_start_date)
        FROM commodity.forecast.distributions
        WHERE commodity = 'Sugar'
          AND model_version = '{model}'
          AND forecast_start_date BETWEEN '2018-01-01' AND '2025-11-10'
        """
        cursor.execute(query)
        count = cursor.fetchone()[0]
        pct = (count/2871)*100
        print(f"  {model:25s}: {count:4d}/2871 ({pct:5.1f}%)")
    
    signal.alarm(0)
    connection.close()
    
except Exception as e:
    signal.alarm(0)
    print(f"\n⚠️  Database check failed: {e}")

PYEOF
    
    # Show last 5 lines of each log
    echo ""
    echo "=== Coffee Log (last 5 lines) ==="
    tail -5 coffee_backfill_fixed.log 2>/dev/null || echo "No log yet"
    
    echo ""
    echo "=== Sugar Log (last 5 lines) ==="
    tail -5 sugar_backfill_fixed.log 2>/dev/null || echo "No log yet"
    
    # Sleep for 5 minutes
    sleep 300
done

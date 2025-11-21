#!/bin/bash
set -a && source ../infra/.env && set +a

echo "Starting completion monitor v2..."
echo "Watching PIDs: 17532 (Coffee), 17544 (Sugar)"
echo "Will check every 10 minutes and run evaluation when all backfills complete"

while true; do
    result=$(python3 << 'PYEOF'
from databricks import sql
import os
import sys

try:
    connection = sql.connect(
        server_hostname=os.environ['DATABRICKS_HOST'].replace('https://', ''),
        http_path=os.environ['DATABRICKS_HTTP_PATH'],
        access_token=os.environ['DATABRICKS_TOKEN']
    )
    
    cursor = connection.cursor()
    
    total_complete = 0
    total_needed = 6 * 2871
    
    for commodity in ['Coffee', 'Sugar']:
        for model in ['naive', 'xgboost', 'sarimax_auto_weather']:
            query = f"""
            SELECT COUNT(DISTINCT forecast_start_date)
            FROM commodity.forecast.distributions
            WHERE commodity = '{commodity}'
              AND model_version = '{model}'
              AND forecast_start_date BETWEEN '2018-01-01' AND '2025-11-10'
            """
            cursor.execute(query)
            count = cursor.fetchone()[0]
            total_complete += count
    
    connection.close()
    
    print(f"{total_complete}/{total_needed}")
    
    if total_complete >= total_needed:
        sys.exit(100)
    else:
        sys.exit(0)
        
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
PYEOF
)
    
    exit_code=$?
    
    echo "[$(date)] Status: $result"
    
    if [ $exit_code -eq 100 ]; then
        echo ""
        echo "========================================="
        echo "🎉 ALL BACKFILLS COMPLETE!"
        echo "========================================="
        echo ""
        echo "Running evaluation scripts..."
        
        for commodity in Coffee Sugar; do
            for model in naive xgboost sarimax_auto_weather; do
                echo ""
                echo "Evaluating $commodity - $model..."
                python evaluate_historical_forecasts.py \
                    --commodity $commodity \
                    --model $model \
                    2>&1 | tee eval_${commodity}_${model}.log
            done
        done
        
        echo ""
        echo "========================================="
        echo "✅ EVALUATION COMPLETE"
        echo "========================================="
        
        python3 << 'PYEOF'
from databricks import sql
import os

connection = sql.connect(
    server_hostname=os.environ['DATABRICKS_HOST'].replace('https://', ''),
    http_path=os.environ['DATABRICKS_HTTP_PATH'],
    access_token=os.environ['DATABRICKS_TOKEN']
)

cursor = connection.cursor()

query = """
SELECT commodity, model_version, COUNT(*)
FROM commodity.forecast.distributions
WHERE day_1 IS NULL
GROUP BY commodity, model_version
"""
cursor.execute(query)
null_results = cursor.fetchall()

if null_results:
    print("⚠️  Found NULL forecasts:")
    for row in null_results:
        print(f"  {row[0]} - {row[1]}: {row[2]} NULL values")
else:
    print("✅ No NULL forecasts found")

query = """
SELECT commodity, model_version, COUNT(*)
FROM commodity.forecast.distributions
WHERE day_1 > 1000 OR day_1 < 0
GROUP BY commodity, model_version
"""
cursor.execute(query)
unrealistic_results = cursor.fetchall()

if unrealistic_results:
    print("⚠️  Found unrealistic forecasts:")
    for row in unrealistic_results:
        print(f"  {row[0]} - {row[1]}: {row[2]} unrealistic values")
else:
    print("✅ No unrealistic forecasts found")

connection.close()
PYEOF
        
        break
    fi
    
    # Check if processes are still running
    coffee_running=$(ps aux | grep "17532" | grep -v grep | wc -l)
    sugar_running=$(ps aux | grep "17544" | grep -v grep | wc -l)
    
    if [ $coffee_running -eq 0 ]; then
        echo "⚠️  Coffee backfill (PID 17532) appears to have stopped!"
    fi
    
    if [ $sugar_running -eq 0 ]; then
        echo "⚠️  Sugar backfill (PID 17544) appears to have stopped!"
    fi
    
    sleep 600
done

echo "Completion monitor exiting"

#!/bin/bash
# Complete Naive Model Backfill for Trading Agent
# Run this script to backfill remaining forecasts: 2023-06-11 to 2025-11-11

cd "/Users/connorwatson/Documents/Data Science/DS210/ucberkeley-capstone/forecast_agent"

# Load credentials
set -a
source ../infra/.env
set +a

echo "========================================="
echo "BACKFILL: Jun 2023 - Dec 2023"
echo "========================================="
python backfill_rolling_window.py --commodity Coffee --models naive \
  --train-frequency semiannually --start-date 2023-06-11 --end-date 2023-12-31

echo ""
echo "========================================="
echo "BACKFILL: Jan 2024 - Jun 2024"
echo "========================================="
python backfill_rolling_window.py --commodity Coffee --models naive \
  --train-frequency semiannually --start-date 2024-01-01 --end-date 2024-06-30

echo ""
echo "========================================="
echo "BACKFILL: Jul 2024 - Dec 2024"
echo "========================================="
python backfill_rolling_window.py --commodity Coffee --models naive \
  --train-frequency semiannually --start-date 2024-07-01 --end-date 2024-12-31

echo ""
echo "========================================="
echo "BACKFILL: Jan 2025 - Nov 2025"
echo "========================================="
python backfill_rolling_window.py --commodity Coffee --models naive \
  --train-frequency semiannually --start-date 2025-01-01 --end-date 2025-11-11

echo ""
echo "========================================="
echo "BACKFILL COMPLETE!"
echo "========================================="

# Check final status
python -c "
from databricks import sql
import os
connection = sql.connect(
    server_hostname=os.getenv('DATABRICKS_HOST').replace('https://', ''),
    http_path=os.getenv('DATABRICKS_HTTP_PATH'),
    access_token=os.getenv('DATABRICKS_TOKEN')
)
cursor = connection.cursor()
cursor.execute('''
SELECT COUNT(DISTINCT forecast_start_date) as dates,
       MIN(forecast_start_date) as first,
       MAX(forecast_start_date) as last
FROM commodity.forecast.distributions
WHERE commodity = \"Coffee\" AND model_version = \"naive\"
''')
dates, first, last = cursor.fetchone()
print(f'\n✅ Final Status: {dates:,} forecast dates')
print(f'   Range: {first} to {last}')
cursor.close()
connection.close()
"

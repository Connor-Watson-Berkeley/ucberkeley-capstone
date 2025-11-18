#!/usr/bin/env python3
"""Check backfill coverage by model."""

import os
from databricks import sql

connection = sql.connect(
    server_hostname=os.getenv('DATABRICKS_HOST'),
    http_path=os.getenv('DATABRICKS_HTTP_PATH'),
    access_token=os.getenv('DATABRICKS_TOKEN')
)
cursor = connection.cursor()

print('📊 Current Backfill Coverage by Model:')
print()
print(f"{'Model':<25} {'Total':<8} {'Has Metrics':<12} {'Date Range':<40}")
print('-' * 90)

cursor.execute('''
SELECT
    model_version,
    COUNT(*) as total_forecasts,
    COUNT(CASE WHEN mae_14d IS NOT NULL THEN 1 END) as has_metrics,
    MIN(forecast_start_date) as earliest,
    MAX(forecast_start_date) as latest
FROM commodity.forecast.forecast_metadata
WHERE commodity = 'Coffee'
GROUP BY model_version
ORDER BY total_forecasts DESC, model_version
''')

for row in cursor.fetchall():
    model = row[0]
    total = row[1]
    has_metrics = row[2]
    date_range = f'{row[3]} to {row[4]}' if row[3] else 'N/A'
    print(f'{model:<25} {total:<8} {has_metrics:<12} {date_range:<40}')

print()
print('💡 Note: "Has Metrics" shows how many forecasts have been evaluated (have MAE)')

cursor.close()
connection.close()

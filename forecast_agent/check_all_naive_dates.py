#!/usr/bin/env python3
"""Check all dates for naive model"""
import os
from databricks import sql

connection = sql.connect(
    server_hostname=os.environ['DATABRICKS_HOST'],
    http_path=os.environ['DATABRICKS_HTTP_PATH'],
    access_token=os.environ['DATABRICKS_TOKEN']
)
cursor = connection.cursor()

# Check all naive forecasts
cursor.execute('''
SELECT
    forecast_start_date,
    COUNT(*) as num_paths
FROM commodity.forecast.distributions
WHERE model_version = 'naive'
  AND is_actuals = FALSE
  AND commodity = 'Coffee'
GROUP BY forecast_start_date
ORDER BY forecast_start_date
''')

print('📊 All Naive Forecasts for Coffee:')
print()
rows = cursor.fetchall()
for row in rows:
    print(f'  {row[0]} | {row[1]:5} paths')

print(f'\nTotal: {len(rows)} forecast dates')

cursor.close()
connection.close()

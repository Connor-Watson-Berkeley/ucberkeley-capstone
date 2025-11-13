#!/usr/bin/env python3
"""Verify backfill data in distributions table"""
import os
from databricks import sql

connection = sql.connect(
    server_hostname=os.environ['DATABRICKS_HOST'],
    http_path=os.environ['DATABRICKS_HTTP_PATH'],
    access_token=os.environ['DATABRICKS_TOKEN']
)
cursor = connection.cursor()

# Check the 3 test forecasts we just inserted
cursor.execute('''
SELECT
    forecast_start_date,
    model_version,
    COUNT(*) as num_paths,
    AVG(day_1) as avg_day1,
    AVG(day_14) as avg_day14
FROM commodity.forecast.distributions
WHERE model_version = 'naive'
  AND is_actuals = FALSE
  AND forecast_start_date BETWEEN '2018-01-01' AND '2018-01-03'
GROUP BY forecast_start_date, model_version
ORDER BY forecast_start_date
''')

print('📊 Test Backfill Verification:')
print()
for row in cursor.fetchall():
    print(f'  {row[0]} | {row[1]:10} | {row[2]:5} paths | day_1: ${row[3]:.2f} | day_14: ${row[4]:.2f}')

cursor.close()
connection.close()

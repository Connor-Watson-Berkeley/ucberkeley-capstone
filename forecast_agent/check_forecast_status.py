#!/usr/bin/env python3
"""Quick check of what forecasts and metrics we have."""

import os
from databricks import sql

connection = sql.connect(
    server_hostname=os.getenv('DATABRICKS_HOST'),
    http_path=os.getenv('DATABRICKS_HTTP_PATH'),
    access_token=os.getenv('DATABRICKS_TOKEN')
)
cursor = connection.cursor()

# Check forecast_metadata table
print('📊 Checking forecast_metadata table:')
print()
try:
    cursor.execute('''
    SELECT
        commodity,
        model_version,
        COUNT(*) as num_forecasts,
        AVG(mae_1d) as avg_mae_1d,
        AVG(mae_14d) as avg_mae_14d,
        COUNT(CASE WHEN mae_14d IS NOT NULL THEN 1 END) as has_metrics
    FROM commodity.forecast.forecast_metadata
    GROUP BY commodity, model_version
    ORDER BY commodity, model_version
    LIMIT 20
    ''')
    rows = cursor.fetchall()
    if rows:
        print(f"{'Commodity':<10} {'Model':<20} {'Forecasts':<10} {'MAE 1d':<10} {'MAE 14d':<10} {'Has Metrics':<12}")
        print('-' * 85)
        for row in rows:
            mae_1d = f'{row[3]:.2f}' if row[3] else 'NULL'
            mae_14d = f'{row[4]:.2f}' if row[4] else 'NULL'
            print(f'{row[0]:<10} {row[1]:<20} {row[2]:<10} {mae_1d:<10} {mae_14d:<10} {row[5]:<12}')
    else:
        print('  ⚠️  Table exists but has no data')
except Exception as e:
    print(f'  ⚠️  Table may not exist or error: {str(e)[:100]}')

cursor.close()
connection.close()

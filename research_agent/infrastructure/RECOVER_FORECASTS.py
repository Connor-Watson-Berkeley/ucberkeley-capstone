#!/usr/bin/env python3
"""
EMERGENCY: Recover dropped forecast tables from S3 Unity Storage

The forecast tables were dropped but underlying parquet files may still exist.
This script attempts to recreate the tables from S3 data.
"""
import os
from databricks import sql
from dotenv import load_dotenv

env_path = '../../infra/.env'
load_dotenv(env_path)

token = os.getenv('DATABRICKS_TOKEN')
server_hostname = os.getenv('DATABRICKS_HOST').replace('https://', '')
http_path = os.getenv('DATABRICKS_HTTP_PATH')

print("="*80)
print("EMERGENCY FORECAST RECOVERY")
print("="*80)
print()

connection = sql.connect(
    server_hostname=server_hostname,
    http_path=http_path,
    access_token=token
)
cursor = connection.cursor()

# List all files in Unity Storage for forecast schema
print("1. Searching for forecast data in S3 Unity Storage...")
print()

# Try to find the S3 location for the forecast schema
try:
    cursor.execute("DESCRIBE SCHEMA EXTENDED commodity.forecast")
    schema_info = cursor.fetchall()

    print("Schema info:")
    for row in schema_info:
        print(f"  {row[0]}: {row[1]}")
        if row[0] == 'Location':
            storage_location = row[1]
            print(f"\n  📂 Storage location: {storage_location}")
    print()
except Exception as e:
    print(f"  ERROR: {e}")
    print()

# Check what tables currently exist
print("2. Current tables in commodity.forecast:")
cursor.execute("SHOW TABLES IN commodity.forecast")
existing_tables = cursor.fetchall()

if existing_tables:
    for table in existing_tables:
        print(f"  - {table[1]}")
else:
    print("  (none - all dropped)")
print()

print("="*80)
print("RECOVERY OPTIONS")
print("="*80)
print()
print("Option 1: RESTORE FROM S3 PARQUET FILES")
print("  - Unity Storage may still have the parquet files")
print("  - We saw files in __unitystorage/.../tables/5f1fb9ac-2b09-4e8d-ba7e-7d4788cea299/")
print("  - Can recreate tables by reading these files directly")
print()
print("Option 2: REGENERATE FROM FORECAST_AGENT")
print("  - Run forecast_agent/ground_truth/training/train_baseline_models.py")
print("  - This was planned for Phase 7 anyway (with v2 weather data)")
print("  - Takes 1-2 hours but uses correct coordinates")
print()
print("Option 3: CHECK DATABRICKS RECYCLE BIN (if enabled)")
print("  - Some Databricks workspaces have a 7-day retention")
print("  - Go to Data Explorer > Recycle Bin")
print()

cursor.close()
connection.close()

print("="*80)
print("RECOMMENDED: Try Option 1 first (restore from S3)")
print("="*80)

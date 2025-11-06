"""
Backfill historical weather data for all regions (Coffee + Sugar).

This script invokes the weather-data-fetcher Lambda with a date range
covering 2015-01-01 to today.
"""

import boto3
import json
from datetime import datetime, date

# AWS Configuration
LAMBDA_FUNCTION = 'weather-data-fetcher'
AWS_REGION = 'us-west-2'

# Calculate days from 2015-01-01 to today
START_DATE = date(2015, 1, 1)
TODAY = date.today()
DAYS_AGO = (TODAY - START_DATE).days

print("="*80)
print("WEATHER DATA HISTORICAL BACKFILL")
print("="*80)
print(f"\nTarget Lambda: {LAMBDA_FUNCTION}")
print(f"Region: {AWS_REGION}")
print(f"Backfill Range: {START_DATE} to {TODAY}")
print(f"Days to fetch: {DAYS_AGO} days")
print(f"\nThis will fetch weather data for:")
print(f"  - 29 Coffee regions")
print(f"  - 38 Sugar regions")
print(f"  - {DAYS_AGO} days per region")
print(f"  - Total: ~{67 * DAYS_AGO:,} rows")

# Confirm
import sys
if '--confirm' not in sys.argv:
    response = input("\nProceed with weather backfill? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Backfill cancelled.")
        exit(0)
else:
    print("\n--confirm flag detected, proceeding...")

lambda_client = boto3.client('lambda', region_name=AWS_REGION)

# Invoke Lambda with historical date range
# The Lambda uses date_range_from_days_ago() which gets min/max from the list
# So we just pass [0, DAYS_AGO] to get the full range (not the entire list!)
payload = {
    'days_to_fetch': [0, DAYS_AGO]  # 0 = today, DAYS_AGO = 2015-01-01
}

print("\n" + "-"*80)
print("Invoking Lambda...")
print("-"*80)
print(f"  Payload: days_to_fetch = [0, {DAYS_AGO}] (covers {START_DATE} to {TODAY})")
print(f"  Expected duration: 5-10 minutes (API rate limits)")

try:
    response = lambda_client.invoke(
        FunctionName=LAMBDA_FUNCTION,
        InvocationType='Event',  # Asynchronous - don't wait for response
        Payload=json.dumps(payload)
    )

    status_code = response['StatusCode']

    print(f"\n✓ Lambda invoked asynchronously")
    print(f"  Status Code: {status_code}")

    if status_code == 202:
        print("\n✅ Weather backfill started successfully (running in background)")
        print("\nMonitoring:")
        print("  1. Check Lambda logs in CloudWatch (region: us-west-2)")
        print("  2. Expected duration: 10-15 minutes")
        print("  3. Check S3 after completion:")
        print("     aws s3 ls s3://groundtruth-capstone/landing/weather_data/")
    else:
        print(f"\n⚠️ Unexpected status code: {status_code}")

except Exception as e:
    print(f"\n❌ Error invoking Lambda: {e}")
    exit(1)

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. Wait 2-3 minutes for S3 upload to complete")
print("\n2. Rebuild Databricks layers:")
print("   cd research_agent/infrastructure")
print("   python rebuild_all_layers.py")
print("\n3. Validate Sugar data:")
print("   Expected: ~140,000 rows in unified_data")
print("   Coffee: ~75,000 rows (unchanged)")
print("   Sugar: ~140,000 rows (was 380)")
print("\n" + "="*80)

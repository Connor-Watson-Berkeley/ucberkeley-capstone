"""
Backfill weather data in yearly chunks to avoid Lambda timeout.

Strategy: Break 2015-2025 into yearly chunks and invoke Lambda once per year.
"""

import boto3
import json
import time
from datetime import date

# AWS Configuration
LAMBDA_FUNCTION = 'weather-data-fetcher'
AWS_REGION = 'us-west-2'

# Calculate yearly chunks
START_YEAR = 2015
END_YEAR = 2025
TODAY = date.today()

print("="*80)
print("WEATHER DATA HISTORICAL BACKFILL - CHUNKED APPROACH")
print("="*80)
print(f"\nTarget Lambda: {LAMBDA_FUNCTION}")
print(f"Region: {AWS_REGION}")
print(f"Strategy: Yearly chunks from {START_YEAR} to {END_YEAR}")
print(f"Expected: ~26,500 rows per year (67 regions × ~365 days)")

# Generate yearly chunks
chunks = []
for year in range(START_YEAR, END_YEAR + 1):
    # Calculate days ago for start and end of each year
    year_start = date(year, 1, 1)
    year_end = date(year, 12, 31) if year < END_YEAR else TODAY

    days_ago_start = (TODAY - year_start).days
    days_ago_end = (TODAY - year_end).days

    # Lambda expects [min_days, max_days] and gets min/max from that list
    # days_ago_end is smaller (more recent), days_ago_start is larger (further back)
    chunks.append({
        'year': year,
        'days_to_fetch': [days_ago_end, days_ago_start],
        'date_range': f"{year_start} to {year_end}"
    })

print(f"\nWill process {len(chunks)} chunks:")
for chunk in chunks:
    print(f"  {chunk['year']}: days_to_fetch = {chunk['days_to_fetch']} ({chunk['date_range']})")

# Confirm
import sys
if '--confirm' not in sys.argv:
    response = input("\nProceed with chunked backfill? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Backfill cancelled.")
        exit(0)
else:
    print("\n--confirm flag detected, proceeding...")

lambda_client = boto3.client('lambda', region_name=AWS_REGION)

print("\n" + "="*80)
print("INVOKING LAMBDA FOR EACH CHUNK")
print("="*80)

successful = []
failed = []

for i, chunk in enumerate(chunks, 1):
    year = chunk['year']
    payload = {'days_to_fetch': chunk['days_to_fetch']}

    print(f"\n[{i}/{len(chunks)}] Processing {year}...")
    print(f"  Date range: {chunk['date_range']}")
    print(f"  Payload: {payload}")

    try:
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION,
            InvocationType='Event',  # Asynchronous
            Payload=json.dumps(payload)
        )

        status_code = response['StatusCode']

        if status_code == 202:
            print(f"  ✅ Invoked successfully (running in background)")
            successful.append(year)
        else:
            print(f"  ⚠️ Unexpected status: {status_code}")
            failed.append(year)

        # Wait 5 seconds between invocations to avoid rate limits
        if i < len(chunks):
            print(f"  Waiting 5 seconds before next chunk...")
            time.sleep(5)

    except Exception as e:
        print(f"  ❌ Error: {e}")
        failed.append(year)

# Summary
print("\n" + "="*80)
print("BACKFILL INVOCATION SUMMARY")
print("="*80)

print(f"\n✅ Successful: {len(successful)}/{len(chunks)} chunks")
if successful:
    print(f"   Years: {', '.join(map(str, successful))}")

if failed:
    print(f"\n❌ Failed: {len(failed)}/{len(chunks)} chunks")
    print(f"   Years: {', '.join(map(str, failed))}")

print("\n" + "="*80)
print("MONITORING & NEXT STEPS")
print("="*80)

print("\n1. Check Lambda logs (each chunk will complete separately):")
print("   aws logs tail /aws/lambda/weather-data-fetcher --since 20m --region us-west-2")

print("\n2. Monitor S3 for new files (expect ~11 files, one per year):")
print("   aws s3 ls s3://groundtruth-capstone/landing/weather_data/ | grep historical")

print("\n3. Expected completion time:")
print(f"   - Each chunk: 5-10 minutes")
print(f"   - Total: {len(chunks) * 7} minutes (~{len(chunks) * 7 // 60} hours)")

print("\n4. Once all chunks complete:")
print("   cd research_agent/infrastructure")
print("   python rebuild_all_layers.py")

print("\n5. Expected final result:")
print("   - Coffee: ~79,000 rows (unchanged)")
print("   - Sugar: ~265,000 rows (was 380, now full 2015-2025)")
print("   - unified_data: Sugar ~140,000 rows (38 regions × 3,770 days)")

print("\n" + "="*80)

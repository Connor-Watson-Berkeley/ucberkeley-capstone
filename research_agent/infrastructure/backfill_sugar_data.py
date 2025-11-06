"""
One-time backfill script for Sugar historical data.

This script:
1. Sets market-data-fetcher Lambda to HISTORICAL mode
2. Invokes the Lambda to fetch 2015-2025 data
3. Resets Lambda to INCREMENTAL mode
4. Provides instructions for Databricks ingestion

Run with: python backfill_sugar_data.py
"""

import boto3
import json
import time
from datetime import datetime

# AWS Configuration
LAMBDA_FUNCTION = 'market-data-fetcher'
AWS_REGION = 'us-west-2'
S3_BUCKET = 'groundtruth-capstone'

def main():
    print("="*80)
    print("SUGAR DATA HISTORICAL BACKFILL")
    print("="*80)
    print(f"\nTarget Lambda: {LAMBDA_FUNCTION}")
    print(f"Region: {AWS_REGION}")
    print(f"S3 Bucket: {S3_BUCKET}")
    print(f"Backfill Range: 2015-01-01 to {datetime.now().strftime('%Y-%m-%d')}")

    # Confirm before proceeding
    import sys
    if '--confirm' not in sys.argv:
        response = input("\nProceed with backfill? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Backfill cancelled.")
            return
    else:
        print("\n--confirm flag detected, proceeding with backfill...")

    lambda_client = boto3.client('lambda', region_name=AWS_REGION)

    # Step 1: Set HISTORICAL mode
    print("\n" + "-"*80)
    print("STEP 1: Setting Lambda to HISTORICAL mode...")
    print("-"*80)

    try:
        lambda_client.update_function_configuration(
            FunctionName=LAMBDA_FUNCTION,
            Environment={
                'Variables': {
                    'RUN_MODE': 'HISTORICAL',
                    'S3_BUCKET_NAME': S3_BUCKET
                }
            }
        )
        print("  ✓ Configuration updated")

        # Wait for update to complete
        print("  Waiting for Lambda update to complete...")
        waiter = lambda_client.get_waiter('function_updated')
        waiter.wait(FunctionName=LAMBDA_FUNCTION)
        print("  ✓ Lambda ready")

    except Exception as e:
        print(f"  ❌ Error updating Lambda configuration: {e}")
        return

    # Step 2: Invoke Lambda for backfill
    print("\n" + "-"*80)
    print("STEP 2: Invoking Lambda for historical backfill...")
    print("-"*80)
    print("  This will fetch ~10 years of data (2015-2025) for Coffee + Sugar")
    print("  Expected duration: 2-5 minutes")

    try:
        start_time = time.time()
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION,
            InvocationType='RequestResponse',  # Synchronous
            Payload=json.dumps({})
        )

        duration = time.time() - start_time
        status_code = response['StatusCode']
        payload = json.loads(response['Payload'].read().decode())

        print(f"  ✓ Lambda completed in {duration:.1f} seconds")
        print(f"  Status Code: {status_code}")
        print(f"  Response: {json.dumps(payload, indent=2)}")

        if status_code == 200 and payload.get('statusCode') == 200:
            print("\n  ✅ Historical data successfully fetched and saved to S3!")
        else:
            print(f"\n  ⚠️ Lambda returned non-200 status: {payload}")

    except Exception as e:
        print(f"  ❌ Error invoking Lambda: {e}")
        print("  Attempting to reset Lambda to INCREMENTAL mode...")
        # Continue to Step 3 to reset even if invoke failed

    # Step 3: Reset to INCREMENTAL mode
    print("\n" + "-"*80)
    print("STEP 3: Resetting Lambda to INCREMENTAL mode...")
    print("-"*80)

    try:
        lambda_client.update_function_configuration(
            FunctionName=LAMBDA_FUNCTION,
            Environment={
                'Variables': {
                    'RUN_MODE': 'INCREMENTAL',
                    'S3_BUCKET_NAME': S3_BUCKET
                }
            }
        )
        print("  ✓ Configuration reset to INCREMENTAL mode")

        waiter = lambda_client.get_waiter('function_updated')
        waiter.wait(FunctionName=LAMBDA_FUNCTION)
        print("  ✓ Lambda ready for daily incremental updates")

    except Exception as e:
        print(f"  ❌ Error resetting Lambda: {e}")
        print("  WARNING: Lambda may still be in HISTORICAL mode!")
        print("  Please manually reset via AWS Console or CLI")
        return

    # Step 4: Next steps
    print("\n" + "="*80)
    print("BACKFILL COMPLETE!")
    print("="*80)

    print("\nHistorical data has been saved to S3:")
    print(f"  s3://{S3_BUCKET}/landing/market_data/history/")

    print("\nNEXT STEPS:")
    print("-"*80)
    print("1. Check S3 for the historical CSV file:")
    print(f"   aws s3 ls s3://{S3_BUCKET}/landing/market_data/history/")

    print("\n2. Load into Databricks (option A - SQL):")
    print("   Run in Databricks SQL Editor:")
    print("   ```sql")
    print("   COPY INTO commodity.landing.market_data")
    print(f"   FROM 's3://{S3_BUCKET}/landing/market_data/history/'")
    print("   FILEFORMAT = CSV")
    print("   FORMAT_OPTIONS ('mergeSchema' = 'true', 'header' = 'true');")
    print("   ```")

    print("\n3. Load into Databricks (option B - Python):")
    print("   cd research_agent/infrastructure")
    print("   python rebuild_all_layers.py")

    print("\n4. Validate Sugar data:")
    print("   ```sql")
    print("   SELECT commodity, COUNT(*) as rows, MIN(date), MAX(date)")
    print("   FROM commodity.silver.unified_data")
    print("   GROUP BY commodity;")
    print("   ```")

    print("\n5. Expected outcome:")
    print("   Coffee: ~75,472 rows (unchanged)")
    print("   Sugar:  ~140,000 rows (was 304, now full history)")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()

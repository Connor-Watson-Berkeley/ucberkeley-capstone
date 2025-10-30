#!/usr/bin/env python3
"""
Create Databricks Unity Catalog schemas via SQL API
This approach is more reliable than running a full notebook for initial setup
"""
import requests
import time
import json
import os

DATABRICKS_HOST = "https://dbc-fd7b00f3-7a6d.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")  # Set via environment variable
WAREHOUSE_ID = "3cede8561503a13c"  # Serverless Starter Warehouse

def execute_sql(sql_query, wait_for_result=True):
    """Execute SQL query via Databricks SQL API"""

    url = f"{DATABRICKS_HOST}/api/2.0/sql/statements/"
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "statement": sql_query,
        "warehouse_id": WAREHOUSE_ID,
        "wait_timeout": "30s"
    }

    print(f"\nExecuting SQL:")
    print(f"  {sql_query[:100]}...")

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code in [200, 201]:
        result = response.json()
        status = result.get('status', {}).get('state')
        print(f"  Status: {status}")

        if 'manifest' in result:
            rows = result.get('result', {}).get('data_array', [])
            print(f"  Rows returned: {len(rows)}")

        return True, result
    else:
        print(f"  ✗ Failed: {response.status_code}")
        print(f"  Response: {response.text}")
        return False, response.text

def create_unity_catalog_structure():
    """Create the commodity catalog and schemas"""

    print("="*60)
    print("Creating Unity Catalog Structure")
    print("="*60)

    # Step 1: Create catalog
    print("\n[1/4] Creating commodity catalog...")
    success, result = execute_sql("""
        CREATE CATALOG IF NOT EXISTS commodity
        COMMENT 'Commodity price forecasting data catalog'
    """)

    if not success:
        print("✗ Failed to create catalog")
        return False

    # Step 2: Use catalog
    print("\n[2/4] Using commodity catalog...")
    success, result = execute_sql("USE CATALOG commodity")

    if not success:
        print("✗ Failed to use catalog")
        return False

    # Step 3: Create schemas
    print("\n[3/4] Creating schemas...")

    schemas = [
        ("bronze", "Bronze layer - raw data from S3 (market, VIX, weather, GDELT)"),
        ("silver", "Silver layer - cleaned and joined data for forecasting"),
        ("landing", "Landing layer - raw ingestion from Lambda functions")
    ]

    for schema_name, comment in schemas:
        success, result = execute_sql(f"""
            CREATE SCHEMA IF NOT EXISTS commodity.{schema_name}
            COMMENT '{comment}'
        """)

        if not success:
            print(f"✗ Failed to create schema: {schema_name}")
            return False
        print(f"  ✓ Created schema: commodity.{schema_name}")

    # Step 4: Verify
    print("\n[4/4] Verifying schemas...")
    success, result = execute_sql("SHOW SCHEMAS IN commodity")

    if success:
        print("\n✓ Unity Catalog structure created successfully!")
        return True
    else:
        print("✗ Failed to verify schemas")
        return False

def show_tables():
    """Show tables in bronze schema"""
    print("\nChecking existing tables in commodity.bronze...")
    success, result = execute_sql("SHOW TABLES IN commodity.bronze")
    return success

if __name__ == "__main__":

    print("="*60)
    print("Databricks Unity Catalog Setup via SQL API")
    print("="*60)

    # Create catalog and schemas
    if create_unity_catalog_structure():
        print("\n" + "="*60)
        print("Setup Complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run the Auto Loader notebook to ingest S3 data")
        print("2. Create bronze views")
        print("3. Set up daily refresh job")

        # Show existing tables
        show_tables()
    else:
        print("\n" + "="*60)
        print("Setup Failed")
        print("="*60)
        print("\nTroubleshooting:")
        print("1. Check cluster status")
        print("2. Verify Unity Catalog permissions")
        print("3. Try creating catalog manually in Databricks UI")

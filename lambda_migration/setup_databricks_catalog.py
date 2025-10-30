#!/usr/bin/env python3
"""
Set up Databricks catalog, schemas, and bronze layer via SQL API
"""
import requests
import time
import os
import sys

DATABRICKS_HOST = "https://dbc-fd7b00f3-7a6d.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
WAREHOUSE_ID = "3cede8561503a13c"  # Serverless Starter Warehouse

if not DATABRICKS_TOKEN:
    # Try reading from infra/.databrickscfg
    try:
        with open("../infra/.databrickscfg", "r") as f:
            for line in f:
                if line.startswith("token"):
                    DATABRICKS_TOKEN = line.split("=")[1].strip()
                    break
    except:
        print("ERROR: DATABRICKS_TOKEN not found in environment or config file")
        sys.exit(1)

def execute_sql(sql_query, wait_timeout=60):
    """Execute SQL query via Databricks SQL API"""
    
    url = f"{DATABRICKS_HOST}/api/2.0/sql/statements/"
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "statement": sql_query,
        "warehouse_id": WAREHOUSE_ID,
        "wait_timeout": f"{wait_timeout}s"
    }
    
    print(f"\n{'='*60}")
    print(f"SQL: {sql_query[:200]}...")
    print(f"{'='*60}")
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code in [200, 201]:
        result = response.json()
        status = result.get('status', {}).get('state')
        
        if status == 'SUCCEEDED':
            print(f"✓ SUCCESS")
            if 'manifest' in result:
                rows = result.get('result', {}).get('data_array', [])
                if rows:
                    print(f"  Rows: {len(rows)}")
                    for row in rows[:5]:  # Show first 5 rows
                        print(f"    {row}")
            return True, result
        elif status == 'FAILED':
            error = result.get('status', {}).get('error', {})
            print(f"✗ FAILED: {error}")
            return False, result
        else:
            print(f"⚠ Status: {status}")
            return False, result
    else:
        print(f"✗ HTTP Error: {response.status_code}")
        print(f"  {response.text}")
        return False, response.text

def main():
    print("="*60)
    print("Databricks Catalog Setup")
    print("="*60)
    
    # Step 1: Use commodity catalog
    print("\n[1/4] Using commodity catalog...")
    success, _ = execute_sql("USE CATALOG commodity")
    if not success:
        print("ERROR: Cannot use commodity catalog. Does it exist?")
        print("Try creating it first: CREATE CATALOG IF NOT EXISTS commodity")
        return False
    
    # Step 2: Create schemas
    print("\n[2/4] Creating schemas...")
    
    schemas = [
        ("bronze", "Bronze layer - raw data with deduplication views"),
        ("silver", "Silver layer - cleaned and joined data"),
        ("landing", "Landing layer - raw ingestion from S3")
    ]
    
    for schema_name, comment in schemas:
        sql = f"CREATE SCHEMA IF NOT EXISTS commodity.{schema_name} COMMENT '{comment}'"
        success, _ = execute_sql(sql)
        if success:
            print(f"  ✓ commodity.{schema_name}")
        else:
            print(f"  ✗ Failed: commodity.{schema_name}")
            return False
    
    # Step 3: Verify schemas exist
    print("\n[3/4] Verifying schemas...")
    success, result = execute_sql("SHOW SCHEMAS IN commodity")
    
    if success:
        print("\n✓ Schemas created successfully!")
        print("\nNext: Run Auto Loader notebook to ingest S3 data")
        return True
    else:
        print("\n✗ Schema verification failed")
        return False

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)

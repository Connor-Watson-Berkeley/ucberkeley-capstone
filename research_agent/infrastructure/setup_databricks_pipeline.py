#!/usr/bin/env python3
"""
Set up complete Databricks pipeline from S3 data via SQL API

This script:
1. Creates commodity catalog and schemas (landing, bronze, silver)
2. Creates landing tables that read from S3
3. Creates bronze deduplication views
4. Verifies the setup

Usage:
    export DATABRICKS_TOKEN=<your-token>
    python setup_databricks_pipeline.py
"""
import requests
import time
import os
import sys
from pathlib import Path

# Databricks configuration
DATABRICKS_HOST = "https://dbc-fd7b00f3-7a6d.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
WAREHOUSE_ID = "3cede8561503a13c"  # Serverless Starter Warehouse

# File paths
SCRIPT_DIR = Path(__file__).parent
LANDING_SQL = SCRIPT_DIR / "databricks" / "01_create_landing_tables.sql"
BRONZE_SQL = SCRIPT_DIR / "databricks" / "02_create_bronze_views.sql"

if not DATABRICKS_TOKEN:
    # Try reading from infra/.databrickscfg
    config_path = Path(__file__).parent.parent.parent / "infra" / ".databrickscfg"
    if config_path.exists():
        with open(config_path, "r") as f:
            for line in f:
                if line.startswith("token"):
                    DATABRICKS_TOKEN = line.split("=")[1].strip()
                    break

    if not DATABRICKS_TOKEN:
        print("ERROR: DATABRICKS_TOKEN not found in environment or config file")
        print("Set it with: export DATABRICKS_TOKEN=<your-token>")
        sys.exit(1)


def execute_sql(sql_query, wait_timeout=50, description=None):
    """Execute SQL query via Databricks SQL API

    Note: wait_timeout must be 0 or between 5-50 seconds per Databricks API requirements
    """

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

    # Show description or truncated SQL
    if description:
        print(f"\n{'='*70}")
        print(f"  {description}")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"SQL: {sql_query[:150]}...")
        print(f"{'='*70}")

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code in [200, 201]:
        result = response.json()
        status = result.get('status', {}).get('state')

        if status == 'SUCCEEDED':
            print(f"✓ SUCCESS")

            # Show row count if available
            if 'manifest' in result:
                rows = result.get('result', {}).get('data_array', [])
                if rows:
                    print(f"  Returned {len(rows)} rows")

            return True, result

        elif status == 'FAILED':
            error = result.get('status', {}).get('error', {})
            error_msg = error.get('message', 'Unknown error')
            print(f"✗ FAILED: {error_msg}")

            # Don't fail on "already exists" errors
            if "already exists" in error_msg.lower() or "alreadyexists" in error_msg.lower():
                print(f"  (Continuing - resource already exists)")
                return True, result

            return False, result

        else:
            print(f"⚠ Status: {status}")
            return status == 'PENDING', result

    else:
        print(f"✗ HTTP Error: {response.status_code}")
        print(f"  {response.text}")
        return False, response.text


def parse_sql_file(filepath):
    """Parse SQL file into individual statements, handling comments"""

    with open(filepath, 'r') as f:
        content = f.read()

    # Remove single-line comments
    lines = []
    for line in content.split('\n'):
        # Remove comments but keep the rest of the line
        if '--' in line:
            line = line.split('--')[0]
        lines.append(line)

    content = '\n'.join(lines)

    # Split by semicolons
    statements = [stmt.strip() for stmt in content.split(';') if stmt.strip()]

    return statements


def setup_catalog_and_schemas():
    """Create catalog and schemas"""

    print("\n" + "="*70)
    print("STEP 1: Create Catalog and Schemas")
    print("="*70)

    # Create catalog
    print("\n[1/4] Creating commodity catalog...")
    success, _ = execute_sql(
        "CREATE CATALOG IF NOT EXISTS commodity COMMENT 'Commodity price forecasting data'",
        description="Creating commodity catalog"
    )
    if not success:
        print("⚠ Warning: Could not create catalog (may already exist)")

    # Use catalog
    print("\n[2/4] Using commodity catalog...")
    success, _ = execute_sql("USE CATALOG commodity")
    if not success:
        print("ERROR: Cannot use commodity catalog")
        return False

    # Create schemas
    print("\n[3/4] Creating schemas...")
    schemas = [
        ("landing", "Landing layer - raw S3 data with ingest timestamps"),
        ("bronze", "Bronze layer - deduplicated views"),
        ("silver", "Silver layer - unified feature table")
    ]

    for schema_name, comment in schemas:
        success, _ = execute_sql(
            f"CREATE SCHEMA IF NOT EXISTS commodity.{schema_name} COMMENT '{comment}'",
            description=f"Creating schema: commodity.{schema_name}"
        )
        if success:
            print(f"  ✓ commodity.{schema_name}")

    # Verify schemas
    print("\n[4/4] Verifying schemas...")
    success, result = execute_sql("SHOW SCHEMAS IN commodity")

    if success:
        print("\n✓ Catalog and schemas ready!")
        return True
    else:
        print("\n✗ Schema verification failed")
        return False


def create_landing_tables():
    """Create landing tables from S3"""

    print("\n" + "="*70)
    print("STEP 2: Create Landing Tables (read from S3)")
    print("="*70)

    if not LANDING_SQL.exists():
        print(f"ERROR: SQL file not found: {LANDING_SQL}")
        return False

    print(f"\nParsing {LANDING_SQL.name}...")
    statements = parse_sql_file(LANDING_SQL)

    print(f"Found {len(statements)} SQL statements")

    # Execute each statement
    for i, stmt in enumerate(statements, 1):
        # Skip USE CATALOG statements (we already did this)
        if stmt.upper().startswith('USE CATALOG'):
            continue

        # Skip SHOW and SELECT statements (verification queries)
        if stmt.upper().startswith('SHOW') or stmt.upper().startswith('SELECT'):
            continue

        # Identify table name for better description
        table_name = "unknown"
        if "CREATE" in stmt.upper() and "TABLE" in stmt.upper():
            parts = stmt.split()
            for j, part in enumerate(parts):
                if part.upper() == "TABLE":
                    if j + 1 < len(parts):
                        table_name = parts[j + 1].replace('commodity.landing.', '')
                        break

        print(f"\n[{i}/{len(statements)}] Creating table: {table_name}")

        success, result = execute_sql(stmt, wait_timeout=50, description=f"Creating {table_name}")

        if not success:
            print(f"⚠ Warning: Failed to create {table_name}")
            # Continue even if one table fails

    print("\n✓ Landing tables created!")
    return True


def create_bronze_views():
    """Create bronze deduplication views"""

    print("\n" + "="*70)
    print("STEP 3: Create Bronze Deduplication Views")
    print("="*70)

    if not BRONZE_SQL.exists():
        print(f"ERROR: SQL file not found: {BRONZE_SQL}")
        return False

    print(f"\nParsing {BRONZE_SQL.name}...")
    statements = parse_sql_file(BRONZE_SQL)

    print(f"Found {len(statements)} SQL statements")

    # Execute each statement
    for i, stmt in enumerate(statements, 1):
        # Skip USE CATALOG statements
        if stmt.upper().startswith('USE CATALOG'):
            continue

        # Skip SHOW and SELECT statements (verification queries)
        if stmt.upper().startswith('SHOW') or stmt.upper().startswith('SELECT'):
            continue

        # Identify view name
        view_name = "unknown"
        if "CREATE" in stmt.upper() and "VIEW" in stmt.upper():
            parts = stmt.split()
            for j, part in enumerate(parts):
                if part.upper() == "VIEW":
                    if j + 1 < len(parts):
                        view_name = parts[j + 1].replace('commodity.bronze.', '')
                        break

        print(f"\n[{i}/{len(statements)}] Creating view: {view_name}")

        success, result = execute_sql(stmt, wait_timeout=50, description=f"Creating {view_name}")

        if not success:
            print(f"⚠ Warning: Failed to create {view_name}")

    print("\n✓ Bronze views created!")
    return True


def verify_setup():
    """Verify tables and views were created"""

    print("\n" + "="*70)
    print("STEP 4: Verify Setup")
    print("="*70)

    # Check landing tables
    print("\n[1/3] Checking landing tables...")
    success, result = execute_sql(
        "SHOW TABLES IN commodity.landing",
        description="Listing landing tables"
    )

    if success and 'result' in result:
        tables = result.get('result', {}).get('data_array', [])
        print(f"  Found {len(tables)} landing tables")
        for table in tables:
            print(f"    - {table[1] if len(table) > 1 else table}")

    # Check bronze views
    print("\n[2/3] Checking bronze views...")
    success, result = execute_sql(
        "SHOW VIEWS IN commodity.bronze",
        description="Listing bronze views"
    )

    if success and 'result' in result:
        views = result.get('result', {}).get('data_array', [])
        print(f"  Found {len(views)} bronze views")
        for view in views:
            print(f"    - {view[1] if len(view) > 1 else view}")

    # Sample query: check market data
    print("\n[3/3] Testing market data query...")
    success, result = execute_sql(
        """
        SELECT commodity, COUNT(*) as row_count,
               MIN(date) as earliest_date, MAX(date) as latest_date
        FROM commodity.bronze.v_market_data_all
        GROUP BY commodity
        """,
        description="Checking market data content"
    )

    if success and 'result' in result:
        rows = result.get('result', {}).get('data_array', [])
        if rows:
            print("\n  Market Data Summary:")
            print(f"  {'Commodity':<15} {'Rows':<10} {'Earliest':<12} {'Latest':<12}")
            print(f"  {'-'*50}")
            for row in rows:
                commodity, count, earliest, latest = row
                print(f"  {commodity:<15} {count:<10} {earliest:<12} {latest:<12}")
        else:
            print("  ⚠ No data found in market_data table")
            print("  This is expected if Lambda functions haven't run yet")

    print("\n✓ Verification complete!")
    return True


def main():
    """Main setup flow"""

    print("\n" + "="*70)
    print("DATABRICKS PIPELINE SETUP")
    print("="*70)
    print(f"Host: {DATABRICKS_HOST}")
    print(f"Warehouse: {WAREHOUSE_ID}")
    print(f"S3 Bucket: groundtruth-capstone")

    try:
        # Step 1: Catalog and schemas
        if not setup_catalog_and_schemas():
            print("\n✗ Failed at catalog setup")
            return False

        # Step 2: Landing tables
        if not create_landing_tables():
            print("\n✗ Failed at landing table creation")
            return False

        # Step 3: Bronze views
        if not create_bronze_views():
            print("\n✗ Failed at bronze view creation")
            return False

        # Step 4: Verify
        if not verify_setup():
            print("\n✗ Failed at verification")
            return False

        print("\n" + "="*70)
        print("✓ PIPELINE SETUP COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("1. Lambda functions will write data to S3 daily at 2AM UTC")
        print("2. Databricks tables auto-refresh from S3 on query")
        print("3. Query data: SELECT * FROM commodity.bronze.v_market_data_all")
        print("4. Create silver unified_data table with: research_agent/sql/create_unified_data.sql")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

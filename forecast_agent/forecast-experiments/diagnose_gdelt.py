"""
Diagnose GDELT Join Issue

Checks why the LEFT JOIN between unified_data and gdelt_wide_fillforward
is returning all NULLs.
"""

import os
import pandas as pd
from databricks import sql

print("=" * 80)
print("GDELT JOIN DIAGNOSTIC")
print("=" * 80)
print()

# Connect to Databricks
print("Connecting to Databricks...")
conn = sql.connect(
    server_hostname=os.environ['DATABRICKS_HOST'],
    http_path=os.environ['DATABRICKS_HTTP_PATH'],
    access_token=os.environ['DATABRICKS_TOKEN']
)

cursor = conn.cursor()

# 1. Check if gdelt_wide_fillforward table exists and has data
print("1. Checking gdelt_wide_fillforward table...")
query1 = """
SELECT COUNT(*) as row_count
FROM commodity.silver.gdelt_wide_fillforward
"""
cursor.execute(query1)
row_count = cursor.fetchone()[0]
print(f"   Total rows: {row_count:,}")

if row_count == 0:
    print("   ERROR: Table is empty!")
    conn.close()
    exit(1)

# 2. Check column names
print()
print("2. Checking column names...")
query2 = """
SELECT *
FROM commodity.silver.gdelt_wide_fillforward
LIMIT 1
"""
cursor.execute(query2)
columns = [desc[0] for desc in cursor.description]
print(f"   Columns: {columns[:10]}... (showing first 10)")

# Check for date column
date_cols = [c for c in columns if 'date' in c.lower()]
print(f"   Date columns: {date_cols}")

# Check for commodity column
commodity_cols = [c for c in columns if 'commodity' in c.lower()]
print(f"   Commodity columns: {commodity_cols}")

# 3. Check date range
print()
print("3. Checking date range...")
date_col = date_cols[0] if date_cols else 'date'
query3 = f"""
SELECT
    MIN({date_col}) as min_date,
    MAX({date_col}) as max_date,
    COUNT(DISTINCT {date_col}) as unique_dates
FROM commodity.silver.gdelt_wide_fillforward
"""
cursor.execute(query3)
result = cursor.fetchone()
print(f"   Date range: {result[0]} to {result[1]}")
print(f"   Unique dates: {result[2]:,}")

# 4. Check commodity values
print()
print("4. Checking commodity values...")
commodity_col = commodity_cols[0] if commodity_cols else 'commodity'
query4 = f"""
SELECT
    {commodity_col} as commodity,
    COUNT(*) as count
FROM commodity.silver.gdelt_wide_fillforward
GROUP BY {commodity_col}
"""
cursor.execute(query4)
commodities = cursor.fetchall()
print(f"   Commodities:")
for comm, count in commodities:
    print(f"     - {comm}: {count:,} rows")

# 5. Check unified_data date range for comparison
print()
print("5. Checking unified_data for comparison...")
query5 = """
SELECT
    MIN(date) as min_date,
    MAX(date) as max_date,
    COUNT(DISTINCT date) as unique_dates,
    COUNT(DISTINCT commodity) as unique_commodities
FROM commodity.silver.unified_data
WHERE commodity = 'Coffee'
"""
cursor.execute(query5)
result = cursor.fetchone()
print(f"   unified_data (Coffee only):")
print(f"     Date range: {result[0]} to {result[1]}")
print(f"     Unique dates: {result[2]:,}")

# 6. Try to find matching dates
print()
print("6. Testing join condition...")
query6 = f"""
SELECT COUNT(*) as matches
FROM commodity.silver.unified_data ud
INNER JOIN commodity.silver.gdelt_wide_fillforward g
    ON ud.date = g.{date_col}
    AND ud.commodity = g.{commodity_col}
WHERE ud.commodity = 'Coffee'
"""
cursor.execute(query6)
matches = cursor.fetchone()[0]
print(f"   Matching rows: {matches:,}")

if matches == 0:
    # Try with just date match
    print()
    print("   Trying date-only join...")
    query7 = f"""
    SELECT COUNT(*) as matches
    FROM commodity.silver.unified_data ud
    INNER JOIN commodity.silver.gdelt_wide_fillforward g
        ON ud.date = g.{date_col}
    WHERE ud.commodity = 'Coffee'
    """
    cursor.execute(query7)
    date_only_matches = cursor.fetchone()[0]
    print(f"   Date-only matches: {date_only_matches:,}")

    if date_only_matches > 0:
        print()
        print("   DIAGNOSIS: Commodity name mismatch!")
        print("   The dates match but commodity names don't.")

        # Show sample commodity values from each side
        query8 = f"""
        SELECT DISTINCT ud.commodity as ud_commodity, g.{commodity_col} as g_commodity
        FROM commodity.silver.unified_data ud
        INNER JOIN commodity.silver.gdelt_wide_fillforward g
            ON ud.date = g.{date_col}
        WHERE ud.commodity = 'Coffee'
        LIMIT 5
        """
        cursor.execute(query8)
        samples = cursor.fetchall()
        print(f"   Sample mismatches:")
        for ud_comm, g_comm in samples:
            print(f"     unified_data: '{ud_comm}' vs gdelt: '{g_comm}'")
    else:
        print()
        print("   DIAGNOSIS: Date format or range mismatch!")
        print("   The dates don't overlap between tables.")

# 7. Recommend fix
print()
print("=" * 80)
print("RECOMMENDED FIX")
print("=" * 80)
print()

if matches > 0:
    print("JOIN IS WORKING! The issue might be in the download script.")
    print(f"Correct join condition:")
    print(f"  ON ud.date = g.{date_col}")
    print(f"  AND ud.commodity = g.{commodity_col}")
else:
    print("JOIN NEEDS FIXING:")
    print()
    print("Update download_data_with_sentiment.py:")
    print(f"  - Change 'g.article_date' to 'g.{date_col}'")
    print(f"  - Change 'g.commodity' to 'g.{commodity_col}'")
    print()
    print("Or fix commodity name casing:")
    print(f"  - Use: AND UPPER(ud.commodity) = UPPER(g.{commodity_col})")

conn.close()
print()
print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

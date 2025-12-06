"""Test GDELT table access"""
import os
from databricks import sql

print("Testing GDELT table access...")

conn = sql.connect(
    server_hostname=os.environ['DATABRICKS_HOST'],
    http_path=os.environ['DATABRICKS_HTTP_PATH'],
    access_token=os.environ['DATABRICKS_TOKEN']
)

cursor = conn.cursor()

# Try simple queries
queries = [
    "SHOW TABLES IN commodity.silver LIKE 'gdelt*'",
    "DESCRIBE commodity.silver.gdelt_wide_fillforward",
    "SELECT COUNT(*) FROM commodity.silver.gdelt_wide_fillforward LIMIT 1"
]

for i, query in enumerate(queries, 1):
    print(f"\n{i}. {query}")
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        print(f"   SUCCESS: {result[:3]}")  # Show first 3 rows
    except Exception as e:
        print(f"   ERROR: {e}")

conn.close()

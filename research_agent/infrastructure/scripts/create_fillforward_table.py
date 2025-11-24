#!/usr/bin/env python3
"""
Create forward-filled GDELT table with continuous daily data.

This script executes the SQL to create commodity.silver.gdelt_wide_fillforward
which has a record for every date-commodity combination with:
- Count columns set to 0 for missing dates
- Tone columns forward-filled from the previous date with data
"""

from databricks import sql
import sys
from pathlib import Path

def create_fillforward_table():
    """Execute SQL to create forward-filled table."""

    # Read SQL file
    sql_file = Path(__file__).parent.parent / "databricks" / "create_gdelt_fillforward.sql"

    if not sql_file.exists():
        print(f"❌ SQL file not found: {sql_file}")
        return False

    with open(sql_file, 'r') as f:
        sql_commands = f.read()

    print("Connecting to Databricks...")
    connection = sql.connect(
        server_hostname="dbc-5e4780f4-fcec.cloud.databricks.com",
        http_path="/sql/1.0/warehouses/d88ad009595327fd",
        access_token="dapi8f0886905a2b080bc5456595a8746b89"
    )

    cursor = connection.cursor()

    try:
        # Remove comment lines and split on semicolons
        lines = [line for line in sql_commands.split('\n') if not line.strip().startswith('--')]
        clean_sql = '\n'.join(lines)

        # Split on semicolons
        statements = [stmt.strip() for stmt in clean_sql.split(';') if stmt.strip()]

        print(f"\nExecuting {len(statements)} SQL statement(s)...")

        for i, statement in enumerate(statements, 1):
            print(f"\n[{i}/{len(statements)}] Executing statement...")

            # Show preview of statement type
            first_word = statement.split()[0].upper() if statement.split() else ''
            print(f"Type: {first_word}")

            cursor.execute(statement)

            # If this is a SELECT statement, fetch and print results
            if first_word == 'SELECT':
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

                print(f"\nResults:")
                print(f"  {' | '.join(columns)}")
                print(f"  {'-' * (sum(len(c) for c in columns) + len(columns) * 3)}")
                for row in results:
                    print(f"  {' | '.join(str(v) for v in row)}")

        print(f"\n✅ Successfully created commodity.silver.gdelt_wide_fillforward")
        return True

    except Exception as e:
        print(f"\n❌ Error executing SQL: {e}")
        return False

    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    success = create_fillforward_table()
    sys.exit(0 if success else 1)

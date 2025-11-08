"""
Databricks Catalog Migration Script

Migrates catalog structure to clean naming:
- Bronze: vix, cftc, gdelt, macro, market, weather (removing v_, _data, _all)
- Silver: unified_data only
- Forecasts: point_forecasts, distributions, forecast_actuals, forecast_metadata
"""

from databricks import sql
import os
import sys

host = os.getenv("DATABRICKS_HOST", "https://dbc-fd7b00f3-7a6d.cloud.databricks.com")
token = os.getenv("DATABRICKS_TOKEN")
http_path = os.getenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/3cede8561503a13c")

def execute_sql(cursor, sql, description):
    """Execute SQL and handle errors"""
    print(f"\n{'='*80}")
    print(f"  {description}")
    print(f"{'='*80}")
    try:
        cursor.execute(sql)
        print(f"✅ Success")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def get_row_count(cursor, schema, table):
    """Get row count for a table"""
    try:
        cursor.execute(f"SELECT COUNT(*) FROM commodity.{schema}.{table}")
        return cursor.fetchone()[0]
    except:
        return None

connection = sql.connect(
    server_hostname=host.replace("https://", ""),
    http_path=http_path,
    access_token=token
)
cursor = connection.cursor()

print("="*80)
print("DATABRICKS CATALOG MIGRATION")
print("="*80)
print()

# ============================================================================
# STEP 1: Create forecasts schema
# ============================================================================
print("\n🔨 STEP 1: Creating forecasts schema...")
execute_sql(
    cursor,
    "CREATE SCHEMA IF NOT EXISTS commodity.forecasts COMMENT 'Forecast model outputs and metadata'",
    "Create forecasts schema"
)

# ============================================================================
# STEP 2: Migrate forecast tables from silver to forecasts
# ============================================================================
print("\n📦 STEP 2: Migrating forecast tables from silver → forecasts...")

forecast_tables = {
    'point_forecasts': 'Forecast point estimates',
    'distributions': 'Forecast Monte Carlo distributions',
    'forecast_actuals': 'Actual values for forecast evaluation',
    'forecast_metadata': 'Forecast generation metadata'
}

for table, desc in forecast_tables.items():
    # Check if source exists in silver
    source_count = get_row_count(cursor, 'silver', table)
    if source_count is not None:
        print(f"\n  {table}: {source_count:,} rows in silver")
        execute_sql(
            cursor,
            f"CREATE OR REPLACE TABLE commodity.forecast.{table} AS SELECT * FROM commodity.silver.{table}",
            f"Copy {table} to forecasts schema"
        )
    else:
        print(f"\n  {table}: Not found in silver (skipping)")

# ============================================================================
# STEP 3: Rename bronze tables to clean names
# ============================================================================
print("\n✨ STEP 3: Renaming bronze tables to clean names...")

bronze_migrations = {
    'v_vix_data_all': 'vix',
    'v_cftc_data_all': 'cftc',
    'v_gdelt_sentiment_all': 'gdelt',
    'v_macro_data_all': 'macro',
    'v_market_data_all': 'market',
    'v_weather_data_all': 'weather'
}

for old_name, new_name in bronze_migrations.items():
    source_count = get_row_count(cursor, 'bronze', old_name)
    if source_count is not None:
        print(f"\n  {old_name} → {new_name}: {source_count:,} rows")
        execute_sql(
            cursor,
            f"CREATE OR REPLACE TABLE commodity.bronze.{new_name} AS SELECT * FROM commodity.bronze.{old_name}",
            f"Rename {old_name} to {new_name}"
        )
    else:
        print(f"\n  {old_name}: Not found (skipping)")

# ============================================================================
# STEP 4: Validation
# ============================================================================
print("\n📊 STEP 4: Validating migration...")
print("\nExpected Bronze Tables:")
for new_name in bronze_migrations.values():
    count = get_row_count(cursor, 'bronze', new_name)
    status = "✅" if count is not None and count > 0 else "❌"
    print(f"  {status} commodity.bronze.{new_name:<20} {count:>15,} rows" if count else f"  {status} commodity.bronze.{new_name:<20} NOT FOUND")

print("\nExpected Silver Tables:")
silver_count = get_row_count(cursor, 'silver', 'unified_data')
status = "✅" if silver_count and silver_count > 0 else "❌"
print(f"  {status} commodity.silver.unified_data{' ':<12} {silver_count:>15,} rows" if silver_count else f"  {status} commodity.silver.unified_data NOT FOUND")

print("\nExpected Forecasts Tables:")
for table in forecast_tables.keys():
    count = get_row_count(cursor, 'forecasts', table)
    status = "✅" if count is not None else "❌"
    count_str = f"{count:>15,} rows" if count is not None else "NOT FOUND"
    print(f"  {status} commodity.forecast.{table:<20} {count_str}")

# Ask user to confirm before deletion
print("\n" + "="*80)
print("⚠️  VALIDATION COMPLETE - Review the table counts above")
print("="*80)
response = input("\nProceed with deleting old tables? (yes/no): ")

if response.lower() != 'yes':
    print("\n❌ Migration aborted. No tables were deleted.")
    print("   New tables have been created but old ones remain.")
    sys.exit(0)

# ============================================================================
# STEP 5: Drop old tables
# ============================================================================
print("\n🗑️  STEP 5: Cleaning up old tables...")

# Drop old bronze v_ tables
print("\nDropping old bronze tables with v_ prefix...")
for old_name in bronze_migrations.keys():
    execute_sql(
        cursor,
        f"DROP TABLE IF EXISTS commodity.bronze.{old_name}",
        f"Drop commodity.bronze.{old_name}"
    )

# Drop old bronze tables without prefix (duplicates)
old_duplicates = ['vix_data', 'cftc_data', 'gdelt_sentiment', 'macro_data', 'market_data', 'weather_data']
print("\nDropping duplicate bronze tables...")
for old_name in old_duplicates:
    execute_sql(
        cursor,
        f"DROP TABLE IF EXISTS commodity.bronze.{old_name}",
        f"Drop commodity.bronze.{old_name}"
    )

# Drop forecast tables from silver
print("\nDropping forecast tables from silver schema...")
for table in forecast_tables.keys():
    execute_sql(
        cursor,
        f"DROP TABLE IF EXISTS commodity.silver.{table}",
        f"Drop commodity.silver.{table}"
    )

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*80)
print("✅ MIGRATION COMPLETE!")
print("="*80)
print("\nFinal Catalog Structure:")
print("\n📁 commodity.bronze:")
print("   - vix, cftc, gdelt, macro, market, weather")
print("\n📁 commodity.silver:")
print("   - unified_data")
print("\n📁 commodity.forecasts:")
print("   - point_forecasts, distributions, forecast_actuals, forecast_metadata")
print("\n" + "="*80)

cursor.close()
connection.close()

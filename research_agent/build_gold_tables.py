#!/usr/bin/env python3
"""
Build both gold.unified_data tables in Databricks Unity Catalog

Usage:
    python research_agent/build_gold_tables.py

Requirements:
    - Valid DATABRICKS_TOKEN in infra/.env
    - Unity Catalog cluster running
    - databricks-sql-connector installed
"""

from dotenv import load_dotenv
import os
from databricks import sql
import time
from pathlib import Path

# Load credentials
load_dotenv('infra/.env')

token = os.environ.get('DATABRICKS_TOKEN')
server_hostname = os.environ.get('DATABRICKS_HOST', '').replace('https://', '').replace('http://', '')
http_path = os.environ.get('DATABRICKS_CLUSTER_HTTP_PATH')

if not all([token, server_hostname, http_path]):
    print('❌ ERROR: Missing Databricks credentials in infra/.env')
    print(f'   DATABRICKS_TOKEN: {"SET" if token else "NOT SET"}')
    print(f'   DATABRICKS_HOST: {"SET" if server_hostname else "NOT SET"}')
    print(f'   DATABRICKS_CLUSTER_HTTP_PATH: {"SET" if http_path else "NOT SET"}')
    exit(1)

print('🔧 Building Gold Layer Tables')
print('=' * 70)
print(f'Host: {server_hostname}')
print(f'Cluster: Unity Catalog cluster (not serverless)')
print('=' * 70)

# Connect to Databricks
print('\n✓ Connecting to Databricks...')
connection = sql.connect(
    server_hostname=server_hostname,
    http_path=http_path,
    access_token=token
)
cursor = connection.cursor()

# Build Table 1: Production (all forward-filled)
print('\n📊 Building commodity.gold.unified_data (PRODUCTION)')
print('   - All features forward-filled')
print('   - No NULLs except initial rows')
print('   - Use for: Production models, stable pipelines')

sql_file_1 = Path('research_agent/sql/create_gold_unified_data.sql')
with open(sql_file_1, 'r') as f:
    sql_content_1 = f.read()

start = time.time()
cursor.execute(sql_content_1)
elapsed_1 = time.time() - start
print(f'   ✅ Built in {elapsed_1:.1f}s')

# Build Table 2: Experimental (NULLs preserved)
print('\n📊 Building commodity.gold.unified_data_no_imputation (EXPERIMENTAL)')
print('   - Only `close` forward-filled')
print('   - VIX, FX, OHLV, weather, GDELT preserve NULLs')
print('   - Has 3 missingness flags (has_market_data, has_weather_data, has_gdelt_data)')
print('   - Use for: New models, experimentation, imputation flexibility')

sql_file_2 = Path('research_agent/sql/create_gold_unified_data_no_imputation.sql')
with open(sql_file_2, 'r') as f:
    sql_content_2 = f.read()

start = time.time()
cursor.execute(sql_content_2)
elapsed_2 = time.time() - start
print(f'   ✅ Built in {elapsed_2:.1f}s')

# Validation queries
print('\n🔍 Running validation queries...')
print('=' * 70)

# Row counts
cursor.execute("SELECT COUNT(*) as row_count FROM commodity.gold.unified_data")
row_count_1 = cursor.fetchone()[0]
print(f'✓ unified_data row count: {row_count_1:,}')

cursor.execute("SELECT COUNT(*) as row_count FROM commodity.gold.unified_data_no_imputation")
row_count_2 = cursor.fetchone()[0]
print(f'✓ unified_data_no_imputation row count: {row_count_2:,}')

if row_count_1 != row_count_2:
    print(f'⚠️  WARNING: Row counts differ! ({row_count_1} vs {row_count_2})')

# Check for NULLs in production table (should be minimal)
cursor.execute("""
    SELECT
        SUM(CASE WHEN vix IS NULL THEN 1 ELSE 0 END) as vix_nulls,
        SUM(CASE WHEN open IS NULL THEN 1 ELSE 0 END) as open_nulls,
        SUM(CASE WHEN gdelt_themes IS NULL THEN 1 ELSE 0 END) as gdelt_nulls
    FROM commodity.gold.unified_data
""")
prod_nulls = cursor.fetchone()
print(f'\n✓ Production table NULL counts:')
print(f'   - vix: {prod_nulls[0]} (should be ~0 after forward-fill)')
print(f'   - open: {prod_nulls[1]} (should be ~0 after forward-fill)')
print(f'   - gdelt_themes: {prod_nulls[2]} (should be ~0 after forward-fill)')

# Check for NULLs in experimental table (should be ~30% for market data)
cursor.execute("""
    SELECT
        SUM(CASE WHEN vix IS NULL THEN 1 ELSE 0 END) as vix_nulls,
        SUM(CASE WHEN open IS NULL THEN 1 ELSE 0 END) as open_nulls,
        SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as close_nulls,
        SUM(CASE WHEN gdelt_themes IS NULL THEN 1 ELSE 0 END) as gdelt_nulls,
        COUNT(*) as total_rows
    FROM commodity.gold.unified_data_no_imputation
""")
exp_nulls = cursor.fetchone()
vix_null_pct = (exp_nulls[0] / exp_nulls[4]) * 100
open_null_pct = (exp_nulls[1] / exp_nulls[4]) * 100
close_null_pct = (exp_nulls[2] / exp_nulls[4]) * 100
gdelt_null_pct = (exp_nulls[3] / exp_nulls[4]) * 100

print(f'\n✓ Experimental table NULL percentages:')
print(f'   - vix: {vix_null_pct:.1f}% (expect ~30% for weekends/holidays)')
print(f'   - open: {open_null_pct:.1f}% (expect ~30% for weekends/holidays)')
print(f'   - close: {close_null_pct:.1f}% (expect 0% - forward-filled)')
print(f'   - gdelt_themes: {gdelt_null_pct:.1f}% (expect ~73% - days without articles)')

# Check missingness flags
cursor.execute("""
    SELECT
        AVG(has_market_data) as market_pct,
        AVG(has_weather_data) as weather_pct,
        AVG(has_gdelt_data) as gdelt_pct
    FROM commodity.gold.unified_data_no_imputation
""")
flags = cursor.fetchone()
print(f'\n✓ Missingness flags (% of days with data):')
print(f'   - has_market_data: {flags[0]*100:.1f}% (expect ~70% for trading days)')
print(f'   - has_weather_data: {flags[1]*100:.1f}% (expect ~100% - weather daily)')
print(f'   - has_gdelt_data: {flags[2]*100:.1f}% (expect ~27% - days with articles)')

# Check GDELT commodity capitalization (should be 'Coffee', 'Sugar', not 'coffee', 'sugar')
cursor.execute("""
    SELECT DISTINCT commodity
    FROM commodity.gold.unified_data
    WHERE gdelt_themes IS NOT NULL
    ORDER BY commodity
""")
commodities = [row[0] for row in cursor.fetchall()]
print(f'\n✓ GDELT commodities (should be capitalized): {commodities}')
if all(c[0].isupper() for c in commodities):
    print('   ✅ All commodities properly capitalized')
else:
    print('   ⚠️  WARNING: Some commodities not capitalized!')

cursor.close()
connection.close()

print('\n' + '=' * 70)
print('✅ Both gold tables built and validated successfully!')
print('=' * 70)
print('\nNext steps:')
print('  1. Review GOLD_MIGRATION_GUIDE.md to choose which table to use')
print('  2. Update forecast models to query commodity.gold.unified_data OR')
print('     commodity.gold.unified_data_no_imputation')
print('  3. For experimental table: implement ImputationTransformer in your pipeline')

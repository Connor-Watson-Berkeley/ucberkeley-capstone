# Databricks notebook source
"""
Bulk Upload Oracle Forecasts from Parquet to Databricks Tables

COST-OPTIMIZED: Uses Spark to bulk load ~5M rows in ~5 minutes (~$1.33).
Much cheaper than incremental SQL INSERTs (~$16-32 for 1-2 hours).

Prerequisites:
1. Run generate_oracle_forecasts_parquet.py locally
2. Upload Parquet files to DBFS: dbfs:/tmp/oracle_forecasts/

Usage:
- Run this notebook on your ML cluster
- Expected runtime: ~5 minutes
- Cost: ~$1.33 (5 min × $16/hour)
"""

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime

print("="*80)
print("ORACLE FORECASTS - PARQUET BULK UPLOAD")
print("="*80)
print(f"Started: {datetime.now()}")
print()

# COMMAND ----------

# List Parquet files
parquet_dir = "dbfs:/tmp/oracle_forecasts"
files = dbutils.fs.ls(parquet_dir)

print(f"Found {len(files)} Parquet files in {parquet_dir}:")
for f in files:
    size_mb = f.size / (1024 * 1024)
    print(f"  - {f.name} ({size_mb:.2f} MB)")
print()

# COMMAND ----------

# Bulk load distributions
print("="*80)
print("LOADING DISTRIBUTIONS")
print("="*80)

dist_files = [f.path for f in files if f.name.startswith('distributions_')]
print(f"Loading {len(dist_files)} distribution files...")

for file_path in dist_files:
    model_version = file_path.split('/')[-1].replace('distributions_', '').replace('.parquet', '')
    print(f"\n📦 {model_version}")

    # Read Parquet
    df = spark.read.parquet(file_path)
    row_count = df.count()
    print(f"   Rows: {row_count:,}")

    # Write to table (append mode)
    df.write \
        .format("delta") \
        .mode("append") \
        .saveAsTable("commodity.forecast.distributions")

    print(f"   ✅ Loaded to commodity.forecast.distributions")

print()
print("="*80)
print(f"✅ Loaded all {len(dist_files)} distribution files")
print("="*80)

# COMMAND ----------

# Bulk load point forecasts
print()
print("="*80)
print("LOADING POINT FORECASTS")
print("="*80)

point_files = [f.path for f in files if f.name.startswith('point_forecasts_')]
print(f"Loading {len(point_files)} point forecast files...")

for file_path in point_files:
    model_version = file_path.split('/')[-1].replace('point_forecasts_', '').replace('.parquet', '')
    print(f"\n📊 {model_version}")

    # Read Parquet
    df = spark.read.parquet(file_path)
    row_count = df.count()
    print(f"   Rows: {row_count:,}")

    # Write to table (append mode)
    df.write \
        .format("delta") \
        .mode("append") \
        .saveAsTable("commodity.forecast.point_forecasts")

    print(f"   ✅ Loaded to commodity.forecast.point_forecasts")

print()
print("="*80)
print(f"✅ Loaded all {len(point_files)} point forecast files")
print("="*80)

# COMMAND ----------

# Verification
print()
print("="*80)
print("VERIFICATION")
print("="*80)

# Count rows by model
print("\n📊 Distributions by model:")
dist_counts = spark.sql("""
    SELECT model_version, COUNT(*) as row_count
    FROM commodity.forecast.distributions
    WHERE model_version LIKE 'oracle_%'
    GROUP BY model_version
    ORDER BY model_version
""")
dist_counts.show(truncate=False)

print("\n📈 Point forecasts by model:")
point_counts = spark.sql("""
    SELECT model_version, COUNT(*) as row_count
    FROM commodity.forecast.point_forecasts
    WHERE model_version LIKE 'oracle_%'
    GROUP BY model_version
    ORDER BY model_version
""")
point_counts.show(truncate=False)

# COMMAND ----------

print()
print("="*80)
print("✅ ORACLE FORECASTS SUCCESSFULLY LOADED")
print("="*80)
print(f"Completed: {datetime.now()}")
print()
print("Next steps:")
print("  1. Query these forecasts in your trading agent")
print("  2. Measure trading P&L for each oracle model")
print("  3. Determine minimum forecast quality needed for profitability")
print("="*80)

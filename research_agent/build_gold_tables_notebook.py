# Databricks notebook source
# MAGIC %md
# MAGIC # Build Gold Layer Tables - Production & Experimental
# MAGIC
# MAGIC **Purpose**: Create both `commodity.gold.unified_data` tables:
# MAGIC - `unified_data`: Production (all features forward-filled)
# MAGIC - `unified_data_no_imputation`: Experimental (NULLs preserved for imputation flexibility)
# MAGIC
# MAGIC **Cluster**: Run on `unity-catalog-cluster` (NOT serverless - Unity Catalog required)
# MAGIC
# MAGIC **Runtime**: ~2-3 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 1: commodity.gold.unified_data (PRODUCTION)
# MAGIC
# MAGIC - **All features forward-filled** (no NULLs)
# MAGIC - **Use for**: Production models, existing pipelines
# MAGIC - **Benefits**: Proven, stable, no NULL handling needed

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Production table: All features forward-filled
# MAGIC -- Read from: research_agent/sql/create_gold_unified_data.sql
# MAGIC -- Run this in SQL editor by copying the file contents

# COMMAND ----------

print("✓ Run research_agent/sql/create_gold_unified_data.sql in SQL editor")
print("  (Too large to embed in notebook - use SQL editor for DDL)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table 2: commodity.gold.unified_data_no_imputation (EXPERIMENTAL)
# MAGIC
# MAGIC - **Only `close` forward-filled** (all other features preserve NULLs)
# MAGIC - **Use for**: New models, experimentation, imputation control
# MAGIC - **Includes**: 3 missingness flags (has_market_data, has_weather_data, has_gdelt_data)

# COMMAND ----------

print("✓ Run research_agent/sql/create_gold_unified_data_no_imputation.sql in SQL editor")
print("  (Too large to embed in notebook - use SQL editor for DDL)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation Queries

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Row counts (should both be ~7k)
# MAGIC SELECT 'unified_data' as table_name, COUNT(*) as row_count
# MAGIC FROM commodity.gold.unified_data
# MAGIC UNION ALL
# MAGIC SELECT 'unified_data_no_imputation' as table_name, COUNT(*) as row_count
# MAGIC FROM commodity.gold.unified_data_no_imputation

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Production table: Check NULLs (should be minimal after forward-fill)
# MAGIC SELECT
# MAGIC   SUM(CASE WHEN vix IS NULL THEN 1 ELSE 0 END) as vix_nulls,
# MAGIC   SUM(CASE WHEN open IS NULL THEN 1 ELSE 0 END) as open_nulls,
# MAGIC   SUM(CASE WHEN gdelt_themes IS NULL THEN 1 ELSE 0 END) as gdelt_nulls,
# MAGIC   COUNT(*) as total_rows
# MAGIC FROM commodity.gold.unified_data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Experimental table: Check NULL percentages (expect ~30% for market data)
# MAGIC SELECT
# MAGIC   ROUND(100.0 * SUM(CASE WHEN vix IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as vix_null_pct,
# MAGIC   ROUND(100.0 * SUM(CASE WHEN open IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as open_null_pct,
# MAGIC   ROUND(100.0 * SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as close_null_pct,
# MAGIC   ROUND(100.0 * SUM(CASE WHEN gdelt_themes IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as gdelt_null_pct,
# MAGIC   COUNT(*) as total_rows
# MAGIC FROM commodity.gold.unified_data_no_imputation

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Experimental table: Check missingness flags
# MAGIC SELECT
# MAGIC   ROUND(100.0 * AVG(has_market_data), 1) as market_data_pct,
# MAGIC   ROUND(100.0 * AVG(has_weather_data), 1) as weather_data_pct,
# MAGIC   ROUND(100.0 * AVG(has_gdelt_data), 1) as gdelt_data_pct
# MAGIC FROM commodity.gold.unified_data_no_imputation

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check GDELT commodity capitalization (should be 'Coffee', 'Sugar')
# MAGIC SELECT DISTINCT commodity
# MAGIC FROM commodity.gold.unified_data
# MAGIC WHERE gdelt_themes IS NOT NULL
# MAGIC ORDER BY commodity

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Sample data from production table
# MAGIC SELECT date, commodity, close, vix, size(weather_data) as weather_regions, size(gdelt_themes) as gdelt_themes
# MAGIC FROM commodity.gold.unified_data
# MAGIC WHERE commodity = 'Coffee'
# MAGIC ORDER BY date DESC
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Sample data from experimental table (check NULLs and flags)
# MAGIC SELECT
# MAGIC   date, commodity, close, vix, open,
# MAGIC   has_market_data, has_weather_data, has_gdelt_data
# MAGIC FROM commodity.gold.unified_data_no_imputation
# MAGIC WHERE commodity = 'Coffee'
# MAGIC ORDER BY date DESC
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Validation Complete
# MAGIC
# MAGIC **Expected Results**:
# MAGIC - Both tables: ~7k rows (2 commodities × ~3,500 days)
# MAGIC - Production (`unified_data`): Minimal NULLs (only initial rows before forward-fill)
# MAGIC - Experimental (`unified_data_no_imputation`):
# MAGIC   - VIX/open/FX: ~30% NULL (weekends/holidays)
# MAGIC   - close: 0% NULL (forward-filled)
# MAGIC   - GDELT: ~73% NULL (days without articles)
# MAGIC   - Missingness flags: 70% market, 100% weather, 27% GDELT
# MAGIC
# MAGIC **Next Steps**:
# MAGIC 1. Read `GOLD_MIGRATION_GUIDE.md` to choose which table to use
# MAGIC 2. Update forecast models to query `commodity.gold.unified_data` or `commodity.gold.unified_data_no_imputation`
# MAGIC 3. For experimental table: Implement `ImputationTransformer` in your pipeline

# Databricks notebook source
# MAGIC %md
# MAGIC # Build commodity.gold.unified_data
# MAGIC
# MAGIC **Purpose:** Create gold-layer unified data with array-based architecture
# MAGIC
# MAGIC **New Structure:**
# MAGIC - Grain: `(date, commodity)` - ~7k rows (vs ~75k in silver.unified_data)
# MAGIC - Weather: `ARRAY<STRUCT<region, temp, humidity, ...>>`
# MAGIC - GDELT: `ARRAY<STRUCT<theme_group, count, tone_metrics, ...>>`
# MAGIC
# MAGIC **Benefits:**
# MAGIC - 90% fewer rows
# MAGIC - Models can flexibly aggregate regions (mean, weighted, separate features)
# MAGIC - Clean integration with PySpark ML transformers

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Execute SQL to Create Table

# COMMAND ----------

# Read SQL file from repository
sql_path = "/Workspace/Repos/Project_Git/ucberkeley-capstone/research_agent/sql/create_gold_unified_data.sql"

with open(sql_path) as f:
    create_gold_sql = f.read()

print(f"Executing SQL from: {sql_path}")
print(f"SQL size: {len(create_gold_sql):,} characters\n")

# Execute the CREATE TABLE statement
spark.sql(create_gold_sql)

print("✅ Table commodity.gold.unified_data created successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Validate Table Statistics

# COMMAND ----------

# Row count
row_count = spark.sql("SELECT COUNT(*) as cnt FROM commodity.gold.unified_data").collect()[0]['cnt']
print(f"Total rows: {row_count:,}")

# Commodities
commodities = [row['commodity'] for row in spark.sql("SELECT DISTINCT commodity FROM commodity.gold.unified_data").collect()]
print(f"Commodities: {', '.join(commodities)}")

# Date range
date_range = spark.sql("""
    SELECT MIN(date) as min_date, MAX(date) as max_date
    FROM commodity.gold.unified_data
""").collect()[0]
print(f"Date range: {date_range['min_date']} to {date_range['max_date']}")

# Weather array size
weather_size = spark.sql("""
    SELECT size(weather_data) as num_regions
    FROM commodity.gold.unified_data
    WHERE weather_data IS NOT NULL
    LIMIT 1
""").collect()[0]['num_regions']
print(f"Weather regions per row: {weather_size}")

# GDELT array size
gdelt_size = spark.sql("""
    SELECT size(gdelt_themes) as num_themes
    FROM commodity.gold.unified_data
    WHERE gdelt_themes IS NOT NULL
    LIMIT 1
""").collect()[0]['num_themes']
print(f"GDELT theme groups per row: {gdelt_size}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Sample Data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Sample weather array for Coffee on recent date
# MAGIC SELECT
# MAGIC   date,
# MAGIC   commodity,
# MAGIC   close,
# MAGIC   weather_data,
# MAGIC   size(weather_data) as num_regions
# MAGIC FROM commodity.gold.unified_data
# MAGIC WHERE commodity = 'Coffee'
# MAGIC   AND date >= '2024-01-01'
# MAGIC LIMIT 3

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Sample GDELT themes for Coffee
# MAGIC SELECT
# MAGIC   date,
# MAGIC   commodity,
# MAGIC   gdelt_themes,
# MAGIC   size(gdelt_themes) as num_themes
# MAGIC FROM commodity.gold.unified_data
# MAGIC WHERE commodity = 'Coffee'
# MAGIC   AND gdelt_themes IS NOT NULL
# MAGIC   AND date >= '2024-01-01'
# MAGIC LIMIT 3

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Explode Arrays to Verify Data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Explode weather array to see individual regions
# MAGIC SELECT
# MAGIC   date,
# MAGIC   commodity,
# MAGIC   weather.region,
# MAGIC   weather.temp_mean_c,
# MAGIC   weather.precipitation_mm,
# MAGIC   weather.humidity_mean_pct
# MAGIC FROM commodity.gold.unified_data
# MAGIC LATERAL VIEW explode(weather_data) AS weather
# MAGIC WHERE commodity = 'Coffee'
# MAGIC   AND date = '2024-01-01'
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Explode GDELT array to see individual themes
# MAGIC SELECT
# MAGIC   date,
# MAGIC   commodity,
# MAGIC   theme.theme_group,
# MAGIC   theme.article_count,
# MAGIC   theme.tone_avg,
# MAGIC   theme.tone_polarity
# MAGIC FROM commodity.gold.unified_data
# MAGIC LATERAL VIEW explode(gdelt_themes) AS theme
# MAGIC WHERE commodity = 'Coffee'
# MAGIC   AND date >= '2024-01-01'
# MAGIC ORDER BY date DESC, theme.article_count DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ SUCCESS!
# MAGIC
# MAGIC **commodity.gold.unified_data** is ready for use in ml_lib pipelines.
# MAGIC
# MAGIC **Next Steps:**
# MAGIC 1. Create PySpark transformers to unpack weather/GDELT arrays
# MAGIC 2. Implement TimeSeriesForecastCV
# MAGIC 3. Build first baseline model pipeline

# COMMAND ----------

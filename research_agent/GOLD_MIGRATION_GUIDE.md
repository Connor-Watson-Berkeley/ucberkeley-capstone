# Migration Guide: Silver → Gold Unified Data

**Purpose**: Guide for transitioning forecast models from `commodity.silver.unified_data` to `commodity.gold.unified_data`

**Audience**: Forecast Agent developers

**Status**: Gold table is production-ready (Dec 2024)

---

## Why Migrate?

| Benefit | Silver | Gold | Improvement |
|---------|--------|------|-------------|
| **Row count** | ~75k rows | ~7k rows | 90% reduction |
| **Memory usage** | High | Low | 90% reduction |
| **Training speed** | Baseline | Faster | Data loading 90% faster |
| **Regional flexibility** | Fixed (exploded rows) | Flexible (arrays) | Models choose aggregation |
| **GDELT sentiment** | ❌ Not available | ✅ 7 theme groups | New feature source |
| **Query performance** | Slower (large scans) | Faster (smaller table) | 90% fewer rows to scan |

---

## Schema Comparison

### Silver (Legacy)
```
Grain: (date, commodity, region)
Rows: ~75,000

Columns:
  date, commodity, region,           # Keys
  close, open, high, low, volume,    # Market data
  vix,                                # Volatility
  vnd_usd, cop_usd, ... (24 FX),     # Exchange rates
  temp_mean_c,                        # Weather (scalar, one region)
  precipitation_mm,
  humidity_mean_pct
```

### Gold (Recommended)
```
Grain: (date, commodity)
Rows: ~7,000

Columns:
  date, commodity,                    # Keys
  close, open, high, low, volume,     # Market data
  vix,                                # Volatility
  vnd_usd, cop_usd, ... (24 FX),     # Exchange rates
  weather_data,                       # ARRAY<STRUCT> - all regions
  gdelt_themes                        # ARRAY<STRUCT> - 7 theme groups
```

---

## Migration Path

### Option 1: Simple Aggregation (Recommended for Tree-Based Models)

**Use case**: Models that don't need regional granularity (XGBoost, LightGBM, SARIMAX)

**Before (Silver)**:
```python
from pyspark.sql.functions import mean

# Manually aggregate regions (slow, memory intensive)
df = spark.table("commodity.silver.unified_data") \
    .filter("commodity = 'Coffee' AND is_trading_day = 1") \
    .groupBy("date", "commodity") \
    .agg(
        mean("close").alias("close"),
        mean("temp_mean_c").alias("avg_temp"),
        mean("precipitation_mm").alias("avg_precip")
    )
```

**After (Gold)**:
```python
from pyspark.sql.functions import expr

# Use pre-aggregated data (fast, low memory)
df = spark.table("commodity.gold.unified_data") \
    .filter("commodity = 'Coffee' AND is_trading_day = 1") \
    .select(
        "date",
        "commodity",
        "close",
        expr("aggregate(weather_data, 0.0, (acc, w) -> acc + w.temp_mean_c) / size(weather_data)").alias("avg_temp"),
        expr("aggregate(weather_data, 0.0, (acc, w) -> acc + w.precipitation_mm) / size(weather_data)").alias("avg_precip")
    )
```

### Option 2: Exploded Regional Features (For Regional Models)

**Use case**: Models that need per-region features (hierarchical models, attention mechanisms)

**Before (Silver)**:
```python
# Regions already exploded (75k rows)
df = spark.table("commodity.silver.unified_data") \
    .filter("commodity = 'Coffee' AND is_trading_day = 1")
# Each (date, commodity) has ~35 rows (one per region)
```

**After (Gold)**:
```python
from pyspark.sql.functions import explode

# Explode arrays on-demand (more flexible)
df = spark.table("commodity.gold.unified_data") \
    .filter("commodity = 'Coffee' AND is_trading_day = 1") \
    .select(
        "date",
        "commodity",
        "close",
        explode("weather_data").alias("weather")
    ) \
    .select(
        "date",
        "commodity",
        "close",
        "weather.region",
        "weather.temp_mean_c",
        "weather.precipitation_mm"
    )
```

### Option 3: Weighted Aggregation (Advanced)

**Use case**: Weight regions by production volume

**Gold only** (requires array operations):
```python
from pyspark.sql.functions import expr

# Define production weights per region (example)
production_weights = {
    "Sul_de_Minas": 0.35,
    "Cerrado": 0.25,
    "Mogiana": 0.20,
    # ... more regions
}

# Weighted average temperature
df = spark.table("commodity.gold.unified_data") \
    .filter("commodity = 'Coffee'") \
    .select(
        "date",
        "commodity",
        "close",
        expr(f"""
            aggregate(
                weather_data,
                0.0,
                (acc, w) -> acc + CASE
                    WHEN w.region = 'Sul_de_Minas' THEN w.temp_mean_c * {production_weights['Sul_de_Minas']}
                    WHEN w.region = 'Cerrado' THEN w.temp_mean_c * {production_weights['Cerrado']}
                    ELSE 0.0
                END
            )
        """).alias("weighted_temp")
    )
```

---

## GDELT Sentiment Features (New in Gold)

**Array structure**:
```sql
gdelt_themes: ARRAY<STRUCT<
  theme_group STRING,      -- SUPPLY, LOGISTICS, TRADE, MARKET, POLICY, CORE, OTHER
  article_count INT,
  tone_avg DOUBLE,
  tone_positive DOUBLE,
  tone_negative DOUBLE,
  tone_polarity DOUBLE
>>
```

**Usage**:
```python
from pyspark.sql.functions import expr, explode_outer

# Explode GDELT themes
df = spark.table("commodity.gold.unified_data") \
    .select(
        "date",
        "commodity",
        "close",
        explode_outer("gdelt_themes").alias("theme")
    ) \
    .select(
        "date",
        "commodity",
        "close",
        "theme.theme_group",
        "theme.article_count",
        "theme.tone_avg"
    )

# Or aggregate (e.g., mean tone across all themes)
agg_df = spark.table("commodity.gold.unified_data") \
    .select(
        "date",
        "commodity",
        "close",
        expr("aggregate(gdelt_themes, 0.0, (acc, t) -> acc + t.tone_avg) / size(gdelt_themes)").alias("avg_tone")
    )
```

---

## Backward Compatibility

**Silver table will remain available** for:
- ✅ Existing models in production
- ✅ Regional analysis requiring explicit rows
- ✅ Legacy workflows that haven't migrated

**No breaking changes** - both tables are maintained.

---

## Migration Checklist

### For Existing Models

- [ ] Read DATA_CONTRACTS.md to understand gold schema
- [ ] Decide aggregation strategy (simple mean, weighted, regional)
- [ ] Update data loading code to use `commodity.gold.unified_data`
- [ ] Update feature engineering to use array operations
- [ ] Test with small date range to verify correctness
- [ ] Benchmark training speed (should be faster)
- [ ] Run backfill to compare forecast quality
- [ ] Update documentation/comments to reference gold table

### For New Models

- [ ] Start with `commodity.gold.unified_data` (don't use silver)
- [ ] Use array operations for weather/GDELT features
- [ ] Consider GDELT sentiment as additional signal
- [ ] Document which aggregation approach you chose

---

## Performance Tips

### ✅ DO
- Use `commodity.gold.unified_data` for new models
- Filter on `is_trading_day = 1` early to reduce rows further
- Use `aggregate()` SQL function for array operations (faster than UDFs)
- Explode arrays only when needed (not for simple aggregations)

### ❌ DON'T
- Join silver + gold (choose one)
- Explode arrays and then immediately re-aggregate (use aggregate() instead)
- Load entire table into pandas (use Spark for large date ranges)

---

## Example: Migrating SARIMAX Model

**Before (Silver)**:
```python
# ground_truth/models/sarimax_auto_weather.py

def get_training_data(spark, commodity, cutoff_date):
    # Group regions to get single time series
    df = spark.table("commodity.silver.unified_data") \
        .filter(f"commodity = '{commodity}' AND is_trading_day = 1") \
        .filter(f"date <= '{cutoff_date}'") \
        .groupBy("date") \
        .agg(
            first("close").alias("close"),
            mean("temp_mean_c").alias("temp"),
            mean("humidity_mean_pct").alias("humidity"),
            mean("precipitation_mm").alias("precip")
        ) \
        .orderBy("date")
    return df.toPandas()
```

**After (Gold)**:
```python
# ground_truth/models/sarimax_auto_weather.py

def get_training_data(spark, commodity, cutoff_date):
    from pyspark.sql.functions import expr

    # Use gold table with pre-aggregated data
    df = spark.table("commodity.gold.unified_data") \
        .filter(f"commodity = '{commodity}' AND is_trading_day = 1") \
        .filter(f"date <= '{cutoff_date}'") \
        .select(
            "date",
            "close",
            expr("aggregate(weather_data, 0.0, (acc, w) -> acc + w.temp_mean_c) / size(weather_data)").alias("temp"),
            expr("aggregate(weather_data, 0.0, (acc, w) -> acc + w.humidity_mean_pct) / size(weather_data)").alias("humidity"),
            expr("aggregate(weather_data, 0.0, (acc, w) -> acc + w.precipitation_mm) / size(weather_data)").alias("precip")
        ) \
        .orderBy("date")
    return df.toPandas()
```

**Performance gain**: 90% faster data loading (fewer rows to scan and group)

---

## Validation

After migrating, validate that forecasts are equivalent:

```bash
# Run validation script
python research_agent/infrastructure/tests/validate_gold_unified_data.py

# Compare silver vs gold query results
python research_agent/infrastructure/tests/compare_silver_gold_outputs.py
```

---

## Questions?

- **Schema details**: See [docs/DATA_CONTRACTS.md](../docs/DATA_CONTRACTS.md)
- **Architecture**: See [UNIFIED_DATA_ARCHITECTURE.md](UNIFIED_DATA_ARCHITECTURE.md)
- **SQL source**: See [sql/create_gold_unified_data.sql](sql/create_gold_unified_data.sql)
- **Validation**: See [infrastructure/tests/validate_gold_unified_data.py](infrastructure/tests/validate_gold_unified_data.py)

---

**Last Updated**: 2025-12-06
**Owner**: Research Agent Team

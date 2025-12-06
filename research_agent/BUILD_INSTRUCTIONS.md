# Gold Tables Build Instructions

## Quick Start: Build & Validate in Databricks

### Option 1: Databricks SQL Editor (Recommended for DDL)

1. **Open Databricks SQL Editor** in your workspace
2. **Attach to cluster**: Select `unity-catalog-cluster` (must be Unity Catalog enabled)
3. **Build Production Table**:
   - Copy contents of `research_agent/sql/create_gold_unified_data.sql`
   - Paste into SQL Editor
   - Run (takes ~1-2 minutes)

4. **Build Experimental Table**:
   - Copy contents of `research_agent/sql/create_gold_unified_data_no_imputation.sql`
   - Paste into SQL Editor
   - Run (takes ~1-2 minutes)

5. **Validate**:
   ```bash
   python research_agent/validate_gold_tables.py
   ```

### Option 2: Python Script (Local)

```bash
# Ensure unity-catalog-cluster is running
databricks clusters start --cluster-id 1206-035121-fk793i8i

# Build both tables
python research_agent/build_gold_tables.py

# Validate
python research_agent/validate_gold_tables.py
```

### Option 3: Databricks Notebook

1. Upload `research_agent/build_gold_tables_notebook.py` to Databricks workspace
2. Attach to `unity-catalog-cluster`
3. Run all cells

---

## What Gets Created

### Table 1: `commodity.gold.unified_data` (PRODUCTION)
- **Rows**: ~7,000 (2 commodities × ~3,500 days)
- **Imputation**: All features forward-filled (no NULLs)
- **Use for**: Production models, existing pipelines
- **Benefits**: Proven, stable, no NULL handling needed

**Schema**:
```
date DATE
commodity STRING
is_trading_day INT
open, high, low, close, volume DOUBLE  -- All forward-filled
vix DOUBLE                              -- Forward-filled
<24 FX rates>                           -- All forward-filled
weather_data ARRAY<STRUCT<...>>         -- Forward-filled
gdelt_themes ARRAY<STRUCT<...>>         -- Forward-filled
```

---

### Table 2: `commodity.gold.unified_data_no_imputation` (EXPERIMENTAL)
- **Rows**: ~7,000 (same as production)
- **Imputation**: Only `close` forward-filled, all others preserve NULLs
- **Use for**: New models, experimentation, imputation control
- **Benefits**: Flexibility to choose imputation strategy per model

**Schema**:
```
date DATE
commodity STRING
is_trading_day INT
open, high, low, volume DOUBLE          -- NULL on weekends/holidays (~30%)
close DOUBLE                             -- Forward-filled (0% NULL)
vix DOUBLE                               -- NULL on weekends/holidays (~30%)
<24 FX rates>                            -- NULL on weekends/holidays (~30%)
weather_data ARRAY<STRUCT<...>>          -- NULL preserved (rare)
gdelt_themes ARRAY<STRUCT<...>>          -- NULL on days without articles (~73%)
has_market_data INT                      -- 1 if VIX/FX/OHLV present, 0 otherwise
has_weather_data INT                     -- 1 if weather array non-empty, 0 otherwise
has_gdelt_data INT                       -- 1 if GDELT array non-empty, 0 otherwise
```

---

## Validation Checks

The `validate_gold_tables.py` script performs 6 rigorous tests:

1. **Row Counts**: Both tables should have ~7k rows
2. **Production NULLs**: Should have <1% NULLs (only initial rows)
3. **Experimental NULLs**:
   - VIX/Open: 25-35% NULL (weekends/holidays)
   - Close: 0% NULL (forward-filled)
   - GDELT: 60-85% NULL (days without articles)
4. **Missingness Flags**:
   - `has_market_data`: 65-75% (trading days)
   - `has_weather_data`: ≥95% (weather daily)
   - `has_gdelt_data`: 20-35% (days with articles)
5. **GDELT Capitalization**: Should be 'Coffee'/'Sugar', not 'coffee'/'sugar'
6. **Sample Data Inspection**: Visual check of latest rows

---

## Troubleshooting

### Cluster Not Running
```bash
databricks clusters start --cluster-id 1206-035121-fk793i8i
# Wait ~5 minutes for cluster to start
```

### Invalid Access Token
Update `infra/.env` with fresh token:
```bash
DATABRICKS_TOKEN=dapi<your_new_token>
```

### "Table already exists" Error
Drop and recreate:
```sql
DROP TABLE IF EXISTS commodity.gold.unified_data;
DROP TABLE IF EXISTS commodity.gold.unified_data_no_imputation;
```

---

## Next Steps

1. **Read** `GOLD_MIGRATION_GUIDE.md` to choose which table to use
2. **Update** forecast models to query `commodity.gold.unified_data` OR `commodity.gold.unified_data_no_imputation`
3. **For experimental table**: Implement `ImputationTransformer` in your pipeline (see `forecast_agent/ml_lib/transformers/imputation.py`)

---

**Last Updated**: 2024-12-05
**Owner**: Research Agent

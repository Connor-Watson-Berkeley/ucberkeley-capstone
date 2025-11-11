# Weather v2 Bronze Table - Manual Creation Instructions

If the Python script times out, you can create the table manually in Databricks SQL Editor.

## Option 1: Run SQL File in Databricks SQL Editor

1. Go to Databricks SQL Editor: `https://dbc-5e4780f4-fcec.cloud.databricks.com/sql/editor`
2. Copy the contents of `create_weather_v2_bronze_table.sql`
3. Paste into SQL Editor
4. Run each section separately (use Ctrl+Enter or Cmd+Enter)
5. Monitor progress in the Query History tab

**Estimated time**: 10-15 minutes

## Option 2: Run Python Script Locally

```bash
cd research_agent/infrastructure
python create_weather_v2_with_copy_into.py
```

**Estimated time**: 5-10 minutes

## Option 3: Execute via Databricks CLI

```bash
# Upload SQL file to Databricks workspace
databricks workspace import create_weather_v2_bronze_table.sql /Workspace/weather_v2_setup.sql

# Execute via SQL Warehouse
databricks sql execute \
  --warehouse-id d88ad009595327fd \
  --sql-file /Workspace/weather_v2_setup.sql
```

## Verification Queries

After creation, verify the table:

```sql
-- Check row count
SELECT COUNT(*) FROM commodity.bronze.weather_v2;
-- Expected: ~252,000 rows

-- Check regions
SELECT COUNT(DISTINCT region) FROM commodity.bronze.weather_v2;
-- Expected: 67 regions

-- Check date range
SELECT MIN(date), MAX(date) FROM commodity.bronze.weather_v2;
-- Expected: 2015-07-07 to 2025-11-05

-- Check Sul de Minas coordinates
SELECT region, latitude, longitude, temp_min_c, temp_max_c
FROM commodity.bronze.weather_v2
WHERE region LIKE '%Minas%'
ORDER BY date DESC
LIMIT 5;
-- Expected: Latitude around -20.3, Longitude around -45.4
```

## Troubleshooting

### Error: "Path does not exist"
- Verify S3 bucket access: `aws s3 ls s3://groundtruth-capstone/landing/weather_v2/`
- Check that backfill completed successfully

### Error: "Timeout" or "Max retry duration exceeded"
- Use Databricks SQL Editor instead of Python script
- SQL Editor has better timeout handling for long-running queries

### Error: "Column not found"
- Double-check schema matches S3 data
- Inspect sample file: `aws s3 cp s3://groundtruth-capstone/landing/weather_v2/year=2015/month=07/day=07/data.jsonl - | head -1`

## Next Steps After Creation

1. **Validate July 2021 frost event**:
   ```bash
   python tests/validate_july2021_frost.py
   ```

2. **Update unified_data**:
   - Edit `sql/create_unified_data.sql`
   - Replace `commodity.bronze.weather` with `commodity.bronze.weather_v2`

3. **Train new models**:
   - Retrain SARIMAX with corrected weather data
   - Compare accuracy vs old models

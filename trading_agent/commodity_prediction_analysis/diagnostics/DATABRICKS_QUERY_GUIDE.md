# Databricks GDELT Query Guide

**Table:** `commodity.silver.gdelt_wide`
**Workspace:** https://dbc-5e4780f4-fcec.cloud.databricks.com
**Warehouse ID:** d88ad009595327fd

---

## Table Details

- **Location:** s3://groundtruth-capstone/processed/gdelt/silver/gdelt_wide/
- **Rows:** 2,045 (2021-01-01 to 2025-11-22)
- **Commodities:** coffee, sugar
- **Partitions:** article_date + commodity
- **Columns:** 41 total
  - `article_date`, `commodity`
  - 7 groups (SUPPLY, DEMAND, PRICE, WEATHER, LOGISTICS, POLICY, MARKET) × 5 metrics each
  - Metrics: `_count`, `_tone_avg`, `_tone_min`, `_tone_max`, `_tone_stddev`

---

## Quick Start

### SQL Editor (Recommended)
1. Open https://dbc-5e4780f4-fcec.cloud.databricks.com
2. Click "SQL Editor"
3. Select warehouse: `d88ad009595327fd`
4. Run your query

### Basic Queries

**Count rows:**
```sql
SELECT COUNT(*) FROM commodity.silver.gdelt_wide
```

**Latest coffee data:**
```sql
SELECT * FROM commodity.silver.gdelt_wide
WHERE commodity = 'coffee'
ORDER BY article_date DESC LIMIT 10
```

**Supply/demand trends:**
```sql
SELECT article_date,
       group_SUPPLY_count,
       group_SUPPLY_tone_avg,
       group_DEMAND_count,
       group_DEMAND_tone_avg
FROM commodity.silver.gdelt_wide
WHERE commodity = 'coffee'
  AND article_date >= '2024-01-01'
ORDER BY article_date
```

**Compare commodities:**
```sql
SELECT commodity,
       AVG(group_PRICE_tone_avg) as avg_price_sentiment,
       SUM(group_PRICE_count) as total_articles
FROM commodity.silver.gdelt_wide
WHERE article_date >= CURRENT_DATE - 30
GROUP BY commodity
```

---

## Python REST API

```python
import requests

def query_databricks(sql):
    response = requests.post(
        "https://dbc-5e4780f4-fcec.cloud.databricks.com/api/2.0/sql/statements/",
        headers={
            "Authorization": "Bearer YOUR_TOKEN",
            "Content-Type": "application/json"
        },
        json={
            "warehouse_id": "d88ad009595327fd",
            "statement": sql,
            "wait_timeout": "50s"
        }
    )
    result = response.json()
    return result.get('result', {}).get('data_array', [])
```

---

## Performance Tips

1. **Always filter on partitions:** `WHERE commodity = 'coffee' AND article_date >= '2024-01-01'`
2. **Use LIMIT for exploration:** `LIMIT 100`
3. **Refresh if data missing:** `REFRESH TABLE commodity.silver.gdelt_wide`

---

## Common Issues

- **"Table not found"** → Use full name: `commodity.silver.gdelt_wide`
- **Slow query** → Add partition filters (commodity, article_date)
- **No results** → Check filters, verify data exists with `SELECT COUNT(*)`

# Research Agent

**Owners**: Stuart & Francisco
**Status**: ✅ Complete (GDELT integration pending)

## Purpose
Create unified data table from raw market, weather, and macro sources.

## Output
`commodity.silver.unified_data` - Single source of truth for forecasting

## Contents

- `sql/create_unified_data.sql` - Table creation script (includes commented GDELT integration)
- `notebooks/Data Exploration.dbquery.ipynb` - Data validation
- `UNIFIED_DATA_ARCHITECTURE.md` - Data joining strategy, forward-fill approach, hierarchy
- `GDELT_PROCESSING.md` - 🚧 **WIP**: GDELT sentiment processing decisions

## Key Features
- Deduplicated global data (VIX, macro, market)
- Forward-filled to cover non-trading days
- Regional weather data (65+ locations)
- Exchange rates for producer countries
- **Future**: GDELT sentiment (7 features) - code ready, pending enablement

## Current Data Sources

- Market: Coffee/Sugar futures prices
- Weather: Temperature, humidity, precipitation (65 regions)
- Macro: Exchange rates (24 currencies), VIX
- **Available**: GDELT sentiment (`commodity.bronze.bronze_gkg`)

## GDELT Integration (Ready to Enable)

GDELT sentiment data exists but is **not yet integrated**. See `GDELT_PROCESSING.md` for:
- Processing approach (bag-of-articles aggregation)
- 7 features available (tone, polarity, volatility, article count, etc.)
- Alternative approaches under consideration
- Open questions and feedback requests

To enable: Uncomment GDELT sections in `sql/create_unified_data.sql`

## Usage

```sql
-- Load in Databricks
SELECT * FROM commodity.silver.unified_data
WHERE commodity = 'Coffee' AND is_trading_day = 1
ORDER BY date DESC
LIMIT 100;
```

See `agent_instructions/DATA_CONTRACTS.md` for full schema.

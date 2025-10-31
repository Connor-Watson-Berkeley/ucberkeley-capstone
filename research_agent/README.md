# Research Agent

Data pipeline and feature engineering for commodity price forecasting.

## Purpose
Automated data collection pipeline (Lambda → S3 → Databricks) and unified feature table creation.

## Output
`commodity.silver.unified_data` - Single source of truth for forecasting

## Directory Structure

```
research_agent/
├── infrastructure/          # AWS Lambda + Databricks setup
│   ├── lambda/             # 6 data collection functions
│   ├── eventbridge/        # Daily scheduling (2AM UTC)
│   └── databricks/         # Landing + bronze SQL tables
├── ground_truth/           # Feature engineering modules
├── sql/                    # Silver layer (unified_data)
└── notebooks/              # Data exploration
```

## Contents

- `infrastructure/` - **NEW**: Automated data pipeline (see `infrastructure/README.md`)
- `sql/create_unified_data.sql` - Silver layer unified table
- `notebooks/Data Exploration.dbquery.ipynb` - Data validation
- `UNIFIED_DATA_ARCHITECTURE.md` - Data joining strategy
- `GDELT_PROCESSING.md` - GDELT sentiment processing

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

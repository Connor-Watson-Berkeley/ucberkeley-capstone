# Research Agent TODO

Last Updated: 2025-11-07

## In Progress

- [ ] Creating weather_v2 bronze table (running now)
  - Loading 252K+ records with corrected growing region coordinates
  - Expected completion: ~5 minutes

## Completed

- [x] Weather backfill v2 - 10+ years of historical data (2015-2025)
- [x] Full pipeline validation - 20/20 tests passed
- [x] Forecast API Guide validation - 7/7 tests passed
- [x] New Databricks workspace setup (Unity Catalog)
- [x] Fixed hardcoded credentials in git repo

## Pending

- [ ] Validate July 2021 frost event capture in weather_v2
- [ ] Update unified_data SQL to use weather_v2 instead of weather
- [ ] Train new SARIMAX models with corrected weather data
- [ ] Setup Databricks jobs for automated pipeline (manual via UI)

## Blocked

None

## Notes

- New workspace: `https://dbc-5e4780f4-fcec.cloud.databricks.com`
- SQL Warehouse: `d88ad009595327fd`
- All credentials moved to `.env` files (gitignored)

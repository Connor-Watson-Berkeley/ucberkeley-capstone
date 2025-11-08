# Tests & Validation

**All test and validation scripts for the data pipeline.**

## Running Tests

### Data Quality Tests
```bash
# Validate July 2021 frost event captured
python tests/validate_july2021_frost.py

# Check region coordinates are correct
python tests/validate_region_coordinates.py

# General data quality checks
python tests/validate_data_quality.py
python tests/validate_pipeline.py
```

### Catalog & Schema Tests
```bash
# Check Unity Catalog structure
python tests/check_catalog_structure.py

# Validate forecast schemas
python tests/check_forecast_schemas.py
python tests/test_forecast_schema.py
```

### Pipeline Tests
```bash
# Full pipeline test
python tests/test_full_pipeline.py

# Component tests
python tests/test_pipeline.py
python tests/test_unified_data.py
```

### Data Source Tests
```bash
# Check individual data sources
python tests/check_databricks_tables.py
python tests/check_gdelt_data.py
python tests/check_landing_sugar.py
python tests/check_sugar_weather.py
```

## Test Organization

- **validate_*.py** - Data quality validation
- **check_*.py** - Infrastructure/catalog checks
- **test_*.py** - Pipeline component tests

## Notes

- All tests load credentials from `../.env`
- Tests are read-only and safe to run in production
- Run tests after major data pipeline changes

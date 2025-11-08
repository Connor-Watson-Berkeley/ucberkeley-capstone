# Research Agent Infrastructure

**Data pipeline infrastructure for commodity forecasting platform.**

## 🔑 Setup

1. **Create `.env` file** (NEVER commit this!)
```bash
cp .env.example .env
# Edit .env with your Databricks credentials
```

2. **Install dependencies**
```bash
pip install python-dotenv databricks-sql-connector boto3
```

## 📂 Structure

```
infrastructure/
├── .env                                    # Secrets (gitignored)
├── .env.example                            # Template
├── backfill_historical_weather_v2.py       # Weather backfill (production)
├── create_weather_v2_bronze_table.py       # Create bronze table
├── create_unified_data.py                  # Build unified_data table
├── rebuild_all_layers.py                   # Rebuild bronze/silver/forecast
├── unity_catalog_workaround.py             # SQL connector fallback
├── databricks/                             # Databricks configs
│   ├── setup_unity_catalog_credentials.py  # Unity Catalog setup
│   ├── databricks_unity_catalog_*.json     # Cluster configs
│   └── *.sql                               # SQL setup scripts
├── tests/                                  # All tests & validation
│   ├── README.md                           # Test documentation
│   ├── validate_*.py                       # Data quality tests
│   ├── check_*.py                          # Infrastructure checks
│   └── test_*.py                           # Pipeline tests
└── archive/                                # Old scripts (reference only)
```

## 🚀 Key Scripts

### Weather Backfill (Production)
```bash
# Backfill historical weather with corrected coordinates
python backfill_historical_weather_v2.py --start-date 2015-07-07 --end-date 2025-11-05
```

### Unity Catalog Setup
```bash
# Configure Unity Catalog storage credentials
cd databricks
python setup_unity_catalog_credentials.py
```

### Data Pipeline
```bash
# Rebuild all data layers
python rebuild_all_layers.py

# Create/update unified_data
python create_unified_data.py
```

## 🧪 Testing

See [`tests/README.md`](tests/README.md) for test documentation.

```bash
# Run validation
python tests/validate_july2021_frost.py
python tests/validate_data_quality.py

# Check catalog structure
python tests/check_catalog_structure.py

# Full pipeline test
python tests/test_full_pipeline.py
```

## 📋 Utilities

- `dashboard_pipeline_health.py` - Pipeline monitoring dashboard
- `list_databricks_repos.py` - List Databricks repos
- `pull_databricks_repo.py` - Pull repo updates
- `load_historical_to_databricks.py` - Load historical data

## 📖 Documentation

- `DATABRICKS_MIGRATION_GUIDE.md` - Complete migration guide
- `MIGRATION_PREFLIGHT_CHECKLIST.md` - Quick migration checklist
- `CLEANUP_PLAN.md` - Cleanup decisions (this cleanup)

## 🔒 Security

- **NEVER commit `.env`** - Contains secrets
- **NEVER hardcode credentials** - Use environment variables
- All scripts load from `.env` via `python-dotenv`

## 📦 Archive

`archive/` contains old scripts kept for reference. Not used in production.

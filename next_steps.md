# Ground Truth Forecast Agent - Chat Summary for Claude Code

## Project Context
Connor is building "Ground Truth," a commodity price forecasting system for coffee 
and sugar futures as a 4-person capstone project. He's developing the Forecast Agent 
component that generates probabilistic price predictions for a Risk Agent built by 
teammates. The system runs in Databricks with approximately 4 weeks remaining in 
the project timeline.

## Key Achievements in This Session

### 1. Established Modular Architecture Philosophy
- Moved from monolithic V1 SARIMAX to modular, scalable design
- Emphasis on "demonstrate great techniques over perfect accuracy"
- Designed for experimentation with multiple model types (SARIMAX, LSTM, 
  transformers, synthetic forecasts)

### 2. Created Configuration System
Started with comprehensive configs, then refined to minimal necessary code:

Final Minimal Setup:
- config/model_registry.py - Model definitions, hyperparameters, feature 
  engineering function references
- core/feature_engineering.py - Transformation functions for regional aggregation 
  and covariate projection

Key Design Decision: Function-based approach (Connor's idea!)
- Instead of pre-defined feature engineering categories, models reference 
  transformation functions
- Provides infinite flexibility for experimentation
- Example: "feature_fn": feature_engineering.aggregate_weather_simple_mean

### 3. Feature Engineering Strategies Defined
Four regional aggregation approaches:
1. Simple mean across regions (baseline)
2. Weighted mean by production volume
3. Pivot regions as separate features (for LSTM)
4. Regional models (hierarchical approach)

Four covariate projection methods:
1. Persist (forward-fill last known value)
2. Seasonal average (historical patterns)
3. None (pure ARIMA)
4. Perfect foresight (upper bound testing)

### 4. Data Contract Defined
Two output tables:
- commodity.silver.point_forecasts - Daily forecasts with confidence intervals
- commodity.silver.distributions - 2,000 Monte Carlo sample paths per forecast

Input: commodity.silver.unified_data (already exists, created by Risk Agent)

### 5. Started Core Modules
Created core/data_loader.py with functions:
- load_unified_data() - Query unified data table
- prepare_features() - Apply feature engineering functions from config
- get_training_data() - Load with data cutoff (prevents leakage)
- project_covariates() - Project exog variables into forecast horizon

### 6. GitHub Repos Integration
Just established: https://github.com/stuhollandUCB/data-sci-210-capstone
- Connor is now a collaborator
- His code lives in forecast_agent/ folder
- Ready to migrate to proper Python package structure

## Critical Design Decisions

1. Data Grain: One row per (date, commodity, region) requires flexible 
   aggregation strategies
2. Data Leakage Prevention: data_cutoff_date always < forecast_date
3. Daily Forecast Frequency: Better simulates real-world trading than weekly
4. Synthetic Forecasts: For Risk Agent sensitivity analysis 
   (e.g., "60% directional accuracy, 5% MAPE")
5. Runtime Metadata: Store in tables (not pre-configured) - training time, 
   hardware used, performance metrics

## What's Next (Not Yet Built)

Immediate Next Steps:
1. base_forecaster.py - Abstract base class for all models
2. forecast_writer.py - Write to output tables
3. Refactor V1 SARIMAX into modular structure
4. backfill_engine.py - Orchestration (optional, can manual loop for now)

Package Structure for GitHub:
forecast_agent/
├── ground_truth/              # Python package
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── model_registry.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   ├── data_loader.py
│   │   └── (base_forecaster, forecast_writer, backfill_engine - TBD)
│   └── models/
│       ├── __init__.py
│       └── (sarimax_forecaster, lstm_forecaster, synthetic_forecaster - TBD)
└── notebooks/
    └── experiments/

## Files Ready to Use
1. config/model_registry.py - 11 pre-configured models
2. core/feature_engineering.py - All transformation functions
3. core/data_loader.py - Data loading utilities

## Key Principles Established
- Start minimal, build up as needed
- Function references > pre-defined categories
- Runtime logging > pre-configured estimates
- Demonstrate multiple approaches systematically
- Engineering excellence > perfect accuracy

## Technical Details

### Model Registry Structure (function-based)
Each model config includes:
- class: Model class name (e.g., "SARIMAXForecaster")
- hyperparameters: Dict of model-specific params
- features: List of feature column names
- commodity: "Coffee" or "Sugar"
- feature_fn: Reference to feature engineering function
- covariate_projection_fn: Reference to projection function
- training_mode: "rolling" (retrain each date) or "incremental" (fewer retrains)
- table_paths: Output table names
- mlflow_experiment: Experiment name for tracking

### Data Loader Key Functions
- Loads from commodity.silver.unified_data
- Applies feature engineering based on model config
- Handles data cutoff to prevent leakage
- Projects covariates into forecast horizon
- Returns properly formatted data for model training

### Feature Engineering Functions
All functions follow signature:
def function_name(df_spark, commodity, features, cutoff_date=None):
    # Transform data
    return transformed_df

This allows easy swapping of different strategies by changing function reference 
in model registry.

## Current State
- V1 SARIMAX exists and works (not yet refactored)
- Configuration foundation complete
- Data loading module complete
- Ready to build: base_forecaster, forecast_writer, model implementations
- GitHub repo established, ready for package structure migration

# DARTS Forecasting Experiments

This directory contains experiments using the [DARTS](https://unit8co.github.io/darts/) library for time series forecasting on commodity price data.

## Overview

DARTS (Python library for time series forecasting) provides a unified interface for:
- Deep learning models (TFT, N-BEATS, N-HiTS, TCN, Transformer, LSTM)
- Statistical models (ARIMA, Prophet, TBATS, Exponential Smoothing)
- Machine learning models (XGBoost, LightGBM, RandomForest)

These experiments focus on forecasting coffee prices using weather covariates.

## Experiments

### 1. quick_darts_test.py
**Purpose**: Lightweight validation test for DARTS installation
**Model**: N-BEATS (reduced complexity)
**Data**: 90 days of coffee prices
**Training**: 10 epochs (fast execution)

```bash
cd forecast-experiments
set -a && source ../../infra/.env && set +a
python quick_darts_test.py
```

**Expected Runtime**: 2-5 minutes

### 2. darts_nbeats_experiment.py
**Purpose**: Full N-BEATS experiment with production parameters
**Model**: N-BEATS (Neural Basis Expansion Analysis for Time Series)
**Data**: 365 days of coffee prices + 7 weather covariates
**Training**: 100 epochs
**Features**:
- State-of-the-art univariate/multivariate forecasting
- Generic architecture (can also use interpretable)
- Past covariates support (weather data)

```bash
cd forecast-experiments
set -a && source ../../infra/.env && set +a
python darts_nbeats_experiment.py
```

**Expected Runtime**: 20-40 minutes (depending on hardware)

### 3. darts_tft_experiment.py
**Purpose**: Temporal Fusion Transformer experiment
**Model**: TFT (Temporal Fusion Transformer)
**Data**: 365 days of coffee prices + 7 weather covariates
**Training**: 50 epochs
**Features**:
- Multi-horizon probabilistic forecasting
- Attention mechanisms for interpretability
- Quantile regression (10th, 50th, 90th percentiles)
- Weather covariate integration

```bash
cd forecast-experiments
set -a && source ../../infra/.env && set +a
python darts_tft_experiment.py
```

**Expected Runtime**: 30-60 minutes

### 4. databricks_darts_tft_experiment.py
**Purpose**: Databricks notebook version of TFT experiment
**Platform**: Databricks Workspace
**Format**: Notebook with `# COMMAND ----------` cell markers
**Usage**: Upload to Databricks workspace and run

**Upload Steps**:
1. Go to Databricks Workspace
2. Navigate to Repos or Workspace folder
3. Click "Import" → "File"
4. Upload `databricks_darts_tft_experiment.py`
5. Run cells sequentially

**Note**: This version uses Spark SQL (`spark.sql()`) instead of databricks-sql connector.

## Prerequisites

### 1. Install DARTS

```bash
pip3 install darts
```

This installs DARTS v0.38.0+ with PyTorch Lightning and all dependencies.

### 2. Download Local Data (Recommended for Local Experiments)

To avoid Databricks serverless costs during experimentation, download the unified_data once:

```bash
cd forecast-experiments
# Make sure you have Databricks credentials configured first
set -a && source ../../infra/.env && set +a
python3 download_data_local.py
```

This creates a `data/unified_data.parquet` file (~0.76 MB) that contains all Coffee data from 2015-2025 across 22 regions. The data directory is gitignored and won't be committed.

**Note**: The experiments automatically use local data if available. You only need Databricks credentials for the initial download.

### 3. Configure Databricks Credentials (One-Time Setup)

Only needed for initial data download:

**Option A: Interactive Setup (Recommended)**
```bash
cd ../infra
./setup_credentials.sh
```

**Option B: Manual Setup**
```bash
cd ../infra
cp .env.template .env
# Edit .env and replace YOUR_TOKEN_HERE with your actual token
```

See [infra/README.md](../../infra/README.md) for detailed instructions on generating tokens.

## Weather Covariates

All experiments use 7 weather features from the unified data table:

1. `temperature_2m_mean` - Mean temperature (°C)
2. `precipitation_sum` - Total precipitation (mm)
3. `wind_speed_10m_max` - Maximum wind speed (m/s)
4. `relative_humidity_2m_mean` - Mean relative humidity (%)
5. `soil_moisture_0_to_7cm_mean` - Mean soil moisture (m³/m³)
6. `shortwave_radiation_sum` - Solar radiation (MJ/m²)
7. `et0_fao_evapotranspiration_sum` - Evapotranspiration (mm)

These covariates are used as "past covariates" - observed values that influence price.

## Model Comparison

| Model | Strengths | Forecast Horizon | Probabilistic | Interpretable |
|-------|-----------|------------------|---------------|---------------|
| **N-BEATS** | State-of-the-art performance | 14 days | No | Optional |
| **TFT** | Multi-variate, attention-based | 14 days | Yes (quantiles) | Yes (attention) |
| N-HiTS | Hierarchical, efficient | 14 days | No | No |
| TCN | Lightweight, fast | 14 days | No | No |

**Recommendation**: Start with TFT for interpretability and probabilistic forecasts. Use N-BEATS for best performance.

## Output Structure

Each experiment generates:

1. **Console Output**:
   - Training progress
   - Validation metrics (MAPE, RMSE, MAE)
   - Forecast summary table

2. **Metrics**:
   - MAPE (Mean Absolute Percentage Error)
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)

3. **Forecasts**:
   - 14-day point forecasts
   - Confidence intervals (TFT only)
   - Forecast dates and values

## Troubleshooting

### Error: "Invalid access token"

Your Databricks token has expired. Generate a fresh token:
```bash
cd ../infra
./setup_credentials.sh
```

### Error: "No module named 'darts'"

Install DARTS:
```bash
pip3 install darts
```

### Error: "No such file or directory: ../infra/.env"

Create credentials file:
```bash
cd ../infra
cp .env.template .env
# Edit .env with your token
```

### CUDA/GPU Issues

DARTS will automatically use GPU if available (via PyTorch). If you encounter GPU issues:

1. Check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

2. Force CPU mode by adding to model initialization:
```python
pl_trainer_kwargs={"accelerator": "cpu"}
```

## Next Steps

1. **Run Quick Test**: Validate installation with `quick_darts_test.py`
2. **Run Full Experiments**: Execute N-BEATS and TFT experiments
3. **Compare Results**: Evaluate metrics against existing models (LSTM, SARIMAX, XGBoost)
4. **Hyperparameter Tuning**: Adjust model parameters for better performance
5. **Deploy to Databricks**: Upload notebook version for scheduled training

## Resources

- DARTS Documentation: https://unit8co.github.io/darts/
- TFT Paper: https://arxiv.org/abs/1912.09363
- N-BEATS Paper: https://arxiv.org/abs/1905.10437
- DARTS GitHub: https://github.com/unit8co/darts
- DARTS Examples: https://unit8co.github.io/darts/examples/

## Model Files Location

Trained models are saved to:
- **Local**: `~/.darts/checkpoints/` (temporary)
- **Databricks**: Should be saved to Unity Catalog or DBFS

To persist models for production, implement custom save/load logic to store in Databricks.

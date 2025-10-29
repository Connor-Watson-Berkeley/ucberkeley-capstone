# Coffee Futures Forecasting System: V1 Findings & Strategic Roadmap

## Executive Summary

This document provides a comprehensive overview of our **validated coffee futures forecasting system**, current V1 findings, and strategic roadmap for enterprise deployment and enhancement. Our system has achieved **54.95% directional accuracy** on 7-day forecasts with statistical significance (p=0.0003), representing a genuine trading edge validated across 1,204 predictions over 5+ years of out-of-sample data.

### **V1 System Achievements (GitHub Gold Standard)**
- **54.95% directional accuracy** on 7-day forecasts (statistically significant, p=0.0003)
- **SARIMAX(1,1,1)** with 5 dynamic covariates (weather + macro)
- **2,000 sample paths** for risk assessment via Monte Carlo simulation
- **Validated across 1,204 predictions** over 5+ years out-of-sample data
- **Production-ready package** in `coffee_forecast_prototype/` folder
- **Code audited** with 10/10 checks passed
- **Tested 11 different models** comprehensively

### **Immediate Next Step (Phase 1.5)**
**Goal**: Get the E2E pipeline running in Databricks - data ingestion to distribution file output for the risk agent.

**Core Requirements**:
- Migrate GitHub SARIMAX model to Databricks
- Use the new unified dataset from `DATASET_FOR_FORECAST_V2.md`
- Establish complete E2E pipeline in Databricks
- Generate 7-day forecasts with 2,000 sample paths
- Output distribution files for risk agent consumption
- Maintain 54.95% directional accuracy from V1 system

### **Future Phases (2-4)**
2. **Phase 2**: Extend to hierarchical forecasting (Region → Country → Global)
3. **Phase 3**: Generate skewed distributions reflecting real-world price dynamics
4. **Phase 4**: Enterprise scaling (APIs, monitoring, cost optimization)

### **Key Deliverables (Phase 1.5)**
- **Complete E2E pipeline** in Databricks (data ingestion → forecast → output)
- **Distribution output files** for risk agent consumption
- **Maintained performance** (54.95% directional accuracy)
- **Simple file-based interface** for risk agent integration

## Table of Contents

1. [V1 System Overview](#v1-system-overview)
2. [Current Architecture & Findings](#current-architecture)
3. [Strategic Vision & Roadmap](#strategic-vision)
4. [Databricks Migration Guide](#databricks-migration)
5. [Hierarchical Forecasting Framework](#hierarchical-forecasting)
6. [Skewed Distribution Implementation](#skewed-distributions)
7. [Trading Agent Integration](#trading-agent-integration)
8. [API Design & Output Contracts](#api-design)
9. [Implementation Details](#implementation-details)
10. [Testing & Validation Framework](#testing-validation)
11. [Deployment Strategy](#deployment-strategy)
12. [Monitoring & Maintenance](#monitoring-maintenance)
13. [Advanced Considerations](#advanced-considerations)
14. [Cost Optimization](#cost-optimization)
15. [Long-term Strategic Vision](#long-term-vision)

---

## V1 System Overview

### What We've Built

Our coffee futures forecasting system is a **production-ready, statistically validated** solution that:

- **Achieves 54.95% directional accuracy** on 7-day forecasts
- **Beats random chance** by 4.95 percentage points (statistically significant)
- **Uses SARIMAX(1,1,1)** with 5 dynamic covariates
- **Generates 2,000 sample paths** for risk assessment
- **Validated across 172 walk-forward folds** over 5+ years
- **Code audited** with 10/10 checks passed

### Key Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Directional Accuracy** | **54.95%** | Predicts up/down correctly 55% of time |
| **P-value** | **0.0003** | Highly statistically significant |
| **Sample Size** | **1,204** | Based on 172 walk-forward folds |
| **Test Period** | **Dec 2020 - Oct 2025** | 5 years out-of-sample |
| **Edge over Random** | **+4.95 pp** | Meaningful trading advantage |
| **RMSE** | **10.54** | Price error magnitude |

### Models Tested & Results

We tested **11 different models** comprehensively:

| Rank | Model | Directional Accuracy | RMSE | Status |
|------|-------|---------------------|------|--------|
| 🥇 | **SARIMAX + Covariates** | **54.95%** | 10.54 | ✅ **CHAMPION** |
| 🥈 | XGBoost | ~53% | 6.44 | ✅ Production ready |
| 🥉 | Naive (Last Value) | 47.6% | 4.12 | ✅ Best RMSE baseline |
| 4th | Chronos-mini (inverted) | 49.0% | 7.16 | ✅ Tested contrarian |
| 5th | Moving Average | 61.8% | 5.76 | ⚠️ Uses future data |
| 6th+ | Chronos variants | 28-43% | 6-9 | ❌ Foundation models failed |

**Key Insight**: Simple statistical models with domain expertise outperform complex foundation models for commodity forecasting.

---

## Strategic Vision & Roadmap

### **Phase 1: Current V1 System (Completed)**
✅ **Production-ready forecasting system** with 54.95% directional accuracy  
✅ **Comprehensive model testing** (11 models evaluated)  
✅ **Code audit** (10/10 checks passed)  
✅ **GitHub deployment** (`coffee_forecast_prototype/`)  
✅ **Backtesting framework** with data leakage warnings  

### **Phase 1.5: Databricks E2E Pipeline (Next 3 months)**
🔄 **Databricks migration** - get the system running in Databricks  
🔄 **E2E pipeline** - data ingestion to distribution file output  
🔄 **Risk agent integration** - usable output files for trading team  
🔄 **Basic validation** - maintain 54.95% directional accuracy  

### **Phase 2: Advanced Forecasting (6-12 months)**
📋 **Hierarchical forecasting** (Region → Country → Global)  
📋 **Skewed distributions** for realistic price dynamics  
📋 **Multi-commodity support** (Coffee, Sugar, Cocoa)  
📋 **Real-time weather integration** (65+ locations)  

### **Phase 3: Enterprise Scaling (12+ months)**
📋 **14-week ahead API** for strategic planning  
📋 **Enhanced monitoring** and alerting systems  
📋 **Cost optimization** and resource management  
📋 **Trading agent API** with WebSocket support  

---

## Current Architecture (GitHub Gold Standard)

### **GitHub Repository Structure**
```
https://github.com/stuhollandUCB/data-sci-210-capstone/
├── coffee_forecast_prototype/          # Production-ready package
│   ├── forecast.py                     # Core forecasting script
│   ├── data.csv                        # Historical dataset
│   ├── model.pkl                       # Pretrained SARIMAX model
│   ├── README.md                       # Usage instructions
│   ├── example_usage.py                # Integration examples
│   └── backtest_example.py             # Backtesting framework
├── research_agent/                     # Hierarchical forecasting requirements
│   ├── requirements/index.html         # ERD for hierarchical structure
│   └── requirements/SCHEMA.md          # Technical specifications
└── forecasting/                        # Development and testing
    ├── test_7day_forecast.py           # Validation framework
    ├── code_audit.py                   # Quality assurance
    └── results/final_report/           # Performance analysis
```

### **Current Data Sources (Validated)**

| Source | Provider | Frequency | Records | Date Range | Purpose | GitHub Reference |
|--------|----------|-----------|---------|------------|---------|------------------|
| Coffee Futures | Trading Economics API (KC1:COM) | Daily | 2,709 | 2015-2025 | Target variable | `data.csv` |
| Weather (Global) | NASA POWER API | Daily | 19,722 | 2015-2025 | Top 3 covariates | Aggregated in `forecast.py` |
| Exchange Rates | FRED API | Daily | 3,644 | 2015-2025 | Macro covariates | BRL/USD, INR/USD |
| Macro Indicators | FRED API | Daily | 3,644 | 2015-2025 | Economic context | VIX, CFTC data |
| Regional Weather | NASA POWER API | Daily | 65 locations | 2015-2025 | Enhanced features | `collect_enhanced_weather.py` |

### **Current Model Configuration (From GitHub)**
```python
# From coffee_forecast_prototype/forecast.py
model = SARIMAX(
    endog=df[target],
    exog=df[covariates],
    order=(1, 1, 1),                    # Validated configuration
    seasonal_order=(0, 0, 0, 0),       # No seasonality (tested)
    trend=None,                         # Avoid constant conflict
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Top 5 Covariates (from GitHub validation)
covariates = [
    'coffee_temp_c',      # Global coffee temperature
    'coffee_precip_mm',   # Global coffee precipitation  
    'coffee_humidity',    # Global coffee humidity
    'brl_usd',           # Brazilian Real exchange rate
    'inr_usd'            # Indian Rupee exchange rate
]
```

### **Current Output Structure (GitHub Standard)**

**forecast.csv** (from `coffee_forecast_prototype/forecast.py`):
```csv
forecast_date,day_ahead,forecast_mean,forecast_std,lower_95,upper_95
2025-01-15,1,167.23,2.45,162.43,172.03
2025-01-16,2,167.89,3.12,161.77,174.01
2025-01-17,3,168.45,3.78,161.04,175.86
2025-01-18,4,169.12,4.23,160.83,177.41
2025-01-19,5,168.78,4.67,159.63,177.93
2025-01-20,6,169.45,5.12,159.42,179.48
2025-01-21,7,170.12,5.56,159.22,181.02
```

**distribution.csv** (2,000 sample paths):
```csv
path_id,day_1,day_2,day_3,day_4,day_5,day_6,day_7
1,167.45,168.12,166.89,169.23,168.45,167.12,168.89
2,166.78,167.23,168.45,167.89,169.12,168.23,167.45
3,168.12,169.45,167.23,170.12,168.89,169.45,170.12
...
2000,167.89,166.45,168.12,167.23,169.45,168.12,167.89
```

### **GitHub Code Examples (Production Ready)**

**Core Forecasting Logic** (from `coffee_forecast_prototype/forecast.py`):
```python
# Load and prepare data
df = pd.read_csv('data.csv', index_col='date', parse_dates=True)
target = 'coffee_close'
covariates = ['coffee_temp_c', 'coffee_precip_mm', 'coffee_humidity', 'brl_usd', 'inr_usd']

# Clean data (handle inf/nan values)
for col in [target] + covariates:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.ffill().bfill()

# Train SARIMAX model
model = SARIMAX(
    endog=df[target],
    exog=df[covariates],
    order=(1, 1, 1),
    seasonal_order=(0, 0, 0, 0),
    trend=None,
    enforce_stationarity=False,
    enforce_invertibility=False
)
fitted_model = model.fit(disp=False)

# Generate 7-day forecast
recent_covs = df[covariates].iloc[-7:].copy()
recent_covs = recent_covs.replace([np.inf, -np.inf], np.nan).ffill().bfill()

forecast_result = fitted_model.get_forecast(steps=7, exog=recent_covs)
forecast_mean = forecast_result.predicted_mean.values
forecast_se = forecast_result.se_mean.values

# Generate 2,000 sample paths
N_PATHS = 2000
residual_std = np.sqrt(fitted_model.params.get('sigma2', forecast_se.mean()**2))

sample_paths = []
for i in range(N_PATHS):
    path = []
    for day in range(7):
        shock = np.random.normal(0, residual_std)
        next_price = forecast_mean[day] + shock
        path.append(next_price)
    sample_paths.append(path)
```

**Backtesting Framework** (from `coffee_forecast_prototype/backtest_example.py`):
```python
# Backtesting with cutoff dates (no data leakage)
def backtest_forecast(cutoff_date):
    """Generate forecast using only data prior to cutoff_date"""
    
    # Load data up to cutoff (NO FUTURE DATA)
    df = pd.read_csv('data.csv', index_col='date', parse_dates=True)
    df_train = df[df.index <= cutoff_date]
    
    if len(df_train) < 1500:  # Minimum training data
        return None
    
    # Train model on historical data only
    model = SARIMAX(
        endog=df_train[target],
        exog=df_train[covariates],
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        trend=None,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    fitted_model = model.fit(disp=False)
    
    # Generate forecast for next 7 days
    recent_covs = df_train[covariates].iloc[-7:].copy()
    recent_covs = recent_covs.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    forecast_result = fitted_model.get_forecast(steps=7, exog=recent_covs)
    
    return forecast_result.predicted_mean.values
```

---

## Hierarchical Forecasting Vision

### **Research Agent Requirements (index.html)**

Based on the research agent requirements in `research_agent/requirements/index.html`, our long-term vision includes:

**Hierarchical Structure**:
```
Global Level (Coffee Futures Price)
    ↓
Country Level (Brazil, Colombia, Vietnam, etc.)
    ↓  
Region Level (Minas Gerais, Huila, Central Highlands, etc.)
```

**Key Features from ERD**:
- **Multi-commodity support** (Coffee, Sugar, Cocoa, etc.)
- **Regional weather data** (65+ locations)
- **Production aggregation** (Region → Country → Global)
- **Agronomic features** (growth periods, ideal temperatures)
- **Sparse data handling** (gap-filling, quality scores)

### **Current Dataset (Global + Region Pivot)**

**Phase 1: Current Global Dataset**
```python
# Current structure (from GitHub)
df = pd.read_csv('data.csv', index_col='date', parse_dates=True)

# Global aggregated features
global_features = {
    'coffee_temp_c': 'Global coffee temperature (weighted by production)',
    'coffee_precip_mm': 'Global coffee precipitation (weighted by production)', 
    'coffee_humidity': 'Global coffee humidity (weighted by production)',
    'brl_usd': 'Brazilian Real exchange rate',
    'inr_usd': 'Indian Rupee exchange rate'
}
```

**Phase 2: Regional Pivot Dataset**
```python
# Enhanced regional structure (from collect_enhanced_weather.py)
regional_features = {
    'brazil_temp_c': 'Brazil coffee regions temperature',
    'colombia_temp_c': 'Colombia coffee regions temperature', 
    'vietnam_temp_c': 'Vietnam coffee regions temperature',
    'ethiopia_temp_c': 'Ethiopia coffee regions temperature',
    # ... 65 total locations
}

# Regional production weights
production_weights = {
    'brazil': 0.37,      # 37% of global production
    'vietnam': 0.17,     # 17% of global production
    'colombia': 0.08,    # 8% of global production
    'ethiopia': 0.05,    # 5% of global production
    # ... other regions
}
```

### **Hierarchical Forecasting Implementation**

**Level 1: Regional Models**
```python
# Regional SARIMAX models
def train_regional_models():
    """Train separate models for each major coffee region"""
    
    regional_models = {}
    
    for region in ['brazil', 'colombia', 'vietnam', 'ethiopia']:
        # Regional features
        regional_covariates = [
            f'{region}_temp_c',
            f'{region}_precip_mm', 
            f'{region}_humidity',
            'brl_usd',  # Global macro factors
            'inr_usd'
        ]
        
        # Train regional model
        model = SARIMAX(
            endog=df['coffee_close'],  # Global price target
            exog=df[regional_covariates],
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            trend=None,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=False)
        regional_models[region] = fitted_model
    
    return regional_models

# Regional forecasts
def generate_regional_forecasts(regional_models):
    """Generate forecasts from each regional model"""
    
    regional_forecasts = {}
    
    for region, model in regional_models.items():
        regional_covariates = [
            f'{region}_temp_c',
            f'{region}_precip_mm',
            f'{region}_humidity', 
            'brl_usd',
            'inr_usd'
        ]
        
        recent_covs = df[regional_covariates].iloc[-7:].copy()
        recent_covs = recent_covs.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        forecast_result = model.get_forecast(steps=7, exog=recent_covs)
        regional_forecasts[region] = forecast_result.predicted_mean.values
    
    return regional_forecasts
```

**Level 2: Hierarchical Ensemble**
```python
# Hierarchical ensemble with production weights
def hierarchical_ensemble(regional_forecasts, production_weights):
    """Combine regional forecasts using production weights"""
    
    ensemble_forecast = np.zeros(7)  # 7-day forecast
    
    for region, forecast in regional_forecasts.items():
        weight = production_weights.get(region, 0.0)
        ensemble_forecast += weight * forecast
    
    return ensemble_forecast

# Example usage
regional_models = train_regional_models()
regional_forecasts = generate_regional_forecasts(regional_models)
hierarchical_forecast = hierarchical_ensemble(regional_forecasts, production_weights)
```

---

## Skewed Distribution Requirements

### **Why Skewed Distributions Matter**

**Real-World Price Dynamics**:
- **Supply shocks** (frost, drought) → Right skew (price spikes)
- **Demand shocks** (recession, substitution) → Left skew (price drops)
- **Market sentiment** → Asymmetric volatility
- **Weather events** → Fat tails and skewness

**Current Limitation** (GitHub system):
```python
# Current: Normal distribution assumption
shock = np.random.normal(0, residual_std)  # Symmetric
next_price = forecast_mean[day] + shock
```

### **Skewed Distribution Implementation**

**Method 1: Skewed Normal Distribution**
```python
from scipy.stats import skewnorm
import numpy as np

def generate_skewed_sample_paths(forecast_mean, forecast_se, skewness=0.5, N_PATHS=2000):
    """Generate sample paths with skewness"""
    
    sample_paths = []
    
    for i in range(N_PATHS):
        path = []
        for day in range(7):
            # Generate skewed shock
            skew_shock = skewnorm.rvs(skewness, loc=0, scale=forecast_se[day])
            next_price = forecast_mean[day] + skew_shock
            path.append(next_price)
        sample_paths.append(path)
    
    return sample_paths

# Detect skewness from historical residuals
def estimate_skewness(model, historical_data):
    """Estimate skewness from model residuals"""
    
    residuals = model.resid
    skewness = stats.skew(residuals)
    
    return skewness

# Generate skewed forecasts
skewness = estimate_skewness(fitted_model, df)
skewed_paths = generate_skewed_sample_paths(forecast_mean, forecast_se, skewness)
```

**Method 2: Mixture of Distributions**
```python
def generate_mixture_sample_paths(forecast_mean, forecast_se, N_PATHS=2000):
    """Generate sample paths using mixture of normal and extreme distributions"""
    
    sample_paths = []
    
    for i in range(N_PATHS):
        path = []
        for day in range(7):
            # 90% normal, 10% extreme (fat tails)
            if np.random.random() < 0.9:
                # Normal shock
                shock = np.random.normal(0, forecast_se[day])
            else:
                # Extreme shock (3x volatility)
                shock = np.random.normal(0, forecast_se[day] * 3)
            
            next_price = forecast_mean[day] + shock
            path.append(next_price)
        sample_paths.append(path)
    
    return sample_paths
```

**Method 3: Regime-Dependent Skewness**
```python
def generate_regime_skewed_paths(forecast_mean, forecast_se, market_regime, N_PATHS=2000):
    """Generate paths with regime-dependent skewness"""
    
    # Define skewness by market regime
    regime_skewness = {
        'normal': 0.0,      # Symmetric
        'volatile': 0.3,    # Slight right skew
        'crisis': -0.5,     # Left skew (downward pressure)
        'boom': 0.8         # Strong right skew (upward pressure)
    }
    
    skewness = regime_skewness.get(market_regime, 0.0)
    
    sample_paths = []
    for i in range(N_PATHS):
        path = []
        for day in range(7):
            # Generate regime-appropriate shock
            if skewness == 0.0:
                shock = np.random.normal(0, forecast_se[day])
            else:
                shock = skewnorm.rvs(skewness, loc=0, scale=forecast_se[day])
            
            next_price = forecast_mean[day] + shock
            path.append(next_price)
        sample_paths.append(path)
    
    return sample_paths

# Detect market regime
def detect_market_regime(df, lookback=30):
    """Detect current market regime based on recent volatility and trends"""
    
    recent_returns = df['coffee_close'].pct_change().tail(lookback)
    volatility = recent_returns.std()
    trend = recent_returns.mean()
    
    if volatility > recent_returns.std() * 1.5:
        if trend > 0:
            return 'boom'
        else:
            return 'crisis'
    elif volatility > recent_returns.std() * 1.2:
        return 'volatile'
    else:
        return 'normal'

# Generate regime-aware forecasts
market_regime = detect_market_regime(df)
regime_skewed_paths = generate_regime_skewed_paths(forecast_mean, forecast_se, market_regime)
```

### **Enhanced Distribution Output**

**Updated distribution.csv with skewness metadata**:
```csv
path_id,day_1,day_2,day_3,day_4,day_5,day_6,day_7,skewness,regime,distribution_type
1,167.45,168.12,166.89,169.23,168.45,167.12,168.89,0.3,volatile,skewed_normal
2,166.78,167.23,168.45,167.89,169.12,168.23,167.45,0.3,volatile,skewed_normal
...
2000,167.89,166.45,168.12,167.23,169.45,168.12,167.89,0.3,volatile,skewed_normal
```

**Distribution Statistics**:
```python
def calculate_distribution_stats(sample_paths):
    """Calculate distribution statistics for validation"""
    
    stats = {}
    
    for day in range(7):
        day_prices = [path[day] for path in sample_paths]
        
        stats[f'day_{day+1}'] = {
            'mean': np.mean(day_prices),
            'std': np.std(day_prices),
            'skewness': stats.skew(day_prices),
            'kurtosis': stats.kurtosis(day_prices),
            'p5': np.percentile(day_prices, 5),
            'p25': np.percentile(day_prices, 25),
            'p50': np.percentile(day_prices, 50),
            'p75': np.percentile(day_prices, 75),
            'p95': np.percentile(day_prices, 95)
        }
    
    return stats

# Validate skewed distributions
distribution_stats = calculate_distribution_stats(skewed_paths)
print(f"Day 7 skewness: {distribution_stats['day_7']['skewness']:.3f}")
print(f"Day 7 kurtosis: {distribution_stats['day_7']['kurtosis']:.3f}")
```

---

## Databricks Requirements

### Infrastructure Requirements

**Compute**:
- **Minimum**: 2-4 cores, 8GB RAM (for single model)
- **Recommended**: 4-8 cores, 16GB RAM (for ensemble + backfill)
- **For backfill**: Auto-scaling cluster (4-16 cores based on workload)

**Storage**:
- **Delta Lake**: For time series data with versioning
- **Unity Catalog**: For data governance and lineage
- **Volume**: ~10GB for historical data + models + outputs

**Networking**:
- **API Access**: Trading Economics, NASA POWER, FRED APIs
- **External Storage**: S3/GCS for model artifacts
- **VPC**: For secure API connections

### Databricks Features Required

**Core Features**:
- **Databricks Workflows**: For scheduled forecasting
- **Delta Lake**: For ACID transactions on time series
- **MLflow**: For model versioning and tracking
- **Unity Catalog**: For data governance
- **Databricks SQL**: For API endpoints

**Advanced Features**:
- **Delta Live Tables**: For data pipeline orchestration
- **Feature Store**: For covariate management
- **Model Serving**: For real-time API endpoints
- **Alerting**: For model performance monitoring

### Python Environment

**Core Dependencies**:
```python
# Time series modeling
statsmodels>=0.14.0
scipy>=1.10.0
numpy>=1.24.0
pandas>=2.0.0

# Machine learning
scikit-learn>=1.3.0
xgboost>=1.7.0

# Data sources
yfinance>=0.2.0
fredapi>=0.5.0
requests>=2.31.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Databricks specific
databricks-sdk>=0.12.0
mlflow>=2.5.0
delta-spark>=2.4.0
```

---

## Data Pipeline Design

### **Migration from GitHub Structure**

**Current GitHub Data Structure** (from `coffee_forecast_prototype/data.csv`):
```python
# Current CSV structure (GitHub gold standard)
df = pd.read_csv('data.csv', index_col='date', parse_dates=True)

# Columns in current dataset
columns = [
    'coffee_close',      # Target variable (Trading Economics KC1:COM)
    'coffee_temp_c',     # Global coffee temperature (NASA POWER)
    'coffee_precip_mm',  # Global coffee precipitation (NASA POWER)
    'coffee_humidity',   # Global coffee humidity (NASA POWER)
    'brl_usd',          # Brazilian Real exchange rate (FRED)
    'inr_usd',          # Indian Rupee exchange rate (FRED)
    'vix',              # Volatility index (Yahoo Finance)
    'cftc_net_position' # CFTC positioning data
]
```

### **Databricks Delta Lake Schema Design**

**Raw Data Tables** (migrating from GitHub sources):
```sql
-- Raw futures data (from Trading Economics API)
CREATE TABLE raw.coffee_futures (
    date DATE,
    symbol STRING,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    data_source STRING DEFAULT 'trading_economics',
    ingestion_timestamp TIMESTAMP,
    -- GitHub reference
    github_source STRING DEFAULT 'coffee_forecast_prototype/data.csv'
) USING DELTA
PARTITIONED BY (YEAR(date), MONTH(date));

-- Raw weather data  
CREATE TABLE raw.weather_data (
    date DATE,
    location STRING,
    latitude DOUBLE,
    longitude DOUBLE,
    temperature_c DOUBLE,
    precipitation_mm DOUBLE,
    humidity_pct DOUBLE,
    data_source STRING,
    ingestion_timestamp TIMESTAMP
) USING DELTA
PARTITIONED BY (YEAR(date), MONTH(date));

-- Raw exchange rates
CREATE TABLE raw.exchange_rates (
    date DATE,
    currency_pair STRING,
    rate DOUBLE,
    data_source STRING,
    ingestion_timestamp TIMESTAMP
) USING DELTA
PARTITIONED BY (YEAR(date), MONTH(date));
```

**Feature Engineering Tables**:
```sql
-- Aggregated coffee weather features
CREATE TABLE features.coffee_weather_aggregated (
    date DATE,
    coffee_temp_c DOUBLE,
    coffee_precip_mm DOUBLE,
    coffee_humidity DOUBLE,
    feature_engineering_timestamp TIMESTAMP
) USING DELTA
PARTITIONED BY (YEAR(date), MONTH(date));

-- Final modeling dataset
CREATE TABLE features.modeling_dataset (
    date DATE,
    coffee_close DOUBLE,
    coffee_temp_c DOUBLE,
    coffee_precip_mm DOUBLE,
    coffee_humidity DOUBLE,
    brl_usd DOUBLE,
    inr_usd DOUBLE,
    vix DOUBLE,
    cftc_net_position DOUBLE,
    feature_engineering_timestamp TIMESTAMP
) USING DELTA
PARTITIONED BY (YEAR(date), MONTH(date));
```

### Data Pipeline Architecture

**Delta Live Tables Pipeline**:
```python
# 01_raw_futures_data.py
@dlt.table(
    name="coffee_futures",
    table_properties={
        "quality": "bronze",
        "pipelines.autoOptimize.autoCompact.enabled": "true"
    }
)
def raw_coffee_futures():
    """Ingest raw coffee futures data from Trading Economics API"""
    # API call logic
    # Data validation
    # Write to Delta table
    pass

# 02_raw_weather_data.py  
@dlt.table(
    name="weather_data",
    table_properties={
        "quality": "bronze", 
        "pipelines.autoOptimize.autoCompact.enabled": "true"
    }
)
def raw_weather_data():
    """Ingest weather data from NASA POWER API"""
    # API call logic
    # Data validation
    # Write to Delta table
    pass

# 03_features_coffee_weather.py
@dlt.table(
    name="coffee_weather_aggregated",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.autoCompact.enabled": "true"
    }
)
def coffee_weather_features():
    """Aggregate weather data for coffee growing regions"""
    # Spatial aggregation logic
    # Quality checks
    # Write to Delta table
    pass

# 04_features_modeling_dataset.py
@dlt.table(
    name="modeling_dataset", 
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.autoCompact.enabled": "true"
    }
)
def modeling_dataset():
    """Create final modeling dataset with all features"""
    # Join all data sources
    # Handle missing values
    # Feature engineering
    # Write to Delta table
    pass
```

### Data Quality Framework

**Expectations**:
```python
# Data quality expectations
@dlt.expect("valid_price_range", "coffee_close BETWEEN 50 AND 500")
@dlt.expect("no_null_targets", "coffee_close IS NOT NULL")
@dlt.expect("valid_dates", "date >= '2015-01-01'")
@dlt.expect("temperature_range", "coffee_temp_c BETWEEN -10 AND 50")
@dlt.expect("humidity_range", "coffee_humidity BETWEEN 0 AND 100")
def modeling_dataset():
    # Implementation
    pass
```

**Monitoring**:
```python
# Data drift detection
def detect_data_drift():
    """Monitor for data drift in key features"""
    current_stats = spark.sql("""
        SELECT 
            AVG(coffee_temp_c) as avg_temp,
            STDDEV(coffee_temp_c) as std_temp,
            AVG(coffee_close) as avg_price,
            STDDEV(coffee_close) as std_price
        FROM features.modeling_dataset 
        WHERE date >= current_date() - 30
    """).collect()[0]
    
    # Compare with historical baseline
    # Alert if drift detected
    pass
```

---

## Model Architecture

### **GitHub Model Migration**

**Current GitHub Model** (from `coffee_forecast_prototype/model.pkl`):
```python
# Load existing model from GitHub
import pickle
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load GitHub model
with open('model.pkl', 'rb') as f:
    github_model = pickle.load(f)

# Load GitHub data
df = pd.read_csv('data.csv', index_col='date', parse_dates=True)

# GitHub model configuration (validated)
github_config = {
    'order': (1, 1, 1),
    'seasonal_order': (0, 0, 0, 0),
    'trend': None,
    'enforce_stationarity': False,
    'enforce_invertibility': False,
    'covariates': ['coffee_temp_c', 'coffee_precip_mm', 'coffee_humidity', 'brl_usd', 'inr_usd'],
    'target': 'coffee_close',
    'performance': {
        'directional_accuracy': 0.5495,
        'rmse': 10.54,
        'p_value': 0.0003,
        'sample_size': 1204
    }
}
```

### **Databricks MLflow Model Registry**

**Model Registration** (migrating from GitHub):
```python
# 01_model_migration.py
import mlflow
import mlflow.statsmodels
import pandas as pd
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX

def migrate_github_model_to_mlflow():
    """Migrate GitHub model to MLflow model registry"""
    
    # Load GitHub model and data
    with open('model.pkl', 'rb') as f:
        github_model = pickle.load(f)
    
    df = pd.read_csv('data.csv', index_col='date', parse_dates=True)
    
    # Start MLflow run
    with mlflow.start_run(run_name="github_model_migration"):
        
        # Log model
        mlflow.statsmodels.log_model(
            github_model,
            "sarimax_coffee_model",
            registered_model_name="coffee_forecast_sarimax"
        )
        
        # Log GitHub performance metrics
        mlflow.log_metric("directional_accuracy", 0.5495)
        mlflow.log_metric("rmse", 10.54)
        mlflow.log_metric("p_value", 0.0003)
        mlflow.log_metric("sample_size", 1204)
        mlflow.log_metric("validation_period_years", 5.0)
        
        # Log GitHub model parameters
        mlflow.log_param("order", "(1,1,1)")
        mlflow.log_param("seasonal_order", "(0,0,0,0)")
        mlflow.log_param("trend", "None")
        mlflow.log_param("enforce_stationarity", False)
        mlflow.log_param("enforce_invertibility", False)
        mlflow.log_param("covariates", "coffee_temp_c,coffee_precip_mm,coffee_humidity,brl_usd,inr_usd")
        
        # Log GitHub source information
        mlflow.log_param("github_repo", "stuhollandUCB/data-sci-210-capstone")
        mlflow.log_param("github_folder", "coffee_forecast_prototype")
        mlflow.log_param("github_model_file", "model.pkl")
        mlflow.log_param("github_data_file", "data.csv")
        
        # Log GitHub artifacts
        mlflow.log_artifact("forecast.csv")
        mlflow.log_artifact("distribution.csv")
        mlflow.log_artifact("README.md")
        
        # Log model summary
        model_summary = str(github_model.summary())
        mlflow.log_text(model_summary, "model_summary.txt")
        
        # Transition to Production
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="coffee_forecast_sarimax",
            version=1,
            stage="Production"
        )
        
        print("GitHub model successfully migrated to MLflow Production stage")
    
    return github_model

# Execute migration
migrated_model = migrate_github_model_to_mlflow()
```

### **Enhanced Forecasting Pipeline (GitHub + Skewed Distributions)**

**Complete Forecasting System** (combining GitHub standards with skewed distributions):
```python
# 05_enhanced_forecasting_pipeline.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import skewnorm, skew, kurtosis
from databricks import sql

class EnhancedCoffeeForecaster:
    """Enhanced forecasting system combining GitHub standards with skewed distributions"""
    
    def __init__(self):
        self.github_config = {
            'order': (1, 1, 1),
            'seasonal_order': (0, 0, 0, 0),
            'trend': None,
            'enforce_stationarity': False,
            'enforce_invertibility': False,
            'covariates': ['coffee_temp_c', 'coffee_precip_mm', 'coffee_humidity', 'brl_usd', 'inr_usd'],
            'target': 'coffee_close',
            'n_paths': 2000
        }
        
        # Load GitHub model as baseline
        self.github_model = self.load_github_model()
    
    def load_github_model(self):
        """Load GitHub model for baseline comparison"""
        try:
            # Load from MLflow (migrated from GitHub)
            model = mlflow.statsmodels.load_model("models:/coffee_forecast_sarimax/Production")
            print("Loaded GitHub model from MLflow")
            return model
        except:
            print("GitHub model not found, will train new model")
            return None
    
    def detect_market_regime(self, df, lookback=30):
        """Detect market regime for skewed distribution selection"""
        recent_returns = df['coffee_close'].pct_change().tail(lookback)
        volatility = recent_returns.std()
        trend = recent_returns.mean()
        historical_vol = df['coffee_close'].pct_change().std()
        
        if volatility > historical_vol * 1.5:
            if trend > 0:
                return 'boom', 0.8  # Strong right skew
            else:
                return 'crisis', -0.5  # Left skew
        elif volatility > historical_vol * 1.2:
            return 'volatile', 0.3  # Slight right skew
        else:
            return 'normal', 0.0  # Symmetric
    
    def estimate_skewness_from_residuals(self, model, df):
        """Estimate skewness from model residuals (GitHub approach)"""
        residuals = model.resid
        return skew(residuals)
    
    def generate_skewed_sample_paths(self, forecast_mean, forecast_se, skewness, regime, n_paths=2000):
        """Generate sample paths with appropriate skewness"""
        
        sample_paths = []
        
        for i in range(n_paths):
            path = []
            for day in range(7):
                if abs(skewness) < 0.1:  # Nearly symmetric
                    # Use normal distribution (GitHub approach)
                    shock = np.random.normal(0, forecast_se[day])
                else:
                    # Use skewed normal distribution
                    shock = skewnorm.rvs(skewness, loc=0, scale=forecast_se[day])
                
                next_price = forecast_mean[day] + shock
                path.append(next_price)
            sample_paths.append(path)
        
        return sample_paths
    
    def calculate_distribution_stats(self, sample_paths):
        """Calculate comprehensive distribution statistics"""
        stats = {}
        
        for day in range(7):
            day_prices = [path[day] for path in sample_paths]
            
            stats[f'day_{day+1}'] = {
                'mean': np.mean(day_prices),
                'std': np.std(day_prices),
                'skewness': skew(day_prices),
                'kurtosis': kurtosis(day_prices),
                'p5': np.percentile(day_prices, 5),
                'p25': np.percentile(day_prices, 25),
                'p50': np.percentile(day_prices, 50),
                'p75': np.percentile(day_prices, 75),
                'p95': np.percentile(day_prices, 95)
            }
        
        return stats
    
    def generate_enhanced_forecast(self, df, forecast_date=None):
        """Generate enhanced forecast with skewed distributions"""
        
        # Use GitHub model or train new one
        if self.github_model is not None:
            model = self.github_model
        else:
            # Train new model using GitHub configuration
            model = SARIMAX(
                endog=df[self.github_config['target']],
                exog=df[self.github_config['covariates']],
                order=self.github_config['order'],
                seasonal_order=self.github_config['seasonal_order'],
                trend=self.github_config['trend'],
                enforce_stationarity=self.github_config['enforce_stationarity'],
                enforce_invertibility=self.github_config['enforce_invertibility']
            ).fit(disp=False)
        
        # Generate point forecast (GitHub approach)
        recent_covs = df[self.github_config['covariates']].iloc[-7:].copy()
        recent_covs = recent_covs.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        forecast_result = model.get_forecast(steps=7, exog=recent_covs)
        forecast_mean = forecast_result.predicted_mean.values
        forecast_se = forecast_result.se_mean.values
        
        # Detect market regime and estimate skewness
        market_regime, regime_skewness = self.detect_market_regime(df)
        residual_skewness = self.estimate_skewness_from_residuals(model, df)
        
        # Use regime-based skewness if residual skewness is small
        final_skewness = residual_skewness if abs(residual_skewness) > 0.1 else regime_skewness
        
        # Generate skewed sample paths
        sample_paths = self.generate_skewed_sample_paths(
            forecast_mean, forecast_se, final_skewness, market_regime, 
            self.github_config['n_paths']
        )
        
        # Calculate distribution statistics
        dist_stats = self.calculate_distribution_stats(sample_paths)
        
        # Create enhanced forecast DataFrame (GitHub compatible + enhancements)
        forecast_dates = pd.date_range(df.index[-1], periods=8, freq='D')[1:]
        
        forecast_df = pd.DataFrame({
            'forecast_date': forecast_dates,
            'day_ahead': range(1, 8),
            'forecast_mean': forecast_mean,
            'forecast_std': forecast_se,
            'lower_95': forecast_mean - 1.96 * forecast_se,
            'upper_95': forecast_mean + 1.96 * forecast_se,
            'model_version': 'sarimax_v1_enhanced',
            'generation_timestamp': pd.Timestamp.now(),
            'data_cutoff_date': forecast_date or df.index[-1],
            'skewness': [dist_stats[f'day_{i+1}']['skewness'] for i in range(7)],
            'kurtosis': [dist_stats[f'day_{i+1}']['kurtosis'] for i in range(7)],
            'market_regime': market_regime,
            'distribution_type': 'skewed_normal' if abs(final_skewness) > 0.1 else 'normal',
            'github_source': 'coffee_forecast_prototype/forecast.py'
        })
        
        # Create enhanced distribution DataFrame
        paths_df = pd.DataFrame(
            sample_paths,
            columns=[f'day_{i+1}' for i in range(7)]
        )
        paths_df['path_id'] = range(1, self.github_config['n_paths'] + 1)
        paths_df['model_version'] = 'sarimax_v1_enhanced'
        paths_df['generation_timestamp'] = pd.Timestamp.now()
        paths_df['data_cutoff_date'] = forecast_date or df.index[-1]
        paths_df['skewness'] = final_skewness
        paths_df['market_regime'] = market_regime
        paths_df['distribution_type'] = 'skewed_normal' if abs(final_skewness) > 0.1 else 'normal'
        paths_df['github_source'] = 'coffee_forecast_prototype/forecast.py'
        
        return forecast_df, paths_df, dist_stats
    
    def save_to_databricks(self, forecast_df, paths_df):
        """Save enhanced forecasts to Databricks Delta tables"""
        
        # Save point forecasts
        spark.createDataFrame(forecast_df).write \
            .format("delta") \
            .mode("append") \
            .saveAsTable("forecasts.coffee_forecasts")
        
        # Save distributions
        spark.createDataFrame(paths_df).write \
            .format("delta") \
            .mode("append") \
            .saveAsTable("forecasts.coffee_distributions")
        
        print(f"Saved {len(forecast_df)} point forecasts and {len(paths_df)} sample paths to Databricks")

# Usage example
def run_enhanced_forecasting():
    """Run enhanced forecasting pipeline"""
    
    # Load data from Databricks
    df = spark.sql("""
        SELECT * FROM features.modeling_dataset 
        WHERE date >= current_date() - 30
        ORDER BY date
    """).toPandas()
    
    # Initialize enhanced forecaster
    forecaster = EnhancedCoffeeForecaster()
    
    # Generate enhanced forecast
    forecast_df, paths_df, dist_stats = forecaster.generate_enhanced_forecast(df)
    
    # Save to Databricks
    forecaster.save_to_databricks(forecast_df, paths_df)
    
    # Print summary
    print(f"Generated forecast with {len(paths_df)} sample paths")
    print(f"Market regime: {forecast_df['market_regime'].iloc[0]}")
    print(f"Distribution type: {forecast_df['distribution_type'].iloc[0]}")
    print(f"Average skewness: {np.mean(forecast_df['skewness']):.3f}")
    
    return forecast_df, paths_df, dist_stats

# Execute enhanced forecasting
forecast_df, paths_df, dist_stats = run_enhanced_forecasting()
```

### Model Serving Architecture

**Batch Inference**:
```python
# 06_batch_inference.py
def generate_forecasts():
    """Generate daily forecasts and distributions"""
    
    # Load latest model
    model = mlflow.statsmodels.load_model("models:/coffee_forecast_sarimax/Production")
    
    # Load latest data
    df = spark.sql("""
        SELECT * FROM features.modeling_dataset 
        WHERE date >= current_date() - 30
        ORDER BY date
    """).toPandas()
    
    # Generate forecast
    target = 'coffee_close'
    covariates = ['coffee_temp_c', 'coffee_precip_mm', 'coffee_humidity', 'brl_usd', 'inr_usd']
    
    recent_covs = df[covariates].iloc[-7:].copy()
    recent_covs = recent_covs.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    forecast_result = model.get_forecast(steps=7, exog=recent_covs)
    forecast_mean = forecast_result.predicted_mean.values
    forecast_se = forecast_result.se_mean.values
    
    # Generate sample paths
    N_PATHS = 2000
    residual_std = np.sqrt(model.params.get('sigma2', forecast_se.mean()**2))
    
    sample_paths = []
    for i in range(N_PATHS):
        path = []
        for day in range(7):
            shock = np.random.normal(0, residual_std)
            next_price = forecast_mean[day] + shock
            path.append(next_price)
        sample_paths.append(path)
    
    # Save to Delta tables
    forecast_df = pd.DataFrame({
        'forecast_date': pd.date_range(df.index[-1], periods=8, freq='D')[1:],
        'day_ahead': range(1, 8),
        'forecast_mean': forecast_mean,
        'forecast_std': forecast_se,
        'lower_95': forecast_mean - 1.96 * forecast_se,
        'upper_95': forecast_mean + 1.96 * forecast_se,
        'model_version': 'sarimax_v1',
        'generation_timestamp': pd.Timestamp.now()
    })
    
    paths_df = pd.DataFrame(
        sample_paths,
        columns=[f'day_{i+1}' for i in range(7)]
    )
    paths_df['path_id'] = range(1, N_PATHS + 1)
    paths_df['model_version'] = 'sarimax_v1'
    paths_df['generation_timestamp'] = pd.Timestamp.now()
    
    # Write to Delta tables
    spark.createDataFrame(forecast_df).write \
        .format("delta") \
        .mode("append") \
        .saveAsTable("forecasts.coffee_forecasts")
    
    spark.createDataFrame(paths_df).write \
        .format("delta") \
        .mode("append") \
        .saveAsTable("forecasts.coffee_distributions")
    
    return forecast_df, paths_df
```

---

## Output Contract for Trading Agent

### **GitHub Output Requirements (Gold Standard)**

**Current GitHub Outputs** (from `coffee_forecast_prototype/`):
```python
# forecast.csv (7-day point predictions)
forecast_df = pd.DataFrame({
    'forecast_date': forecast_dates,
    'day_ahead': range(1, 8),
    'forecast_mean': forecast_mean,
    'forecast_std': forecast_se,
    'lower_95': forecast_mean - 1.96 * forecast_se,
    'upper_95': forecast_mean + 1.96 * forecast_se
})

# distribution.csv (2,000 sample paths)
paths_df = pd.DataFrame(
    sample_paths,
    columns=[f'day_{i+1}' for i in range(7)]
)
paths_df['path_id'] = range(1, N_PATHS + 1)
```

### **Enhanced Databricks Output Schema**

**Point Forecasts Table** (extending GitHub format):
```sql
CREATE TABLE forecasts.coffee_forecasts (
    forecast_date DATE,
    day_ahead INT,
    forecast_mean DOUBLE,
    forecast_std DOUBLE,
    lower_95 DOUBLE,
    upper_95 DOUBLE,
    -- GitHub compatibility
    model_version STRING DEFAULT 'sarimax_v1',
    generation_timestamp TIMESTAMP,
    data_cutoff_date DATE,  -- For backtesting (no data leakage)
    -- Enhanced features
    skewness DOUBLE,         -- Distribution skewness
    kurtosis DOUBLE,         -- Distribution kurtosis
    market_regime STRING,    -- normal/volatile/crisis/boom
    distribution_type STRING DEFAULT 'normal', -- normal/skewed/mixture
    -- GitHub reference
    github_source STRING DEFAULT 'coffee_forecast_prototype/forecast.py'
) USING DELTA
PARTITIONED BY (YEAR(forecast_date), MONTH(forecast_date));
```

**Distribution Table**:
```sql
CREATE TABLE forecasts.coffee_distributions (
    path_id INT,
    day_1 DOUBLE,
    day_2 DOUBLE,
    day_3 DOUBLE,
    day_4 DOUBLE,
    day_5 DOUBLE,
    day_6 DOUBLE,
    day_7 DOUBLE,
    model_version STRING,
    generation_timestamp TIMESTAMP,
    data_cutoff_date DATE  -- For backtesting
) USING DELTA
PARTITIONED BY (YEAR(generation_timestamp), MONTH(generation_timestamp));
```

### Trading Agent Interface

**Required Outputs**:

1. **Point Forecasts** (`forecast.csv` equivalent):
   - 7-day ahead predictions
   - Confidence intervals (95%)
   - Standard errors
   - Model metadata

2. **Distribution** (`distribution.csv` equivalent):
   - 2,000 sample paths
   - 7-day horizon
   - Monte Carlo scenarios
   - Risk profiling data

3. **Metadata**:
   - Model version
   - Generation timestamp
   - Data cutoff date (for backtesting)
   - Performance metrics

### Sample Output Format

**Point Forecasts**:
```csv
forecast_date,day_ahead,forecast_mean,forecast_std,lower_95,upper_95,model_version,generation_timestamp,data_cutoff_date
2025-01-15,1,167.23,2.45,162.43,172.03,sarimax_v1,2025-01-14T16:00:00Z,2025-01-14
2025-01-16,2,167.89,3.12,161.77,174.01,sarimax_v1,2025-01-14T16:00:00Z,2025-01-14
2025-01-17,3,168.45,3.78,161.04,175.86,sarimax_v1,2025-01-14T16:00:00Z,2025-01-14
2025-01-18,4,169.12,4.23,160.83,177.41,sarimax_v1,2025-01-14T16:00:00Z,2025-01-14
2025-01-19,5,168.78,4.67,159.63,177.93,sarimax_v1,2025-01-14T16:00:00Z,2025-01-14
2025-01-20,6,169.45,5.12,159.42,179.48,sarimax_v1,2025-01-14T16:00:00Z,2025-01-14
2025-01-21,7,170.12,5.56,159.22,181.02,sarimax_v1,2025-01-14T16:00:00Z,2025-01-14
```

**Distribution Sample**:
```csv
path_id,day_1,day_2,day_3,day_4,day_5,day_6,day_7,model_version,generation_timestamp,data_cutoff_date
1,167.45,168.12,166.89,169.23,168.45,167.12,168.89,sarimax_v1,2025-01-14T16:00:00Z,2025-01-14
2,166.78,167.23,168.45,167.89,169.12,168.23,167.45,sarimax_v1,2025-01-14T16:00:00Z,2025-01-14
3,168.12,169.45,167.23,170.12,168.89,169.45,170.12,sarimax_v1,2025-01-14T16:00:00Z,2025-01-14
...
2000,167.89,166.45,168.12,167.23,169.45,168.12,167.89,sarimax_v1,2025-01-14T16:00:00Z,2025-01-14
```

---

## Backfill Distribution Strategy

### Data Leakage Prevention

**Critical Requirement**: Backfill distributions must only use models trained on data **prior to** the forecast date to avoid data leakage.

### Backfill Implementation

**SARIMAX Backfill** (No Data Leakage):
```python
# 07_backfill_distributions.py
def generate_backfill_distributions(start_date, end_date):
    """Generate historical distributions for backtesting"""
    
    # Get all dates to backfill
    dates = pd.date_range(start_date, end_date, freq='D')
    
    backfill_results = []
    
    for forecast_date in dates:
        print(f"Backfilling for {forecast_date}")
        
        # Load data up to forecast_date (NO FUTURE DATA)
        df = spark.sql(f"""
            SELECT * FROM features.modeling_dataset 
            WHERE date < '{forecast_date}'
            ORDER BY date
        """).toPandas()
        
        if len(df) < 1500:  # Minimum training data
            continue
            
        # Train model on historical data only
        target = 'coffee_close'
        covariates = ['coffee_temp_c', 'coffee_precip_mm', 'coffee_humidity', 'brl_usd', 'inr_usd']
        
        model = SARIMAX(
            endog=df[target],
            exog=df[covariates],
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            trend=None,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=False)
        
        # Generate forecast for next 7 days
        recent_covs = df[covariates].iloc[-7:].copy()
        recent_covs = recent_covs.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        forecast_result = fitted_model.get_forecast(steps=7, exog=recent_covs)
        forecast_mean = forecast_result.predicted_mean.values
        forecast_se = forecast_result.se_mean.values
        
        # Generate sample paths
        N_PATHS = 2000
        residual_std = np.sqrt(fitted_model.params.get('sigma2', forecast_se.mean()**2))
        
        sample_paths = []
        for i in range(N_PATHS):
            path = []
            for day in range(7):
                shock = np.random.normal(0, residual_std)
                next_price = forecast_mean[day] + shock
                path.append(next_price)
            sample_paths.append(path)
        
        # Create output DataFrames
        forecast_dates = pd.date_range(forecast_date, periods=8, freq='D')[1:]
        
        forecast_df = pd.DataFrame({
            'forecast_date': forecast_dates,
            'day_ahead': range(1, 8),
            'forecast_mean': forecast_mean,
            'forecast_std': forecast_se,
            'lower_95': forecast_mean - 1.96 * forecast_se,
            'upper_95': forecast_mean + 1.96 * forecast_se,
            'model_version': 'sarimax_v1',
            'generation_timestamp': pd.Timestamp.now(),
            'data_cutoff_date': forecast_date
        })
        
        paths_df = pd.DataFrame(
            sample_paths,
            columns=[f'day_{i+1}' for i in range(7)]
        )
        paths_df['path_id'] = range(1, N_PATHS + 1)
        paths_df['model_version'] = 'sarimax_v1'
        paths_df['generation_timestamp'] = pd.Timestamp.now()
        paths_df['data_cutoff_date'] = forecast_date
        
        backfill_results.append((forecast_df, paths_df))
    
    # Write all results to Delta tables
    for forecast_df, paths_df in backfill_results:
        spark.createDataFrame(forecast_df).write \
            .format("delta") \
            .mode("append") \
            .saveAsTable("forecasts.coffee_forecasts_backfill")
        
        spark.createDataFrame(paths_df).write \
            .format("delta") \
            .mode("append") \
            .saveAsTable("forecasts.coffee_distributions_backfill")
    
    return len(backfill_results)
```

### Advanced Model Backfill (With Data Leakage Warning)

**For Neural Networks/Transformers** (May contain data leakage):
```python
def generate_advanced_backfill_distributions(start_date, end_date, model_type="transformer"):
    """Generate backfill for expensive models (may contain data leakage)"""
    
    # Load pre-trained model (trained on full dataset)
    if model_type == "transformer":
        model = load_transformer_model()  # Trained on full data
        data_leakage_warning = True
    elif model_type == "neural_network":
        model = load_neural_network_model()  # Trained on full data  
        data_leakage_warning = True
    else:
        raise ValueError("Only transformer/neural_network supported for advanced backfill")
    
    # Generate backfill (faster but with data leakage)
    for forecast_date in pd.date_range(start_date, end_date, freq='D'):
        # Use pre-trained model (contains future information)
        forecast_df, paths_df = generate_forecast_with_pretrained_model(
            model, forecast_date, data_leakage_warning=True
        )
        
        # Add data leakage warning to metadata
        forecast_df['data_leakage_warning'] = True
        forecast_df['model_type'] = model_type
        paths_df['data_leakage_warning'] = True
        paths_df['model_type'] = model_type
        
        # Write to separate tables with warning
        spark.createDataFrame(forecast_df).write \
            .format("delta") \
            .mode("append") \
            .saveAsTable("forecasts.coffee_forecasts_advanced_backfill")
        
        spark.createDataFrame(paths_df).write \
            .format("delta") \
            .mode("append") \
            .saveAsTable("forecasts.coffee_distributions_advanced_backfill")
```

### Backfill Table Schema

**Clean Backfill** (No Data Leakage):
```sql
CREATE TABLE forecasts.coffee_forecasts_backfill (
    forecast_date DATE,
    day_ahead INT,
    forecast_mean DOUBLE,
    forecast_std DOUBLE,
    lower_95 DOUBLE,
    upper_95 DOUBLE,
    model_version STRING,
    generation_timestamp TIMESTAMP,
    data_cutoff_date DATE,
    data_leakage_warning BOOLEAN DEFAULT FALSE
) USING DELTA
PARTITIONED BY (YEAR(forecast_date), MONTH(forecast_date));
```

**Advanced Backfill** (With Data Leakage Warning):
```sql
CREATE TABLE forecasts.coffee_forecasts_advanced_backfill (
    forecast_date DATE,
    day_ahead INT,
    forecast_mean DOUBLE,
    forecast_std DOUBLE,
    lower_95 DOUBLE,
    upper_95 DOUBLE,
    model_version STRING,
    generation_timestamp TIMESTAMP,
    data_cutoff_date DATE,
    data_leakage_warning BOOLEAN DEFAULT TRUE,
    model_type STRING
) USING DELTA
PARTITIONED BY (YEAR(forecast_date), MONTH(forecast_date));
```

---

## API Design

### REST API Endpoints

**Forecast API**:
```python
# 08_api_endpoints.py
from flask import Flask, jsonify, request
from databricks import sql
import pandas as pd

app = Flask(__name__)

@app.route('/api/v1/forecast', methods=['GET'])
def get_forecast():
    """Get 14-week ahead forecast for given date"""
    
    # Parse parameters
    forecast_date = request.args.get('date', pd.Timestamp.now().strftime('%Y-%m-%d'))
    horizon_weeks = int(request.args.get('weeks', 14))
    model_version = request.args.get('model_version', 'latest')
    
    # Query forecast data
    query = f"""
        SELECT 
            forecast_date,
            day_ahead,
            forecast_mean,
            forecast_std,
            lower_95,
            upper_95,
            model_version,
            generation_timestamp
        FROM forecasts.coffee_forecasts
        WHERE forecast_date >= '{forecast_date}'
        AND forecast_date <= date_add('{forecast_date}', {horizon_weeks * 7})
        AND model_version = '{model_version}'
        ORDER BY forecast_date
    """
    
    # Execute query
    with sql.connect(server_hostname=os.getenv('DATABRICKS_HOST'),
                    http_path=os.getenv('DATABRICKS_HTTP_PATH'),
                    access_token=os.getenv('DATABRICKS_TOKEN')) as connection:
        
        with connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
    
    # Format response
    forecast_data = []
    for row in results:
        forecast_data.append({
            'forecast_date': row[0].strftime('%Y-%m-%d'),
            'day_ahead': row[1],
            'forecast_mean': float(row[2]),
            'forecast_std': float(row[3]),
            'lower_95': float(row[4]),
            'upper_95': float(row[5]),
            'model_version': row[6],
            'generation_timestamp': row[7].isoformat()
        })
    
    return jsonify({
        'status': 'success',
        'forecast_date': forecast_date,
        'horizon_weeks': horizon_weeks,
        'model_version': model_version,
        'data': forecast_data,
        'count': len(forecast_data)
    })

@app.route('/api/v1/distribution', methods=['GET'])
def get_distribution():
    """Get distribution sample paths for given date"""
    
    # Parse parameters
    forecast_date = request.args.get('date', pd.Timestamp.now().strftime('%Y-%m-%d'))
    model_version = request.args.get('model_version', 'latest')
    max_paths = int(request.args.get('max_paths', 2000))
    
    # Query distribution data
    query = f"""
        SELECT 
            path_id,
            day_1, day_2, day_3, day_4, day_5, day_6, day_7,
            model_version,
            generation_timestamp
        FROM forecasts.coffee_distributions
        WHERE generation_timestamp = (
            SELECT MAX(generation_timestamp) 
            FROM forecasts.coffee_distributions 
            WHERE date(generation_timestamp) = '{forecast_date}'
        )
        AND model_version = '{model_version}'
        LIMIT {max_paths}
    """
    
    # Execute query
    with sql.connect(server_hostname=os.getenv('DATABRICKS_HOST'),
                    http_path=os.getenv('DATABRICKS_HTTP_PATH'),
                    access_token=os.getenv('DATABRICKS_TOKEN')) as connection:
        
        with connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
    
    # Format response
    distribution_data = []
    for row in results:
        distribution_data.append({
            'path_id': row[0],
            'day_1': float(row[1]),
            'day_2': float(row[2]),
            'day_3': float(row[3]),
            'day_4': float(row[4]),
            'day_5': float(row[5]),
            'day_6': float(row[6]),
            'day_7': float(row[7]),
            'model_version': row[8],
            'generation_timestamp': row[9].isoformat()
        })
    
    return jsonify({
        'status': 'success',
        'forecast_date': forecast_date,
        'model_version': model_version,
        'sample_paths': distribution_data,
        'count': len(distribution_data)
    })

@app.route('/api/v1/backtest', methods=['GET'])
def get_backtest_data():
    """Get backtest data for given date range"""
    
    # Parse parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    model_version = request.args.get('model_version', 'latest')
    include_data_leakage = request.args.get('include_data_leakage', 'false').lower() == 'true'
    
    if not start_date or not end_date:
        return jsonify({'error': 'start_date and end_date required'}), 400
    
    # Choose table based on data leakage preference
    table_name = "forecasts.coffee_forecasts_advanced_backfill" if include_data_leakage else "forecasts.coffee_forecasts_backfill"
    
    # Query backtest data
    query = f"""
        SELECT 
            forecast_date,
            day_ahead,
            forecast_mean,
            forecast_std,
            lower_95,
            upper_95,
            model_version,
            generation_timestamp,
            data_cutoff_date,
            data_leakage_warning
        FROM {table_name}
        WHERE forecast_date >= '{start_date}'
        AND forecast_date <= '{end_date}'
        AND model_version = '{model_version}'
        ORDER BY forecast_date, day_ahead
    """
    
    # Execute query and return results
    # (Implementation similar to above)
    
    return jsonify({
        'status': 'success',
        'start_date': start_date,
        'end_date': end_date,
        'model_version': model_version,
        'include_data_leakage': include_data_leakage,
        'data': backtest_data,
        'count': len(backtest_data)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### API Response Examples

**Forecast Response**:
```json
{
  "status": "success",
  "forecast_date": "2025-01-14",
  "horizon_weeks": 14,
  "model_version": "sarimax_v1",
  "data": [
    {
      "forecast_date": "2025-01-15",
      "day_ahead": 1,
      "forecast_mean": 167.23,
      "forecast_std": 2.45,
      "lower_95": 162.43,
      "upper_95": 172.03,
      "model_version": "sarimax_v1",
      "generation_timestamp": "2025-01-14T16:00:00Z"
    },
    {
      "forecast_date": "2025-01-16", 
      "day_ahead": 2,
      "forecast_mean": 167.89,
      "forecast_std": 3.12,
      "lower_95": 161.77,
      "upper_95": 174.01,
      "model_version": "sarimax_v1",
      "generation_timestamp": "2025-01-14T16:00:00Z"
    }
  ],
  "count": 98
}
```

**Distribution Response**:
```json
{
  "status": "success",
  "forecast_date": "2025-01-14",
  "model_version": "sarimax_v1",
  "sample_paths": [
    {
      "path_id": 1,
      "day_1": 167.45,
      "day_2": 168.12,
      "day_3": 166.89,
      "day_4": 169.23,
      "day_5": 168.45,
      "day_6": 167.12,
      "day_7": 168.89,
      "model_version": "sarimax_v1",
      "generation_timestamp": "2025-01-14T16:00:00Z"
    },
    {
      "path_id": 2,
      "day_1": 166.78,
      "day_2": 167.23,
      "day_3": 168.45,
      "day_4": 167.89,
      "day_5": 169.12,
      "day_6": 168.23,
      "day_7": 167.45,
      "model_version": "sarimax_v1",
      "generation_timestamp": "2025-01-14T16:00:00Z"
    }
  ],
  "count": 2000
}
```

---

## Implementation Details

### Databricks Workflow Configuration

**Daily Forecast Workflow**:
```json
{
  "name": "coffee_forecast_daily",
  "schedule": {
    "quartz_cron_expression": "0 0 16 * * ?",
    "timezone_id": "UTC"
  },
  "tasks": [
    {
      "task_key": "data_pipeline",
      "notebook_task": {
        "notebook_path": "/pipelines/01_data_pipeline",
        "base_parameters": {
          "target_date": "{{job.triggered_time}}"
        }
      },
      "libraries": [
        {
          "pypi": {
            "package": "pandas>=2.0.0"
          }
        },
        {
          "pypi": {
            "package": "statsmodels>=0.14.0"
          }
        }
      ]
    },
    {
      "task_key": "model_training",
      "depends_on": [
        {
          "task_key": "data_pipeline"
        }
      ],
      "notebook_task": {
        "notebook_path": "/models/05_model_training"
      }
    },
    {
      "task_key": "forecast_generation",
      "depends_on": [
        {
          "task_key": "model_training"
        }
      ],
      "notebook_task": {
        "notebook_path": "/forecasting/06_batch_inference"
      }
    },
    {
      "task_key": "api_update",
      "depends_on": [
        {
          "task_key": "forecast_generation"
        }
      ],
      "notebook_task": {
        "notebook_path": "/api/08_api_endpoints"
      }
    }
  ],
  "max_concurrent_runs": 1,
  "timeout_seconds": 3600
}
```

**Backfill Workflow**:
```json
{
  "name": "coffee_forecast_backfill",
  "tasks": [
    {
      "task_key": "backfill_distributions",
      "notebook_task": {
        "notebook_path": "/backfill/07_backfill_distributions",
        "base_parameters": {
          "start_date": "2020-01-01",
          "end_date": "2024-12-31"
        }
      },
      "libraries": [
        {
          "pypi": {
            "package": "pandas>=2.0.0"
          }
        },
        {
          "pypi": {
            "package": "statsmodels>=0.14.0"
          }
        }
      ]
    }
  ],
  "max_concurrent_runs": 1,
  "timeout_seconds": 7200
}
```

### Environment Configuration

**Cluster Configuration**:
```json
{
  "cluster_name": "coffee-forecast-cluster",
  "spark_version": "13.3.x-scala2.12",
  "node_type_id": "i3.xlarge",
  "driver_node_type_id": "i3.xlarge",
  "num_workers": 2,
  "autoscale": {
    "min_workers": 1,
    "max_workers": 4
  },
  "spark_conf": {
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer"
  },
  "custom_tags": {
    "project": "coffee-forecasting",
    "environment": "production"
  }
}
```

**Library Installation**:
```python
# Install required packages
%pip install statsmodels>=0.14.0
%pip install scipy>=1.10.0
%pip install numpy>=1.24.0
%pip install pandas>=2.0.0
%pip install scikit-learn>=1.3.0
%pip install yfinance>=0.2.0
%pip install fredapi>=0.5.0
%pip install requests>=2.31.0
%pip install matplotlib>=3.7.0
%pip install seaborn>=0.12.0
%pip install mlflow>=2.5.0
%pip install delta-spark>=2.4.0
```

### Basic Configuration

**Environment Setup**:
```python
# Basic environment variables
import os
DATABRICKS_HOST = os.getenv('DATABRICKS_HOST')
DATABRICKS_HTTP_PATH = os.getenv('DATABRICKS_HTTP_PATH')
DATABRICKS_TOKEN = os.getenv('DATABRICKS_TOKEN')
```

**Note**: Security, authentication, and access control configurations will be addressed in future phases as the system scales.

---

## Testing & Validation

### Unit Tests

**Model Testing**:
```python
# tests/test_sarimax_model.py
import unittest
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

class TestSARIMAXModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        n_obs = 1000
        
        # Generate synthetic data
        self.test_data = pd.DataFrame({
            'coffee_close': np.cumsum(np.random.normal(0, 1, n_obs)) + 100,
            'coffee_temp_c': np.random.normal(20, 5, n_obs),
            'coffee_precip_mm': np.random.exponential(2, n_obs),
            'coffee_humidity': np.random.uniform(30, 90, n_obs),
            'brl_usd': np.random.normal(5, 0.5, n_obs),
            'inr_usd': np.random.normal(75, 5, n_obs)
        })
        
        self.target = 'coffee_close'
        self.covariates = ['coffee_temp_c', 'coffee_precip_mm', 'coffee_humidity', 'brl_usd', 'inr_usd']
    
    def test_model_training(self):
        """Test model training"""
        model = SARIMAX(
            endog=self.test_data[self.target],
            exog=self.test_data[self.covariates],
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            trend=None,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=False)
        
        # Check model converged
        self.assertTrue(fitted_model.mle_retvals['converged'])
        
        # Check parameters are reasonable
        self.assertGreater(fitted_model.params['ar.L1'], -1)
        self.assertLess(fitted_model.params['ar.L1'], 1)
        self.assertGreater(fitted_model.params['ma.L1'], -1)
        self.assertLess(fitted_model.params['ma.L1'], 1)
    
    def test_forecast_generation(self):
        """Test forecast generation"""
        # Train model
        model = SARIMAX(
            endog=self.test_data[self.target],
            exog=self.test_data[self.covariates],
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            trend=None,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=False)
        
        # Generate forecast
        recent_covs = self.test_data[self.covariates].iloc[-7:].copy()
        forecast_result = fitted_model.get_forecast(steps=7, exog=recent_covs)
        
        # Check forecast structure
        self.assertEqual(len(forecast_result.predicted_mean), 7)
        self.assertEqual(len(forecast_result.se_mean), 7)
        
        # Check forecast values are reasonable
        for i in range(7):
            self.assertGreater(forecast_result.predicted_mean[i], 0)
            self.assertGreater(forecast_result.se_mean[i], 0)
    
    def test_sample_path_generation(self):
        """Test sample path generation"""
        # Train model
        model = SARIMAX(
            endog=self.test_data[self.target],
            exog=self.test_data[self.covariates],
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            trend=None,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=False)
        
        # Generate sample paths
        N_PATHS = 100
        recent_covs = self.test_data[self.covariates].iloc[-7:].copy()
        forecast_result = fitted_model.get_forecast(steps=7, exog=recent_covs)
        
        forecast_mean = forecast_result.predicted_mean.values
        forecast_se = forecast_result.se_mean.values
        residual_std = np.sqrt(fitted_model.params.get('sigma2', forecast_se.mean()**2))
        
        sample_paths = []
        for i in range(N_PATHS):
            path = []
            for day in range(7):
                shock = np.random.normal(0, residual_std)
                next_price = forecast_mean[day] + shock
                path.append(next_price)
            sample_paths.append(path)
        
        # Check sample paths structure
        self.assertEqual(len(sample_paths), N_PATHS)
        self.assertEqual(len(sample_paths[0]), 7)
        
        # Check sample paths are reasonable
        for path in sample_paths:
            for price in path:
                self.assertGreater(price, 0)

if __name__ == '__main__':
    unittest.main()
```

**API Testing**:
```python
# tests/test_api.py
import unittest
import requests
import json

class TestForecastAPI(unittest.TestCase):
    
    def setUp(self):
        """Set up test client"""
        self.base_url = "http://localhost:5000/api/v1"
        self.headers = {'Content-Type': 'application/json'}
    
    def test_forecast_endpoint(self):
        """Test forecast endpoint"""
        response = requests.get(f"{self.base_url}/forecast")
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('data', data)
        self.assertGreater(len(data['data']), 0)
        
        # Check forecast structure
        forecast = data['data'][0]
        required_fields = ['forecast_date', 'day_ahead', 'forecast_mean', 'forecast_std', 'lower_95', 'upper_95']
        for field in required_fields:
            self.assertIn(field, forecast)
    
    def test_distribution_endpoint(self):
        """Test distribution endpoint"""
        response = requests.get(f"{self.base_url}/distribution")
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('sample_paths', data)
        self.assertGreater(len(data['sample_paths']), 0)
        
        # Check distribution structure
        path = data['sample_paths'][0]
        required_fields = ['path_id', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7']
        for field in required_fields:
            self.assertIn(field, path)
    
    def test_backtest_endpoint(self):
        """Test backtest endpoint"""
        response = requests.get(f"{self.base_url}/backtest?start_date=2024-01-01&end_date=2024-01-07")
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('data', data)
        self.assertIn('include_data_leakage', data)

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

**End-to-End Testing**:
```python
# tests/test_integration.py
import unittest
import pandas as pd
import numpy as np
from databricks import sql
import os

class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.connection_params = {
            'server_hostname': os.getenv('DATABRICKS_HOST'),
            'http_path': os.getenv('DATABRICKS_HTTP_PATH'),
            'access_token': os.getenv('DATABRICKS_TOKEN')
        }
    
    def test_data_pipeline_integration(self):
        """Test complete data pipeline"""
        with sql.connect(**self.connection_params) as connection:
            with connection.cursor() as cursor:
                # Check raw data exists
                cursor.execute("SELECT COUNT(*) FROM raw.coffee_futures")
                raw_count = cursor.fetchone()[0]
                self.assertGreater(raw_count, 0)
                
                # Check features exist
                cursor.execute("SELECT COUNT(*) FROM features.modeling_dataset")
                features_count = cursor.fetchone()[0]
                self.assertGreater(features_count, 0)
                
                # Check data quality
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_rows,
                        COUNT(CASE WHEN coffee_close IS NULL THEN 1 END) as null_targets,
                        COUNT(CASE WHEN coffee_temp_c IS NULL THEN 1 END) as null_temp
                    FROM features.modeling_dataset
                """)
                quality = cursor.fetchone()
                self.assertEqual(quality[1], 0)  # No null targets
                self.assertLess(quality[2] / quality[0], 0.05)  # <5% null temp
    
    def test_forecast_generation_integration(self):
        """Test forecast generation pipeline"""
        with sql.connect(**self.connection_params) as connection:
            with connection.cursor() as cursor:
                # Check forecasts exist
                cursor.execute("SELECT COUNT(*) FROM forecasts.coffee_forecasts")
                forecast_count = cursor.fetchone()[0]
                self.assertGreater(forecast_count, 0)
                
                # Check distributions exist
                cursor.execute("SELECT COUNT(*) FROM forecasts.coffee_distributions")
                distribution_count = cursor.fetchone()[0]
                self.assertGreater(distribution_count, 0)
                
                # Check forecast quality
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_forecasts,
                        COUNT(CASE WHEN forecast_mean IS NULL THEN 1 END) as null_forecasts,
                        COUNT(CASE WHEN forecast_std <= 0 THEN 1 END) as invalid_std
                    FROM forecasts.coffee_forecasts
                """)
                quality = cursor.fetchone()
                self.assertEqual(quality[1], 0)  # No null forecasts
                self.assertEqual(quality[2], 0)  # No invalid std
    
    def test_backfill_integration(self):
        """Test backfill pipeline"""
        with sql.connect(**self.connection_params) as connection:
            with connection.cursor() as cursor:
                # Check backfill data exists
                cursor.execute("SELECT COUNT(*) FROM forecasts.coffee_forecasts_backfill")
                backfill_count = cursor.fetchone()[0]
                self.assertGreater(backfill_count, 0)
                
                # Check data leakage warnings
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_backfill,
                        COUNT(CASE WHEN data_leakage_warning = TRUE THEN 1 END) as with_leakage
                    FROM forecasts.coffee_forecasts_backfill
                """)
                leakage = cursor.fetchone()
                self.assertEqual(leakage[1], 0)  # No data leakage in clean backfill

if __name__ == '__main__':
    unittest.main()
```

### Performance Testing

**Load Testing**:
```python
# tests/test_performance.py
import unittest
import time
import concurrent.futures
import requests

class TestPerformance(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.base_url = "http://localhost:5000/api/v1"
        self.num_requests = 100
    
    def test_api_response_time(self):
        """Test API response time"""
        start_time = time.time()
        
        response = requests.get(f"{self.base_url}/forecast")
        
        end_time = time.time()
        response_time = end_time - start_time
        
        self.assertEqual(response.status_code, 200)
        self.assertLess(response_time, 5.0)  # <5 seconds
    
    def test_concurrent_requests(self):
        """Test concurrent request handling"""
        def make_request():
            return requests.get(f"{self.base_url}/forecast")
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(self.num_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check all requests succeeded
        for result in results:
            self.assertEqual(result.status_code, 200)
        
        # Check performance
        avg_time = total_time / self.num_requests
        self.assertLess(avg_time, 2.0)  # <2 seconds average
    
    def test_large_distribution_request(self):
        """Test large distribution request"""
        start_time = time.time()
        
        response = requests.get(f"{self.base_url}/distribution?max_paths=2000")
        
        end_time = time.time()
        response_time = end_time - start_time
        
        self.assertEqual(response.status_code, 200)
        self.assertLess(response_time, 10.0)  # <10 seconds for 2000 paths
        
        data = response.json()
        self.assertEqual(len(data['sample_paths']), 2000)

if __name__ == '__main__':
    unittest.main()
```

---

## Deployment Strategy

### Immediate Implementation Plan (Phase 1.5)

**Week 1: Basic Setup**
- Set up Databricks workspace
- Create basic Delta Lake tables
- Configure MLflow model registry
- Load GitHub model and new dataset

**Week 2: E2E Pipeline**
- Migrate SARIMAX model to Databricks
- Implement complete E2E pipeline using new dataset
- Generate 7-day forecasts with 2,000 sample paths
- Output distribution files for risk agent

**Week 3: Integration & Testing**
- Test forecast accuracy (maintain 54.95% directional accuracy)
- Validate distribution file outputs
- Simple file-based interface for risk agent
- Basic validation of E2E pipeline

**Future Phases**: Hierarchical forecasting, skewed distributions, APIs, monitoring, and scaling will be addressed in subsequent phases.

### Phase 1.5 Checklist

**Basic Infrastructure**:
- [ ] Databricks workspace configured
- [ ] Basic Delta Lake tables created
- [ ] MLflow model registry configured
- [ ] GitHub model loaded

**E2E Pipeline**:
- [ ] SARIMAX model migrated to Databricks
- [ ] New unified dataset integrated
- [ ] Complete E2E pipeline working (data → forecast → output)
- [ ] 7-day forecasting pipeline working
- [ ] 2,000 sample paths generated
- [ ] Distribution files output correctly

**Risk Agent Integration**:
- [ ] Distribution files accessible to risk agent
- [ ] File format matches risk agent requirements
- [ ] Forecast accuracy maintained (54.95% directional accuracy)
- [ ] Basic validation of E2E pipeline working

**Future Considerations**: Hierarchical forecasting, skewed distributions, APIs, monitoring, security, and scaling features will be implemented in subsequent phases.

### Rollback Strategy

**Model Rollback**:
```python
# Rollback to previous model version
def rollback_model():
    client = mlflow.tracking.MlflowClient()
    
    # Get current production model
    current_version = client.get_latest_versions("coffee_forecast_sarimax", stages=["Production"])[0]
    
    # Get previous version
    all_versions = client.get_latest_versions("coffee_forecast_sarimax", stages=["None"])
    previous_version = all_versions[1] if len(all_versions) > 1 else None
    
    if previous_version:
        # Transition previous version to Production
        client.transition_model_version_stage(
            name="coffee_forecast_sarimax",
            version=previous_version.version,
            stage="Production"
        )
        
        # Transition current version to Archived
        client.transition_model_version_stage(
            name="coffee_forecast_sarimax",
            version=current_version.version,
            stage="Archived"
        )
        
        print(f"Rolled back from version {current_version.version} to {previous_version.version}")
    else:
        print("No previous version available for rollback")
```

**Data Rollback**:
```sql
-- Rollback to previous data version
RESTORE TABLE features.modeling_dataset TO VERSION AS OF 10;

-- Rollback forecasts
RESTORE TABLE forecasts.coffee_forecasts TO VERSION AS OF 5;
```

---

## Monitoring & Maintenance

### Performance Monitoring

**Model Performance Metrics**:
```python
# monitoring/model_performance.py
def monitor_model_performance():
    """Monitor model performance in production"""
    
    # Get recent forecasts
    recent_forecasts = spark.sql("""
        SELECT 
            forecast_date,
            forecast_mean,
            model_version,
            generation_timestamp
        FROM forecasts.coffee_forecasts
        WHERE forecast_date >= current_date() - 30
        ORDER BY forecast_date
    """).toPandas()
    
    # Get actual prices
    actual_prices = spark.sql("""
        SELECT 
            date,
            coffee_close
        FROM features.modeling_dataset
        WHERE date >= current_date() - 30
        ORDER BY date
    """).toPandas()
    
    # Calculate performance metrics
    merged = recent_forecasts.merge(actual_prices, left_on='forecast_date', right_on='date')
    
    if len(merged) > 0:
        # Calculate directional accuracy
        actual_directions = np.sign(merged['coffee_close'].diff())
        pred_directions = np.sign(merged['forecast_mean'].diff())
        directional_accuracy = np.mean(actual_directions == pred_directions) * 100
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((merged['coffee_close'] - merged['forecast_mean'])**2))
        
        # Check if performance degraded
        if directional_accuracy < 50:  # Below random
            send_alert(f"Model performance degraded: {directional_accuracy:.2f}% directional accuracy")
        
        if rmse > 15:  # High error
            send_alert(f"Model RMSE too high: {rmse:.2f}")
        
        # Log metrics
        log_metric("directional_accuracy", directional_accuracy)
        log_metric("rmse", rmse)
        log_metric("sample_size", len(merged))
    
    return directional_accuracy, rmse
```

**Data Quality Monitoring**:
```python
# monitoring/data_quality.py
def monitor_data_quality():
    """Monitor data quality metrics"""
    
    # Check for missing data
    missing_data = spark.sql("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(CASE WHEN coffee_close IS NULL THEN 1 END) as null_targets,
            COUNT(CASE WHEN coffee_temp_c IS NULL THEN 1 END) as null_temp,
            COUNT(CASE WHEN coffee_precip_mm IS NULL THEN 1 END) as null_precip
        FROM features.modeling_dataset
        WHERE date >= current_date() - 7
    """).collect()[0]
    
    # Check for data drift
    current_stats = spark.sql("""
        SELECT 
            AVG(coffee_temp_c) as avg_temp,
            STDDEV(coffee_temp_c) as std_temp,
            AVG(coffee_close) as avg_price,
            STDDEV(coffee_close) as std_price
        FROM features.modeling_dataset
        WHERE date >= current_date() - 30
    """).collect()[0]
    
    historical_stats = spark.sql("""
        SELECT 
            AVG(coffee_temp_c) as avg_temp,
            STDDEV(coffee_temp_c) as std_temp,
            AVG(coffee_close) as avg_price,
            STDDEV(coffee_close) as std_price
        FROM features.modeling_dataset
        WHERE date >= current_date() - 365 AND date < current_date() - 30
    """).collect()[0]
    
    # Check for anomalies
    if missing_data.null_targets > 0:
        send_alert(f"Missing target data: {missing_data.null_targets} rows")
    
    if missing_data.null_temp > missing_data.total_rows * 0.1:
        send_alert(f"High missing temperature data: {missing_data.null_temp} rows")
    
    # Check for drift
    temp_drift = abs(current_stats.avg_temp - historical_stats.avg_temp) / historical_stats.std_temp
    price_drift = abs(current_stats.avg_price - historical_stats.avg_price) / historical_stats.std_price
    
    if temp_drift > 2:  # 2 standard deviations
        send_alert(f"Temperature drift detected: {temp_drift:.2f} std devs")
    
    if price_drift > 2:
        send_alert(f"Price drift detected: {price_drift:.2f} std devs")
    
    return {
        'missing_targets': missing_data.null_targets,
        'missing_temp': missing_data.null_temp,
        'temp_drift': temp_drift,
        'price_drift': price_drift
    }
```

### Alerting System

**Alert Configuration**:
```python
# monitoring/alerts.py
def send_alert(message, severity="WARNING"):
    """Send alert to monitoring system"""
    
    alert_data = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "severity": severity,
        "message": message,
        "service": "coffee-forecasting",
        "environment": "production"
    }
    
    # Send to monitoring system (e.g., PagerDuty, Slack, email)
    if severity == "CRITICAL":
        send_pagerduty_alert(alert_data)
        send_slack_alert(alert_data)
        send_email_alert(alert_data)
    elif severity == "WARNING":
        send_slack_alert(alert_data)
        send_email_alert(alert_data)
    else:
        send_slack_alert(alert_data)
    
    # Log to monitoring system
    log_alert(alert_data)

def check_system_health():
    """Check overall system health"""
    
    issues = []
    
    # Check data pipeline
    try:
        latest_data = spark.sql("""
            SELECT MAX(date) as latest_date
            FROM features.modeling_dataset
        """).collect()[0]
        
        if latest_data.latest_date < pd.Timestamp.now() - pd.Timedelta(days=2):
            issues.append("Data pipeline delayed: Latest data is 2+ days old")
    except Exception as e:
        issues.append(f"Data pipeline error: {str(e)}")
    
    # Check model performance
    try:
        directional_accuracy, rmse = monitor_model_performance()
        
        if directional_accuracy < 50:
            issues.append(f"Model performance degraded: {directional_accuracy:.2f}% accuracy")
        
        if rmse > 15:
            issues.append(f"Model RMSE too high: {rmse:.2f}")
    except Exception as e:
        issues.append(f"Model monitoring error: {str(e)}")
    
    # Check API health
    try:
        response = requests.get("http://localhost:5000/api/v1/forecast", timeout=10)
        if response.status_code != 200:
            issues.append(f"API health check failed: {response.status_code}")
    except Exception as e:
        issues.append(f"API health check error: {str(e)}")
    
    # Send alerts for issues
    if issues:
        for issue in issues:
            send_alert(issue, severity="WARNING")
    
    return len(issues) == 0
```

### Basic Maintenance (Phase 1)

**Daily Tasks**:
- Check forecast generation
- Validate distribution file outputs
- Monitor basic model performance

**Weekly Tasks**:
- Review forecast accuracy
- Check data pipeline status
- Validate trading agent file consumption

**Future Considerations**: Advanced monitoring, automated retraining, security audits, and comprehensive performance analysis will be implemented in subsequent phases.

---

## Summary & Next Steps

### What We've Accomplished

Our coffee futures forecasting system represents a **production-ready, statistically validated** solution that:

✅ **Achieves 54.95% directional accuracy** (statistically significant, p=0.0003)  
✅ **Validated across 1,204 predictions** over 5+ years of out-of-sample data  
✅ **Code audited** with 10/10 checks passed  
✅ **Tested 11 different models** comprehensively  
✅ **Ready for Databricks migration** with detailed implementation guide  

### Key Technical Achievements

1. **Model Performance**: SARIMAX(1,1,1) with 5 dynamic covariates outperforms all tested alternatives
2. **Validation Rigor**: 172 walk-forward folds across all market conditions
3. **Data Quality**: Comprehensive data pipeline with quality monitoring
4. **Output Contract**: Standardized forecast and distribution outputs for trading
5. **Backfill Strategy**: Data leakage prevention for accurate backtesting
6. **API Design**: RESTful endpoints for 14-week ahead forecasts

### Migration Benefits

**Scalability**: Databricks provides auto-scaling compute for backfill operations  
**Reliability**: Delta Lake ensures ACID transactions and data versioning  
**Governance**: Unity Catalog provides data lineage and access control  
**MLOps**: MLflow enables model versioning and deployment  
**Monitoring**: Built-in monitoring and alerting capabilities  

### Immediate Next Steps

1. **Set up Databricks workspace** with Unity Catalog and Delta Lake
2. **Implement data pipeline** using Delta Live Tables
3. **Deploy SARIMAX model** to MLflow model registry
4. **Develop API endpoints** for trading agent integration
5. **Generate backfill distributions** for historical validation
6. **Set up monitoring** and alerting systems

### Expected Outcomes

**For Trading Team**:
- Daily 7-day ahead forecasts with 54.95% directional accuracy
- 2,000 sample paths for risk assessment
- 14-week ahead API for strategic planning
- Historical backfill for strategy validation

**For Data Science Team**:
- Scalable model training and deployment
- Automated data pipeline with quality monitoring
- Model versioning and performance tracking
- Comprehensive testing and validation framework

**For Business**:
- Production-ready forecasting system
- Reduced manual effort through automation
- Improved risk management through distributions
- Scalable foundation for additional commodities

---

## Advanced Considerations

### Ensemble Model Implementation

**Multi-Model Ensemble**:
```python
# 09_ensemble_model.py
def create_ensemble_forecast():
    """Create ensemble forecast from multiple models"""
    
    # Load models
    sarimax_model = mlflow.statsmodels.load_model("models:/coffee_forecast_sarimax/Production")
    xgboost_model = mlflow.xgboost.load_model("models:/coffee_forecast_xgboost/Production")
    
    # Generate individual forecasts
    sarimax_forecast = generate_sarimax_forecast(sarimax_model)
    xgboost_forecast = generate_xgboost_forecast(xgboost_model)
    
    # Ensemble weights (based on historical performance)
    weights = {
        'sarimax': 0.60,  # 54.95% directional accuracy
        'xgboost': 0.40   # ~53% directional accuracy
    }
    
    # Weighted ensemble
    ensemble_forecast = (
        weights['sarimax'] * sarimax_forecast +
        weights['xgboost'] * xgboost_forecast
    )
    
    # Generate ensemble distributions
    ensemble_paths = generate_ensemble_sample_paths(
        sarimax_forecast, xgboost_forecast, weights
    )
    
    return ensemble_forecast, ensemble_paths

def generate_ensemble_sample_paths(sarimax_paths, xgboost_paths, weights):
    """Generate ensemble sample paths"""
    
    N_PATHS = 2000
    ensemble_paths = []
    
    for i in range(N_PATHS):
        # Sample from each model's distribution
        sarimax_path = sarimax_paths[i]
        xgboost_path = xgboost_paths[i]
        
        # Weighted combination
        ensemble_path = (
            weights['sarimax'] * sarimax_path +
            weights['xgboost'] * xgboost_path
        )
        
        ensemble_paths.append(ensemble_path)
    
    return ensemble_paths
```

### Model Retraining Strategies

**Adaptive Retraining**:
```python
# 10_adaptive_retraining.py
def adaptive_model_retraining():
    """Retrain model based on performance degradation"""
    
    # Check recent performance
    recent_accuracy = get_recent_directional_accuracy(days=30)
    baseline_accuracy = 0.5495  # Our validated performance
    
    # Retrain if performance degraded significantly
    if recent_accuracy < baseline_accuracy - 0.02:  # 2% degradation
        print(f"Performance degraded: {recent_accuracy:.3f} vs {baseline_accuracy:.3f}")
        
        # Retrain model
        new_model = retrain_sarimax_model()
        
        # Validate new model
        new_accuracy = validate_model_performance(new_model)
        
        if new_accuracy > recent_accuracy:
            # Deploy new model
            deploy_model_to_production(new_model)
            print(f"New model deployed with accuracy: {new_accuracy:.3f}")
        else:
            print("New model not better, keeping current model")
    
    return recent_accuracy

def retrain_sarimax_model():
    """Retrain SARIMAX model with latest data"""
    
    # Load latest data
    df = spark.sql("""
        SELECT * FROM features.modeling_dataset 
        WHERE date >= '2015-01-01'
        ORDER BY date
    """).toPandas()
    
    # Prepare features
    target = 'coffee_close'
    covariates = ['coffee_temp_c', 'coffee_precip_mm', 'coffee_humidity', 'brl_usd', 'inr_usd']
    
    # Train model
    model = SARIMAX(
        endog=df[target],
        exog=df[covariates],
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        trend=None,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    fitted_model = model.fit(disp=False)
    
    return fitted_model
```

### A/B Testing Framework

**Model A/B Testing**:
```python
# 11_ab_testing.py
def setup_ab_test(model_a, model_b, traffic_split=0.5):
    """Set up A/B test between two models"""
    
    # Create A/B test configuration
    ab_config = {
        'model_a': {
            'model': model_a,
            'traffic_percentage': traffic_split,
            'name': 'sarimax_v1'
        },
        'model_b': {
            'model': model_b,
            'traffic_percentage': 1 - traffic_split,
            'name': 'sarimax_v2'
        },
        'start_date': pd.Timestamp.now(),
        'duration_days': 30
    }
    
    # Store configuration
    spark.createDataFrame([ab_config]).write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable("experiments.ab_test_config")
    
    return ab_config

def generate_ab_test_forecast(forecast_date, user_id=None):
    """Generate forecast using A/B test logic"""
    
    # Get A/B test configuration
    ab_config = spark.sql("""
        SELECT * FROM experiments.ab_test_config
        WHERE start_date <= current_date()
        AND start_date + INTERVAL duration_days DAYS >= current_date()
    """).collect()
    
    if not ab_config:
        # No active A/B test, use production model
        return generate_production_forecast()
    
    config = ab_config[0]
    
    # Determine which model to use based on user_id hash
    if user_id:
        hash_value = hash(user_id) % 100
        use_model_b = hash_value < (config['model_b']['traffic_percentage'] * 100)
    else:
        use_model_b = np.random.random() < config['model_b']['traffic_percentage']
    
    # Generate forecast with selected model
    if use_model_b:
        model = config['model_b']['model']
        model_name = config['model_b']['name']
    else:
        model = config['model_a']['model']
        model_name = config['model_a']['name']
    
    forecast = generate_forecast_with_model(model, forecast_date)
    
    # Log A/B test result
    log_ab_test_result(forecast_date, model_name, user_id)
    
    return forecast, model_name

def analyze_ab_test_results():
    """Analyze A/B test results"""
    
    # Get A/B test results
    results = spark.sql("""
        SELECT 
            model_name,
            COUNT(*) as forecast_count,
            AVG(forecast_accuracy) as avg_accuracy,
            STDDEV(forecast_accuracy) as std_accuracy
        FROM experiments.ab_test_results
        WHERE test_date >= current_date() - 30
        GROUP BY model_name
    """).toPandas()
    
    # Statistical significance test
    model_a_results = results[results['model_name'] == 'sarimax_v1']['forecast_accuracy']
    model_b_results = results[results['model_name'] == 'sarimax_v2']['forecast_accuracy']
    
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(model_a_results, model_b_results)
    
    # Determine winner
    if p_value < 0.05:  # Statistically significant
        if model_a_results.mean() > model_b_results.mean():
            winner = 'sarimax_v1'
        else:
            winner = 'sarimax_v2'
    else:
        winner = 'inconclusive'
    
    return {
        'results': results,
        't_statistic': t_stat,
        'p_value': p_value,
        'winner': winner
    }
```

---

## Cost Optimization

### Compute Optimization

**Cluster Configuration for Different Workloads**:
```json
{
  "data_pipeline_cluster": {
    "node_type_id": "i3.large",
    "num_workers": 2,
    "autoscale": {
      "min_workers": 1,
      "max_workers": 4
    },
    "spark_conf": {
      "spark.sql.adaptive.enabled": "true",
      "spark.sql.adaptive.coalescePartitions.enabled": "true"
    }
  },
  "model_training_cluster": {
    "node_type_id": "i3.xlarge",
    "num_workers": 1,
    "autoscale": {
      "min_workers": 0,
      "max_workers": 2
    },
    "spark_conf": {
      "spark.sql.adaptive.enabled": "true",
      "spark.sql.adaptive.skewJoin.enabled": "true"
    }
  },
  "backfill_cluster": {
    "node_type_id": "i3.2xlarge",
    "num_workers": 4,
    "autoscale": {
      "min_workers": 0,
      "max_workers": 8
    },
    "spark_conf": {
      "spark.sql.adaptive.enabled": "true",
      "spark.sql.adaptive.localShuffleReader.enabled": "true"
    }
  }
}
```

**Cost Optimization Strategies**:
```python
# 12_cost_optimization.py
def optimize_cluster_costs():
    """Optimize cluster costs based on workload patterns"""
    
    # Analyze workload patterns
    workload_analysis = spark.sql("""
        SELECT 
            HOUR(start_time) as hour,
            AVG(duration_seconds) as avg_duration,
            COUNT(*) as job_count
        FROM system.compute.cluster_usage
        WHERE start_time >= current_date() - 30
        GROUP BY HOUR(start_time)
        ORDER BY hour
    """).toPandas()
    
    # Identify peak and off-peak hours
    peak_hours = workload_analysis[workload_analysis['job_count'] > workload_analysis['job_count'].quantile(0.8)]['hour'].tolist()
    off_peak_hours = workload_analysis[workload_analysis['job_count'] < workload_analysis['job_count'].quantile(0.2)]['hour'].tolist()
    
    # Optimize cluster configuration
    optimization_config = {
        'peak_hours': {
            'min_workers': 2,
            'max_workers': 8,
            'node_type': 'i3.xlarge'
        },
        'off_peak_hours': {
            'min_workers': 0,
            'max_workers': 2,
            'node_type': 'i3.large'
        }
    }
    
    return optimization_config

def implement_cost_controls():
    """Implement cost controls and monitoring"""
    
    # Set up cost alerts
    cost_alerts = {
        'daily_budget': 100,  # $100/day
        'monthly_budget': 2000,  # $2000/month
        'alert_thresholds': {
            'daily': 0.8,  # Alert at 80% of daily budget
            'monthly': 0.9  # Alert at 90% of monthly budget
        }
    }
    
    # Monitor costs
    daily_cost = get_daily_cost()
    monthly_cost = get_monthly_cost()
    
    if daily_cost > cost_alerts['daily_budget'] * cost_alerts['alert_thresholds']['daily']:
        send_cost_alert(f"Daily cost approaching limit: ${daily_cost:.2f}")
    
    if monthly_cost > cost_alerts['monthly_budget'] * cost_alerts['alert_thresholds']['monthly']:
        send_cost_alert(f"Monthly cost approaching limit: ${monthly_cost:.2f}")
    
    return {
        'daily_cost': daily_cost,
        'monthly_cost': monthly_cost,
        'budget_utilization': {
            'daily': daily_cost / cost_alerts['daily_budget'],
            'monthly': monthly_cost / cost_alerts['monthly_budget']
        }
    }
```

### Storage Optimization

**Delta Lake Optimization**:
```python
# 13_storage_optimization.py
def optimize_delta_tables():
    """Optimize Delta Lake tables for cost and performance"""
    
    # Optimize forecast tables
    spark.sql("OPTIMIZE forecasts.coffee_forecasts ZORDER BY (forecast_date, day_ahead)")
    spark.sql("OPTIMIZE forecasts.coffee_distributions ZORDER BY (generation_timestamp, path_id)")
    
    # Vacuum old data
    spark.sql("VACUUM forecasts.coffee_forecasts RETAIN 90 DAYS")
    spark.sql("VACUUM forecasts.coffee_distributions RETAIN 90 DAYS")
    
    # Compress old partitions
    spark.sql("""
        ALTER TABLE forecasts.coffee_forecasts 
        SET TBLPROPERTIES (
            'delta.autoOptimize.optimizeWrite' = 'true',
            'delta.autoOptimize.autoCompact' = 'true'
        )
    """)
    
    return "Delta tables optimized"

def implement_data_lifecycle():
    """Implement data lifecycle management"""
    
    # Archive old data
    archive_date = pd.Timestamp.now() - pd.Timedelta(days=365)
    
    # Move old forecasts to archive
    spark.sql(f"""
        INSERT INTO forecasts.coffee_forecasts_archive
        SELECT * FROM forecasts.coffee_forecasts
        WHERE forecast_date < '{archive_date}'
    """)
    
    # Delete old data from main table
    spark.sql(f"""
        DELETE FROM forecasts.coffee_forecasts
        WHERE forecast_date < '{archive_date}'
    """)
    
    return f"Archived data older than {archive_date}"
```

---

## Trading Agent Integration

### Real-Time Integration

**WebSocket API for Real-Time Updates**:
```python
# 14_realtime_integration.py
import asyncio
import websockets
import json
from datetime import datetime

class RealtimeForecastServer:
    def __init__(self):
        self.clients = set()
        self.latest_forecast = None
    
    async def register_client(self, websocket, path):
        """Register new client"""
        self.clients.add(websocket)
        print(f"Client connected: {websocket.remote_address}")
        
        # Send latest forecast immediately
        if self.latest_forecast:
            await websocket.send(json.dumps(self.latest_forecast))
    
    async def unregister_client(self, websocket):
        """Unregister client"""
        self.clients.remove(websocket)
        print(f"Client disconnected: {websocket.remote_address}")
    
    async def broadcast_forecast(self, forecast_data):
        """Broadcast forecast to all connected clients"""
        self.latest_forecast = forecast_data
        
        if self.clients:
            message = json.dumps(forecast_data)
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
    
    async def handle_client_message(self, websocket, path):
        """Handle client messages"""
        await self.register_client(websocket, path)
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'subscribe':
                    # Send latest forecast
                    if self.latest_forecast:
                        await websocket.send(json.dumps(self.latest_forecast))
                
                elif data['type'] == 'request_forecast':
                    # Generate new forecast
                    forecast = await generate_realtime_forecast()
                    await websocket.send(json.dumps(forecast))
        
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)

async def generate_realtime_forecast():
    """Generate real-time forecast"""
    
    # Load latest model
    model = mlflow.statsmodels.load_model("models:/coffee_forecast_sarimax/Production")
    
    # Load latest data
    df = spark.sql("""
        SELECT * FROM features.modeling_dataset 
        WHERE date >= current_date() - 30
        ORDER BY date
    """).toPandas()
    
    # Generate forecast
    target = 'coffee_close'
    covariates = ['coffee_temp_c', 'coffee_precip_mm', 'coffee_humidity', 'brl_usd', 'inr_usd']
    
    recent_covs = df[covariates].iloc[-7:].copy()
    recent_covs = recent_covs.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    forecast_result = model.get_forecast(steps=7, exog=recent_covs)
    forecast_mean = forecast_result.predicted_mean.values
    forecast_se = forecast_result.se_mean.values
    
    # Generate sample paths
    N_PATHS = 1000  # Reduced for real-time
    residual_std = np.sqrt(model.params.get('sigma2', forecast_se.mean()**2))
    
    sample_paths = []
    for i in range(N_PATHS):
        path = []
        for day in range(7):
            shock = np.random.normal(0, residual_std)
            next_price = forecast_mean[day] + shock
            path.append(next_price)
        sample_paths.append(path)
    
    return {
        'timestamp': datetime.now().isoformat(),
        'forecast_mean': forecast_mean.tolist(),
        'forecast_std': forecast_se.tolist(),
        'sample_paths': sample_paths,
        'model_version': 'sarimax_v1'
    }

# Start WebSocket server
async def main():
    server = RealtimeForecastServer()
    
    # Start WebSocket server
    start_server = websockets.serve(
        server.handle_client_message,
        "localhost",
        8765
    )
    
    # Start forecast generation task
    async def forecast_generator():
        while True:
            forecast = await generate_realtime_forecast()
            await server.broadcast_forecast(forecast)
            await asyncio.sleep(300)  # Update every 5 minutes
    
    await asyncio.gather(start_server, forecast_generator())

if __name__ == "__main__":
    asyncio.run(main())
```

### Error Handling and Fallback

**Robust Error Handling**:
```python
# 15_error_handling.py
class ForecastError(Exception):
    """Custom exception for forecast errors"""
    pass

class ForecastService:
    def __init__(self):
        self.primary_model = "sarimax_v1"
        self.fallback_model = "naive_baseline"
        self.circuit_breaker = CircuitBreaker()
    
    def get_forecast(self, forecast_date, retry_count=3):
        """Get forecast with error handling and fallback"""
        
        for attempt in range(retry_count):
            try:
                # Check circuit breaker
                if self.circuit_breaker.is_open():
                    return self.get_fallback_forecast(forecast_date)
                
                # Try primary model
                forecast = self.generate_primary_forecast(forecast_date)
                
                # Validate forecast
                self.validate_forecast(forecast)
                
                # Reset circuit breaker on success
                self.circuit_breaker.record_success()
                
                return forecast
            
            except ForecastError as e:
                print(f"Forecast error (attempt {attempt + 1}): {e}")
                
                if attempt == retry_count - 1:
                    # Final attempt failed, use fallback
                    return self.get_fallback_forecast(forecast_date)
                
                # Wait before retry
                time.sleep(2 ** attempt)  # Exponential backoff
            
            except Exception as e:
                print(f"Unexpected error (attempt {attempt + 1}): {e}")
                self.circuit_breaker.record_failure()
                
                if attempt == retry_count - 1:
                    return self.get_fallback_forecast(forecast_date)
                
                time.sleep(2 ** attempt)
    
    def get_fallback_forecast(self, forecast_date):
        """Get fallback forecast using naive model"""
        
        try:
            # Load latest price
            latest_price = spark.sql("""
                SELECT coffee_close 
                FROM features.modeling_dataset 
                ORDER BY date DESC 
                LIMIT 1
            """).collect()[0]['coffee_close']
            
            # Generate naive forecast (no change)
            forecast = {
                'forecast_date': forecast_date,
                'forecast_mean': [latest_price] * 7,
                'forecast_std': [2.0] * 7,  # Conservative estimate
                'model_version': 'naive_fallback',
                'fallback_used': True,
                'error_message': 'Primary model failed, using naive fallback'
            }
            
            return forecast
        
        except Exception as e:
            # Ultimate fallback
            return {
                'forecast_date': forecast_date,
                'forecast_mean': [150.0] * 7,  # Default price
                'forecast_std': [5.0] * 7,
                'model_version': 'emergency_fallback',
                'fallback_used': True,
                'error_message': f'All models failed: {str(e)}'
            }
    
    def validate_forecast(self, forecast):
        """Validate forecast quality"""
        
        # Check for reasonable price range
        for price in forecast['forecast_mean']:
            if price < 50 or price > 500:
                raise ForecastError(f"Price out of range: {price}")
        
        # Check for reasonable standard deviation
        for std in forecast['forecast_std']:
            if std <= 0 or std > 20:
                raise ForecastError(f"Invalid standard deviation: {std}")
        
        # Check for missing values
        if any(pd.isna(forecast['forecast_mean'])):
            raise ForecastError("Missing forecast values")
        
        return True

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def is_open(self):
        """Check if circuit breaker is open"""
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
                return False
            return True
        return False
    
    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
```

### Trading Agent API Contract

**Complete API Specification**:
```yaml
# api_specification.yaml
openapi: 3.0.0
info:
  title: Coffee Futures Forecast API
  version: 1.0.0
  description: API for coffee futures forecasting and distribution generation

paths:
  /api/v1/forecast:
    get:
      summary: Get coffee futures forecast
      parameters:
        - name: date
          in: query
          description: Forecast date (YYYY-MM-DD)
          schema:
            type: string
            format: date
        - name: weeks
          in: query
          description: Forecast horizon in weeks
          schema:
            type: integer
            default: 14
            minimum: 1
            maximum: 52
        - name: model_version
          in: query
          description: Model version to use
          schema:
            type: string
            default: latest
      responses:
        '200':
          description: Successful forecast
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: success
                  forecast_date:
                    type: string
                    format: date
                  horizon_weeks:
                    type: integer
                  model_version:
                    type: string
                  data:
                    type: array
                    items:
                      type: object
                      properties:
                        forecast_date:
                          type: string
                          format: date
                        day_ahead:
                          type: integer
                        forecast_mean:
                          type: number
                        forecast_std:
                          type: number
                        lower_95:
                          type: number
                        upper_95:
                          type: number
                        model_version:
                          type: string
                        generation_timestamp:
                          type: string
                          format: date-time
                  count:
                    type: integer
        '400':
          description: Bad request
        '500':
          description: Internal server error

  /api/v1/distribution:
    get:
      summary: Get forecast distribution
      parameters:
        - name: date
          in: query
          description: Forecast date (YYYY-MM-DD)
          schema:
            type: string
            format: date
        - name: model_version
          in: query
          description: Model version to use
          schema:
            type: string
            default: latest
        - name: max_paths
          in: query
          description: Maximum number of sample paths
          schema:
            type: integer
            default: 2000
            maximum: 5000
      responses:
        '200':
          description: Successful distribution
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: success
                  forecast_date:
                    type: string
                    format: date
                  model_version:
                    type: string
                  sample_paths:
                    type: array
                    items:
                      type: object
                      properties:
                        path_id:
                          type: integer
                        day_1:
                          type: number
                        day_2:
                          type: number
                        day_3:
                          type: number
                        day_4:
                          type: number
                        day_5:
                          type: number
                        day_6:
                          type: number
                        day_7:
                          type: number
                        model_version:
                          type: string
                        generation_timestamp:
                          type: string
                          format: date-time
                  count:
                    type: integer

  /api/v1/backtest:
    get:
      summary: Get backtest data
      parameters:
        - name: start_date
          in: query
          required: true
          description: Start date for backtest (YYYY-MM-DD)
          schema:
            type: string
            format: date
        - name: end_date
          in: query
          required: true
          description: End date for backtest (YYYY-MM-DD)
          schema:
            type: string
            format: date
        - name: model_version
          in: query
          description: Model version to use
          schema:
            type: string
            default: latest
        - name: include_data_leakage
          in: query
          description: Include models with data leakage
          schema:
            type: boolean
            default: false
      responses:
        '200':
          description: Successful backtest data
        '400':
          description: Bad request

  /api/v1/health:
    get:
      summary: Health check
      responses:
        '200':
          description: Service is healthy
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: healthy
                  timestamp:
                    type: string
                    format: date-time
                  version:
                    type: string
                  uptime:
                    type: number
```

---

## Long-term Strategic Vision

### **5-Year Vision: Global Commodity Intelligence Platform**

Our long-term vision extends beyond coffee futures to create a comprehensive **Global Commodity Intelligence Platform** that:

#### **Multi-Commodity Forecasting**
- **Coffee, Sugar, Cocoa, Cotton, Wheat** - all major agricultural commodities
- **Cross-commodity correlations** and substitution effects
- **Supply chain disruption** modeling and early warning systems
- **Climate impact assessment** on global food security

#### **Hierarchical Global Coverage**
- **Regional level**: 200+ coffee regions worldwide
- **Country level**: 50+ producing countries
- **Global level**: Aggregate supply and demand forecasting
- **Real-time updates**: Weather, production, and market data

#### **Advanced AI Integration**
- **Foundation models** for natural language processing of news/sentiment
- **Computer vision** for satellite imagery analysis
- **Reinforcement learning** for dynamic strategy optimization
- **Federated learning** across multiple trading firms

#### **Trading Ecosystem Integration**
- **Risk management** systems for portfolio optimization
- **Regulatory compliance** for multiple jurisdictions
- **ESG scoring** for sustainable investing
- **Carbon footprint** tracking and optimization

### **Technology Evolution**

#### **Current (V1)**: Statistical Models
- SARIMAX with weather covariates
- 54.95% directional accuracy
- 2,000 sample paths for risk assessment

#### **Phase 2**: Enterprise Scaling
- Databricks infrastructure
- 14-week ahead forecasting
- Real-time API integration

#### **Phase 3**: Advanced Forecasting
- Hierarchical models
- Skewed distributions
- Multi-commodity support

#### **Phase 4**: AI Integration
- Foundation model integration
- Computer vision for satellite data
- Reinforcement learning for strategy optimization

#### **Phase 5**: Global Platform
- Multi-commodity intelligence
- Cross-asset correlation modeling
- Real-time global monitoring

### **Market Impact Potential**

#### **For Coffee Producers**
- **Risk management**: Hedge against price volatility
- **Production planning**: Optimize planting and harvesting
- **Market timing**: Best times to sell inventory
- **Climate adaptation**: Prepare for weather changes

#### **For Trading Firms**
- **Alpha generation**: Consistent directional accuracy
- **Risk reduction**: Better understanding of price distributions
- **Portfolio optimization**: Cross-commodity correlations
- **Regulatory compliance**: Transparent and auditable models

#### **For Global Markets**
- **Price discovery**: More accurate commodity pricing
- **Supply chain resilience**: Early warning systems
- **Food security**: Better understanding of global food supply
- **Climate adaptation**: Data-driven agricultural planning

### **Success Metrics**

#### **Technical Metrics**
- **Directional accuracy**: Maintain >55% on 7-day forecasts
- **API uptime**: 99.9% availability
- **Latency**: <100ms for real-time forecasts
- **Scalability**: Support 1000+ concurrent users

#### **Business Metrics**
- **User adoption**: 100+ active trading firms
- **Revenue impact**: $10M+ in trading alpha generated
- **Market share**: 25% of commodity forecasting market
- **Geographic reach**: 50+ countries using the platform

#### **Impact Metrics**
- **Risk reduction**: 30% reduction in portfolio volatility
- **Efficiency gains**: 50% faster decision making
- **Cost savings**: $100M+ in avoided losses
- **Sustainability**: 20% improvement in supply chain efficiency

---

## Conclusion

This strategic roadmap provides a comprehensive vision for our coffee futures forecasting system, from the current V1 implementation to a global commodity intelligence platform. Our validated 54.95% directional accuracy provides a strong foundation for scaling to enterprise infrastructure and expanding to multi-commodity forecasting.

### **Immediate Next Steps (Phase 1.5)**

1. **Set up Databricks workspace** with basic Delta Lake tables
2. **Migrate GitHub SARIMAX model** to MLflow model registry
3. **Integrate new unified dataset** from `DATASET_FOR_FORECAST_V2.md`
4. **Implement complete E2E pipeline** that outputs distribution files
5. **Validate risk agent integration** with file-based interface

### **Future Steps (Phases 2-4)**

1. **Implement hierarchical forecasting** using the research agent requirements
2. **Deploy skewed distribution modeling** for realistic risk assessment
3. **Build 14-week ahead API** for strategic planning
4. **Add enterprise scaling features** (monitoring, cost optimization, security)

### **Key Success Factors**

1. **Maintain statistical rigor** while scaling to enterprise infrastructure
2. **Implement proper data leakage prevention** in all backtesting
3. **Generate realistic skewed distributions** reflecting market dynamics
4. **Build hierarchical forecasting** for multi-level insights
5. **Provide robust APIs** for seamless trading agent integration

The combination of our validated statistical approach with enterprise infrastructure, advanced forecasting capabilities, and long-term strategic vision will create a world-class commodity forecasting system that transforms how agricultural markets operate globally.

---

**This comprehensive strategic roadmap provides everything needed to successfully deploy our validated coffee forecasting system to Databricks, maintain the 54.95% directional accuracy while gaining enterprise-scale capabilities, and build toward a global commodity intelligence platform that transforms agricultural markets worldwide.**

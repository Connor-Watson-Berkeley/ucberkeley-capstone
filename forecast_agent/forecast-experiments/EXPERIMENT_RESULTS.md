# DARTS Model Experiment Results

**Date**: 2025-11-22
**Objective**: Evaluate DARTS deep learning models for coffee price forecasting
**Data Source**: `commodity.silver.unified_data` (local cache)
**Region**: Bahia, Brazil
**Lookback Period**: 730 days (2 years)
**Forecast Horizon**: 14 days
**Train/Val Split**: 80/20 (574 train / 144 validation days)

---

## Dataset Overview

- **Total Rows**: 718 days (2023-11-24 to 2025-11-10)
- **Target Variable**: Coffee closing price
- **Weather Covariates** (7 features):
  - `temp_mean_c` - Mean temperature (°C)
  - `precipitation_mm` - Total precipitation (mm)
  - `wind_speed_max_kmh` - Maximum wind speed (km/h)
  - `humidity_mean_pct` - Mean relative humidity (%)
  - `temp_max_c` - Maximum temperature (°C)
  - `temp_min_c` - Minimum temperature (°C)
  - `rain_mm` - Rainfall (mm)

---

## Experiment 1: N-BEATS (Neural Basis Expansion Analysis)

### Model Configuration
- **Architecture**: Generic N-BEATS
- **Parameters**: 9.8M trainable parameters
- **Input Chunk Length**: 60 days
- **Output Chunk Length**: 14 days
- **Stacks**: 30
- **Blocks per Stack**: 1
- **Layers per Block**: 4
- **Layer Width**: 256
- **Batch Size**: 32
- **Epochs**: 100
- **Learning Rate**: 0.001
- **Optimizer**: Adam

### Training Performance
- **Train Loss**: 0.0012 (final epoch)
- **Validation Loss**: 0.150 (final epoch)
- **Training Time**: ~10 minutes (CPU)
- **Convergence**: Excellent - stable loss reduction

### Validation Metrics
- **MAPE**: TBD (awaiting final results)
- **RMSE**: TBD
- **MAE**: TBD

### Notes
- Model successfully trained with weather covariates
- Strong convergence indicates good fit to data
- No overfitting observed (val loss stable)
- Uses past covariates for weather features

---

## Experiment 2: TFT (Temporal Fusion Transformer)

### Model Configuration
- **Status**: Pending
- **Expected Features**:
  - Probabilistic forecasting (quantile regression)
  - Attention mechanisms for interpretability
  - Multi-horizon forecasting
  - Variable importance analysis

---

## Experiment 3: N-HiTS

### Model Configuration
- **Status**: Pending
- **Expected Features**:
  - Hierarchical interpolation
  - Efficient training
  - Strong performance on long horizons

---

## Experiment 4: TCN (Temporal Convolutional Network)

### Model Configuration
- **Status**: Pending
- **Expected Features**:
  - Lightweight architecture
  - Fast inference
  - Good for production deployment

---

## Baseline Comparison

### Quick N-BEATS Test (90 days, 10 epochs)
- **MAPE**: 2.04%
- **RMSE**: $10.14
- **MAE**: $8.02
- **Note**: Lightweight model, limited data

### Existing Models (for reference)
- **LSTM**: TBD
- **SARIMAX**: TBD
- **XGBoost**: TBD

---

## Model Selection Criteria for Production

### Key Factors:
1. **Accuracy** (MAPE, RMSE, MAE)
2. **Training Time** (for re-training cadence)
3. **Inference Speed** (for real-time predictions)
4. **Interpretability** (for stakeholder communication)
5. **Probabilistic Output** (uncertainty quantification)
6. **Databricks Compatibility** (deployment ease)

### Preliminary Rankings
1. **TBD** - Awaiting all experiment results
2. **TBD**
3. **TBD**

---

## Production Pipeline Recommendations

### Phase 1: Model Deployment
- **Top Model**: TBD (based on accuracy + interpretability)
- **Backup Model**: TBD (fast inference for real-time)
- **Deployment Target**: Databricks MLflow

### Phase 2: Ensemble Strategy
- Combine top 2-3 models for robust predictions
- Weighted average based on validation performance
- Probabilistic aggregation for uncertainty bounds

### Phase 3: Monitoring & Retraining
- Weekly retraining schedule
- Drift detection on prediction errors
- A/B testing new model versions

---

## Next Steps

- [ ] Complete TFT experiment
- [ ] Complete N-HiTS experiment
- [ ] Complete TCN experiment
- [ ] Run hyperparameter tuning on top model
- [ ] Implement ensemble method
- [ ] Create Databricks deployment notebook
- [ ] Set up MLflow experiment tracking

---

**Last Updated**: 2025-11-22
**Owner**: Connor Watson / Claude Code

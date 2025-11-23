# DARTS Model Selection for Databricks Production Pipeline

**Date**: 2025-11-22
**Status**: Experiments In Progress
**Objective**: Select top-performing deep learning models for production deployment

---

## Executive Summary

This document tracks our evaluation of state-of-the-art deep learning models from the DARTS library for coffee price forecasting. The goal is to select the best models for deployment to our Databricks production pipeline.

### Key Findings (Preliminary)

- ✅ **Quick N-BEATS Test**: MAPE 2.04% (lightweight baseline)
- 🏃 **Full Experiments**: Running (N-BEATS, TFT, N-HiTS)
- 📊 **Expected Completion**: ~35-40 minutes
- 🎯 **Target**: Select top 2-3 models for production

---

## Experiment Configuration

### Data Setup
- **Source**: `commodity.silver.unified_data` (Bahia, Brazil region)
- **Lookback**: 730 days (2 years: 2023-11-24 to 2025-11-10)
- **Total Rows**: 718 days
- **Train/Val Split**: 574 / 144 days (80/20)
- **Forecast Horizon**: 14 days
- **Target Variable**: Coffee closing price

### Weather Covariates (7 features)
1. Mean temperature (°C)
2. Total precipitation (mm)
3. Maximum wind speed (km/h)
4. Mean relative humidity (%)
5. Maximum temperature (°C)
6. Minimum temperature (°C)
7. Rainfall (mm)

---

## Models Under Evaluation

### 1. N-BEATS (Neural Basis Expansion Analysis)

**Architecture**:
- 9.8M parameters
- 30 stacks, 1 block/stack, 4 layers/block
- Layer width: 256
- Input/Output chunks: 60/14 days

**Training**:
- 100 epochs
- Batch size: 32
- Learning rate: 0.001
- Status: 🏃 Running (Epoch 1/100)

**Expected Strengths**:
- State-of-the-art performance
- Pure deep learning (no statistical assumptions)
- Generic architecture (learns patterns from data)

**Results**: *Pending*

---

### 2. TFT (Temporal Fusion Transformer)

**Architecture**:
- Transformer-based with attention mechanisms
- Hidden size: 64
- 2 LSTM layers
- 4 attention heads
- Input/Output chunks: 60/14 days

**Training**:
- 50 epochs (fewer due to complexity)
- Batch size: 32
- Learning rate: 0.001
- Quantile regression: 10th, 50th, 90th percentiles
- Status: ⏳ Queued

**Expected Strengths**:
- Probabilistic forecasts (uncertainty quantification)
- Attention mechanisms (interpretability)
- Variable importance analysis
- Multi-horizon predictions

**Results**: *Pending*

---

### 3. N-HiTS (Neural Hierarchical Interpolation)

**Architecture**:
- Hierarchical interpolation structure
- 3 stacks, 1 block/stack, 2 layers/block
- Layer width: 512
- Input/Output chunks: 60/14 days

**Training**:
- 100 epochs
- Batch size: 32
- Learning rate: 0.001
- Status: ⏳ Queued

**Expected Strengths**:
- Efficient training
- Strong long-horizon performance
- Lightweight deployment
- Fast inference

**Results**: *Pending*

---

## Selection Criteria

### Primary Metrics (Ranked by Importance)
1. **MAPE** (Mean Absolute Percentage Error) - Primary accuracy metric
2. **RMSE** (Root Mean Squared Error) - Penalizes large errors
3. **MAE** (Mean Absolute Error) - Absolute accuracy

### Secondary Considerations
4. **Training Time** - Re-training cadence feasibility
5. **Inference Speed** - Real-time prediction requirements
6. **Interpretability** - Stakeholder communication
7. **Probabilistic Output** - Risk management capability
8. **Databricks Compatibility** - Deployment ease

---

## Production Pipeline Recommendations

### Phase 1: Single Model Deployment (Week 1)

**Approach**: Deploy best-performing model

**Selection Process**:
1. Rank models by validation MAPE
2. Verify RMSE/MAE consistency
3. Test inference speed
4. Document model interpretability

**Expected Winner**: TBD (likely N-BEATS or TFT)

**Deployment Steps**:
1. Save model to MLflow Model Registry
2. Create Databricks notebook for inference
3. Set up scheduled retraining (weekly)
4. Implement monitoring dashboard

---

### Phase 2: Ensemble Strategy (Week 2-3)

**Approach**: Combine top 2-3 models for robust predictions

**Ensemble Methods**:
1. **Simple Average**: Equal weight to all models
2. **Weighted Average**: Weight by inverse validation MAPE
3. **Stacking**: Meta-model learns optimal combination

**Probabilistic Ensemble** (if TFT performs well):
- Use TFT for uncertainty bounds
- Use N-BEATS/N-HiTS for point forecasts
- Combine for robust probabilistic output

---

### Phase 3: A/B Testing & Optimization (Week 4+)

**Monitoring Metrics**:
- Daily MAPE tracking
- Prediction drift detection
- Model degradation alerts

**Optimization**:
- Hyperparameter tuning (Optuna/Hyperopt)
- Regional model specialization
- Feature engineering experiments

---

## Databricks Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│ 1. Data Preparation (Spark SQL)                        │
│    └─ Query commodity.silver.unified_data              │
│    └─ Filter region, date range, features              │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Model Training (MLflow)                             │
│    └─ Load DARTS model                                 │
│    └─ Train on historical data                         │
│    └─ Log metrics, parameters, artifacts               │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Model Registry (MLflow)                             │
│    └─ Register trained model                           │
│    └─ Tag with metadata (date, metrics, version)       │
│    └─ Promote to Production stage                      │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Inference (Scheduled Job)                           │
│    └─ Load production model from Registry              │
│    └─ Generate 14-day forecasts                        │
│    └─ Write to commodity.forecast.distributions        │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Monitoring (Dashboard)                              │
│    └─ Track prediction accuracy                        │
│    └─ Alert on drift/degradation                       │
│    └─ Trigger retraining if needed                     │
└─────────────────────────────────────────────────────────┘
```

---

## Next Steps

### Immediate (This Week)
- [ ] Wait for experiment completion (~35-40 min)
- [ ] Document all model results
- [ ] Select top model based on MAPE
- [ ] Create Databricks deployment notebook

### Short-Term (Next 2 Weeks)
- [ ] Deploy top model to Databricks
- [ ] Set up MLflow experiment tracking
- [ ] Implement weekly retraining schedule
- [ ] Create monitoring dashboard

### Medium-Term (Next Month)
- [ ] Implement ensemble strategy
- [ ] Run hyperparameter optimization
- [ ] Test multi-region models
- [ ] A/B test new model versions

---

## Risk Assessment

### Technical Risks
1. **Model Overfitting**: Mitigated by validation split, regularization
2. **Covariate Availability**: Future weather requires forecasts
3. **Concept Drift**: Coffee prices influenced by macro events
4. **Training Time**: Long epochs may limit retraining frequency

### Mitigation Strategies
- Regular validation monitoring
- Weather forecast integration for future covariates
- Ensemble methods for robustness
- Incremental training / transfer learning

---

## Cost Analysis

### Local Experimentation
- **Cost**: $0 (using local data cache)
- **Benefit**: Rapid iteration, no Databricks serverless charges

### Production Deployment (Estimated)
- **Training**: ~$0.50/week (compute for model training)
- **Inference**: ~$0.10/day (daily forecast generation)
- **Storage**: ~$0.01/month (model artifacts in MLflow)
- **Total**: ~$5-10/month

**ROI**: Trading strategy informed by accurate forecasts >> infrastructure costs

---

## References

- DARTS Documentation: https://unit8co.github.io/darts/
- N-BEATS Paper: https://arxiv.org/abs/1905.10437
- TFT Paper: https://arxiv.org/abs/1912.09363
- N-HiTS Paper: https://arxiv.org/abs/2201.12886

---

**Document Owner**: Connor Watson / Claude Code
**Last Updated**: 2025-11-22
**Status**: Experiments In Progress
**Next Update**: Upon experiment completion

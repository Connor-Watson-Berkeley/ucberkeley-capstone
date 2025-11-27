# Model Recommendations for Databricks

## Quick Start: Recommended Models

**Start with these 2 models (no dependencies):**
```python
models = ["naive", "random_walk"]
```

**Why:**
- ✅ No package dependencies (just pandas/numpy - already in Databricks)
- ✅ Fast to train (< 1 second each)
- ✅ Gets distributions table populated quickly
- ✅ Easy to debug

---

## Model Tiers (Add Gradually)

### Tier 1: No Dependencies ✅ (Start Here)

```python
models = ["naive", "random_walk"]
```

**Requirements:** None (pandas/numpy included in Databricks)

**Training Time:** < 1 second each

**Good For:**
- Getting the system working
- Populating distributions table fast
- Baseline comparison

---

### Tier 2: Statistical Models (Needs statsmodels)

```python
models = ["naive", "random_walk", "arima_111", "sarimax_auto_weather"]
```

**Requirements:**
```python
%pip install statsmodels pmdarima
```

**Training Time:** 10-30 seconds per model

**Models:**
- `arima_111` - Classic ARIMA (1,1,1)
- `sarimax_auto_weather` - Auto-ARIMA with weather features

**Good For:**
- Better accuracy than naive/random_walk
- Statistical baselines
- Works without GPU

---

### Tier 3: ML Models (Needs xgboost)

```python
models = ["naive", "random_walk", "arima_111", "sarimax_auto_weather", "xgboost"]
```

**Requirements:**
```python
%pip install xgboost statsmodels pmdarima
```

**Training Time:** 30-60 seconds per model

**Models:**
- `xgboost` - Gradient boosting with engineered features
- `xgboost_weather` - XGBoost with weather features
- `xgboost_deep_lags` - XGBoost with deeper feature engineering

**Good For:**
- Better performance than statistical models
- Handles non-linear patterns
- Still no GPU needed

---

### Tier 4: Prophet Models (Needs prophet)

```python
models = ["naive", "random_walk", "arima_111", "sarimax_auto_weather", "xgboost", "prophet"]
```

**Requirements:**
```python
%pip install prophet xgboost statsmodels pmdarima
```

**Training Time:** 1-2 minutes per model

**Models:**
- `prophet` - Facebook Prophet (handles seasonality well)
- `prophet_weather` - Prophet with weather regressors

**Good For:**
- Strong seasonal patterns
- Automatic holiday detection
- Good interpretability

---

### Tier 5: Deep Learning (Needs PyTorch + GPU) ⚠️

**Not recommended for initial deployment:**

```python
# Only add after core models work
models = [..., "tft", "nhits"]
```

**Requirements:**
- GPU cluster (much more expensive)
- PyTorch dependencies
- More complex setup

**Models from experiments folder:**
- `nhits` - N-HiTS (best performer: 1.12% MAPE)
- `nbeats` - N-BEATS (1.81% MAPE)
- `tft` - Temporal Fusion Transformer (probabilistic)

**When to Add:**
- After core models work end-to-end
- When you need best possible accuracy
- When budget allows GPU cluster

---

## Recommended Progression

### Phase 1: Get It Working (Day 1)
```python
models = ["naive", "random_walk"]
```
- Goal: End-to-end working
- Verify: Distributions table populated

### Phase 2: Add Statistical (Day 2)
```python
models = ["naive", "random_walk", "arima_111", "sarimax_auto_weather"]
```
- Install: `%pip install statsmodels pmdarima`
- Goal: Better accuracy

### Phase 3: Add ML (Day 3)
```python
models = ["naive", "random_walk", "arima_111", "sarimax_auto_weather", "xgboost"]
```
- Install: `%pip install xgboost`
- Goal: Best non-DL performance

### Phase 4: Add Prophet (Optional)
```python
models = [..., "prophet"]
```
- Install: `%pip install prophet`
- Goal: Seasonality handling

### Phase 5: Deep Learning (Future)
```python
# Only if needed and after core models stable
models = [..., "nhits"]
```
- Requires: GPU cluster
- Goal: Best possible accuracy

---

## Package Installation Cheat Sheet

**In Databricks Notebook:**
```python
# Install all at once (or individually as needed)
%pip install statsmodels pmdarima xgboost prophet
```

**Or add to cluster libraries:**
- Databricks UI → Clusters → Libraries → Install New
- More persistent (survives cluster restarts)

---

## Fail-Open Behavior

**Good News:** If a package is missing, the notebook will:
1. Skip that model
2. Continue with other models
3. Report which models failed

**Example Output:**
```
✅ naive: Trained
✅ random_walk: Trained
⚠️  xgboost: Missing package dependency - skipping
✅ arima_111: Trained
```

**You'll still get:**
- ✅ 3 models trained (naive, random_walk, arima_111)
- ✅ Distributions table populated
- ⚠️ 1 model skipped (xgboost) - can fix later

---

## Current Model Registry

See `ground_truth/config/model_registry.py` for full list of available models.

**Quick Reference:**
- `naive` - Last value persistence
- `random_walk` - Random walk with drift
- `arima_111` - ARIMA(1,1,1)
- `sarimax_auto_weather` - Auto-ARIMA with weather
- `xgboost` - XGBoost baseline
- `xgboost_weather` - XGBoost with weather
- `xgboost_sentiment` - XGBoost with GDELT sentiment
- `prophet` - Prophet baseline
- `prophet_weather` - Prophet with weather

---

## Questions?

- **Which models should I use?** Start with `["naive", "random_walk"]`, add more gradually
- **What if package missing?** Model will be skipped, others continue
- **GPU needed?** No! Only for deep learning (add later)
- **How many models?** Start with 2-3, add more as needed

**Goal: Get distributions table populated quickly!** ✅


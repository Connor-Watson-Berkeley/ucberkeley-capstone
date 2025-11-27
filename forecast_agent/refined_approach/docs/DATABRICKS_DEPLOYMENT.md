# Databricks Deployment Guide

## Quick Setup for Getting Started

### 1. Cluster Configuration

**Recommendation: ML Cluster (No GPU)**

**Why:**
- ✅ Most models don't need GPUs (naive, random_walk, XGBoost, SARIMAX)
- ✅ GPUs only needed for deep learning (TFT, LSTM, N-HiTS)
- ✅ Much cheaper to run
- ✅ Faster cluster startup

**Cluster Settings:**
- **Type**: Standard (ML Runtime)
- **Driver**: Standard_DS3_v2 (4 cores, 14 GB)
- **Workers**: 2-4 nodes (Standard_DS3_v2)
- **Auto-termination**: 30 minutes (saves costs)
- **Spark Version**: Latest LTS

### 2. Recommended Starting Models

**Start Simple - Get It Working First:**

```python
models = ["naive", "random_walk"]
```

**Why start simple:**
- ✅ No dependencies (just pandas/numpy)
- ✅ Fast to train
- ✅ Easy to debug
- ✅ Gets distributions table populated quickly

**Then Add More Models:**

```python
# Round 1: Simple models (no extra dependencies)
models = ["naive", "random_walk"]

# Round 2: Add statistical models (needs statsmodels)
models = ["naive", "random_walk", "arima_111", "sarimax_auto_weather"]

# Round 3: Add ML models (needs xgboost)
models = ["naive", "random_walk", "arima_111", "sarimax_auto_weather", "xgboost"]

# Round 4: Add Prophet (needs prophet package)
models = ["naive", "random_walk", "arima_111", "sarimax_auto_weather", "xgboost", "prophet"]
```

### 3. Package Dependencies

**Standard Databricks ML Runtime Includes:**
- ✅ pandas, numpy, scipy
- ✅ scikit-learn
- ✅ pyspark

**May Need to Install:**
- ⚠️ xgboost (if using XGBoost models)
- ⚠️ statsmodels (if using ARIMA/SARIMAX)
- ⚠️ prophet (if using Prophet models)
- ⚠️ pmdarima (for auto-ARIMA)

**Installation Options:**

**Option 1: Cluster Libraries (Recommended)**
- Add libraries to cluster configuration
- Persists across sessions
- Easy to manage

**Option 2: Notebook Install (Per-Run)**
```python
%pip install xgboost statsmodels prophet pmdarima
```

**Option 3: Fail-Open (Current Approach)**
- Notebook handles missing packages gracefully
- Skips models that fail to import
- Continues with available models

### 4. Fail-Open Package Handling

The notebooks are designed to handle missing packages gracefully:

**In Training Notebook:**
```python
for model_key in models:
    try:
        model = create_model_from_registry(model_key)
        # Train model...
    except ImportError as e:
        print(f"⚠️  {model_key}: Missing package - skipping")
        failed_count += 1
        continue  # Move to next model
```

**What This Means:**
- Missing xgboost? XGBoost models skip, others continue ✅
- Missing prophet? Prophet models skip, others continue ✅
- Only have pandas/numpy? Can still run naive/random_walk ✅

### 5. Recommended Model Progression

**Phase 1: Get It Working (Now)**
```python
models = ["naive", "random_walk"]
```
- No dependencies
- Fast
- Gets distributions table populated

**Phase 2: Add Statistical Models**
```python
models = ["naive", "random_walk", "arima_111", "sarimax_auto_weather"]
```
- Needs: statsmodels, pmdarima
- Still no GPU needed
- Good baseline coverage

**Phase 3: Add ML Models**
```python
models = ["naive", "random_walk", "arima_111", "sarimax_auto_weather", "xgboost"]
```
- Needs: xgboost
- Still no GPU needed
- Better performance

**Phase 4: Add More (Later)**
```python
models = [..., "prophet", "xgboost_weather"]
```
- Needs: prophet
- More features

**Phase 5: Deep Learning (Future)**
```python
# Only if needed and packages available
models = [..., "tft", "nhits"]
```
- Needs: pytorch, pytorch-forecasting, darts
- **Requires GPU cluster** (slower, more expensive)
- Best performance but complex

### 6. Deployment Steps

#### Step 1: Push to Databricks Repos

1. Connect repo to GitHub in Databricks
2. Pull latest changes
3. Verify `forecast_agent/refined_approach/` folder exists

#### Step 2: Create/Configure Cluster

1. Create ML cluster (no GPU)
2. Add libraries if needed (or install in notebook)
3. Test cluster starts successfully

#### Step 3: Run Training Notebook

1. Open `notebooks/01_train_models.py`
2. Set parameters:
   ```python
   commodity = "Coffee"
   models = "naive,random_walk"  # Start simple
   train_frequency = "semiannually"
   start_date = "2020-01-01"
   end_date = "2024-01-01"
   ```
3. Run all cells
4. Check summary for trained/skipped/failed counts

#### Step 4: Run Inference Notebook

1. Open `notebooks/02_generate_forecasts.py` (when created)
2. Set parameters
3. Run all cells
4. Verify distributions table populated

#### Step 5: Set Up Daily Job

1. Create Databricks Job
2. Point to `00_daily_production.py`
3. Schedule daily (e.g., 6 AM UTC)
4. Set parameters
5. Test run manually first

### 7. Troubleshooting

#### Package Import Failures

**Symptom:** Model fails with ImportError

**Solution:**
- Install package in notebook: `%pip install package_name`
- Or add to cluster libraries
- Model will be skipped (fail-open), others continue

#### Out of Memory

**Symptom:** Cluster runs out of memory

**Solution:**
- Use smaller date ranges for testing
- Reduce number of models
- Increase cluster size

#### Long Training Times

**Symptom:** Training takes too long

**Solution:**
- Start with simple models (naive, random_walk)
- Use shorter date ranges for testing
- Increase cluster size

### 8. Best Practices

1. **Start Small**
   - Use 2-3 simple models first
   - Test with small date range
   - Verify everything works

2. **Add Models Gradually**
   - Add one model type at a time
   - Verify it works before adding more
   - Check for package dependencies

3. **Monitor Costs**
   - Use auto-termination (30 min)
   - Stop cluster when not in use
   - Monitor DBU usage

4. **Fail-Open Always**
   - One model fails? Others continue
   - Missing package? Skip that model
   - Get maximum coverage

## Quick Answer to Your Questions

**Q: ML cluster without GPUs appropriate?**
**A: ✅ Yes!** Most models don't need GPUs. Only deep learning models (TFT, LSTM) need GPUs.

**Q: What models are currently running?**
**A: Check `ground_truth/config/model_registry.py` - start with:**
- `naive`, `random_walk` (no dependencies)
- `arima_111`, `sarimax_auto_weather` (needs statsmodels)
- `xgboost` (needs xgboost)

**Q: Include models from experiments folder?**
**A: ⚠️ Not yet** - N-HiTS, TFT, etc. need:
- PyTorch dependencies
- GPU cluster
- More complex setup
- Add later after core models work

**Q: Risk of slowing us down?**
**A: No** - fail-open design means:
- Missing package? Skip that model, continue
- One model fails? Continue with others
- Get maximum coverage

**Q: Package dependency handling?**
**A: ✅ Already built in** - notebooks handle missing packages gracefully and skip models that can't load.

## Ready to Deploy!

1. Start with: `models = ["naive", "random_walk"]`
2. Get it working end-to-end
3. Add more models gradually
4. Populate distributions table ✅


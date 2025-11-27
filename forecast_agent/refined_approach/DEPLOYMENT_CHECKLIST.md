# Deployment Checklist - Ready to Push to Databricks! ✅

## Quick Answers to Your Questions

### ✅ ML Cluster Without GPUs - Perfect!

**Yes, ML cluster without GPUs is appropriate!**

**Why:**
- Most models don't need GPUs (naive, random_walk, XGBoost, SARIMAX, Prophet)
- Only deep learning models (TFT, LSTM, N-HiTS) need GPUs
- Much cheaper to run
- Faster cluster startup

**Cluster Recommendation:**
- Type: Standard (ML Runtime)
- Driver: Standard_DS3_v2 (4 cores, 14 GB)
- Workers: 2-4 nodes
- Auto-termination: 30 minutes

---

### 📦 Current Models from Model Config

**Start with these (no dependencies):**
```python
models = ["naive", "random_walk"]
```

**Available in model registry:**
- `naive` - Last value persistence ✅ (no deps)
- `random_walk` - Random walk with drift ✅ (no deps)
- `arima_111` - ARIMA(1,1,1) (needs statsmodels)
- `sarimax_auto_weather` - Auto-ARIMA with weather (needs statsmodels)
- `xgboost` - XGBoost (needs xgboost package)
- `xgboost_weather` - XGBoost with weather (needs xgboost)
- `prophet` - Prophet (needs prophet package)
- `prophet_weather` - Prophet with weather (needs prophet)

**Full list:** See `ground_truth/config/model_registry.py`

---

### ⚠️ Models from Experiments Folder

**Recommendation: Skip for now, add later**

**Why:**
- N-HiTS, N-BEATS, TFT require PyTorch dependencies
- Need GPU cluster (more expensive, slower)
- More complex setup
- **Goal: Get core models working first, populate distributions table**

**From experiments folder:**
- `nhits` - N-HiTS (best: 1.12% MAPE) - needs PyTorch + GPU
- `nbeats` - N-BEATS (1.81% MAPE) - needs PyTorch + GPU
- `tft` - Temporal Fusion Transformer - needs PyTorch + GPU

**Add later when:**
1. Core models working end-to-end
2. Distributions table populated
3. Budget allows GPU cluster

---

### ⚠️ Risk of Slowing Us Down?

**No risk! Fail-open design protects you:**

✅ **Package Missing?**
- Model skipped automatically
- Others continue
- Clear error message

✅ **One Model Fails?**
- Training continues
- Others still train
- Summary shows what worked

✅ **Example:**
```
✅ naive: Trained
✅ random_walk: Trained
⚠️  xgboost: Missing package - skipping
✅ arima_111: Trained

Summary: 3 trained, 1 skipped
```

**You'll always get maximum coverage!**

---

### ✅ Package Dependency Fail-Open

**Already built in!** The notebooks handle missing packages gracefully:

**In Training Notebook:**
- Tries to create model
- If ImportError → logs warning, skips model
- Continues with other models

**What You'll See:**
```
⚠️  xgboost: Missing package dependency - skipping
   Error: No module named 'xgboost'
   💡 Install missing package or remove xgboost from models list
```

**You can:**
1. Install package: `%pip install xgboost`
2. Or remove from models list
3. Or leave it - others will still work

---

## Deployment Steps

### 1. Push to Databricks Repos ✅

**If using Databricks Repos:**
- Push code to GitHub
- Pull in Databricks Repos
- Verify `forecast_agent/refined_approach/` exists

**If using DBFS:**
- Upload files to `/dbfs/FileStore/forecast_agent/refined_approach/`

### 2. Create Cluster ✅

- **Type:** Standard (ML Runtime)
- **No GPU** ✅
- **Auto-termination:** 30 minutes
- **Libraries:** Optional (or install in notebook)

### 3. Run Training Notebook ✅

**Open:** `notebooks/01_train_models.py`

**Parameters:**
```
commodity: Coffee
models: naive,random_walk
train_frequency: semiannually
model_version: v1.0
start_date: 2020-01-01
end_date: 2024-01-01
```

**Run all cells**

**Expected Output:**
```
✅ Models Trained: 2
⏩ Models Skipped: 0
❌ Models Failed: 0
```

### 4. (Optional) Install Packages

If you want to add more models, install packages:

```python
%pip install statsmodels pmdarima  # For ARIMA models
%pip install xgboost               # For XGBoost models
%pip install prophet               # For Prophet models
```

Then re-run with more models:
```
models: naive,random_walk,arima_111,xgboost
```

### 5. Run Inference (Next Step)

Once models trained, use `02_generate_forecasts.py` to populate distributions table.

---

## Recommended Model Progression

### Phase 1: Get It Working (Now)
```python
models = ["naive", "random_walk"]
```
- ✅ No dependencies
- ✅ Fast (< 1 second each)
- ✅ Gets distributions table populated

### Phase 2: Add Statistical (Optional)
```python
models = ["naive", "random_walk", "arima_111", "sarimax_auto_weather"]
```
- Install: `%pip install statsmodels pmdarima`
- Better accuracy

### Phase 3: Add ML (Optional)
```python
models = ["naive", "random_walk", "arima_111", "sarimax_auto_weather", "xgboost"]
```
- Install: `%pip install xgboost`
- Best non-DL performance

### Phase 4: Deep Learning (Future)
```python
# Only after core models work
models = [..., "nhits"]
```
- Requires GPU cluster
- Best accuracy (but complex)

---

## Troubleshooting

### Missing Packages

**Symptom:** Model shows "Missing package dependency"

**Solution:**
```python
%pip install package_name
```

**Or:** Remove from models list, others continue ✅

### Import Errors

**Symptom:** Cannot import refined_approach modules

**Solution:**
- Verify path in notebook (should auto-detect)
- Check repo structure
- Verify files exist in Databricks

### Model Training Fails

**Symptom:** Model fails to train

**Solution:**
- Check error message (full traceback printed)
- Verify data exists for date range
- Check minimum training days (365*2)
- Model will be skipped, others continue ✅

---

## Summary

✅ **ML Cluster without GPU:** Perfect!

✅ **Starting Models:** `["naive", "random_walk"]` (no dependencies)

⚠️ **Experiment Models:** Skip for now (need GPU, add later)

✅ **Package Dependencies:** Fail-open built in (skip models that fail)

✅ **Risk of Slowing Down:** None - fail-open design protects you

**Ready to deploy!** 🚀

---

## Files to Push

```
forecast_agent/refined_approach/
├── notebooks/
│   ├── 00_setup_and_imports.py        ✅ New (fail-open imports)
│   ├── 01_train_models.py              ✅ Updated (better error handling)
│   └── 00_daily_production.py          ✅ Ready
├── data_loader.py                       ✅ Ready
├── evaluator.py                         ✅ Ready
├── cross_validator.py                   ✅ Ready
├── model_pipeline.py                    ✅ Updated (fail-open)
├── model_persistence.py                 ✅ Ready
├── distributions_writer.py              ✅ Ready
├── daily_production.py                  ✅ Ready
├── docs/
│   └── DATABRICKS_DEPLOYMENT.md        ✅ New (deployment guide)
├── MODEL_RECOMMENDATIONS.md             ✅ New (model selection)
└── DEPLOYMENT_CHECKLIST.md              ✅ This file
```

**All files ready to push!** ✅


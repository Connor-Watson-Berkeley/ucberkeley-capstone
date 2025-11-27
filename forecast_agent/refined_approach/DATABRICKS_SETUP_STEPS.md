# Step-by-Step: Pull into Databricks and Run

## ✅ Step 1: Pull Repo in Databricks

### If Using Databricks Repos:

1. Open Databricks workspace
2. Go to **Repos** (left sidebar)
3. Find your repo: `ucberkeley-capstone`
4. Click **Pull** button (or three dots → Pull)
5. Verify you see latest commit: `ab290ba Add refined forecast agent approach...`

### If Using DBFS/Workspace:

1. Go to **Workspace** (left sidebar)
2. Navigate to your project folder
3. Upload files or use git sync

---

## ✅ Step 2: Create/Configure Cluster

1. Go to **Clusters** (left sidebar)
2. Click **Create Cluster** (or use existing)
3. **Settings:**
   - **Cluster Name**: `forecast-training` (or your name)
   - **Cluster Mode**: Standard
   - **Databricks Runtime**: ML Runtime (latest LTS)
   - **Node Type**: Standard_DS3_v2 (or similar - no GPU needed)
   - **Workers**: 2-4 nodes
   - **Auto-termination**: 30 minutes
4. Click **Create Cluster**
5. Wait for cluster to start (green status)

---

## ✅ Step 3: Open Training Notebook

1. In **Repos**, navigate to:
   ```
   ucberkeley-capstone
   └── forecast_agent
       └── refined_approach
           └── notebooks
               └── 01_train_models.py
   ```
2. Right-click `01_train_models.py` → **Open in Notebook**
3. **Attach notebook to cluster** (top right) → select your cluster

---

## ✅ Step 4: Configure Parameters

In the notebook, find the widget section (near top). Set:

```
commodity: Coffee
models: naive,random_walk
train_frequency: semiannually
model_version: v1.0
start_date: 2020-01-01
end_date: 2024-01-01
```

**Start Simple!**
- `models: naive,random_walk` (no dependencies)
- Get it working first
- Add more models later

---

## ✅ Step 5: Run All Cells

1. Click **Run All** (top right)
2. Watch for:
   - ✅ Path detection: `Added /Workspace/Repos/.../refined_approach to path`
   - ✅ Module imports successful
   - ⚠️  Any package warnings (OK - will skip those models)
3. Wait for training to complete

**Expected Output:**
```
✅ Models Trained: 2
⏩ Models Skipped: 0
❌ Models Failed: 0
```

---

## ✅ Step 6: Verify Models Saved

1. In notebook, run this cell (add at end):

```python
# Check trained models
from pyspark.sql.functions import col

display(spark.sql("""
    SELECT 
        commodity,
        model_name,
        training_date,
        model_version,
        is_active
    FROM commodity.forecast.trained_models
    WHERE commodity = 'Coffee'
    ORDER BY training_date DESC, model_name
    LIMIT 20
"""))
```

You should see rows with:
- `commodity: Coffee`
- `model_name: Naive` or `RandomWalk`
- `training_date: 2020-06-01, 2020-12-01, ...` (semiannual dates)
- `is_active: true`

---

## 🎉 Success!

If you see models in the table, **you're done!** 

**Next Steps (later):**
1. Run inference notebook (`02_generate_forecasts.py`) to populate distributions table
2. Set up daily job (`00_daily_production.py`)
3. Add more models (XGBoost, ARIMA, etc.)

---

## 🐛 Troubleshooting

### "Cannot find module" errors

**Solution:** Check path detection in notebook. Should see:
```
✅ Added /Workspace/Repos/.../forecast_agent/refined_approach to path
```

If not, verify repo structure matches.

### Package import failures

**This is OK!** Models will be skipped. You'll see:
```
⚠️  xgboost: Missing package dependency - skipping
```

To fix later: Install package or remove from models list.

### Cluster issues

**Solution:**
- Check cluster is running (green status)
- Verify notebook is attached to cluster
- Try restarting cluster

### No data found

**Solution:**
- Check `start_date` - ensure data exists
- Verify table: `commodity.silver.unified_data`
- Check commodity name matches (case-sensitive)

---

## 📋 Quick Checklist

- [ ] Repo pulled in Databricks
- [ ] Cluster created and running
- [ ] Notebook opened and attached to cluster
- [ ] Parameters set: `models: naive,random_walk`
- [ ] All cells run successfully
- [ ] Models appear in `trained_models` table
- [ ] Summary shows: `✅ Models Trained: 2+`

**You're ready to go! Happy Thanksgiving! 🦃**


# ML Forecast Clusters

Databricks cluster configurations for ML forecasting workflows.

---

## Available Clusters

### 1. ML Testing Cluster (`ml-testing-cluster`)

**Purpose:** Small cluster for testing, validation, and development

**Configuration:**
- **Node Type:** `i3.xlarge` (4 cores, 30.5 GB RAM, 950 GB NVMe SSD)
- **Workers:** 1-2 (autoscale)
- **Driver:** i3.xlarge
- **Auto-termination:** 30 minutes
- **Spark Version:** 13.3.x-scala2.12
- **Cost:** ~$0.30/hour (with spot instances)

**Use Cases:**
- Running validation notebooks (e.g., `validate_gold_unified_data.py`)
- Testing end-to-end pipeline with small date ranges
- Exploratory data analysis
- Debugging pipeline issues
- Interactive notebook development

**Estimated Runtime:**
- End-to-end example (2 models, Coffee 2024): ~10-15 minutes
- Validation notebook: ~5 minutes

---

### 2. ML Training Cluster (`ml-training-cluster`)

**Purpose:** Large cluster for full cross-validation training and backfills

**Configuration:**
- **Node Type:** `i3.2xlarge` (8 cores, 61 GB RAM, 1.9 TB NVMe SSD)
- **Workers:** 2-8 (autoscale)
- **Driver:** i3.2xlarge
- **Auto-termination:** 60 minutes
- **Spark Version:** 13.3.x-scala2.12
- **Cost:** ~$1.20-4.80/hour (with spot instances, depending on workers)

**Use Cases:**
- Full 5-fold cross-validation on 10 years of data
- Training multiple models in parallel
- Backfilling historical forecasts (2015-2024)
- Production model training
- Large-scale feature engineering

**Estimated Runtime:**
- 5-fold CV for 1 model (Coffee, 2015-2024): ~30-45 minutes
- Training 4 models in sequence: ~2-3 hours
- Full backfill (all models, all dates): ~6-8 hours

**Spark Configuration:**
- Adaptive query execution enabled
- Skew join optimization enabled
- Default parallelism: 112 tasks (7 cores × 8 workers × 2)
- Executor memory: 48 GB

---

## Creating Clusters

### Create Both Clusters

```bash
python forecast_agent/infrastructure/databricks/clusters/create_ml_clusters.py
```

### Create Only Testing Cluster

```bash
python forecast_agent/infrastructure/databricks/clusters/create_ml_clusters.py --cluster testing
```

### Create Only Training Cluster

```bash
python forecast_agent/infrastructure/databricks/clusters/create_ml_clusters.py --cluster training
```

### Create Without Waiting for Startup

```bash
# Create clusters but don't wait for them to start (returns immediately)
python forecast_agent/infrastructure/databricks/clusters/create_ml_clusters.py --no-start
```

---

## Cluster Selection Guide

**Use Testing Cluster When:**
- ✅ Testing code changes
- ✅ Running validation notebooks
- ✅ Exploring data (< 1 year)
- ✅ Training 1-2 models on recent data (e.g., 2024 only)
- ✅ Cost is a concern

**Use Training Cluster When:**
- ✅ Running full 5-fold CV on all historical data
- ✅ Training multiple models in parallel
- ✅ Backfilling forecasts for production
- ✅ Performance is critical
- ✅ Working with large feature sets

---

## Cost Optimization

Both clusters use **SPOT_WITH_FALLBACK** instances:
- First node is on-demand (guaranteed availability)
- Remaining workers are spot instances (up to 90% savings)
- Auto-terminates after inactivity to prevent wasted spend

**Estimated Monthly Costs** (assuming 40 hours/month usage):

| Cluster | Configuration | Hourly Cost | Monthly Cost |
|---------|--------------|-------------|--------------|
| Testing | 1-2 workers | $0.30 | $12 |
| Training | 2-8 workers (avg 4) | $2.40 | $96 |

**Total:** ~$108/month for moderate usage

**Tips to Reduce Costs:**
1. Always use testing cluster for development
2. Only use training cluster for production runs
3. Verify auto-termination is working (check Databricks UI)
4. Use `--no-start` flag if creating clusters for later use
5. Terminate clusters manually when done with large jobs

---

## Monitoring Cluster Usage

### Check Cluster Status

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# List all clusters
for cluster in w.clusters.list():
    print(f"{cluster.cluster_name}: {cluster.state}")
```

### View Cluster Metrics in Databricks UI

1. Go to **Compute** in left sidebar
2. Select cluster
3. View **Metrics** tab for:
   - CPU utilization
   - Memory usage
   - Shuffle read/write
   - Task execution times

### Terminate Clusters

```python
# Terminate testing cluster
w.clusters.delete(cluster_id="<cluster_id>")

# Or use CLI
databricks clusters delete --cluster-id <cluster_id>
```

---

## Troubleshooting

### Cluster Fails to Start

**Check:**
1. AWS capacity issues (switch availability zone in config)
2. Databricks workspace limits (contact support to increase)
3. IAM permissions (ensure Databricks can create EC2 instances)

**Solution:**
- Try different availability zone in `aws_attributes.zone_id`
- Reduce `autoscale.max_workers` if hitting limits
- Contact Databricks support for workspace quota increase

### Out of Memory Errors

**Symptoms:**
- `java.lang.OutOfMemoryError: Java heap space`
- Tasks failing with memory errors

**Solutions:**
1. Increase cluster size (use training cluster instead of testing)
2. Reduce data parallelism: `spark.sql.shuffle.partitions = 200` → `100`
3. Enable adaptive query execution (already enabled in configs)
4. Persist intermediate results: `df.cache()`

### Slow Performance

**Symptoms:**
- CV taking hours instead of minutes
- High shuffle write/read times

**Solutions:**
1. Check Spark UI for skewed partitions
2. Enable skew join optimization (already enabled in training cluster)
3. Increase parallelism: `spark.default.parallelism = 200`
4. Repartition data by commodity: `df.repartition("commodity")`

---

## Configuration Files

- **Testing:** `ml_testing_cluster.json`
- **Training:** `ml_training_cluster.json`
- **Creation Script:** `create_ml_clusters.py`

**To modify:**
1. Edit JSON config (e.g., change `node_type_id` or `autoscale` settings)
2. Delete existing cluster in Databricks UI
3. Re-run creation script

---

## Integration with ML Pipeline

### Testing Workflow

```python
# 1. Attach to testing cluster
# 2. Run validation
%run research_agent/infrastructure/databricks/validate_gold_unified_data.py

# 3. Test end-to-end on small date range
%run forecast_agent/ml_lib/examples/end_to_end_example.py
```

### Production Training Workflow

```python
# 1. Attach to training cluster
# 2. Train models with full CV
%run forecast_agent/ml_lib/train.py \
    --commodity Coffee \
    --models naive_baseline linear_weather_min_max ridge_top_regions \
    --n-folds 5 \
    --window-type expanding

# 3. Generate forecasts
%run forecast_agent/ml_lib/inference.py \
    --commodity Coffee \
    --models linear_weather_min_max \
    --n-paths 2000
```

---

## Cluster Specifications Comparison

| Feature | Testing Cluster | Training Cluster |
|---------|----------------|------------------|
| **Node Type** | i3.xlarge | i3.2xlarge |
| **Cores per Node** | 4 | 8 |
| **RAM per Node** | 30.5 GB | 61 GB |
| **Storage per Node** | 950 GB NVMe | 1.9 TB NVMe |
| **Min Workers** | 1 | 2 |
| **Max Workers** | 2 | 8 |
| **Total Cores (max)** | 12 | 72 |
| **Total RAM (max)** | 91.5 GB | 549 GB |
| **Parallelism (max)** | 24 tasks | 112 tasks |
| **Auto-terminate** | 30 min | 60 min |
| **Hourly Cost** | $0.30 | $1.20-4.80 |

---

**Last Updated:** 2024-12-05
**Maintained By:** Connor Watson
**Status:** Active

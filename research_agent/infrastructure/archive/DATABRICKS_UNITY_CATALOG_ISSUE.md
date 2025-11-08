# Databricks Unity Catalog Issue - Root Cause Analysis

**Date**: 2025-11-07
**Status**: ⚠️ BLOCKED - Awaiting Databricks Support
**Severity**: High - Blocks compute cluster access to Unity Catalog

## Problem

All Unity Catalog operations hang indefinitely when run from compute clusters (not SQL Warehouses).

**Affected Commands**:
- `spark.sql("USE CATALOG commodity")` - Hangs forever
- `df = spark.table("commodity.bronze.weather")` - Hangs forever
- Any query accessing Unity Catalog tables

**Working**:
- ✅ SQL Warehouses can access Unity Catalog
- ✅ Basic Spark operations work on compute clusters
- ✅ Network connectivity is fine (`curl https://www.google.com` works)

## Root Cause

**Broken PrivateLink/Secure Cluster Connectivity Configuration**

The `/etc/hosts` file on compute clusters contains:
```
10.53.184.99 oregon.cloud.databricks.com
```

This redirects Unity Catalog requests to a private IP (`10.53.184.99`) where no service is running, causing all Unity Catalog operations to hang.

**Validation**:
```python
# Run in notebook on any cluster
dbutils.fs.head("file:///etc/hosts")

# Output shows the problematic redirect:
# 10.53.184.99 oregon.cloud.databricks.com
```

## Why This Happened

This is a **Databricks control plane configuration issue**:
1. Databricks Account has broken "Private Access Settings"
2. This configuration is pushed to all compute clusters
3. The redirect points to a non-functioning PrivateLink endpoint

## Why You Can't Fix It

1. Setting is controlled in: **Account Console** → **Security** → **Private access settings**
2. Attempting to access this page returns: **403 Forbidden**
3. Only Databricks Support can modify control plane configurations

## Solution

**Contact Databricks Support** with the following:

---

**Subject**: Unity Catalog hangs due to broken Private Access Settings

**Message**:
```
Our clusters cannot access Unity Catalog - all UC operations hang indefinitely.

Root Cause:
/etc/hosts on clusters redirects oregon.cloud.databricks.com to private IP 10.53.184.99
where no service is responding.

Evidence:
- Cluster /etc/hosts contains: 10.53.184.99 oregon.cloud.databricks.com
- All Unity Catalog commands (USE CATALOG, spark.table, etc.) hang indefinitely
- SQL Warehouses work correctly (bypass this configuration)
- Network connectivity verified (curl https://www.google.com succeeds)

Affected Clusters:
- 1107-041742-dpvkvaj6 (unity-catalog-cluster)
- 1030-040527-3do4v2at (general-purpose-mid-compute)

Account Issue:
- Cannot access Account Console → Security → Private access settings (403 Forbidden)
- Account admin permissions confirmed but page inaccessible

Request:
Please remove the broken Private Access Settings from our account
(Workspace: dbc-fd7b00f3-7a6d.cloud.databricks.com)
```

---

## Workarounds

While waiting for Databricks Support:

### Option 1: Use SQL Warehouse (Recommended)

SQL Warehouses bypass the broken cluster configuration.

**Access**:
- SQL Editor: Works perfectly
- Notebooks: Attach to SQL Warehouse instead of compute cluster

**Limitations**:
- More expensive (serverless)
- Cannot use for long-running jobs

**Example**:
```python
# In Databricks SQL Editor or notebook attached to SQL Warehouse
spark.sql("USE CATALOG commodity")
df = spark.sql("SELECT * FROM commodity.bronze.weather LIMIT 10")
display(df)
```

### Option 2: Use Databricks SQL Python Library

Access Unity Catalog via the `databricks-sql-connector` library.

**Setup**:
```python
from databricks import sql

connection = sql.connect(
    server_hostname="dbc-fd7b00f3-7a6d.cloud.databricks.com",
    http_path="/sql/1.0/warehouses/3cede8561503a13c",
    access_token="<your-token>"
)

cursor = connection.cursor()
cursor.execute("SELECT * FROM commodity.bronze.weather LIMIT 10")
results = cursor.fetchall()
cursor.close()
connection.close()
```

**Usage**: Scripts in `research_agent/infrastructure/` use this approach

### Option 3: Direct S3 Access (No Unity Catalog)

Access Delta tables directly from S3 without Unity Catalog.

**Example**:
```python
# Read Delta table directly from S3
df = spark.read.format("delta").load("s3://groundtruth-capstone/delta/bronze/weather/")
display(df)
```

**Limitations**:
- Bypasses Unity Catalog governance
- No catalog/schema organization
- Must know exact S3 paths

## Affected Workflows

### Currently Working
- ✅ Weather backfill (writes to S3, doesn't need UC)
- ✅ SQL Editor queries (uses SQL Warehouse)
- ✅ Databricks SQL Python scripts (bypass cluster config)

### Blocked
- ❌ Notebook queries on compute clusters
- ❌ Trading agent development (if using compute clusters)
- ❌ Forecast agent (if using compute clusters)

## Affected Clusters

| Cluster ID | Name | Status | UC Access |
|------------|------|--------|-----------|
| 1107-041742-dpvkvaj6 | unity-catalog-cluster | RUNNING | ❌ Hangs |
| 1030-040527-3do4v2at | general-purpose-mid-compute | RUNNING | ❌ Hangs |
| 1106-170051-fzr9yk1u | unity-test-cluster | TERMINATED | ❌ Would hang |

## Timeline

- **Nov 7, 2025 20:00**: Issue reported - Unity Catalog queries hanging
- **Nov 7, 2025 20:17**: Created unity-catalog-cluster with SINGLE_USER mode
- **Nov 7, 2025 20:30**: Tested cluster - confirmed USE CATALOG hangs
- **Nov 7, 2025 20:45**: Root cause identified - broken PrivateLink config
- **Nov 7, 2025 20:50**: Documented issue, created workarounds

## Next Steps

1. **Immediate**: Contact Databricks Support (see message template above)
2. **Short-term**: Use SQL Warehouse for Unity Catalog access
3. **Long-term**: Once fixed, validate compute clusters can access UC
4. **Follow-up**: Remove workarounds once issue resolved

## Testing After Fix

Once Databricks Support confirms the fix:

```bash
# Test Unity Catalog access from compute cluster
cd research_agent/infrastructure
python test_cluster_unity_catalog.py

# Expected: All 3 tests should PASS
```

## References

- Databricks PrivateLink: https://docs.databricks.com/security/network/classic/privatelink.html
- Unity Catalog Troubleshooting: https://docs.databricks.com/data-governance/unity-catalog/troubleshooting.html
- Secure Cluster Connectivity: https://docs.databricks.com/security/network/classic/secure-cluster-connectivity.html

---

**Last Updated**: 2025-11-07
**Updated By**: Claude Code
**Support Ticket**: [Pending - contact support]

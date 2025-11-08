"""
Diagnose Unity Catalog Cluster Issues

Run this in your Databricks notebook to diagnose why queries are hanging.
"""

import json

print("=" * 80)
print("Unity Catalog Cluster Diagnostics")
print("=" * 80)

# Check 1: Which cluster are you connected to?
print("\n1. Checking current cluster...")
try:
    cluster_info = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
    print(f"   Connected to cluster: {cluster_info}")

    cluster_name = spark.conf.get("spark.databricks.clusterUsageTags.clusterName")
    print(f"   Cluster name: {cluster_name}")

    if cluster_name != "unity-catalog-cluster":
        print(f"   ⚠️  WARNING: You're not on 'unity-catalog-cluster'!")
        print(f"   Please switch to: unity-catalog-cluster (ID: 1107-041742-dpvkvaj6)")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Check 2: Unity Catalog enabled?
print("\n2. Checking Unity Catalog status...")
try:
    uc_enabled = spark.conf.get("spark.databricks.unityCatalog.enabled", "false")
    print(f"   Unity Catalog enabled: {uc_enabled}")

    if uc_enabled != "true":
        print(f"   ❌ Unity Catalog is NOT enabled on this cluster!")
        print(f"   Switch to cluster: unity-catalog-cluster")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Check 3: Data security mode
print("\n3. Checking cluster security mode...")
try:
    # This will show in cluster details
    security_mode = spark.conf.get("spark.databricks.dataLineage.enabled", "unknown")
    print(f"   Data lineage: {security_mode}")

    # Try to get user info
    current_user = spark.sql("SELECT current_user()").collect()[0][0]
    print(f"   Current user: {current_user}")
except Exception as e:
    print(f"   ⚠️  Warning: {e}")

# Check 4: Can we access catalogs?
print("\n4. Testing catalog access...")
try:
    catalogs = spark.sql("SHOW CATALOGS").collect()
    print(f"   ✅ Found {len(catalogs)} catalogs:")
    for cat in catalogs:
        print(f"      - {cat[0]}")
except Exception as e:
    print(f"   ❌ Cannot list catalogs: {e}")

# Check 5: Can we USE commodity catalog?
print("\n5. Testing 'USE CATALOG commodity'...")
try:
    spark.sql("USE CATALOG commodity")
    print(f"   ✅ Successfully set catalog to 'commodity'")

    # Check current catalog
    current_cat = spark.sql("SELECT current_catalog()").collect()[0][0]
    print(f"   Current catalog: {current_cat}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    print(f"   This is where queries hang!")

# Check 6: Can we list schemas?
print("\n6. Testing schema listing...")
try:
    schemas = spark.sql("SHOW SCHEMAS IN commodity").collect()
    print(f"   ✅ Found {len(schemas)} schemas:")
    for schema in schemas:
        print(f"      - {schema[0]}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Check 7: Can we query a table?
print("\n7. Testing table query...")
try:
    count = spark.sql("SELECT COUNT(*) FROM commodity.bronze.weather").collect()[0][0]
    print(f"   ✅ Query successful! Row count: {count:,}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\n" + "=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)

print("\n✅ If all checks passed:")
print("   Your cluster is working! The issue might be intermittent.")

print("\n⚠️  If checks failed:")
print("   1. Verify you're on cluster: unity-catalog-cluster")
print("   2. In notebook, click cluster dropdown at top")
print("   3. Select 'unity-catalog-cluster' from the list")
print("   4. Wait for notebook to reconnect")
print("   5. Re-run this diagnostic script")

print("\n❌ If Unity Catalog is not enabled:")
print("   1. Your notebook is on the wrong cluster")
print("   2. Create a new cluster with these settings:")
print("      - Access Mode: Single User")
print("      - Runtime: 13.3.x LTS or higher")
print("      - Enable Unity Catalog")
print("   3. Or switch to existing cluster: unity-catalog-cluster")

print("\n" + "=" * 80)

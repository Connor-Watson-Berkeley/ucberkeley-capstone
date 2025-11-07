"""
Create Databricks Clusters for Unity Catalog and S3 Ingestion

This script creates two clusters with different purposes:
1. Unity Catalog Cluster - For querying Bronze/Silver/Forecast tables
2. S3 Ingestion Cluster - For Auto Loader jobs writing to landing zone

Prerequisites:
- Databricks CLI installed: pip install databricks-cli
- Databricks configured: databricks configure --token
- Admin permissions in Databricks workspace

Usage:
    python create_databricks_clusters.py

    # Or create specific cluster:
    python create_databricks_clusters.py --cluster unity
    python create_databricks_clusters.py --cluster s3-ingestion
"""

import json
import subprocess
import sys
import argparse
import time
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
UNITY_CATALOG_CONFIG = SCRIPT_DIR / "databricks_unity_catalog_cluster.json"
S3_INGESTION_CONFIG = SCRIPT_DIR / "databricks_s3_ingestion_cluster.json"

def run_command(cmd, description):
    """Run a shell command and return output"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"✅ Success")
        if result.stdout:
            print(result.stdout)
        return result.stdout
    else:
        print(f"❌ Failed")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return None


def check_databricks_cli():
    """Check if Databricks CLI is installed and configured"""
    print("\n" + "="*80)
    print("Checking Databricks CLI")
    print("="*80)

    # Check if databricks CLI is installed
    result = subprocess.run(
        ["databricks", "--version"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ Databricks CLI not found")
        print("\nInstall with: pip install databricks-cli")
        print("Configure with: databricks configure --token")
        return False

    print(f"✅ Databricks CLI installed: {result.stdout.strip()}")

    # Check if configured
    result = subprocess.run(
        ["databricks", "workspace", "ls", "/"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ Databricks CLI not configured")
        print("\nConfigure with: databricks configure --token")
        print("  Host: https://dbc-fd7b00f3-7a6d.cloud.databricks.com")
        print("  Token: <your personal access token>")
        return False

    print("✅ Databricks CLI configured")
    return True


def create_cluster(config_file, cluster_type):
    """Create a Databricks cluster from JSON config"""
    print(f"\n{'='*80}")
    print(f"Creating {cluster_type} Cluster")
    print(f"{'='*80}")

    # Load cluster config
    with open(config_file, 'r') as f:
        config = json.load(f)

    cluster_name = config['cluster_name']
    print(f"\nCluster Name: {cluster_name}")
    print(f"Config File: {config_file}")
    print(f"Runtime: {config['spark_version']}")
    print(f"Node Type: {config['node_type_id']}")
    print(f"Workers: {config.get('num_workers', 'auto')} (autoscale: {config.get('autoscale', {})})")
    print(f"Access Mode: {config.get('data_security_mode', 'NONE')}")

    # Check if cluster already exists
    result = subprocess.run(
        ["databricks", "clusters", "list", "--output", "JSON"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        clusters = json.loads(result.stdout) if result.stdout else {"clusters": []}
        existing = [c for c in clusters.get('clusters', []) if c['cluster_name'] == cluster_name]

        if existing:
            cluster_id = existing[0]['cluster_id']
            state = existing[0]['state']
            print(f"\n⚠️  Cluster '{cluster_name}' already exists")
            print(f"   Cluster ID: {cluster_id}")
            print(f"   State: {state}")

            response = input("\n   Delete and recreate? (yes/no): ")
            if response.lower() == 'yes':
                print(f"\n   Deleting cluster {cluster_id}...")
                subprocess.run(
                    ["databricks", "clusters", "delete", "--cluster-id", cluster_id],
                    capture_output=True
                )
                print("   Waiting for deletion...")
                time.sleep(10)
            else:
                print("   Skipping cluster creation")
                return cluster_id

    # Create cluster
    print(f"\nCreating cluster...")

    # Write config to temp file
    temp_config = Path("/tmp/databricks_cluster_config.json")
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)

    result = subprocess.run(
        ["databricks", "clusters", "create", "--json-file", str(temp_config)],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        response = json.loads(result.stdout)
        cluster_id = response.get('cluster_id')
        print(f"\n✅ Cluster created successfully!")
        print(f"   Cluster ID: {cluster_id}")
        print(f"   Name: {cluster_name}")

        # Wait for cluster to start
        print(f"\n   Waiting for cluster to start...")
        print(f"   (This may take 3-5 minutes)")

        max_wait = 300  # 5 minutes
        waited = 0
        while waited < max_wait:
            result = subprocess.run(
                ["databricks", "clusters", "get", "--cluster-id", cluster_id],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                cluster_info = json.loads(result.stdout)
                state = cluster_info.get('state')
                print(f"   Status: {state}")

                if state == 'RUNNING':
                    print(f"\n   ✅ Cluster is running!")
                    break
                elif state in ['ERROR', 'TERMINATED']:
                    print(f"\n   ❌ Cluster failed to start (state: {state})")
                    if 'state_message' in cluster_info:
                        print(f"   Error: {cluster_info['state_message']}")
                    break

            time.sleep(10)
            waited += 10

        return cluster_id
    else:
        print(f"\n❌ Failed to create cluster")
        print(f"   Error: {result.stderr}")
        return None


def test_unity_catalog_cluster(cluster_id):
    """Test Unity Catalog connectivity on the cluster"""
    print(f"\n{'='*80}")
    print("Testing Unity Catalog Connectivity")
    print(f"{'='*80}")

    # Create a test notebook command
    test_commands = [
        "spark.sql('USE CATALOG commodity')",
        "spark.sql('SELECT current_catalog(), current_schema()').show()",
        "spark.sql('SELECT COUNT(*) FROM commodity.bronze.weather').show()"
    ]

    print("\nTest Commands to Run Manually:")
    print("-" * 80)
    for cmd in test_commands:
        print(f"  {cmd}")
    print("-" * 80)
    print(f"\nAttach a notebook to cluster ID: {cluster_id}")
    print("Run the commands above to verify Unity Catalog access")


def main():
    parser = argparse.ArgumentParser(
        description='Create Databricks clusters for Unity Catalog and S3 ingestion'
    )
    parser.add_argument(
        '--cluster',
        choices=['unity', 's3-ingestion', 'both'],
        default='both',
        help='Which cluster to create (default: both)'
    )

    args = parser.parse_args()

    print("="*80)
    print("Databricks Cluster Creation Script")
    print("="*80)

    # Check prerequisites
    if not check_databricks_cli():
        print("\n❌ Prerequisites not met. Exiting.")
        sys.exit(1)

    # Create clusters
    created_clusters = []

    if args.cluster in ['unity', 'both']:
        cluster_id = create_cluster(UNITY_CATALOG_CONFIG, "Unity Catalog")
        if cluster_id:
            created_clusters.append(("Unity Catalog", cluster_id))
            test_unity_catalog_cluster(cluster_id)

    if args.cluster in ['s3-ingestion', 'both']:
        cluster_id = create_cluster(S3_INGESTION_CONFIG, "S3 Ingestion")
        if cluster_id:
            created_clusters.append(("S3 Ingestion", cluster_id))

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")

    if created_clusters:
        print(f"\n✅ Created {len(created_clusters)} cluster(s):")
        for name, cluster_id in created_clusters:
            print(f"   - {name}: {cluster_id}")

        print(f"\n📝 Next Steps:")
        print(f"   1. Run Unity Catalog storage setup:")
        print(f"      - Open Databricks SQL Editor")
        print(f"      - Run: infra/databricks_unity_catalog_storage_setup.sql")
        print(f"   2. Test Unity Catalog cluster:")
        print(f"      - Attach notebook to 'unity-catalog-cluster'")
        print(f"      - Run: spark.sql('USE CATALOG commodity')")
        print(f"      - Run: spark.sql('SELECT * FROM commodity.bronze.weather LIMIT 5').show()")
        print(f"   3. Use correct cluster for each task:")
        print(f"      - Unity Catalog queries → unity-catalog-cluster")
        print(f"      - Auto Loader jobs → s3-ingestion-cluster")
    else:
        print(f"\n⚠️  No clusters created")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()

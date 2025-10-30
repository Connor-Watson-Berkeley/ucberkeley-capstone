#!/usr/bin/env python3
"""
Upload the ETL setup notebook to Databricks workspace
"""
import base64
import json
import requests
import sys
import os

# Databricks configuration
DATABRICKS_HOST = "https://dbc-fd7b00f3-7a6d.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")  # Set via environment variable or read from infra/.databrickscfg
CLUSTER_ID = "1030-040527-3do4v2at"  # general-purpose-mid-compute

def upload_notebook(notebook_path, workspace_path):
    """Upload notebook to Databricks workspace"""

    # Read and encode notebook
    with open(notebook_path, 'rb') as f:
        notebook_content = f.read()

    encoded_content = base64.b64encode(notebook_content).decode('utf-8')

    # API request
    url = f"{DATABRICKS_HOST}/api/2.0/workspace/import"
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "path": workspace_path,
        "format": "SOURCE",
        "language": "PYTHON",
        "content": encoded_content,
        "overwrite": True
    }

    print(f"Uploading notebook to: {workspace_path}")
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        print(f"✓ Notebook uploaded successfully!")
        return True
    else:
        print(f"✗ Failed to upload notebook")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def start_cluster(cluster_id):
    """Start a Databricks cluster"""

    url = f"{DATABRICKS_HOST}/api/2.0/clusters/start"
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {"cluster_id": cluster_id}

    print(f"\nStarting cluster: {cluster_id}")
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        print(f"✓ Cluster start initiated")
        return True
    else:
        print(f"✗ Failed to start cluster")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def run_notebook(cluster_id, notebook_path):
    """Run a notebook on a cluster"""

    url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/submit"
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "run_name": "Commodity ETL Setup - Bronze Layer",
        "existing_cluster_id": cluster_id,
        "notebook_task": {
            "notebook_path": notebook_path,
            "source": "WORKSPACE"
        },
        "timeout_seconds": 3600
    }

    print(f"\nSubmitting notebook run...")
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        run_id = result.get('run_id')
        print(f"✓ Notebook run submitted!")
        print(f"  Run ID: {run_id}")
        print(f"  View at: {DATABRICKS_HOST}/#job/runs/{run_id}")
        return run_id
    else:
        print(f"✗ Failed to submit notebook run")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def check_run_status(run_id):
    """Check the status of a notebook run"""

    url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get"
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}"
    }

    params = {"run_id": run_id}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        result = response.json()
        state = result.get('state', {})
        life_cycle_state = state.get('life_cycle_state')
        result_state = state.get('result_state')

        print(f"\nRun Status:")
        print(f"  Life Cycle State: {life_cycle_state}")
        if result_state:
            print(f"  Result State: {result_state}")

        return result
    else:
        print(f"✗ Failed to get run status")
        print(f"Response: {response.text}")
        return None

if __name__ == "__main__":

    print("="*60)
    print("Databricks ETL Setup - Deployment")
    print("="*60)

    # Paths
    notebook_path = "/Users/connorwatson/Documents/Data Science/DS210/ucberkeley-capstone/lambda_migration/setup_bronze_layer.py"
    workspace_path = "/Workspace/Users/ground.truth.datascience@gmail.com/setup_bronze_layer"

    # Step 1: Upload notebook
    print("\n[1/3] Uploading notebook to Databricks...")
    if not upload_notebook(notebook_path, workspace_path):
        sys.exit(1)

    # Step 2: Start cluster
    print("\n[2/3] Starting cluster...")
    if not start_cluster(CLUSTER_ID):
        print("Note: Cluster might already be starting/running")

    # Step 3: Run notebook
    print("\n[3/3] Running ETL setup notebook...")
    run_id = run_notebook(CLUSTER_ID, workspace_path)

    if run_id:
        print("\n" + "="*60)
        print("Deployment Initiated Successfully!")
        print("="*60)
        print(f"\nMonitor the run at:")
        print(f"  {DATABRICKS_HOST}/#job/runs/{run_id}")
        print(f"\nThe notebook will:")
        print(f"  1. Create Unity Catalog structure (commodity.bronze, commodity.silver, commodity.landing)")
        print(f"  2. Set up Auto Loader streaming jobs for 5 data sources")
        print(f"  3. Create bronze views with deduplication")
        print(f"  4. Create GDELT bronze table")
        print(f"  5. Run verification queries")
        print(f"\nEstimated time: 5-10 minutes")
        print(f"\nTo check status, run:")
        print(f"  python3 -c 'from upload_notebook_to_databricks import check_run_status; check_run_status({run_id})'")
    else:
        print("\n✗ Failed to start notebook run")
        sys.exit(1)

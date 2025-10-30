#!/usr/bin/env python3
"""
Monitor Databricks ETL setup notebook run
"""
import requests
import time
import sys
import os

DATABRICKS_HOST = "https://dbc-fd7b00f3-7a6d.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")  # Set via environment variable or .databrickscfg

def get_run_status(run_id):
    """Get detailed status of a notebook run"""

    url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get"
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}
    params = {"run_id": run_id}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"✗ Failed to get run status: {response.text}")
        return None

def print_run_status(run_data):
    """Print formatted run status"""

    if not run_data:
        return

    state = run_data.get('state', {})
    life_cycle_state = state.get('life_cycle_state')
    result_state = state.get('result_state')
    state_message = state.get('state_message', '')

    print("\n" + "="*60)
    print(f"Run ID: {run_data.get('run_id')}")
    print(f"Run Name: {run_data.get('run_name')}")
    print("="*60)

    print(f"\nStatus: {life_cycle_state}")

    if result_state:
        print(f"Result: {result_state}")

    if state_message:
        print(f"Message: {state_message}")

    # Show start/end times if available
    if 'start_time' in run_data:
        start_time = run_data['start_time'] / 1000
        print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    if 'end_time' in run_data:
        end_time = run_data['end_time'] / 1000
        print(f"Ended: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

        # Calculate duration
        if 'start_time' in run_data:
            duration_seconds = (run_data['end_time'] - run_data['start_time']) / 1000
            duration_minutes = duration_seconds / 60
            print(f"Duration: {duration_minutes:.2f} minutes")

    # Show cluster info
    cluster_instance = run_data.get('cluster_instance', {})
    if cluster_instance.get('cluster_id'):
        print(f"\nCluster: {cluster_instance['cluster_id']}")

    # Show task info
    tasks = run_data.get('tasks', [])
    if tasks:
        print(f"\nTasks:")
        for task in tasks:
            task_state = task.get('state', {})
            task_key = task.get('task_key', 'unknown')
            task_life_cycle = task_state.get('life_cycle_state', 'unknown')
            task_result = task_state.get('result_state', '')
            print(f"  - {task_key}: {task_life_cycle} {task_result}")

    # Show output URL
    run_id = run_data.get('run_id')
    print(f"\nView full output: {DATABRICKS_HOST}/#job/runs/{run_id}")

    return life_cycle_state, result_state

def monitor_run(run_id, poll_interval=30, max_wait_minutes=15):
    """Monitor a run until completion or timeout"""

    print(f"Monitoring run {run_id}...")
    print(f"Poll interval: {poll_interval}s")
    print(f"Max wait time: {max_wait_minutes} minutes")

    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60

    while True:
        run_data = get_run_status(run_id)
        life_cycle_state, result_state = print_run_status(run_data)

        # Check if completed
        if life_cycle_state in ['TERMINATED', 'INTERNAL_ERROR']:
            if result_state == 'SUCCESS':
                print("\n✓ Run completed successfully!")
                return True
            else:
                print(f"\n✗ Run failed with state: {result_state}")
                return False

        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            print(f"\n⚠ Timeout reached ({max_wait_minutes} minutes)")
            print("Run is still in progress. Check the Databricks UI for updates.")
            return None

        # Wait before next check
        print(f"\nWaiting {poll_interval}s before next check...")
        time.sleep(poll_interval)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python3 monitor_databricks_run.py <run_id>")
        print("\nDefault run ID (from latest deployment): 434014329903945")
        run_id = 434014329903945
    else:
        run_id = int(sys.argv[1])

    print("="*60)
    print("Databricks Run Monitor")
    print("="*60)

    success = monitor_run(run_id, poll_interval=30, max_wait_minutes=15)

    if success:
        print("\n" + "="*60)
        print("ETL Setup Complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Verify bronze tables were created")
        print("2. Check row counts in bronze views")
        print("3. Set up daily refresh job")
        print("4. Deploy Lambda functions for daily updates")
    elif success is False:
        print("\n" + "="*60)
        print("Setup Failed")
        print("="*60)
        print("\nTroubleshooting:")
        print("1. Check the Databricks UI for error details")
        print("2. Verify S3 bucket access (groundtruth-capstone)")
        print("3. Ensure cluster has S3 instance profile")
        print("4. Review notebook logs in Databricks")
    else:
        print("\n" + "="*60)
        print("Monitoring Timeout")
        print("="*60)
        print("\nThe run is still in progress.")
        print("Check status later with:")
        print(f"  python3 monitor_databricks_run.py {run_id}")

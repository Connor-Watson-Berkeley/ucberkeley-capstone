"""Autonomous LSTM Experiment Runner for Databricks.

Uploads notebook and runs on ML cluster with live monitoring.
Based on patterns from docs/DATABRICKS_API_GUIDE.md
"""

import urllib.request
import urllib.error
import json
import os
import base64
import time
from datetime import datetime


class DatabricksLSTMRunner:
    """Autonomous runner for LSTM experiment on Databricks ML cluster."""

    def __init__(self, host, token):
        """
        Initialize runner with Databricks credentials.

        Args:
            host: Databricks workspace URL
            token: Personal access token
        """
        self.host = host.rstrip('/')
        self.token = token
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

    def find_ml_cluster(self):
        """
        Find ML cluster ID (searches for ML Runtime clusters).

        Returns:
            Cluster ID if found, None otherwise
        """
        url = f"{self.host}/api/2.0/clusters/list"
        req = urllib.request.Request(url, headers=self.headers)

        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                clusters = result.get('clusters', [])

                # Look for ML Runtime cluster (by name pattern)
                for cluster in clusters:
                    state = cluster.get('state')
                    cluster_name = cluster.get('cluster_name', '')

                    if 'ML Runtime' in cluster_name and state == 'RUNNING':
                        print(f"  Found ML cluster: {cluster_name} ({cluster['cluster_id']})")
                        return cluster['cluster_id']

                # Fallback: any running cluster
                for cluster in clusters:
                    if cluster.get('state') == 'RUNNING':
                        cluster_name = cluster.get('cluster_name', '')
                        print(f"  Found running cluster: {cluster_name} ({cluster['cluster_id']})")
                        return cluster['cluster_id']

                # If no running clusters, return first available
                if clusters:
                    cluster = clusters[0]
                    print(f"  Found cluster: {cluster['cluster_name']} ({cluster['cluster_id']}) [state: {cluster.get('state')}]")
                    return cluster['cluster_id']

                return None

        except urllib.error.HTTPError as e:
            print(f"  Error listing clusters: {e}")
            return None

    def upload_notebook(self, local_path, workspace_path):
        """
        Upload notebook to Databricks workspace.

        Args:
            local_path: Path to local notebook file
            workspace_path: Destination path in workspace

        Returns:
            True if successful, False otherwise
        """
        with open(local_path, 'r') as f:
            content = f.read()

        encoded_content = base64.b64encode(content.encode()).decode()

        data = {
            "path": workspace_path,
            "content": encoded_content,
            "language": "PYTHON",
            "overwrite": True,
            "format": "SOURCE"
        }

        # NOTE: Workspace import uses API v2.0 (NOT v2.1!)
        url = f"{self.host}/api/2.0/workspace/import"
        req = urllib.request.Request(url,
                                      data=json.dumps(data).encode(),
                                      headers=self.headers)

        try:
            with urllib.request.urlopen(req) as response:
                print(f"  ✓ Uploaded to {workspace_path}")
                return True
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            print(f"  ✗ Upload failed: {e.code} - {error_body}")
            return False

    def submit_job(self, notebook_path, cluster_id, job_name):
        """
        Submit notebook job to cluster.

        Args:
            notebook_path: Workspace path to notebook
            cluster_id: Cluster ID to run on
            job_name: Name for the job run

        Returns:
            Run ID if successful, None otherwise
        """
        job_config = {
            "run_name": job_name,
            "tasks": [{
                "task_key": "lstm_experiment",
                "notebook_task": {
                    "notebook_path": notebook_path,
                    "source": "WORKSPACE"
                },
                "existing_cluster_id": cluster_id,
                "timeout_seconds": 7200,  # 2 hour timeout
                "email_notifications": {}
            }]
        }

        url = f"{self.host}/api/2.1/jobs/runs/submit"
        req = urllib.request.Request(url,
                                      data=json.dumps(job_config).encode(),
                                      headers=self.headers)

        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                run_id = result['run_id']
                print(f"  ✓ Submitted job (run_id: {run_id})")
                return run_id
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            print(f"  ✗ Job submission failed: {e.code} - {error_body}")
            return None

    def monitor_job(self, run_id, poll_interval=30):
        """
        Monitor job execution with live status updates.

        Args:
            run_id: Run ID to monitor
            poll_interval: Seconds between status checks

        Returns:
            Final job status dict
        """
        print()
        print(f"Monitoring run {run_id} (polling every {poll_interval}s)...")
        print(f"View in Databricks: {self.host}/#job/{run_id}")
        print()

        while True:
            url = f"{self.host}/api/2.1/jobs/runs/get?run_id={run_id}"
            req = urllib.request.Request(url, headers=self.headers)

            try:
                with urllib.request.urlopen(req) as response:
                    status = json.loads(response.read().decode())
                    state = status.get('state', {})
                    life_cycle = state.get('life_cycle_state')
                    result_state = state.get('result_state')
                    state_message = state.get('state_message', '')

                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{timestamp}] Status: {life_cycle} / {result_state or 'N/A'}")

                    if state_message:
                        print(f"  Message: {state_message}")

                    # Check for terminal states
                    if life_cycle == 'TERMINATED':
                        print()
                        if result_state == 'SUCCESS':
                            print("✓ Job completed successfully!")
                            self._print_output(run_id)
                            return status
                        else:
                            print(f"✗ Job failed with state: {result_state}")
                            self._print_error_logs(run_id)
                            return status

                    elif life_cycle in ['INTERNAL_ERROR', 'SKIPPED']:
                        print()
                        print(f"✗ Job error: {life_cycle}")
                        self._print_error_logs(run_id)
                        return status

                    # Still running
                    time.sleep(poll_interval)

            except urllib.error.HTTPError as e:
                print(f"Error checking status: {e}")
                time.sleep(poll_interval)

    def _print_output(self, parent_run_id):
        """
        Retrieve and print job output.

        Args:
            parent_run_id: Parent run ID
        """
        # Get task details first
        url = f"{self.host}/api/2.1/jobs/runs/get?run_id={parent_run_id}"
        req = urllib.request.Request(url, headers=self.headers)

        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                tasks = result.get('tasks', [])

                if not tasks:
                    print("\nNo task output available")
                    return

                for task in tasks:
                    task_run_id = task.get('run_id')
                    task_key = task.get('task_key')

                    # Get task output (NOTE: use task run ID, not parent run ID!)
                    output_url = f"{self.host}/api/2.1/jobs/runs/get-output?run_id={task_run_id}"
                    output_req = urllib.request.Request(output_url, headers=self.headers)

                    with urllib.request.urlopen(output_req) as output_response:
                        output_result = json.loads(output_response.read().decode())

                        print(f"\nOutput from task: {task_key}")
                        print("="*80)

                        if 'notebook_output' in output_result:
                            notebook_output = output_result['notebook_output']
                            if 'result' in notebook_output:
                                print(notebook_output['result'])

        except Exception as e:
            print(f"\nCould not fetch output: {e}")

    def _print_error_logs(self, parent_run_id):
        """
        Retrieve and print error logs from job.

        Args:
            parent_run_id: Parent run ID
        """
        # Get task details
        url = f"{self.host}/api/2.1/jobs/runs/get?run_id={parent_run_id}"
        req = urllib.request.Request(url, headers=self.headers)

        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                tasks = result.get('tasks', [])

                for task in tasks:
                    task_run_id = task.get('run_id')
                    task_key = task.get('task_key')

                    # Get task output (NOTE: use task run ID, not parent run ID!)
                    output_url = f"{self.host}/api/2.1/jobs/runs/get-output?run_id={task_run_id}"
                    output_req = urllib.request.Request(output_url, headers=self.headers)

                    try:
                        with urllib.request.urlopen(output_req) as output_response:
                            output_result = json.loads(output_response.read().decode())

                            if 'error' in output_result:
                                print(f"\nError from task: {task_key}")
                                print("="*80)
                                print(output_result['error'])
                            elif 'error_trace' in output_result:
                                print(f"\nError trace from task: {task_key}")
                                print("="*80)
                                print(output_result['error_trace'])

                    except Exception as e:
                        print(f"Could not fetch task output: {e}")

        except Exception as e:
            print(f"Could not fetch error logs: {e}")


def main():
    """Run LSTM experiment on Databricks."""
    print("="*80)
    print("LSTM Experiment - Databricks Autonomous Runner")
    print("="*80)
    print()

    # Step 1: Load credentials from environment
    print("Step 1: Loading credentials...")
    host = os.environ.get('DATABRICKS_HOST')
    token = os.environ.get('DATABRICKS_TOKEN')

    if not host:
        raise ValueError("DATABRICKS_HOST environment variable not set")
    if not token:
        raise ValueError("DATABRICKS_TOKEN environment variable not set")

    print(f"  Host: {host}")
    print()

    # Step 2: Initialize runner
    print("Step 2: Initializing runner...")
    runner = DatabricksLSTMRunner(host, token)
    print("  ✓ Runner initialized")
    print()

    # Step 3: Use LSTM Experiment cluster
    print("Step 3: Using LSTM Experiment cluster...")
    cluster_id = '1122-033549-98odeyoi'  # LSTM Experiment Cluster with Unity Catalog
    print(f"  Using cluster: {cluster_id}")
    print()

    # Step 4: Upload notebook
    print("Step 4: Uploading LSTM experiment notebook...")
    local_path = 'databricks_lstm_experiment.py'
    workspace_path = '/Shared/lstm_experiment'

    success = runner.upload_notebook(local_path, workspace_path)

    if not success:
        print("  ✗ Upload failed, aborting")
        return

    print()

    # Step 5: Submit job
    print("Step 5: Submitting job to ML cluster...")
    job_name = f"LSTM_Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_id = runner.submit_job(workspace_path, cluster_id, job_name)

    if not run_id:
        print("  ✗ Job submission failed")
        return

    print()

    # Step 6: Monitor execution
    print("Step 6: Monitoring job execution...")
    runner.monitor_job(run_id, poll_interval=30)

    print()
    print("="*80)
    print("LSTM Experiment Complete!")
    print("="*80)


if __name__ == '__main__':
    main()

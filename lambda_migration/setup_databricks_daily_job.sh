#!/bin/bash
# Setup Daily Databricks Job to Refresh Bronze Tables
# This job runs daily to pick up new CSV files from Lambda functions

set -e

# Configuration
DATABRICKS_HOST="https://dbc-fd7b00f3-7a6d.cloud.databricks.com"
DATABRICKS_TOKEN="${DATABRICKS_TOKEN:-}"  # Set via environment variable (export DATABRICKS_TOKEN=your_token)
CLUSTER_ID="<YOUR_CLUSTER_ID>"  # Replace with general-purpose-mid-compute cluster ID
NOTEBOOK_PATH="/Workspace/Repos/<YOUR_USERNAME>/ucberkeley-capstone/lambda_migration/databricks_etl_setup"

echo "======================================"
echo "Databricks Daily Refresh Job Setup"
echo "======================================"
echo ""

# Create job using Databricks REST API
JOB_CONFIG=$(cat << EOF
{
  "name": "Commodity Data - Daily Bronze Refresh",
  "description": "Daily job to ingest new CSV files from Lambda functions into bronze layer",
  "timeout_seconds": 3600,
  "max_concurrent_runs": 1,
  "tasks": [
    {
      "task_key": "refresh_bronze_tables",
      "description": "Run Auto Loader to ingest new files from S3",
      "existing_cluster_id": "$CLUSTER_ID",
      "notebook_task": {
        "notebook_path": "$NOTEBOOK_PATH",
        "source": "GIT"
      },
      "timeout_seconds": 3600
    }
  ],
  "schedule": {
    "quartz_cron_expression": "0 0 3 * * ?",
    "timezone_id": "UTC",
    "pause_status": "UNPAUSED"
  },
  "email_notifications": {
    "on_failure": [],
    "on_success": []
  }
}
EOF
)

echo "Creating Databricks job..."
echo ""

RESPONSE=$(curl -s -X POST "${DATABRICKS_HOST}/api/2.1/jobs/create" \
  -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "$JOB_CONFIG")

JOB_ID=$(echo $RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('job_id', 'ERROR'))")

if [ "$JOB_ID" = "ERROR" ]; then
    echo "✗ Failed to create job"
    echo "Response: $RESPONSE"
    exit 1
fi

echo "✓ Job created successfully!"
echo "  Job ID: $JOB_ID"
echo "  Job Name: Commodity Data - Daily Bronze Refresh"
echo "  Schedule: Daily at 3 AM UTC (after Lambda functions run at 2 AM)"
echo ""
echo "View job: ${DATABRICKS_HOST}/#job/$JOB_ID"
echo ""

# Test run the job immediately
echo "Running test execution..."
RUN_RESPONSE=$(curl -s -X POST "${DATABRICKS_HOST}/api/2.1/jobs/run-now" \
  -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"job_id\": $JOB_ID}")

RUN_ID=$(echo $RUN_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin).get('run_id', 'ERROR'))")

if [ "$RUN_ID" != "ERROR" ]; then
    echo "✓ Test run started (Run ID: $RUN_ID)"
    echo "  Monitor: ${DATABRICKS_HOST}/#job/$JOB_ID/run/$RUN_ID"
else
    echo "⚠ Could not start test run (you can manually trigger from UI)"
fi

echo ""
echo "======================================"
echo "Daily Refresh Job Setup Complete!"
echo "======================================"
echo ""
echo "The bronze tables will auto-refresh daily at 3 AM UTC"
echo "(1 hour after Lambda functions finish writing new data)"
echo ""

# Save job ID for reference
echo $JOB_ID > databricks_job_id.txt
echo "Job ID saved to: databricks_job_id.txt"

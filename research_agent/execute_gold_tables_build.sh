#!/bin/bash
# Execute gold table builds in Databricks
# Run this script after pushing to git and pulling in Databricks

set -e

echo "🔧 Building Gold Layer Tables in Databricks"
echo "============================================"

# Build production table
echo ""
echo "📊 Building commodity.gold.unified_data (PRODUCTION)..."
databricks workspace import research_agent/sql/create_gold_unified_data.sql \
  /Users/$(whoami)/gold_build_production.sql \
  --language SQL \
  --overwrite

databricks runs submit \
  --json '{
    "run_name": "Build gold.unified_data (production)",
    "existing_cluster_id": "1206-035121-fk793i8i",
    "notebook_task": {
      "notebook_path": "/Users/$(whoami)/gold_build_production.sql"
    }
  }'

echo "✅ Production table build submitted"

# Build experimental table
echo ""
echo "📊 Building commodity.gold.unified_data_no_imputation (EXPERIMENTAL)..."
databricks workspace import research_agent/sql/create_gold_unified_data_no_imputation.sql \
  /Users/$(whoami)/gold_build_experimental.sql \
  --language SQL \
  --overwrite

databricks runs submit \
  --json '{
    "run_name": "Build gold.unified_data_no_imputation (experimental)",
    "existing_cluster_id": "1206-035121-fk793i8i",
    "notebook_task": {
      "notebook_path": "/Users/$(whoami)/gold_build_experimental.sql"
    }
  }'

echo "✅ Experimental table build submitted"
echo ""
echo "Check job status in Databricks workspace"

#!/bin/bash
# Cleanup old GDELT tables in Databricks

set -e

# Read Databricks configuration
DATABRICKS_HOST=$(grep -A 2 "\[DEFAULT\]" ~/.databrickscfg | grep "^host" | cut -d'=' -f2 | tr -d ' ')
DATABRICKS_TOKEN=$(grep -A 2 "\[DEFAULT\]" ~/.databrickscfg | grep "^token" | cut -d'=' -f2 | tr -d ' ')

# Get warehouse ID
WAREHOUSE_ID=$(curl -s -X GET \
    -H "Authorization: Bearer $DATABRICKS_TOKEN" \
    "$DATABRICKS_HOST/api/2.0/sql/warehouses" | python3 -c "import sys, json; print(json.load(sys.stdin)['warehouses'][0]['id'])" 2>/dev/null)

echo "=========================================="
echo "Cleaning up old GDELT tables"
echo "=========================================="
echo ""
echo "This will remove:"
echo "  - commodity.landing.gdelt_sentiment_inc (old JSONL-based table)"
echo "  - commodity.bronze.gdelt (old bronze table)"
echo ""
echo "New tables (will be kept):"
echo "  - commodity.bronze.gdelt_bronze (new Parquet external table)"
echo "  - commodity.silver.gdelt_wide (current)"
echo ""

# Function to execute SQL
execute_sql() {
    local sql_statement=$1
    local description=$2

    echo "$description..."

    PAYLOAD=$(cat <<EOF
{
    "warehouse_id": "$WAREHOUSE_ID",
    "statement": $(echo "$sql_statement" | python3 -c "import sys, json; print(json.dumps(sys.stdin.read()))")
}
EOF
)

    RESPONSE=$(curl -s -X POST \
        -H "Authorization: Bearer $DATABRICKS_TOKEN" \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD" \
        "$DATABRICKS_HOST/api/2.0/sql/statements/")

    STATEMENT_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('statement_id', ''))" 2>/dev/null)

    if [ -z "$STATEMENT_ID" ]; then
        echo "✗ Failed to execute: $RESPONSE"
        return 1
    fi

    # Poll for completion
    for i in {1..10}; do
        sleep 1
        STATUS_RESPONSE=$(curl -s -X GET \
            -H "Authorization: Bearer $DATABRICKS_TOKEN" \
            "$DATABRICKS_HOST/api/2.0/sql/statements/$STATEMENT_ID")

        STATE=$(echo "$STATUS_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', {}).get('state', ''))" 2>/dev/null)

        if [ "$STATE" = "SUCCEEDED" ]; then
            echo "✓ $description completed"
            return 0
        elif [ "$STATE" = "FAILED" ]; then
            ERROR=$(echo "$STATUS_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', {}).get('error', {}).get('message', 'Unknown error'))" 2>/dev/null)
            echo "✗ Failed: $ERROR"
            return 1
        fi
    done

    echo "✗ Timeout"
    return 1
}

# Drop old landing table
execute_sql "DROP TABLE IF EXISTS commodity.landing.gdelt_sentiment_inc" "Dropping commodity.landing.gdelt_sentiment_inc"
echo ""

# Drop old bronze table
execute_sql "DROP TABLE IF EXISTS commodity.bronze.gdelt" "Dropping commodity.bronze.gdelt"
echo ""

echo "=========================================="
echo "✓ Cleanup Complete"
echo "=========================================="
echo ""
echo "Remaining GDELT tables:"
echo "  - commodity.bronze.gdelt_bronze (active - Parquet external table)"
echo "  - commodity.silver.gdelt_wide (active - wide-format aggregations)"

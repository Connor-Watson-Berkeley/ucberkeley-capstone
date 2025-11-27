#!/bin/bash
#
# Upload refined_approach to Databricks DBFS
# This bypasses workspace/Unity Catalog AWS validation issues
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Uploading refined_approach to DBFS...${NC}\n"

LOCAL_PATH="forecast_agent/refined_approach"
DBFS_PATH="dbfs:/FileStore/forecast_agent/refined_approach"

# Check authentication
if ! databricks fs ls dbfs:/ &> /dev/null; then
    echo -e "${RED}❌ Authentication failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Authentication OK${NC}\n"

# Create directory structure
echo "Creating directory structure..."
databricks fs mkdirs "$DBFS_PATH/notebooks" || true
databricks fs mkdirs "$DBFS_PATH/docs" || true

# Upload Python files
echo "Uploading Python modules..."
for file in "$LOCAL_PATH"/*.py; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "  📄 $filename"
        databricks fs cp "$file" "$DBFS_PATH/$filename" --overwrite
    fi
done

# Upload notebooks
echo ""
echo "Uploading notebooks..."
for file in "$LOCAL_PATH/notebooks"/*.py; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "  📓 $filename"
        databricks fs cp "$file" "$DBFS_PATH/notebooks/$filename" --overwrite
    fi
done

# Upload markdown files
echo ""
echo "Uploading documentation..."
for file in "$LOCAL_PATH"/*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "  📝 $filename"
        databricks fs cp "$file" "$DBFS_PATH/$filename" --overwrite
    fi
done

# Upload docs
echo ""
echo "Uploading docs folder..."
for file in "$LOCAL_PATH/docs"/*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "  📚 docs/$filename"
        databricks fs cp "$file" "$DBFS_PATH/docs/$filename" --overwrite
    fi
done

echo ""
echo -e "${GREEN}✅ Upload complete!${NC}"
echo ""
echo "Files are now available at:"
echo "  $DBFS_PATH"
echo ""
echo "In notebooks, add to path:"
echo "  import sys; sys.path.insert(0, '/dbfs/FileStore/forecast_agent/refined_approach')"


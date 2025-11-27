#!/bin/bash
#
# Upload refined_approach to Databricks Repos
# 
# Prerequisites:
#   1. Databricks CLI installed and configured
#   2. Token refreshed (run: databricks configure --token)
#   3. Repo exists at /Repos/Project_Git/ucberkeley-capstone
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Uploading refined_approach to Databricks Repos...${NC}\n"

# Check if CLI is installed
if ! command -v databricks &> /dev/null; then
    echo -e "${RED}❌ Databricks CLI not found. Install it first:${NC}"
    echo "   pip install databricks-cli"
    exit 1
fi

# Check authentication
echo "Checking authentication..."
if ! databricks workspace ls /Repos &> /dev/null; then
    echo -e "${YELLOW}⚠️  Authentication failed. Refreshing token...${NC}"
    echo ""
    echo "Please configure your Databricks CLI:"
    echo "  databricks configure --token"
    echo ""
    echo "Enter your workspace URL and a new token from:"
    echo "  Databricks → Settings → User Settings → Access Tokens"
    exit 1
fi

echo -e "${GREEN}✅ Authentication OK${NC}\n"

# Target paths
REPO_BASE="/Repos/Project_Git/ucberkeley-capstone"
TARGET_PATH="$REPO_BASE/forecast_agent/refined_approach"
LOCAL_PATH="forecast_agent/refined_approach"

# Check if repo exists
echo "Checking if repo exists..."
if ! databricks workspace ls "$REPO_BASE" &> /dev/null; then
    echo -e "${YELLOW}⚠️  Repo not found at $REPO_BASE${NC}"
    echo ""
    echo "Available repos:"
    databricks workspace ls /Repos/Project_Git 2>&1 || echo "  (Unable to list repos)"
    echo ""
    echo "Please create the repo in Databricks first, or adjust the REPO_BASE path above."
    exit 1
fi

echo -e "${GREEN}✅ Repo found${NC}\n"

# Create directory structure
echo "Creating directory structure..."
databricks workspace mkdirs "$TARGET_PATH/notebooks" || true
databricks workspace mkdirs "$TARGET_PATH/docs" || true
echo -e "${GREEN}✅ Directories created${NC}\n"

# Upload Python modules
echo "Uploading Python modules..."
for file in "$LOCAL_PATH"/*.py; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "  📄 Uploading $filename..."
        databricks workspace import \
            "$file" \
            "$TARGET_PATH/$filename" \
            --language PYTHON
    fi
done

# Upload notebooks
echo ""
echo "Uploading notebooks..."
for file in "$LOCAL_PATH/notebooks"/*.py; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "  📓 Uploading $filename..."
        databricks workspace import \
            "$file" \
            "$TARGET_PATH/notebooks/$filename" \
            --language PYTHON
    fi
done

# Upload markdown files
echo ""
echo "Uploading documentation..."
for file in "$LOCAL_PATH"/*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "  📝 Uploading $filename..."
        databricks workspace import \
            "$file" \
            "$TARGET_PATH/$filename" \
            --language MARKDOWN
    fi
done

# Upload docs folder
echo ""
echo "Uploading docs folder..."
for file in "$LOCAL_PATH/docs"/*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "  📚 Uploading docs/$filename..."
        databricks workspace import \
            "$file" \
            "$TARGET_PATH/docs/$filename" \
            --language MARKDOWN
    fi
done

echo ""
echo -e "${GREEN}✅ Upload complete!${NC}"
echo ""
echo "Files are now available at:"
echo "  $TARGET_PATH"
echo ""
echo "Next steps:"
echo "  1. Open Databricks workspace"
echo "  2. Navigate to: Repos → Project_Git → ucberkeley-capstone → forecast_agent → refined_approach"
echo "  3. Open notebooks/01_train_models.py"
echo "  4. Run the training notebook!"


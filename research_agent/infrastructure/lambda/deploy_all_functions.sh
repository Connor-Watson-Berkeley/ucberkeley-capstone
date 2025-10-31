#!/bin/bash
# Lambda Function Deployment Script
# Deploys all critical data fetcher functions to new AWS account

set -e  # Exit on error

# Configuration
NEW_ACCOUNT_ID="534150427458"
NEW_REGION="us-west-2"
ROLE_NAME="groundtruth-lambda-execution-role"
ROLE_ARN="arn:aws:iam::${NEW_ACCOUNT_ID}:role/${ROLE_NAME}"

echo "======================================"
echo "Lambda Function Deployment"
echo "Account: ${NEW_ACCOUNT_ID}"
echo "Region: ${NEW_REGION}"
echo "======================================"
echo ""

# Check if role exists, create if not
echo "Checking IAM role..."
if ! aws iam get-role --role-name $ROLE_NAME --region $NEW_REGION 2>/dev/null; then
    echo "Creating IAM role..."

    # Create trust policy
    cat > /tmp/lambda-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    aws iam create-role \
        --role-name $ROLE_NAME \
        --assume-role-policy-document file:///tmp/lambda-trust-policy.json \
        --region $NEW_REGION

    # Attach policies
    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole \
        --region $NEW_REGION

    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess \
        --region $NEW_REGION

    echo "Waiting 10 seconds for role to propagate..."
    sleep 10
else
    echo "IAM role already exists"
fi

echo ""

echo "======================================"
echo "Deploying: market-data-fetcher"
echo "======================================"

# Delete if exists
if aws lambda get-function --function-name market-data-fetcher --region $NEW_REGION 2>/dev/null; then
    echo "Function exists, deleting..."
    aws lambda delete-function --function-name market-data-fetcher --region $NEW_REGION
    sleep 2
fi

# Create function
aws lambda create-function \
    --function-name market-data-fetcher \
    --runtime python3.12 \
    --handler app.lambda_handler \
    --role $ROLE_ARN \
    --zip-file fileb://updated/market-data-fetcher.zip \
    --memory-size 128 \
    --timeout 60 \
    --region $NEW_REGION \
    --architectures arm64

echo "Waiting for function to be active..."
aws lambda wait function-active --function-name market-data-fetcher --region $NEW_REGION

echo "✓ market-data-fetcher deployed successfully"
echo ""

echo "======================================"
echo "Deploying: weather-data-fetcher"
echo "======================================"

# Delete if exists
if aws lambda get-function --function-name weather-data-fetcher --region $NEW_REGION 2>/dev/null; then
    echo "Function exists, deleting..."
    aws lambda delete-function --function-name weather-data-fetcher --region $NEW_REGION
    sleep 2
fi

# Create function
aws lambda create-function \
    --function-name weather-data-fetcher \
    --runtime python3.12 \
    --handler app.lambda_handler \
    --role $ROLE_ARN \
    --zip-file fileb://updated/weather-data-fetcher.zip \
    --memory-size 512 \
    --timeout 60 \
    --region $NEW_REGION \
    --architectures arm64

echo "Waiting for function to be active..."
aws lambda wait function-active --function-name weather-data-fetcher --region $NEW_REGION


# Set environment variables
aws lambda update-function-configuration \
    --function-name weather-data-fetcher \
    --environment "Variables={OPENWEATHER_API_KEY=c7d0e1449305a2f2b1da6eacdd6d4607,S3_BUCKET_NAME=groundtruth-capstone}" \
    --region $NEW_REGION

echo "Waiting for update to complete..."
aws lambda wait function-updated --function-name weather-data-fetcher --region $NEW_REGION

echo "✓ weather-data-fetcher deployed successfully"
echo ""

echo "======================================"
echo "Deploying: vix-data-fetcher"
echo "======================================"

# Delete if exists
if aws lambda get-function --function-name vix-data-fetcher --region $NEW_REGION 2>/dev/null; then
    echo "Function exists, deleting..."
    aws lambda delete-function --function-name vix-data-fetcher --region $NEW_REGION
    sleep 2
fi

# Create function
aws lambda create-function \
    --function-name vix-data-fetcher \
    --runtime python3.12 \
    --handler app.lambda_handler \
    --role $ROLE_ARN \
    --zip-file fileb://updated/vix-data-fetcher.zip \
    --memory-size 128 \
    --timeout 300 \
    --region $NEW_REGION \
    --architectures arm64

echo "Waiting for function to be active..."
aws lambda wait function-active --function-name vix-data-fetcher --region $NEW_REGION


# Set environment variables
aws lambda update-function-configuration \
    --function-name vix-data-fetcher \
    --environment "Variables={FRED_API_KEY=23e399e854cd920b8c34172dbb9c9f7b,S3_BUCKET_NAME=groundtruth-capstone,RUN_MODE=INCREMENTAL}" \
    --region $NEW_REGION

echo "Waiting for update to complete..."
aws lambda wait function-updated --function-name vix-data-fetcher --region $NEW_REGION

echo "✓ vix-data-fetcher deployed successfully"
echo ""

echo "======================================"
echo "Deploying: fx-calculator-fetcher"
echo "======================================"

# Delete if exists
if aws lambda get-function --function-name fx-calculator-fetcher --region $NEW_REGION 2>/dev/null; then
    echo "Function exists, deleting..."
    aws lambda delete-function --function-name fx-calculator-fetcher --region $NEW_REGION
    sleep 2
fi

# Create function
aws lambda create-function \
    --function-name fx-calculator-fetcher \
    --runtime python3.12 \
    --handler app.lambda_handler \
    --role $ROLE_ARN \
    --zip-file fileb://updated/fx-calculator-fetcher.zip \
    --memory-size 128 \
    --timeout 300 \
    --region $NEW_REGION \
    --architectures arm64

echo "Waiting for function to be active..."
aws lambda wait function-active --function-name fx-calculator-fetcher --region $NEW_REGION


# Set environment variables
aws lambda update-function-configuration \
    --function-name fx-calculator-fetcher \
    --environment "Variables={FRED_API_KEY=23e399e854cd920b8c34172dbb9c9f7b,S3_BUCKET_NAME=groundtruth-capstone,RUN_MODE=INCREMENTAL}" \
    --region $NEW_REGION

echo "Waiting for update to complete..."
aws lambda wait function-updated --function-name fx-calculator-fetcher --region $NEW_REGION

echo "✓ fx-calculator-fetcher deployed successfully"
echo ""

echo "======================================"
echo "Deploying: cftc-data-fetcher"
echo "======================================"

# Delete if exists
if aws lambda get-function --function-name cftc-data-fetcher --region $NEW_REGION 2>/dev/null; then
    echo "Function exists, deleting..."
    aws lambda delete-function --function-name cftc-data-fetcher --region $NEW_REGION
    sleep 2
fi

# Create function
aws lambda create-function \
    --function-name cftc-data-fetcher \
    --runtime python3.12 \
    --handler app.lambda_handler \
    --role $ROLE_ARN \
    --zip-file fileb://updated/cftc-data-fetcher.zip \
    --memory-size 128 \
    --timeout 300 \
    --region $NEW_REGION \
    --architectures arm64

echo "Waiting for function to be active..."
aws lambda wait function-active --function-name cftc-data-fetcher --region $NEW_REGION


# Set environment variables
aws lambda update-function-configuration \
    --function-name cftc-data-fetcher \
    --environment "Variables={S3_KEY_PREFIX=landing/cftc_data/,S3_BUCKET_NAME=groundtruth-capstone}" \
    --region $NEW_REGION

echo "Waiting for update to complete..."
aws lambda wait function-updated --function-name cftc-data-fetcher --region $NEW_REGION

echo "✓ cftc-data-fetcher deployed successfully"
echo ""

echo "======================================"
echo "Deploying: gdelt-processor"
echo "======================================"

# Delete if exists
if aws lambda get-function --function-name gdelt-processor --region $NEW_REGION 2>/dev/null; then
    echo "Function exists, deleting..."
    aws lambda delete-function --function-name gdelt-processor --region $NEW_REGION
    sleep 2
fi

# Create function
aws lambda create-function \
    --function-name gdelt-processor \
    --runtime python3.11 \
    --handler lambda_function.lambda_handler \
    --role $ROLE_ARN \
    --zip-file fileb://updated/gdelt-processor.zip \
    --memory-size 2048 \
    --timeout 900 \
    --region $NEW_REGION \
    --architectures x86_64

echo "Waiting for function to be active..."
aws lambda wait function-active --function-name gdelt-processor --region $NEW_REGION


# Set environment variables
aws lambda update-function-configuration \
    --function-name gdelt-processor \
    --environment "Variables={S3_BUCKET=groundtruth-capstone,S3_RAW_PREFIX=landing/gdelt/raw/,TRACKING_TABLE=groundtruth-capstone-file-tracking,S3_FILTERED_PREFIX=landing/gdelt/filtered/}" \
    --region $NEW_REGION

echo "Waiting for update to complete..."
aws lambda wait function-updated --function-name gdelt-processor --region $NEW_REGION

echo "✓ gdelt-processor deployed successfully"
echo ""

echo "======================================"
echo "Deployment Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Set up EventBridge schedule for gdelt-processor"
echo "2. Test each function with: aws lambda invoke --function-name <name> --region us-west-2 response.json"
echo "3. Check CloudWatch logs for any errors"

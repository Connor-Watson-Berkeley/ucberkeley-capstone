#!/usr/bin/env python3
"""
Lambda Function Migration Script
Extracts, updates, and prepares Lambda functions for deployment to new AWS account
"""

import json
import os
import zipfile
import shutil
from pathlib import Path
import re

# Migration configuration
OLD_BUCKET = "berkeley-datasci210-capstone"
NEW_BUCKET = "groundtruth-capstone"
OLD_REGION = "us-east-1"
NEW_REGION = "us-west-2"
NEW_ACCOUNT_ID = "534150427458"

# API Keys from old environment
API_KEYS = {
    "OPENWEATHER_API_KEY": "c7d0e1449305a2f2b1da6eacdd6d4607",
    "FRED_API_KEY": "23e399e854cd920b8c34172dbb9c9f7b"
}

# Priority functions to migrate
PRIORITY_FUNCTIONS = {
    'market-data-MarketDataFunction-KDrn3b6ML3vk': {
        'new_name': 'market-data-fetcher',
        'priority': 1,
        'description': 'Fetches coffee and sugar market data from yfinance'
    },
    'weather-data-WeatherFetcherFunction-R0gqBD2JibqQ': {
        'new_name': 'weather-data-fetcher',
        'priority': 1,
        'description': 'Fetches weather data for coffee/sugar regions'
    },
    'vix-data-VIXCalculatorFunction-Xhomphw9v29c': {
        'new_name': 'vix-data-fetcher',
        'priority': 1,
        'description': 'Fetches VIX volatility data from FRED'
    },
    'fx-calculator-FXCalculatorFunction-UNrpc5EgxsSn': {
        'new_name': 'fx-calculator-fetcher',
        'priority': 1,
        'description': 'Fetches exchange rate data (COP/USD) from FRED'
    },
    'cftc-data-CFTCCalculatorFunction-bDFWiwe08UoF': {
        'new_name': 'cftc-data-fetcher',
        'priority': 1,
        'description': 'Fetches CFTC commitment of traders data'
    },
    'berkeley-datasci210-capstone-processor': {
        'new_name': 'gdelt-processor',
        'priority': 2,
        'description': 'Processes GDELT news data daily'
    }
}

def extract_function_code(function_name, exports_dir, output_dir):
    """Extract and update Lambda function code"""

    zip_path = Path(exports_dir) / f"{function_name}.zip"
    if not zip_path.exists():
        print(f"  ✗ No zip file found for {function_name}")
        return None

    extract_dir = Path(output_dir) / "extracted" / function_name
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Extract zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"  ✓ Extracted to {extract_dir}")

    # Update bucket references in all Python files
    updated_files = []
    for py_file in extract_dir.rglob("*.py"):
        with open(py_file, 'r') as f:
            content = f.read()

        original_content = content

        # Replace bucket name
        content = content.replace(OLD_BUCKET, NEW_BUCKET)

        # Replace region if hardcoded
        content = content.replace(f'region_name="{OLD_REGION}"', f'region_name="{NEW_REGION}"')
        content = content.replace(f"region_name='{OLD_REGION}'", f"region_name='{NEW_REGION}'")

        if content != original_content:
            with open(py_file, 'w') as f:
                f.write(content)
            updated_files.append(str(py_file.relative_to(extract_dir)))

    if updated_files:
        print(f"  ✓ Updated {len(updated_files)} files: {', '.join(updated_files[:3])}")

    return extract_dir

def repackage_function(extract_dir, output_dir, new_name):
    """Repackage function code into new zip file"""

    zip_path = Path(output_dir) / "updated" / f"{new_name}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in extract_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(extract_dir)
                zipf.write(file_path, arcname)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Packaged to {zip_path} ({size_mb:.2f} MB)")

    return zip_path

def load_function_config(function_name, exports_dir):
    """Load function configuration from JSON"""

    config_path = Path(exports_dir) / f"{function_name}-config.json"
    env_path = Path(exports_dir) / f"{function_name}-env.json"

    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            env_vars = json.load(f)

    return config, env_vars

def generate_deployment_script(functions_info, output_dir):
    """Generate AWS CLI deployment script"""

    script_path = Path(output_dir) / "deploy_all_functions.sh"

    script = """#!/bin/bash
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

    aws iam create-role \\
        --role-name $ROLE_NAME \\
        --assume-role-policy-document file:///tmp/lambda-trust-policy.json \\
        --region $NEW_REGION

    # Attach policies
    aws iam attach-role-policy \\
        --role-name $ROLE_NAME \\
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole \\
        --region $NEW_REGION

    aws iam attach-role-policy \\
        --role-name $ROLE_NAME \\
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess \\
        --region $NEW_REGION

    echo "Waiting 10 seconds for role to propagate..."
    sleep 10
else
    echo "IAM role already exists"
fi

echo ""
"""

    for func_info in functions_info:
        func_name = func_info['new_name']
        script += f"""
echo "======================================"
echo "Deploying: {func_name}"
echo "======================================"

# Delete if exists
if aws lambda get-function --function-name {func_name} --region $NEW_REGION 2>/dev/null; then
    echo "Function exists, deleting..."
    aws lambda delete-function --function-name {func_name} --region $NEW_REGION
    sleep 2
fi

# Create function
aws lambda create-function \\
    --function-name {func_name} \\
    --runtime {func_info['runtime']} \\
    --handler {func_info['handler']} \\
    --role $ROLE_ARN \\
    --zip-file fileb://updated/{func_name}.zip \\
    --memory-size {func_info['memory']} \\
    --timeout {func_info['timeout']} \\
    --region $NEW_REGION \\
    --architectures {func_info['architecture']}

echo "Waiting for function to be active..."
aws lambda wait function-active --function-name {func_name} --region $NEW_REGION

"""

        if func_info['env_vars']:
            env_vars_str = ','.join([f'{k}={v}' for k, v in func_info['env_vars'].items()])
            script += f"""
# Set environment variables
aws lambda update-function-configuration \\
    --function-name {func_name} \\
    --environment "Variables={{{env_vars_str}}}" \\
    --region $NEW_REGION

echo "Waiting for update to complete..."
aws lambda wait function-updated --function-name {func_name} --region $NEW_REGION

"""

        script += f'echo "✓ {func_name} deployed successfully"\necho ""\n'

    script += """
echo "======================================"
echo "Deployment Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Set up EventBridge schedule for gdelt-processor"
echo "2. Test each function with: aws lambda invoke --function-name <name> --region us-west-2 response.json"
echo "3. Check CloudWatch logs for any errors"
"""

    with open(script_path, 'w') as f:
        f.write(script)

    os.chmod(script_path, 0o755)  # Make executable

    print(f"\n✓ Deployment script created: {script_path}")
    return script_path

def main():
    """Main migration workflow"""

    print("="*80)
    print("LAMBDA FUNCTION MIGRATION")
    print("="*80)
    print(f"\nOld Account: us-east-1")
    print(f"New Account: {NEW_ACCOUNT_ID} (us-west-2)")
    print(f"Old Bucket: {OLD_BUCKET}")
    print(f"New Bucket: {NEW_BUCKET}")
    print()

    base_dir = Path(__file__).parent
    exports_dir = base_dir / "lambdas" / "lambda-exports"
    output_dir = base_dir / "migrated_functions"
    output_dir.mkdir(exist_ok=True)

    functions_info = []

    # Process each priority function
    for old_name, info in PRIORITY_FUNCTIONS.items():
        print(f"\nProcessing: {old_name} → {info['new_name']}")
        print(f"  Priority: {info['priority']}")
        print(f"  Description: {info['description']}")

        # Load config
        config, env_vars = load_function_config(old_name, exports_dir)

        if not config:
            print(f"  ✗ No config found for {old_name}, skipping")
            continue

        # Extract and update code
        extract_dir = extract_function_code(old_name, exports_dir, output_dir)

        if not extract_dir:
            continue

        # Repackage
        zip_path = repackage_function(extract_dir, output_dir, info['new_name'])

        # Update environment variables
        updated_env_vars = {}
        for key, value in (env_vars or {}).items():
            if key == 'S3_BUCKET' or key == 'S3_BUCKET_NAME':
                updated_env_vars[key] = NEW_BUCKET
            elif key in API_KEYS:
                updated_env_vars[key] = API_KEYS[key]
            elif 'TRACKING_TABLE' in key:
                # Update DynamoDB table name if needed
                updated_env_vars[key] = value.replace(OLD_BUCKET, NEW_BUCKET)
            else:
                updated_env_vars[key] = value

        # Collect deployment info
        functions_info.append({
            'old_name': old_name,
            'new_name': info['new_name'],
            'runtime': config.get('Runtime', 'python3.12'),
            'handler': config.get('Handler', 'lambda_function.lambda_handler'),
            'memory': config.get('MemorySize', 512),
            'timeout': config.get('Timeout', 300),
            'architecture': config.get('Architectures', ['arm64'])[0],
            'env_vars': updated_env_vars,
            'priority': info['priority']
        })

        print(f"  ✓ Migration complete")

    # Generate deployment script
    print("\n" + "="*80)
    print("Generating deployment script...")
    print("="*80)

    script_path = generate_deployment_script(functions_info, output_dir)

    # Save migration summary
    summary_path = output_dir / "migration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'old_bucket': OLD_BUCKET,
            'new_bucket': NEW_BUCKET,
            'new_account': NEW_ACCOUNT_ID,
            'new_region': NEW_REGION,
            'functions': functions_info
        }, f, indent=2)

    print(f"\n✓ Migration summary saved: {summary_path}")

    print("\n" + "="*80)
    print("MIGRATION COMPLETE!")
    print("="*80)
    print(f"\nUpdated Lambda packages: {output_dir}/updated/")
    print(f"Deployment script: {script_path}")
    print(f"\nTo deploy all functions:")
    print(f"  cd {output_dir}")
    print(f"  ./deploy_all_functions.sh")
    print()

if __name__ == "__main__":
    main()

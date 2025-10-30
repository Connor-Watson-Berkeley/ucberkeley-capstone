#!/usr/bin/env python3
"""
Lambda Function Inventory and Analysis
Parses all Lambda config JSON files and creates comprehensive report
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_lambda_configs(exports_dir):
    """Analyze all Lambda config files"""

    configs = []
    exports_path = Path(exports_dir)

    # Find all config JSON files
    config_files = sorted(exports_path.glob("*-config.json"))

    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Extract key information
        function_info = {
            'name': config.get('FunctionName', ''),
            'runtime': config.get('Runtime', ''),
            'handler': config.get('Handler', ''),
            'memory': config.get('MemorySize', 0),
            'timeout': config.get('Timeout', 0),
            'code_size': config.get('CodeSize', 0),
            'architecture': config.get('Architectures', [''])[0],
            'has_vpc': bool(config.get('VpcConfig', {}).get('VpcId')),
            'has_env_vars': bool(config.get('Environment', {}).get('Variables')),
            'env_vars': config.get('Environment', {}).get('Variables', {}),
            'last_modified': config.get('LastModified', ''),
            'role': config.get('Role', ''),
        }

        configs.append(function_info)

    return configs

def categorize_functions(configs):
    """Categorize Lambda functions by purpose"""

    categories = {
        'critical_data_fetchers': [],
        'processors': [],
        'integrations': [],
        'test_dev': [],
        'unknown': []
    }

    for config in configs:
        name = config['name'].lower()

        # Critical data fetchers
        if any(x in name for x in ['market-data', 'vix-data', 'weather-data', 'fx-calculator', 'cftc-data']):
            categories['critical_data_fetchers'].append(config)
        # Processors
        elif any(x in name for x in ['processor', 'clean_data', 'create-table']):
            categories['processors'].append(config)
        # Integrations
        elif any(x in name for x in ['s3-trigger', 'mysql', 'connector']):
            categories['integrations'].append(config)
        # Test/Dev
        elif any(x in name for x in ['test', 'hello', 'staging', 'stock-quote', 'create_file', 'authorizer']):
            categories['test_dev'].append(config)
        else:
            categories['unknown'].append(config)

    return categories

def print_inventory_report(categories):
    """Print comprehensive inventory report"""

    print("=" * 100)
    print("LAMBDA FUNCTION INVENTORY REPORT")
    print("=" * 100)
    print()

    total_functions = sum(len(funcs) for funcs in categories.values())
    print(f"Total Functions Found: {total_functions}")
    print()

    for category, functions in categories.items():
        if not functions:
            continue

        print("-" * 100)
        print(f"{category.upper().replace('_', ' ')}: {len(functions)} functions")
        print("-" * 100)
        print()

        for func in functions:
            print(f"Name:         {func['name']}")
            print(f"Runtime:      {func['runtime']}")
            print(f"Handler:      {func['handler']}")
            print(f"Memory:       {func['memory']} MB")
            print(f"Timeout:      {func['timeout']} sec")
            print(f"Code Size:    {func['code_size'] / 1024 / 1024:.2f} MB")
            print(f"Architecture: {func['architecture']}")
            print(f"Has VPC:      {func['has_vpc']}")
            print(f"Has Env Vars: {func['has_env_vars']}")

            if func['has_env_vars']:
                print(f"Env Vars:     {', '.join(func['env_vars'].keys())}")

            print(f"Last Modified: {func['last_modified'][:10]}")
            print()

        print()

def check_for_bucket_references(exports_dir):
    """Check code for S3 bucket references"""

    print("=" * 100)
    print("S3 BUCKET REFERENCE ANALYSIS")
    print("=" * 100)
    print()

    exports_path = Path(exports_dir)
    old_bucket = "berkeley-datasci210-capstone"

    # Find all .zip files (function code)
    zip_files = sorted(exports_path.glob("*.zip"))

    functions_with_bucket_refs = []

    for zip_file in zip_files:
        # Extract function name from filename
        func_name = zip_file.stem

        # Note: We'd need to unzip and search code, but for now just list the zips
        functions_with_bucket_refs.append(func_name)

    print(f"Found {len(zip_files)} function code packages (.zip files)")
    print()
    print("Functions that may contain bucket references:")
    for func in functions_with_bucket_refs:
        print(f"  - {func}")

    print()
    print(f"Will need to search and replace: '{old_bucket}' → 'groundtruth-capstone'")
    print()

def generate_migration_recommendations(categories):
    """Generate migration priority recommendations"""

    print("=" * 100)
    print("MIGRATION RECOMMENDATIONS")
    print("=" * 100)
    print()

    print("PRIORITY 1 - CRITICAL DATA FETCHERS:")
    print("These functions fetch external data and populate S3 buckets")
    for func in categories['critical_data_fetchers']:
        print(f"  ✓ {func['name']}")
    print()

    print("PRIORITY 2 - DATA PROCESSORS:")
    print("These functions process data from S3 and create tables")
    for func in categories['processors']:
        print(f"  ✓ {func['name']}")
    print()

    print("PRIORITY 3 - INTEGRATIONS (IF NEEDED):")
    print("These may be legacy MySQL/RDS integrations")
    for func in categories['integrations']:
        print(f"  ? {func['name']}")
    print()

    print("SKIP - TEST/DEV FUNCTIONS:")
    print("These are not needed for production")
    for func in categories['test_dev']:
        print(f"  ✗ {func['name']}")
    print()

if __name__ == "__main__":
    # Run analysis
    exports_dir = "lambdas/lambda-exports"

    configs = analyze_lambda_configs(exports_dir)
    categories = categorize_functions(configs)

    # Print reports
    print_inventory_report(categories)
    check_for_bucket_references(exports_dir)
    generate_migration_recommendations(categories)

    print("=" * 100)
    print("NEXT STEPS")
    print("=" * 100)
    print()
    print("1. Review categorization and confirm which functions are actually needed")
    print("2. Examine function code to identify S3 bucket references")
    print("3. Create migration plan for critical functions only")
    print("4. Update code with new bucket names and AWS account IDs")
    print("5. Deploy to new AWS account (534150427458) in us-west-2")
    print()

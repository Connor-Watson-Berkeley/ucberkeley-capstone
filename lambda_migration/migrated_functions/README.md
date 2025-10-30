# Lambda Function Migration - Ready for Deployment

## Migration Complete

All 6 critical Lambda functions have been extracted, updated for the new AWS environment, and packaged for deployment.

### Migrated Functions

**Priority 1 - Data Fetchers** (5 functions):
1. **market-data-fetcher** (49.17 MB)
   - Fetches coffee and sugar market data from Yahoo Finance
   - No API key required
   - Writes to: `s3://groundtruth-capstone/`

2. **weather-data-fetcher** (49.18 MB)
   - Fetches weather data for coffee/sugar producing regions
   - API Key: OpenWeather (configured)
   - Writes to: `s3://groundtruth-capstone/`

3. **vix-data-fetcher** (49.18 MB)
   - Fetches VIX volatility index from FRED
   - API Key: FRED (configured)
   - Writes to: `s3://groundtruth-capstone/`

4. **fx-calculator-fetcher** (49.18 MB)
   - Fetches COP/USD exchange rate from FRED
   - API Key: FRED (configured)
   - Writes to: `s3://groundtruth-capstone/`

5. **cftc-data-fetcher** (40.33 MB)
   - Fetches CFTC commitment of traders data
   - No API key required
   - Writes to: `s3://groundtruth-capstone/landing/cftc_data/`

**Priority 2 - Data Processor** (1 function):
6. **gdelt-processor** (0.01 MB)
   - Processes GDELT news data incrementally
   - Triggered daily by EventBridge at 2 AM UTC
   - Writes to: `s3://groundtruth-capstone/landing/gdelt/`

### Changes Made

All functions have been updated with:
- **S3 Bucket**: `berkeley-datasci210-capstone` → `groundtruth-capstone`
- **Region**: `us-east-1` → `us-west-2`
- **Account ID**: 471112877453 → 534150427458
- **API Keys**: Configured from old environment

### AWS Configuration

**New Account Details**:
- Account ID: 534150427458
- Region: us-west-2
- S3 Bucket: groundtruth-capstone

**IAM Role** (will be created automatically):
- Name: `groundtruth-lambda-execution-role`
- Policies:
  - AWSLambdaBasicExecutionRole (CloudWatch Logs)
  - AmazonS3FullAccess (S3 read/write)

**API Keys Configured**:
- OPENWEATHER_API_KEY: c7d0e1449305a2f2b1da6eacdd6d4607
- FRED_API_KEY: 23e399e854cd920b8c34172dbb9c9f7b

---

## Deployment Instructions

### Prerequisites

1. AWS CLI installed and configured
2. Credentials for new AWS account (534150427458)
3. Permissions to create Lambda functions and IAM roles

### Step 1: Configure AWS CLI

```bash
aws configure
# Enter credentials for account 534150427458
# Default region: us-west-2
```

### Step 2: Deploy All Functions

```bash
cd /path/to/migrated_functions
./deploy_all_functions.sh
```

This script will:
1. Create IAM role `groundtruth-lambda-execution-role` (if needed)
2. Deploy all 6 Lambda functions to us-west-2
3. Configure environment variables with API keys
4. Wait for each function to become active

**Estimated time**: 5-10 minutes

### Step 3: Set Up EventBridge Schedule

```bash
./setup_eventbridge_schedule.sh
```

This sets up a daily schedule (2 AM UTC) for the GDELT processor.

### Step 4: Verify Deployment

Check that all functions were created:

```bash
aws lambda list-functions --region us-west-2 --query 'Functions[?starts_with(FunctionName, `market-data`) || starts_with(FunctionName, `weather-data`) || starts_with(FunctionName, `vix-data`) || starts_with(FunctionName, `fx-calculator`) || starts_with(FunctionName, `cftc-data`) || starts_with(FunctionName, `gdelt-processor`)].FunctionName'
```

Expected output:
```
[
    "market-data-fetcher",
    "weather-data-fetcher",
    "vix-data-fetcher",
    "fx-calculator-fetcher",
    "cftc-data-fetcher",
    "gdelt-processor"
]
```

---

## Testing

### Test Individual Function

```bash
# Test market data fetcher
aws lambda invoke \
    --function-name market-data-fetcher \
    --region us-west-2 \
    response.json

cat response.json
```

### Test All Functions

```bash
for func in market-data-fetcher weather-data-fetcher vix-data-fetcher fx-calculator-fetcher cftc-data-fetcher; do
    echo "Testing $func..."
    aws lambda invoke \
        --function-name $func \
        --region us-west-2\
        output-$func.json
    echo "Response:"
    cat output-$func.json
    echo -e "\n---\n"
done
```

### Test GDELT Processor

```bash
# Test with incremental mode (same as EventBridge will use)
aws lambda invoke \
    --function-name gdelt-processor \
    --payload '{"mode":"incremental"}' \
    --region us-west-2 \
    gdelt-response.json

cat gdelt-response.json
```

### Check CloudWatch Logs

```bash
# View logs for a specific function
aws logs tail /aws/lambda/market-data-fetcher --follow --region us-west-2

# View all log groups
aws logs describe-log-groups --region us-west-2 --query 'logGroups[?starts_with(logGroupName, `/aws/lambda`)].logGroupName'
```

---

## Monitoring

### CloudWatch Dashboards

After deployment, monitor function execution in CloudWatch:

1. Go to: https://console.aws.amazon.com/cloudwatch/home?region=us-west-2
2. Navigate to **Dashboards** → **Lambda**
3. Check:
   - Invocation count
   - Error rate
   - Duration
   - Throttles

### S3 Data Verification

Check that data is being written to S3:

```bash
# List recent files in S3
aws s3 ls s3://groundtruth-capstone/landing/ --recursive --region us-west-2 --human-readable | tail -20

# Check specific data folders
aws s3 ls s3://groundtruth-capstone/landing/cftc_data/ --region us-west-2
aws s3 ls s3://groundtruth-capstone/landing/gdelt/filtered/ --region us-west-2
```

---

## Troubleshooting

### Issue: Function fails with S3 access denied

**Solution**: Verify IAM role has S3FullAccess policy attached:

```bash
aws iam list-attached-role-policies \
    --role-name groundtruth-lambda-execution-role
```

### Issue: API key errors for FRED or OpenWeather

**Solution**: Verify environment variables are set correctly:

```bash
aws lambda get-function-configuration \
    --function-name weather-data-fetcher \
    --region us-west-2 \
    --query 'Environment.Variables'
```

### Issue: EventBridge not triggering GDELT processor

**Solution**: Check rule status and targets:

```bash
aws events describe-rule \
    --name groundtruth-gdelt-daily-update \
    --region us-west-2

aws events list-targets-by-rule \
    --rule groundtruth-gdelt-daily-update \
    --region us-west-2
```

---

## Rollback

If you need to delete all functions:

```bash
for func in market-data-fetcher weather-data-fetcher vix-data-fetcher fx-calculator-fetcher cftc-data-fetcher gdelt-processor; do
    aws lambda delete-function --function-name $func --region us-west-2
done

# Delete EventBridge rule
aws events remove-targets --rule groundtruth-gdelt-daily-update --ids "1" --region us-west-2
aws events delete-rule --name groundtruth-gdelt-daily-update --region us-west-2

# Delete IAM role
aws iam detach-role-policy --role-name groundtruth-lambda-execution-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam detach-role-policy --role-name groundtruth-lambda-execution-role --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam delete-role --role-name groundtruth-lambda-execution-role
```

---

## Next Steps

After successful deployment:

1. **Monitor First Run**: Watch CloudWatch logs for the first execution of each function
2. **Verify Data in S3**: Confirm data files are being created in `s3://groundtruth-capstone/`
3. **Check Databricks**: Ensure Databricks can read from the new S3 bucket
4. **Update Forecast Pipeline**: Verify the forecast agent can access the updated data
5. **Set Up Alerts**: Create CloudWatch alarms for function failures

---

## Support

For issues or questions:
- Check CloudWatch Logs first
- Review migration_summary.json for function configurations
- Verify S3 bucket permissions in Databricks
- Contact team for API key issues

---

**Migration Date**: October 2025
**Source Account**: 471112877453 (us-east-1)
**Target Account**: 534150427458 (us-west-2)

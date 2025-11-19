"""
Lambda function to incrementally convert GDELT JSONL files to Bronze Parquet format.

Reads filtered JSONL files from S3, converts to Parquet with schema transformations,
and writes to Bronze layer. Uses DynamoDB tracking to avoid reprocessing files.
"""

import json
import boto3
import pandas as pd
import awswrangler as wr
from datetime import datetime, timedelta
from typing import Dict, List
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Configuration
S3_BUCKET = 'groundtruth-capstone'
S3_JSONL_PREFIX = 'landing/gdelt/filtered/'
S3_BRONZE_PATH = 's3://groundtruth-capstone/processed/gdelt/bronze/gdelt/'
TRACKING_TABLE = 'groundtruth-capstone-bronze-tracking'


def lambda_handler(event, context):
    """
    Main Lambda handler - converts JSONL to Bronze Parquet incrementally.

    Event structure:
    {
        "mode": "incremental" | "backfill",
        "offset": 0,  # Optional - for batch processing
        "limit": 100   # Optional - files per run (default 100)
    }
    """
    logger.info(f"Event: {json.dumps(event)}")

    mode = event.get('mode', 'incremental')
    offset = event.get('offset', 0)
    limit = event.get('limit', 100)

    try:
        # List all JSONL files in S3
        logger.info(f"Listing JSONL files from s3://{S3_BUCKET}/{S3_JSONL_PREFIX}")
        jsonl_files = list_s3_files(S3_BUCKET, S3_JSONL_PREFIX)
        logger.info(f"Found {len(jsonl_files)} total JSONL files")

        # Filter to unprocessed files
        files_to_process = []
        for file_key in jsonl_files:
            file_name = file_key.split('/')[-1]
            if not is_file_processed(file_name):
                files_to_process.append({
                    'key': file_key,
                    'name': file_name
                })

        logger.info(f"Found {len(files_to_process)} unprocessed files")

        # Sort by name for consistent processing order
        files_to_process.sort(key=lambda x: x['name'])

        # Apply offset and limit for batch processing
        total_unprocessed = len(files_to_process)
        end_index = min(offset + limit, total_unprocessed)
        batch_files = files_to_process[offset:end_index]

        logger.info(f"Processing batch: files {offset} to {end_index-1} ({len(batch_files)} files)")

        # Process each file
        processed_count = 0
        total_records = 0

        for file_info in batch_files:
            try:
                records = process_jsonl_file(file_info['key'], file_info['name'])
                processed_count += 1
                total_records += records

                # Mark as processed only on success
                mark_file_processed(file_info['name'])

                if (processed_count % 10) == 0:
                    logger.info(f"Progress: {processed_count}/{len(batch_files)} files in this batch")

            except Exception as e:
                logger.error(f"Error processing {file_info['name']}: {e}", exc_info=True)
                # Don't mark as processed - will retry next run

        next_offset = end_index
        remaining_files = total_unprocessed - next_offset

        result = {
            'processed_files': processed_count,
            'total_records': total_records,
            'next_offset': next_offset,
            'remaining_files': remaining_files,
            'mode': mode
        }

        logger.info(f"Batch complete: {result}")

        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }

    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def list_s3_files(bucket: str, prefix: str) -> List[str]:
    """List all files with given prefix in S3 bucket."""
    files = []
    paginator = s3.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.jsonl'):
                    files.append(key)

    return files


def process_jsonl_file(s3_key: str, file_name: str) -> int:
    """
    Read JSONL file from S3, transform to Bronze schema, write as Parquet.
    Returns number of records processed.
    """
    logger.info(f"Processing {file_name}")

    # Read JSONL from S3
    s3_path = f"s3://{S3_BUCKET}/{s3_key}"
    df = wr.s3.read_json(path=s3_path, lines=True)

    if df.empty:
        logger.warning(f"No data in {file_name}")
        return 0

    # Transform to Bronze schema
    df_bronze = transform_to_bronze(df)

    # Write to Bronze location as Parquet
    wr.s3.to_parquet(
        df=df_bronze,
        path=S3_BRONZE_PATH,
        dataset=True,
        mode='append',
        compression='snappy'
    )

    record_count = len(df_bronze)
    logger.info(f"✓ Wrote {record_count} records to Bronze for {file_name}")

    return record_count


def transform_to_bronze(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform JSONL schema to Bronze Parquet schema.
    Matches the schema from create_bronze_table.sql
    """
    df_bronze = pd.DataFrame()

    # Parse article_date from date field (format: yyyyMMddHHmmss)
    # Convert to string first to handle mixed types
    df_bronze['article_date'] = pd.to_datetime(
        df['date'].astype(str).str[:8],
        format='%Y%m%d',
        errors='coerce'
    ).dt.date

    df_bronze['source_url'] = df['source_url'].astype(str)
    df_bronze['themes'] = df['themes'].astype(str)
    df_bronze['locations'] = df['locations'].astype(str)
    df_bronze['all_names'] = df['all_names'].astype(str)

    # Parse tone fields (comma-separated: avg,positive,negative,polarity)
    tone_split = df['tone'].astype(str).str.split(',', expand=True)
    df_bronze['tone_avg'] = pd.to_numeric(tone_split[0], errors='coerce')
    df_bronze['tone_positive'] = pd.to_numeric(tone_split[1], errors='coerce')
    df_bronze['tone_negative'] = pd.to_numeric(tone_split[2], errors='coerce')
    df_bronze['tone_polarity'] = pd.to_numeric(tone_split[3], errors='coerce')

    # Flag commodities
    all_names_lower = df['all_names'].astype(str).str.lower()

    df_bronze['has_coffee'] = all_names_lower.str.contains(
        'coffee|arabica|robusta',
        regex=True,
        na=False
    )

    # Sugar: match sugarcane/sugar cane, or 'sugar' but NOT 'sugar ray'
    df_bronze['has_sugar'] = (
        all_names_lower.str.contains('sugarcane|sugar cane', regex=True, na=False) |
        (all_names_lower.str.contains('sugar', regex=False, na=False) &
         ~all_names_lower.str.contains('sugar ray', regex=False, na=False))
    )

    # Drop rows with invalid dates
    df_bronze = df_bronze.dropna(subset=['article_date'])

    return df_bronze


def is_file_processed(file_name: str) -> bool:
    """Check if file has already been processed using DynamoDB."""
    try:
        table = dynamodb.Table(TRACKING_TABLE)
        response = table.get_item(Key={'file_name': file_name})
        return 'Item' in response
    except Exception as e:
        logger.warning(f"Error checking file status for {file_name}: {e}")
        return False


def mark_file_processed(file_name: str):
    """Mark file as processed in DynamoDB."""
    try:
        table = dynamodb.Table(TRACKING_TABLE)
        table.put_item(
            Item={
                'file_name': file_name,
                'processed_at': datetime.utcnow().isoformat(),
                'ttl': int((datetime.utcnow() + timedelta(days=90)).timestamp())
            }
        )
    except Exception as e:
        logger.error(f"Error marking file processed: {e}")


# For local testing
if __name__ == "__main__":
    test_event = {
        "mode": "incremental",
        "offset": 0,
        "limit": 10
    }

    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))

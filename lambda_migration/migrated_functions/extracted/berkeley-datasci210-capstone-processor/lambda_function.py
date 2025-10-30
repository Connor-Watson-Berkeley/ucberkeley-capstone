"""
AWS Lambda Function for GDELT GKG Data Collection
Handles both historical backfill and daily incremental updates
Optimized for commodity price prediction (coffee/sugar)
"""

import json
import boto3
import requests
import zipfile
from io import BytesIO, StringIO
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Set
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Configuration
S3_BUCKET = 'groundtruth-capstone'
S3_RAW_PREFIX = 'landing/gdelt/raw/'
S3_FILTERED_PREFIX = 'landing/gdelt/filtered/'
TRACKING_TABLE = 'gdelt-file-tracking'

# Commodity-specific filters
CORE_THEMES = {
    'AGRICULTURE', 'FOOD_STAPLE', 'FOOD_SECURITY'
}

DRIVER_THEMES = {
    'NATURAL_DISASTER', 'CLIMATE_CHANGE', 'TAX_DISEASE', 'TAX_PLANTDISEASE',
    'TAX_PESTS', 'ECON_SUBSIDIES', 'WB_2044_RURAL_WATER', 'STRIKE',
    'ECON_UNIONS', 'CRISIS_LOGISTICS', 'BLOCKADE', 'DELAY', 'CLOSURE',
    'BORDER', 'GENERAL_HEALTH', 'ECON_DEBT', 'ECON_INTEREST_RATES',
    'ECON_CURRENCY_EXCHANGE_RATE', 'ECON_STOCKMARKET', 'ECON_COST_OF_LIVING',
    'ENERGY', 'OIL', 'ECON_FREETRADE', 'ECON_TRADE_DISPUTE', 'TAX_TARIFFS',
    'LEGISLATION', 'GOV_REFORM', 'NEGOTIATIONS', 'ALLIANCE', 'CEASEFIRE',
    'STATE_OF_EMERGENCY', 'ELECTION', 'CORRUPTION', 'GENERAL_GOVERNMENT',
    'ECON_EARNINGSREPORT', 'ECON_IPO'
}

COMMODITY_KEYWORDS = {
    'coffee', 'arabica', 'robusta', 'sugar', 'sugarcane', 'sugar beet'
}

DRIVER_KEYWORDS = {
    'drought', 'frost', 'flood', 'rainfall', 'la nina', 'el nino',
    'fertilizer', 'pesticide', 'water scarcity', 'labor shortage',
    'port congestion', 'shipping', 'inflation', 'recession',
    'ethanol', 'biofuel', 'crude oil', 'tariff', 'trade deal',
    'geopolitical', 'political instability'
}

ALL_THEMES = CORE_THEMES | DRIVER_THEMES
ALL_KEYWORDS = COMMODITY_KEYWORDS | DRIVER_KEYWORDS


def lambda_handler(event, context):
    """
    Main Lambda handler - routes to historical backfill or incremental update
    
    Event structure:
    {
        "mode": "incremental" | "backfill",
        "start_date": "2023-09-30",  # For backfill only
        "end_date": "2023-10-01"      # For backfill only
    }
    """
    mode = event.get('mode', 'incremental')
    
    try:
        if mode == 'backfill':
            start_date = datetime.strptime(event['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(event['end_date'], '%Y-%m-%d')
            result = process_historical_backfill(start_date, end_date)
        else:
            result = process_incremental_update()
        
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


def process_incremental_update() -> Dict:
    """
    Process the latest GDELT files (last 15 minutes or last day)
    Uses lastupdate.txt to find new files
    """
    logger.info("Starting incremental update")
    
    # Get list of recent files from GDELT
    lastupdate_url = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
    response = requests.get(lastupdate_url, timeout=30)
    response.raise_for_status()
    
    files_to_process = []
    for line in response.text.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 3 and 'gkg.csv' in parts[2]:
            file_url = parts[2]
            file_size = int(parts[0])
            files_to_process.append({
                'url': file_url,
                'size': file_size,
                'name': file_url.split('/')[-1]
            })
    
    logger.info(f"Found {len(files_to_process)} GKG files to process")
    
    # Process each file
    processed_count = 0
    filtered_records = 0
    total_records = 0
    
    for file_info in files_to_process:
        if not is_file_processed(file_info['name']):
            stats = download_and_filter_file(
                file_info['url'], 
                file_info['name']
            )
            processed_count += 1
            filtered_records += stats['filtered']
            total_records += stats['total']
            mark_file_processed(file_info['name'])
    
    result = {
        'processed_files': processed_count,
        'total_records': total_records,
        'filtered_records': filtered_records,
        'filter_rate': f"{(filtered_records/total_records*100):.2f}%" if total_records > 0 else "0%"
    }
    
    logger.info(f"Incremental update complete: {result}")
    return result


def process_historical_backfill(start_date: datetime, end_date: datetime) -> Dict:
    """
    Process historical GDELT files for a date range
    Generates 15-minute interval URLs for the date range
    """
    logger.info(f"Starting historical backfill from {start_date} to {end_date}")
    
    file_urls = generate_file_urls(start_date, end_date)
    logger.info(f"Generated {len(file_urls)} file URLs to process")
    
    processed_count = 0
    filtered_records = 0
    total_records = 0
    skipped_count = 0
    
    # Process files (Lambda has 15min timeout, so limit batch size)
    max_files_per_invocation = 50
    for i, url in enumerate(file_urls[:max_files_per_invocation]):
        file_name = url.split('/')[-1]
        
        if is_file_processed(file_name):
            skipped_count += 1
            continue
        
        try:
            stats = download_and_filter_file(url, file_name)
            processed_count += 1
            filtered_records += stats['filtered']
            total_records += stats['total']
            mark_file_processed(file_name)
            
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(file_urls)} files")
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"File not found: {file_name}")
                mark_file_processed(file_name)  # Mark as processed to skip in future
            else:
                raise
    
    result = {
        'processed_files': processed_count,
        'skipped_files': skipped_count,
        'total_records': total_records,
        'filtered_records': filtered_records,
        'filter_rate': f"{(filtered_records/total_records*100):.2f}%" if total_records > 0 else "0%",
        'remaining_files': len(file_urls) - max_files_per_invocation
    }
    
    logger.info(f"Backfill batch complete: {result}")
    return result


def generate_file_urls(start_date: datetime, end_date: datetime) -> List[str]:
    """
    Generate GDELT GKG file URLs for date range
    GKG 2.0 updates every 15 minutes
    """
    urls = []
    current = start_date
    
    while current <= end_date:
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                timestamp = current.replace(hour=hour, minute=minute, second=0)
                url = f"http://data.gdeltproject.org/gdeltv2/{timestamp.strftime('%Y%m%d%H%M%S')}.gkg.csv.zip"
                urls.append(url)
        current += timedelta(days=1)
    
    return urls


def download_and_filter_file(url: str, file_name: str) -> Dict:
    """
    Download GDELT file, filter for commodity-relevant records, save to S3
    Returns statistics about processing
    """
    logger.info(f"Processing file: {file_name}")
    
    # Download and decompress
    response = requests.get(url, timeout=60, stream=True)
    response.raise_for_status()
    
    zip_data = BytesIO(response.content)
    
    with zipfile.ZipFile(zip_data, 'r') as zip_ref:
        csv_filename = zip_ref.namelist()[0]
        csv_data = zip_ref.read(csv_filename).decode('utf-8')
    
    # Parse CSV and filter
    csv_reader = csv.reader(StringIO(csv_data), delimiter='\t')
    
    filtered_rows = []
    total_count = 0
    
    for row in csv_reader:
        total_count += 1
        
        if len(row) < 27:  # GKG has 27 columns
            continue
        
        # Extract relevant fields
        gkg_record = parse_gkg_row(row)
        
        # Apply filters
        if should_include_record(gkg_record):
            filtered_rows.append(gkg_record)
    
    # Save raw file to S3 (optional, for audit trail)
    # s3.put_object(
    #     Bucket=S3_BUCKET,
    #     Key=f"{S3_RAW_PREFIX}{file_name}",
    #     Body=response.content
    # )
    
    # Save filtered data as JSON Lines (easier for Databricks)
    if filtered_rows:
        filtered_data = '\n'.join([json.dumps(record) for record in filtered_rows])
        
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=f"{S3_FILTERED_PREFIX}{file_name.replace('.zip', '.jsonl')}",
            Body=filtered_data.encode('utf-8'),
            ContentType='application/x-ndjson'
        )
    
    return {
        'total': total_count,
        'filtered': len(filtered_rows)
    }


def parse_gkg_row(row: List[str]) -> Dict:
    """
    Parse a GKG CSV row into structured dict
    Only extracts fields we need for commodity analysis
    """
    return {
        'date': row[1] if len(row) > 1 else '',
        'source_url': row[4] if len(row) > 4 else '',
        'themes': row[7] if len(row) > 7 else '',
        'locations': row[9] if len(row) > 9 else '',
        'persons': row[11] if len(row) > 11 else '',
        'organizations': row[13] if len(row) > 13 else '',
        'tone': row[15] if len(row) > 15 else '',
        'all_names': row[23] if len(row) > 23 else ''
    }


def should_include_record(record: Dict) -> bool:
    """
    Filter logic for commodity-relevant records
    Returns True if record matches any of our criteria
    """
    themes = record.get('themes', '').upper()
    all_names = record.get('all_names', '').lower()
    
    # Check themes
    for theme in ALL_THEMES:
        if f';{theme}' in themes or themes.startswith(theme):
            return True
    
    # Check keywords in all_names
    for keyword in ALL_KEYWORDS:
        if keyword in all_names:
            return True
    
    return False


def is_file_processed(file_name: str) -> bool:
    """
    Check if file has already been processed
    Uses DynamoDB for tracking
    """
    try:
        table = dynamodb.Table(TRACKING_TABLE)
        response = table.get_item(Key={'file_name': file_name})
        return 'Item' in response
    except Exception as e:
        logger.warning(f"Error checking file status: {e}")
        return False


def mark_file_processed(file_name: str):
    """
    Mark file as processed in DynamoDB
    """
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
    # Test incremental mode
    test_event_incremental = {
        "mode": "incremental"
    }
    
    # Test backfill mode
    test_event_backfill = {
        "mode": "backfill",
        "start_date": "2023-09-30",
        "end_date": "2023-10-01"
    }
    
    result = lambda_handler(test_event_incremental, None)
    print(json.dumps(result, indent=2))

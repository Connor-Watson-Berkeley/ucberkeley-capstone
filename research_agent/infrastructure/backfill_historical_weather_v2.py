"""
Historical Weather Data Backfill with CORRECT Coordinates (v2)

⚠️  CRITICAL DATA QUALITY FIX:
This script regenerates ALL historical weather data (2015-2025) using the CORRECT
coordinates from region_coordinates.json (actual growing regions) instead of the
WRONG coordinates (state capitals/administrative centers) used in the current Lambda.

Why this matters:
- Current Lambda missed July 2021 Brazil frost that spiked coffee prices 70%
- Coordinates were ~100-200km off from actual growing regions
- Weather data had no predictive value for commodity forecasting

Data Source: Open-Meteo Historical Weather API (FREE, up to 10,000 requests/day)
API Docs: https://open-meteo.com/en/docs/historical-weather-api

Output: s3://groundtruth-capstone/landing/weather_v2/
Schema: Same as current weather table + latitude/longitude columns for transparency

Estimated Runtime: 6-12 hours (67 regions × 3,800+ days, rate-limited)
"""

import json
import boto3
import requests
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
import logging
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('weather_backfill_v2.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
S3_BUCKET = 'groundtruth-capstone'
S3_PREFIX_V2 = 'landing/weather_v2'  # New prefix for corrected data
OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
RATE_LIMIT_DELAY = 0.1  # Seconds between requests (10 req/sec = 864,000 req/day >> 67×3800 = 254,600)
MAX_RETRIES = 3
CHUNK_DAYS = 90  # Fetch weather in 90-day chunks to avoid API limits

s3_client = boto3.client('s3')

def load_region_coordinates() -> List[Dict]:
    """
    Load CORRECT region coordinates from S3.

    These coordinates target actual growing regions (e.g., Sul de Minas coffee)
    instead of state capitals (e.g., Belo Horizonte).
    """
    logger.info("Loading CORRECT region coordinates from S3...")
    try:
        response = s3_client.get_object(
            Bucket=S3_BUCKET,
            Key='config/region_coordinates.json'
        )
        coordinates = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"✅ Loaded {len(coordinates)} regions with CORRECT coordinates")

        # Log sample for verification
        minas = [r for r in coordinates if r['region'] == 'Minas_Gerais_Brazil'][0]
        logger.info(f"📍 Sample - Minas Gerais: ({minas['latitude']}, {minas['longitude']}) - {minas['description']}")

        return coordinates
    except Exception as e:
        logger.error(f"❌ Failed to load coordinates: {e}")
        raise


def fetch_weather_chunk(
    region: Dict,
    start_date: date,
    end_date: date,
    retry_count: int = 0
) -> List[Dict]:
    """
    Fetch historical weather data for a single region and date range.

    Uses Open-Meteo Historical Weather API (archive-api.open-meteo.com).
    This is actual historical weather data (NOT synthetic, NO data leakage).
    """
    region_name = region['region']
    latitude = region['latitude']
    longitude = region['longitude']
    commodity = region['commodity']

    try:
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': str(start_date),
            'end_date': str(end_date),
            'daily': ','.join([
                'temperature_2m_max',
                'temperature_2m_min',
                'temperature_2m_mean',
                'precipitation_sum',
                'rain_sum',
                'snowfall_sum',
                'windspeed_10m_max',
                'relative_humidity_2m_mean'
            ]),
            'timezone': 'UTC'
        }

        response = requests.get(
            OPEN_METEO_HISTORICAL_URL,
            params=params,
            timeout=30
        )
        response.raise_for_status()

        data = response.json()

        # Check for errors in response
        if 'error' in data:
            raise Exception(f"API error: {data.get('reason', 'Unknown error')}")

        daily = data.get('daily', {})
        if not daily or 'time' not in daily:
            logger.warning(f"⚠️  {region_name} {start_date} to {end_date}: No data returned")
            return []

        # Convert to our schema
        weather_records = []
        for i in range(len(daily['time'])):
            record = {
                # Core weather data
                'date': daily['time'][i],
                'region': region_name,
                'commodity': commodity,
                'temp_max_c': round(daily['temperature_2m_max'][i], 2) if daily['temperature_2m_max'][i] is not None else None,
                'temp_min_c': round(daily['temperature_2m_min'][i], 2) if daily['temperature_2m_min'][i] is not None else None,
                'temp_mean_c': round(daily['temperature_2m_mean'][i], 2) if daily['temperature_2m_mean'][i] is not None else None,
                'precipitation_mm': round(daily['precipitation_sum'][i], 2) if daily['precipitation_sum'][i] is not None else 0.0,
                'rain_mm': round(daily['rain_sum'][i], 2) if daily['rain_sum'][i] is not None else 0.0,
                'snowfall_cm': round(daily['snowfall_sum'][i], 2) if daily['snowfall_sum'][i] is not None else 0.0,
                'wind_speed_max_kmh': round(daily['windspeed_10m_max'][i], 2) if daily['windspeed_10m_max'][i] is not None else None,
                'humidity_mean_pct': round(daily['relative_humidity_2m_mean'][i], 2) if daily['relative_humidity_2m_mean'][i] is not None else None,

                # NEW: Record coordinates for transparency
                'latitude': latitude,
                'longitude': longitude,

                # Metadata
                'country': region.get('country', ''),
                'elevation_m': region.get('elevation_m'),
                'description': region.get('description', ''),
                'ingest_ts': datetime.now().isoformat(),
                'data_version': 'v2_corrected_coordinates',
                'coordinate_source': 'region_coordinates.json'
            }
            weather_records.append(record)

        time.sleep(RATE_LIMIT_DELAY)  # Rate limiting
        return weather_records

    except requests.exceptions.RequestException as e:
        if retry_count < MAX_RETRIES:
            logger.warning(f"⚠️  {region_name} {start_date}-{end_date}: Retry {retry_count + 1}/{MAX_RETRIES} after error: {e}")
            time.sleep(2 ** retry_count)  # Exponential backoff
            return fetch_weather_chunk(region, start_date, end_date, retry_count + 1)
        else:
            logger.error(f"❌ {region_name} {start_date}-{end_date}: Failed after {MAX_RETRIES} retries: {e}")
            return []
    except Exception as e:
        logger.error(f"❌ {region_name} {start_date}-{end_date}: {str(e)}")
        return []


def write_to_s3(records: List[Dict], batch_date: date, dry_run: bool = False) -> bool:
    """
    Write weather records to S3 in JSON Lines format.

    Path: s3://groundtruth-capstone/landing/weather_v2/year=YYYY/month=MM/day=DD/data.jsonl
    """
    if not records:
        return False

    # Partition by date (using first record's date for batch)
    year = batch_date.year
    month = f"{batch_date.month:02d}"
    day = f"{batch_date.day:02d}"

    s3_key = f"{S3_PREFIX_V2}/year={year}/month={month}/day={day}/data.jsonl"

    if dry_run:
        logger.info(f"[DRY RUN] Would write {len(records)} records to s3://{S3_BUCKET}/{s3_key}")
        return True

    try:
        # Convert to JSON Lines
        jsonl_content = '\n'.join(json.dumps(r) for r in records)

        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=jsonl_content.encode('utf-8'),
            ContentType='application/x-ndjson'
        )

        logger.info(f"✅ Wrote {len(records)} records to {s3_key}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to write to S3: {e}")
        return False


def backfill_region_date_range(
    region: Dict,
    start_date: date,
    end_date: date,
    dry_run: bool = False
) -> Dict:
    """
    Backfill weather data for a single region across a date range.
    Fetches in chunks to avoid API limits.
    """
    region_name = region['region']
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {region_name} ({start_date} to {end_date})")
    logger.info(f"Coordinates: ({region['latitude']}, {region['longitude']}) - {region['description']}")
    logger.info(f"{'='*80}")

    all_records = []
    successful_chunks = 0
    failed_chunks = 0

    # Split into chunks
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS - 1), end_date)

        logger.info(f"  Fetching {region_name}: {current} to {chunk_end}...")
        records = fetch_weather_chunk(region, current, chunk_end)

        if records:
            all_records.extend(records)
            successful_chunks += 1
            logger.info(f"    ✅ {len(records)} records")
        else:
            failed_chunks += 1
            logger.warning(f"    ❌ No data")

        current = chunk_end + timedelta(days=1)

    # Write all records for this region
    if all_records:
        # Group by date for partitioning
        by_date = {}
        for record in all_records:
            record_date = datetime.strptime(record['date'], '%Y-%m-%d').date()
            if record_date not in by_date:
                by_date[record_date] = []
            by_date[record_date].append(record)

        # Write each date partition
        writes_successful = 0
        for batch_date, batch_records in by_date.items():
            if write_to_s3(batch_records, batch_date, dry_run):
                writes_successful += 1

        logger.info(f"\n📊 {region_name} Summary:")
        logger.info(f"  Total records: {len(all_records)}")
        logger.info(f"  Successful chunks: {successful_chunks}")
        logger.info(f"  Failed chunks: {failed_chunks}")
        logger.info(f"  S3 writes: {writes_successful}/{len(by_date)}")

    return {
        'region': region_name,
        'total_records': len(all_records),
        'successful_chunks': successful_chunks,
        'failed_chunks': failed_chunks,
        'start_date': str(start_date),
        'end_date': str(end_date)
    }


def validate_july_2021_frost(dry_run: bool = False):
    """
    Validate that the corrected weather data now detects the July 2021 frost.
    Expected: Minas Gerais temps should be -2°C to -4°C on July 20-21, 2021.
    """
    logger.info("\n" + "="*80)
    logger.info("VALIDATION: July 2021 Brazil Frost")
    logger.info("="*80)

    if dry_run:
        logger.info("[DRY RUN] Skipping validation")
        return

    try:
        # Check S3 for July 20-21, 2021 data
        s3_key = f"{S3_PREFIX_V2}/year=2021/month=07/day=20/data.jsonl"

        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        content = response['Body'].read().decode('utf-8')

        # Find Minas Gerais record
        for line in content.split('\n'):
            if not line.strip():
                continue
            record = json.loads(line)
            if record['region'] == 'Minas_Gerais_Brazil':
                temp_min = record['temp_min_c']
                logger.info(f"\n✅ VALIDATION RESULT:")
                logger.info(f"   Date: 2021-07-20")
                logger.info(f"   Region: Minas Gerais")
                logger.info(f"   Coordinates: ({record['latitude']}, {record['longitude']})")
                logger.info(f"   Min Temp: {temp_min}°C")

                if temp_min is not None and temp_min < 0:
                    logger.info(f"   ✅ FROST DETECTED! (Expected: -2°C to -4°C)")
                else:
                    logger.warning(f"   ⚠️  No frost detected (temps {temp_min}°C)")

                break
    except Exception as e:
        logger.warning(f"⚠️  Validation failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Backfill historical weather with CORRECT coordinates (v2)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2015-07-07',
        help='Start date (YYYY-MM-DD, default: 2015-07-07)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=str(date.today()),
        help='End date (YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode (no S3 writes)'
    )
    parser.add_argument(
        '--regions',
        type=str,
        nargs='+',
        help='Specific regions to backfill (default: all)'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run July 2021 frost validation'
    )

    args = parser.parse_args()

    if args.validate_only:
        validate_july_2021_frost()
        return

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()

    print("="*80)
    print("HISTORICAL WEATHER BACKFILL v2 - CORRECTED COORDINATES")
    print("="*80)
    print(f"\n⚠️  DATA QUALITY FIX:")
    print(f"   This regenerates ALL weather data with CORRECT coordinates")
    print(f"   (actual growing regions, not state capitals)")
    print()
    print(f"Configuration:")
    print(f"  Start Date: {start_date}")
    print(f"  End Date: {end_date}")
    print(f"  Total Days: {(end_date - start_date).days + 1}")
    print(f"  S3 Bucket: {S3_BUCKET}")
    print(f"  S3 Prefix: {S3_PREFIX_V2}")
    print(f"  Dry Run: {args.dry_run}")
    print()

    # Load coordinates
    regions = load_region_coordinates()

    # Filter regions if specified
    if args.regions:
        regions = [r for r in regions if r['region'] in args.regions]
        print(f"  Regions: {len(regions)} (filtered)")
    else:
        print(f"  Regions: {len(regions)} (all)")

    total_days = (end_date - start_date).days + 1
    total_requests = len(regions) * (total_days // CHUNK_DAYS + 1)
    estimated_time_hours = (total_requests * 0.5) / 3600  # 0.5 sec per request

    print(f"  Estimated Requests: ~{total_requests:,}")
    print(f"  Estimated Time: {estimated_time_hours:.1f} - {estimated_time_hours*2:.1f} hours")
    print()

    # Confirm
    if not args.dry_run:
        print(f"⚠️  ABOUT TO REGENERATE HISTORICAL WEATHER DATA:")
        print(f"   This will write to: s3://{S3_BUCKET}/{S3_PREFIX_V2}/")
        print()
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Aborted")
            return

    # Backfill
    print(f"\n🚀 Starting backfill...\n")
    start_time = time.time()

    results = []
    for i, region in enumerate(regions, 1):
        print(f"\n[{i}/{len(regions)}] Processing {region['region']}...")
        result = backfill_region_date_range(region, start_date, end_date, args.dry_run)
        results.append(result)

        # Progress update
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (len(regions) - i)
            print(f"\n⏱️  Progress: {i}/{len(regions)} ({100*i/len(regions):.1f}%)")
            print(f"   Elapsed: {elapsed/3600:.1f}h, Remaining: ~{remaining/3600:.1f}h")

    # Final summary
    elapsed = time.time() - start_time
    total_records = sum(r['total_records'] for r in results)
    successful_regions = sum(1 for r in results if r['total_records'] > 0)

    print(f"\n{'='*80}")
    print("BACKFILL COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Final Summary:")
    print(f"  Regions processed: {len(regions)}")
    print(f"  Successful regions: {successful_regions}/{len(regions)}")
    print(f"  Total records: {total_records:,}")
    print(f"  Total time: {elapsed/3600:.1f} hours")
    print()
    print(f"✅ Data written to: s3://{S3_BUCKET}/{S3_PREFIX_V2}/")
    print()
    print(f"📝 Next steps:")
    print(f"  1. Validate July 2021 frost: python backfill_historical_weather_v2.py --validate-only")
    print(f"  2. Create weather_v2 bronze table in Databricks")
    print(f"  3. Update unified_data to use weather_v2")
    print(f"  4. Train SARIMAX models and measure accuracy improvement")
    print("="*80)

    # Run validation
    if not args.dry_run:
        validate_july_2021_frost()


if __name__ == '__main__':
    main()

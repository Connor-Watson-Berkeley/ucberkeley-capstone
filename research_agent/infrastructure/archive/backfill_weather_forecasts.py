"""
⚠️  DEPRECATED - Use backfill_synthetic_weather_forecasts.py instead

This script was originally intended to fetch real historical forecasts from
Open-Meteo, but after testing we discovered that historical forecast archives
are not available from the free API.

CURRENT STRATEGY (as of 2025-11-05):
- Historical forecasts (2015-2025): Use backfill_synthetic_weather_forecasts.py
  → Generates synthetic forecasts with documented data leakage
- Daily forecasts (2025+): Deploy Lambda function (weather_forecast_fetcher.py)
  → Collects real forecasts with no data leakage

See: research_agent/infrastructure/WEATHER_FORECAST_LIMITATION.md

This file is kept for reference but should NOT be used for production.
"""

import json
import boto3
import requests
from datetime import datetime, timedelta, date
from typing import List, Dict
import logging
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

s3_client = boto3.client('s3')

# Configuration
S3_BUCKET = 'groundtruth-capstone'
S3_PREFIX = 'landing/weather_forecast'
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
MAX_WORKERS = 5  # Parallel requests (be nice to free API)
RATE_LIMIT_DELAY = 0.2  # Seconds between requests

def load_region_coordinates() -> List[Dict]:
    """Load region coordinates from S3."""
    logger.info("Loading region coordinates from S3...")
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key='config/region_coordinates.json')
        coordinates = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"✅ Loaded {len(coordinates)} regions")
        return coordinates
    except Exception as e:
        logger.error(f"❌ Failed to load coordinates: {e}")
        raise


def fetch_forecast_for_region_date(region: Dict, forecast_date: date) -> List[Dict]:
    """
    Fetch 16-day forecast for a specific region and forecast date.

    Note: Open-Meteo Historical Forecast API provides past model runs.
    """
    region_name = region['region']
    latitude = region['latitude']
    longitude = region['longitude']

    try:
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'daily': [
                'temperature_2m_max',
                'temperature_2m_min',
                'precipitation_sum',
                'relative_humidity_2m_mean',
                'wind_speed_10m_max'
            ],
            'start_date': str(forecast_date),
            'end_date': str(forecast_date + timedelta(days=15)),  # 16 days total
            'timezone': 'UTC'
        }

        response = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        daily = data['daily']

        forecasts = []
        for i in range(len(daily['time'])):
            target_date = datetime.strptime(daily['time'][i], '%Y-%m-%d').date()
            days_ahead = (target_date - forecast_date).days

            temp_max = daily['temperature_2m_max'][i]
            temp_min = daily['temperature_2m_min'][i]
            temp_mean = (temp_max + temp_min) / 2 if temp_max and temp_min else None

            forecast = {
                'forecast_date': str(forecast_date),
                'target_date': str(target_date),
                'days_ahead': days_ahead,
                'region': region_name,
                'temp_max_c': round(temp_max, 2) if temp_max else None,
                'temp_min_c': round(temp_min, 2) if temp_min else None,
                'temp_mean_c': round(temp_mean, 2) if temp_mean else None,
                'precipitation_mm': round(daily['precipitation_sum'][i], 2) if daily['precipitation_sum'][i] else 0.0,
                'humidity_pct': round(daily['relative_humidity_2m_mean'][i], 2) if daily['relative_humidity_2m_mean'][i] else None,
                'wind_speed_kmh': round(daily['wind_speed_10m_max'][i], 2) if daily['wind_speed_10m_max'][i] else None,
                'ingest_ts': datetime.now().isoformat()
            }

            forecasts.append(forecast)

        time.sleep(RATE_LIMIT_DELAY)  # Rate limiting
        return forecasts

    except Exception as e:
        logger.warning(f"⚠️  {region_name} @ {forecast_date}: {str(e)}")
        return []


def write_to_s3(forecasts: List[Dict], forecast_date: date, dry_run: bool = False) -> bool:
    """Write forecasts to S3 in JSON Lines format."""
    if not forecasts:
        return False

    year = forecast_date.year
    month = f"{forecast_date.month:02d}"
    day = f"{forecast_date.day:02d}"

    s3_key = f"{S3_PREFIX}/year={year}/month={month}/day={day}/forecast.jsonl"

    jsonl_content = '\n'.join(json.dumps(f) for f in forecasts)

    if dry_run:
        logger.info(f"[DRY RUN] Would write {len(forecasts)} forecasts to s3://{S3_BUCKET}/{s3_key}")
        return True

    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=jsonl_content.encode('utf-8'),
            ContentType='application/x-ndjson'
        )
        logger.info(f"✅ Wrote {len(forecasts)} forecasts to {s3_key}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to write to S3: {e}")
        return False


def backfill_single_date(forecast_date: date, regions: List[Dict], dry_run: bool = False) -> Dict:
    """Backfill forecasts for a single date across all regions."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {forecast_date}")
    logger.info(f"{'='*80}")

    all_forecasts = []
    successful = 0
    failed = 0

    # Sequential processing with rate limiting (being nice to free API)
    for region in regions:
        forecasts = fetch_forecast_for_region_date(region, forecast_date)
        if forecasts:
            all_forecasts.extend(forecasts)
            successful += 1
        else:
            failed += 1

    # Write to S3
    if all_forecasts:
        write_success = write_to_s3(all_forecasts, forecast_date, dry_run)
    else:
        write_success = False

    result = {
        'date': str(forecast_date),
        'regions_successful': successful,
        'regions_failed': failed,
        'total_forecasts': len(all_forecasts),
        'write_success': write_success
    }

    logger.info(f"\n📊 Summary for {forecast_date}:")
    logger.info(f"  ✅ Successful regions: {successful}/{len(regions)}")
    logger.info(f"  ❌ Failed regions: {failed}/{len(regions)}")
    logger.info(f"  📝 Total forecasts: {len(all_forecasts)}")
    logger.info(f"  💾 Write to S3: {'✅' if write_success else '❌'}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Backfill weather forecasts from 2021+')
    parser.add_argument('--start-date', type=str, default='2021-01-01',
                       help='Start date (YYYY-MM-DD, default: 2021-01-01)')
    parser.add_argument('--end-date', type=str, default=str(date.today()),
                       help='End date (YYYY-MM-DD, default: today)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode (no S3 writes)')
    parser.add_argument('--sample-size', type=int,
                       help='Only process N random dates (for testing)')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()

    # Validate start date
    if start_date < date(2021, 1, 1):
        logger.warning("⚠️  Start date is before 2021-01-01. Open-Meteo historical forecasts only available from 2021+")
        start_date = date(2021, 1, 1)
        logger.info(f"   Adjusted start date to: {start_date}")

    print("="*80)
    print("WEATHER FORECAST BACKFILL")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Start Date: {start_date}")
    print(f"  End Date: {end_date}")
    print(f"  Total Days: {(end_date - start_date).days + 1}")
    print(f"  S3 Bucket: {S3_BUCKET}")
    print(f"  S3 Prefix: {S3_PREFIX}")
    print(f"  Dry Run: {args.dry_run}")
    print()

    # Load regions
    regions = load_region_coordinates()
    print(f"  Regions: {len(regions)}")
    print(f"  Expected forecasts per day: {len(regions) * 16}")
    print()

    # Generate date range
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    # Sample if requested
    if args.sample_size and args.sample_size < len(dates):
        import random
        dates = random.sample(dates, args.sample_size)
        dates.sort()
        print(f"📍 Sampling {args.sample_size} random dates for testing")
        print()

    # Confirm before proceeding
    if not args.dry_run:
        estimated_forecasts = len(dates) * len(regions) * 16
        estimated_size_mb = estimated_forecasts * 0.0003  # ~300 bytes per forecast
        print(f"⚠️  ABOUT TO WRITE TO S3:")
        print(f"   Dates: {len(dates)}")
        print(f"   Estimated forecasts: ~{estimated_forecasts:,}")
        print(f"   Estimated size: ~{estimated_size_mb:.1f} MB")
        print()
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Aborted")
            return

    # Process all dates
    print(f"\n🚀 Starting backfill...\n")
    start_time = time.time()

    results = []
    for i, forecast_date in enumerate(dates, 1):
        print(f"\n[{i}/{len(dates)}] Processing {forecast_date}...")
        result = backfill_single_date(forecast_date, regions, args.dry_run)
        results.append(result)

        # Progress update every 10 dates
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (len(dates) - i)
            print(f"\n⏱️  Progress: {i}/{len(dates)} ({100*i/len(dates):.1f}%)")
            print(f"   Elapsed: {elapsed/60:.1f} min, Remaining: ~{remaining/60:.1f} min")

    # Final summary
    elapsed = time.time() - start_time
    total_forecasts = sum(r['total_forecasts'] for r in results)
    successful_writes = sum(1 for r in results if r['write_success'])

    print(f"\n{'='*80}")
    print("BACKFILL COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Final Summary:")
    print(f"  Dates processed: {len(dates)}")
    print(f"  Successful writes: {successful_writes}/{len(dates)}")
    print(f"  Total forecasts: {total_forecasts:,}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Avg time per date: {elapsed/len(dates):.1f} seconds")
    print()
    print(f"✅ Data written to: s3://{S3_BUCKET}/{S3_PREFIX}/")
    print(f"   Date range: {start_date} to {end_date}")
    print()
    print(f"📝 Next steps:")
    print(f"  1. Create Databricks tables: Run weather_forecast_setup.sql")
    print(f"  2. Update unified_data: Add weather forecast integration")
    print(f"  3. Train models with data >= 2021-01-01")
    print("="*80)


if __name__ == '__main__':
    main()

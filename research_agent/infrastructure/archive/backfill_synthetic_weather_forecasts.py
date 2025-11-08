"""
Backfill SYNTHETIC weather forecasts from 2015-07-07 to 2025-11-05.

⚠️  CRITICAL WARNING - DATA LEAKAGE:
    This script generates SYNTHETIC forecasts by adding realistic error to
    historical ACTUAL weather data. This means the "forecast" knows the actual
    outcome and only adds noise.

    ⚠️  HAS DATA LEAKAGE - May overestimate forecast utility
    ⚠️  For proof-of-concept only
    ⚠️  Models using this data MUST document the limitation

Purpose:
    Evaluate whether weather forecasts improve SARIMAX models enough to justify
    purchasing real historical forecast data (~$195 from Visual Crossing).

Strategy:
    1. Generate synthetic forecasts for immediate experimentation (2015-2025)
    2. Deploy daily Lambda to collect real forecasts (2025+)
    3. If significant improvement → Consider purchasing real historical data
    4. Transition to real-only once enough accumulated (6+ months)

See: research_agent/infrastructure/WEATHER_FORECAST_LIMITATION.md

Usage:
    python backfill_synthetic_weather_forecasts.py [--start-date 2015-07-07] [--end-date 2025-11-05] [--dry-run]
"""

import os
import json
import boto3
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict
import logging
import argparse
from databricks import sql
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

s3_client = boto3.client('s3')

# Configuration
S3_BUCKET = 'groundtruth-capstone'
S3_PREFIX = 'landing/weather_forecast'

# Databricks connection parameters
DATABRICKS_HOST = os.getenv('DATABRICKS_HOST', 'https://dbc-fd7b00f3-7a6d.cloud.databricks.com')
DATABRICKS_TOKEN = os.getenv('DATABRICKS_TOKEN')
DATABRICKS_HTTP_PATH = os.getenv('DATABRICKS_HTTP_PATH', '/sql/1.0/warehouses/3cede8561503a13c')

def get_databricks_connection():
    """Create Databricks SQL connection."""
    if not DATABRICKS_TOKEN:
        raise ValueError("DATABRICKS_TOKEN environment variable not set")

    return sql.connect(
        server_hostname=DATABRICKS_HOST.replace('https://', ''),
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )

def fetch_historical_weather_for_date_range(start_date: date, end_date: date) -> Dict[str, List[Dict]]:
    """
    Fetch historical weather actuals from Databricks.

    Returns: Dictionary keyed by region, containing list of daily weather records.
    """
    logger.info(f"Fetching historical weather from Databricks: {start_date} to {end_date}")

    # We need to fetch 16 extra days after end_date for forecast horizon
    extended_end_date = end_date + timedelta(days=16)

    query = f"""
    SELECT
        Date as date,
        Region as region,
        Temp_Mean_C as temp_mean_c,
        Temp_Max_C as temp_max_c,
        Temp_Min_C as temp_min_c,
        Rain_mm as rain_mm,
        Snowfall_cm as snowfall_cm,
        Humidity_Mean_Pct as humidity_pct,
        Wind_Speed_Max_kmh as wind_speed_kmh
    FROM commodity.bronze.weather
    WHERE Date >= '{start_date}'
      AND Date <= '{extended_end_date}'
    ORDER BY Region, Date
    """

    with get_databricks_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

    # Organize by region
    weather_by_region = {}
    for row in rows:
        record = dict(zip(columns, row))
        region = record['region']

        if region not in weather_by_region:
            weather_by_region[region] = []

        weather_by_region[region].append(record)

    logger.info(f"✅ Fetched {len(rows)} weather records for {len(weather_by_region)} regions")
    return weather_by_region

def generate_realistic_forecast_error(value: float, days_ahead: int, variable: str) -> float:
    """
    Generate realistic forecast error based on forecast horizon and variable type.

    Error increases with forecast horizon:
    - Temperature: ±1.5°C at day 1, ±3.6°C at day 16
    - Precipitation: Higher relative error (40% + 5% per day)
    - Humidity: ±5% at day 1, ±12% at day 16
    - Wind: ±2 km/h at day 1, ±5 km/h at day 16

    Args:
        value: The actual value
        days_ahead: Forecast horizon (1-16)
        variable: Variable type (temp, precip, humidity, wind)

    Returns:
        Forecasted value (actual + realistic error)
    """
    np.random.seed(int(time.time() * 1000) % 2**32)  # Time-based seed for variation

    if value is None:
        return None

    if variable == 'temp':
        # Temperature: 1.5°C base error + 0.15°C per day ahead
        error_std = 1.5 + (days_ahead * 0.15)
        error = np.random.normal(0, error_std)
        return value + error

    elif variable == 'precip':
        # Precipitation: More complex - higher relative error
        # Base error: 40% + 5% per day
        relative_error = 0.40 + (days_ahead * 0.05)

        if value < 0.1:  # Dry day
            # Small chance of false positive rain
            return max(0, np.random.exponential(0.5) if np.random.random() < 0.1 else 0)
        else:
            # Add multiplicative error
            error_factor = np.random.lognormal(0, relative_error)
            return max(0, value * error_factor)

    elif variable == 'humidity':
        # Humidity: 5% base error + 0.5% per day ahead
        error_std = 5.0 + (days_ahead * 0.5)
        error = np.random.normal(0, error_std)
        return np.clip(value + error, 0, 100)  # Constrain to 0-100%

    elif variable == 'wind':
        # Wind speed: 2 km/h base error + 0.2 km/h per day ahead
        error_std = 2.0 + (days_ahead * 0.2)
        error = np.random.normal(0, error_std)
        return max(0, value + error)

    else:
        return value

def generate_synthetic_forecasts_for_date(
    forecast_date: date,
    weather_by_region: Dict[str, List[Dict]]
) -> List[Dict]:
    """
    Generate 16-day synthetic forecasts for a single forecast date.

    For each region, creates forecasts for days 1-16 by adding realistic error
    to the actual weather that occurred.

    ⚠️ DATA LEAKAGE: Forecasts "know" the actual outcome (just add noise)
    """
    forecasts = []

    for region, weather_records in weather_by_region.items():
        # Create lookup by date for this region
        weather_by_date = {rec['date']: rec for rec in weather_records}

        # Generate 16-day forecast
        for days_ahead in range(1, 17):
            target_date = forecast_date + timedelta(days=days_ahead)

            # Find actual weather for this target date
            if target_date not in weather_by_date:
                continue  # Skip if no actual data available

            actual = weather_by_date[target_date]

            # Generate synthetic forecast by adding realistic error
            temp_max_forecast = generate_realistic_forecast_error(
                actual['temp_max_c'], days_ahead, 'temp'
            )
            temp_min_forecast = generate_realistic_forecast_error(
                actual['temp_min_c'], days_ahead, 'temp'
            )

            # Mean temperature
            if temp_max_forecast is not None and temp_min_forecast is not None:
                temp_mean_forecast = (temp_max_forecast + temp_min_forecast) / 2
            else:
                temp_mean_forecast = generate_realistic_forecast_error(
                    actual['temp_mean_c'], days_ahead, 'temp'
                )

            # Precipitation (rain + snow converted to mm)
            actual_precip = (actual['rain_mm'] or 0) + (actual['snowfall_cm'] or 0) * 10
            precip_forecast = generate_realistic_forecast_error(
                actual_precip, days_ahead, 'precip'
            )

            # Humidity
            humidity_forecast = generate_realistic_forecast_error(
                actual['humidity_pct'], days_ahead, 'humidity'
            )

            # Wind speed
            wind_forecast = generate_realistic_forecast_error(
                actual['wind_speed_kmh'], days_ahead, 'wind'
            )

            # Create forecast record
            forecast = {
                'forecast_date': str(forecast_date),
                'target_date': str(target_date),
                'days_ahead': days_ahead,
                'region': region,
                'temp_max_c': round(temp_max_forecast, 2) if temp_max_forecast else None,
                'temp_min_c': round(temp_min_forecast, 2) if temp_min_forecast else None,
                'temp_mean_c': round(temp_mean_forecast, 2) if temp_mean_forecast else None,
                'precipitation_mm': round(precip_forecast, 2) if precip_forecast else 0.0,
                'humidity_pct': round(humidity_forecast, 2) if humidity_forecast else None,
                'wind_speed_kmh': round(wind_forecast, 2) if wind_forecast else None,
                'ingest_ts': datetime.now().isoformat(),
                # ⚠️ METADATA: Mark as synthetic with data leakage
                'is_synthetic': True,
                'has_data_leakage': True,
                'generation_method': 'actual_weather_plus_realistic_error'
            }

            forecasts.append(forecast)

    return forecasts

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
        logger.info(f"[DRY RUN] Sample forecast: {json.dumps(forecasts[0], indent=2)}")
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

def backfill_synthetic_forecasts(
    start_date: date,
    end_date: date,
    dry_run: bool = False
) -> Dict:
    """
    Main backfill logic.

    1. Fetch all historical weather actuals from Databricks
    2. For each forecast date, generate 16-day synthetic forecasts
    3. Write to S3
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"⚠️  GENERATING SYNTHETIC WEATHER FORECASTS WITH DATA LEAKAGE")
    logger.info(f"{'='*80}\n")

    # Fetch all weather data upfront (more efficient than querying per date)
    logger.info("Step 1: Fetching historical weather actuals from Databricks...")
    weather_by_region = fetch_historical_weather_for_date_range(start_date, end_date)

    # Generate date range
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    logger.info(f"\nStep 2: Generating synthetic forecasts for {len(dates)} dates...")
    logger.info(f"  Date range: {start_date} to {end_date}")
    logger.info(f"  Regions: {len(weather_by_region)}")
    logger.info(f"  Expected forecasts per date: {len(weather_by_region) * 16}")
    logger.info(f"  Total expected: ~{len(dates) * len(weather_by_region) * 16:,}\n")

    if not dry_run:
        response = input("⚠️  Continue with synthetic forecast generation (has data leakage)? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("❌ Aborted")
            return {}

    # Process each date
    results = []
    start_time = time.time()

    for i, forecast_date in enumerate(dates, 1):
        logger.info(f"\n[{i}/{len(dates)}] Generating forecasts for {forecast_date}...")

        forecasts = generate_synthetic_forecasts_for_date(forecast_date, weather_by_region)

        if forecasts:
            write_success = write_to_s3(forecasts, forecast_date, dry_run)
            logger.info(f"  ✅ Generated {len(forecasts)} synthetic forecasts")
        else:
            write_success = False
            logger.warning(f"  ⚠️  No forecasts generated (missing weather data?)")

        results.append({
            'date': str(forecast_date),
            'forecasts_generated': len(forecasts),
            'write_success': write_success
        })

        # Progress update every 50 dates
        if i % 50 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (len(dates) - i)
            total_forecasts = sum(r['forecasts_generated'] for r in results)

            logger.info(f"\n⏱️  Progress: {i}/{len(dates)} ({100*i/len(dates):.1f}%)")
            logger.info(f"   Forecasts generated so far: {total_forecasts:,}")
            logger.info(f"   Elapsed: {elapsed/60:.1f} min, Remaining: ~{remaining/60:.1f} min")

    # Final summary
    elapsed = time.time() - start_time
    total_forecasts = sum(r['forecasts_generated'] for r in results)
    successful_writes = sum(1 for r in results if r['write_success'])

    logger.info(f"\n{'='*80}")
    logger.info("SYNTHETIC FORECAST BACKFILL COMPLETE")
    logger.info(f"{'='*80}\n")
    logger.info(f"📊 Summary:")
    logger.info(f"  Dates processed: {len(dates)}")
    logger.info(f"  Successful writes: {successful_writes}/{len(dates)}")
    logger.info(f"  Total synthetic forecasts: {total_forecasts:,}")
    logger.info(f"  Total time: {elapsed/60:.1f} minutes")
    logger.info(f"  Avg time per date: {elapsed/len(dates):.2f} seconds\n")
    logger.info(f"✅ Data written to: s3://{S3_BUCKET}/{S3_PREFIX}/")
    logger.info(f"   Date range: {start_date} to {end_date}\n")
    logger.info(f"⚠️  REMINDER - DATA LEAKAGE:")
    logger.info(f"  These are SYNTHETIC forecasts with DATA LEAKAGE")
    logger.info(f"  Models using this data MUST document the limitation")
    logger.info(f"  See: research_agent/infrastructure/WEATHER_FORECAST_LIMITATION.md\n")
    logger.info(f"📝 Next steps:")
    logger.info(f"  1. Create Databricks tables: Run weather_forecast_setup.sql")
    logger.info(f"  2. Update unified_data: Integrate weather forecasts")
    logger.info(f"  3. Deploy daily Lambda: Start collecting real forecasts")
    logger.info(f"  4. Evaluate: Does synthetic show improvement? Consider Visual Crossing")
    logger.info(f"{'='*80}")

    return {
        'total_dates': len(dates),
        'total_forecasts': total_forecasts,
        'successful_writes': successful_writes,
        'elapsed_minutes': elapsed / 60
    }

def main():
    parser = argparse.ArgumentParser(
        description='⚠️  Generate SYNTHETIC weather forecasts with DATA LEAKAGE (2015-2025)'
    )
    parser.add_argument('--start-date', type=str, default='2015-07-07',
                       help='Start date (YYYY-MM-DD, default: 2015-07-07)')
    parser.add_argument('--end-date', type=str, default='2025-11-05',
                       help='End date (YYYY-MM-DD, default: 2025-11-05)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode (no S3 writes, show sample)')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()

    print("\n" + "="*80)
    print("⚠️  SYNTHETIC WEATHER FORECAST BACKFILL - DATA LEAKAGE WARNING")
    print("="*80)
    print("\nThis script generates SYNTHETIC forecasts by adding realistic error")
    print("to historical ACTUAL weather data.")
    print("\n⚠️  CRITICAL: These forecasts have DATA LEAKAGE")
    print("   - The 'forecast' knows the actual outcome (just adds noise)")
    print("   - May OVERESTIMATE forecast utility vs real forecasts")
    print("   - For proof-of-concept ONLY")
    print("\nPurpose: Test if forecasts improve SARIMAX before buying real data (~$195)")
    print("\nConfiguration:")
    print(f"  Start Date: {start_date}")
    print(f"  End Date: {end_date}")
    print(f"  Total Days: {(end_date - start_date).days + 1}")
    print(f"  S3 Bucket: {S3_BUCKET}")
    print(f"  S3 Prefix: {S3_PREFIX}")
    print(f"  Dry Run: {args.dry_run}")
    print("="*80 + "\n")

    backfill_synthetic_forecasts(start_date, end_date, args.dry_run)

if __name__ == '__main__':
    main()

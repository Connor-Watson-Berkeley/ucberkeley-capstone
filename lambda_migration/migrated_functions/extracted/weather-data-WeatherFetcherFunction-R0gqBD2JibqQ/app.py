import requests
import csv
import io
import datetime as dt
import time
import os
import json
import boto3
from botocore.exceptions import ClientError
from itertools import islice

# ----------------------------
# Open-Meteo endpoints (no key)
# ----------------------------
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

# Tune batching / retries via env
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))   # how many locations per request
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "6"))
BASE_BACKOFF = float(os.environ.get("BASE_BACKOFF", "0.8"))  # seconds
HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "60"))

# ----------------------------
# Your regions (unchanged)
# ----------------------------
COMMODITY_REGIONS = {
    # Brazil - 4 specific regions (30.78%)
    'Minas_Gerais_Brazil': (-18.5122, -44.5550, 'Coffee'),
    'Sao_Paulo_Brazil': (-23.5505, -46.6333, 'Coffee'),
    'Espirito_Santo_Brazil': (-19.5224, -40.6718, 'Coffee'),
    'Bahia_Brazil': (-12.9714, -38.5014, 'Coffee'),

    # Vietnam - 1 region (17.69%)
    'Central_Highlands_Vietnam': (12.6667, 108.0500, 'Coffee'),

    # Indonesia - 2 regions (6.87%)
    'Sumatra_Indonesia': (-0.5897, 101.3431, 'Coffee'),
    'Java_Indonesia': (-7.6145, 110.7121, 'Coffee'),

    # Colombia - 2 regions (6.15%)
    'Eje_Cafetero_Colombia': (4.8133, -75.6961, 'Coffee'),
    'Huila_Colombia': (2.9273, -75.2819, 'Coffee'),

    # Ethiopia - 2 regions (5.06%)
    'Sidamo_Ethiopia': (6.1621, 38.0762, 'Coffee'),
    'Yirgacheffe_Ethiopia': (6.1630, 38.2050, 'Coffee'),

    # Honduras - 1 region (3.47%)
    'Copan_Honduras': (14.8333, -88.9000, 'Coffee'),

    # Uganda - 2 regions (3.47%)
    'Bugisu_Uganda': (1.3490, 34.3300, 'Coffee'),
    'Rwenzori_Mountains_Uganda': (0.3906, 29.8970, 'Coffee'),

    # Peru - 1 region (3.34%)
    'Cajamarca_Peru': (-7.1638, -78.5003, 'Coffee'),

    # India - 2 regions (3.01%)
    'Karnataka_India': (15.3173, 75.7139, 'Coffee'),
    'Kerala_India': (10.8505, 76.2711, 'Coffee'),

    # Central African Republic (2.86%)
    'CAR_Country_Average': (6.6111, 20.9394, 'Coffee'),

    # Guatemala - 1 region (2.04%)
    'Antigua_Guatemala': (14.5611, -90.7333, 'Coffee'),

    # Guinea (1.81%)
    'Guinea_Country_Average': (9.9456, -9.6966, 'Coffee'),

    # Mexico - 2 regions (1.76%)
    'Chiapas_Mexico': (16.7569, -93.1292, 'Coffee'),
    'Veracruz_Mexico': (19.1738, -96.1342, 'Coffee'),

    # Laos (1.61%)
    'Laos_Country_Average': (19.8563, 102.4955, 'Coffee'),

    # Nicaragua (1.29%)
    'Nicaragua_Country_Average': (12.8654, -85.2072, 'Coffee'),

    # China - Yunnan (0.98%)
    'Yunnan_China_Coffee': (25.0422, 102.7063, 'Coffee'),

    # Ivory Coast (0.83%)
    'Cote_dIvoire_Country_Average': (7.5400, -5.5471, 'Coffee'),

    # Costa Rica (0.71%)
    'Costa_Rica_Country_Average': (9.7489, -83.7534, 'Coffee'),

    # Tanzania (0.57%)
    'Tanzania_Country_Average': (-6.3690, 34.8888, 'Coffee'),

    # Kenya (0.44%)
    'Kenya_Country_Average': (-0.0236, 37.9062, 'Coffee'),

    # ============================================================================
    # SUGAR CANE REGIONS (20 locations - 92% of cane production)
    # ============================================================================

    # Brazil (39.13%)
    'Sao_Paulo_Brazil_Sugar': (-23.5505, -46.6333, 'Sugar'),

    # India - 2 regions (24.53%)
    'Uttar_Pradesh_India': (26.8467, 80.9462, 'Sugar'),
    'Maharashtra_India': (19.7515, 75.7139, 'Sugar'),

    # China - 2 regions (5.23%)
    'Guangxi_China': (23.8322, 108.3201, 'Sugar'),
    'Yunnan_China_Sugar': (25.0422, 102.7063, 'Sugar'),

    # Thailand - 2 regions (4.70%)
    'Nakhon_Sawan_Thailand': (15.7110, 100.1260, 'Sugar'),
    'Khon_Kaen_Thailand': (16.4420, 102.8330, 'Sugar'),

    # Pakistan - 2 regions (4.38%)
    'Punjab_Pakistan': (31.5204, 74.3587, 'Sugar'),
    'Sindh_Pakistan': (25.8943, 68.5247, 'Sugar'),

    # Mexico - 2 regions (2.80%)
    'Veracruz_Mexico_Sugar': (19.1738, -96.1342, 'Sugar'),
    'Jalisco_Mexico': (20.6597, -103.3496, 'Sugar'),

    # Indonesia (1.74%)
    'Java_Indonesia_Sugar': (-7.6145, 110.7121, 'Sugar'),

    # Australia (1.63%)
    'Queensland_Australia': (-20.9176, 142.7028, 'Sugar'),

    # Colombia (1.62%)
    'Valle_del_Cauca_Colombia': (3.4206, -76.5222, 'Sugar'),

    # USA - Cane (1.50%)
    'South_Florida_USA': (26.6631, -80.6431, 'Sugar'),
    'Louisiana_USA': (30.9843, -91.9623, 'Sugar'),

    # Guatemala (1.32%)
    'Escuintla_Guatemala': (14.3050, -90.7850, 'Sugar'),

    # Philippines (1.09%)
    'Negros_Occidental_Philippines': (10.2523, 122.8687, 'Sugar'),

    # South Africa (0.90%)
    'KwaZulu_Natal_South_Africa': (-28.5306, 30.8958, 'Sugar'),

    # Argentina (0.77%)
    'Argentina_Country_Average': (-38.4161, -63.6167, 'Sugar'),

    # Egypt (0.77%) - NOTE: Egypt produces both cane AND beet
    'Qena_Egypt_Cane': (26.1607, 32.7260, 'Sugar'),

    # Cuba (0.42%)
    'Cuba_Country_Average': (21.5218, -77.7812, 'Sugar'),

    # ============================================================================
    # SUGAR BEET REGIONS (16 locations - 95% of beet production)
    # ============================================================================

    # Russia - 2 regions (17.59%)
    'Voronezh_Russia': (51.67, 39.18, 'Sugar'),
    'Tambov_Russia': (52.80, 41.33, 'Sugar'),

    # USA - Beet (11.54%)
    'Red_River_Valley_USA': (47.0, -97.0, 'Sugar'),

    # Germany (11.39%)
    'Germany_Country_Average': (51.1657, 10.4515, 'Sugar'),

    # France (11.03%)
    'France_Country_Average': (46.6034, 1.8883, 'Sugar'),

    # Turkey (9.12%)
    'Turkey_Country_Average': (38.9637, 35.2433, 'Sugar'),

    # Poland (6.10%)
    'Poland_Country_Average': (51.9194, 19.1451, 'Sugar'),

    # Ukraine (4.72%)
    'Ukraine_Country_Average': (48.3794, 31.1656, 'Sugar'),

    # Egypt - Beet (4.61%) - Egypt's northern region for beet
    'Nile_Delta_Egypt_Beet': (30.8, 31.2, 'Sugar'),

    # China - Beet (3.32%) - Northern China beet belt
    'North_China_Beet': (41.0, 115.0, 'Sugar'),

    # United Kingdom (2.78%)
    'UK_Country_Average': (52.3555, -1.1743, 'Sugar'),

    # Netherlands (2.49%)
    'Netherlands_Country_Average': (52.1326, 5.2913, 'Sugar'),

    # Iran (1.84%)
    'Iran_Country_Average': (32.4279, 53.6880, 'Sugar'),

    # Belarus (1.73%)
    'Belarus_Country_Average': (53.7098, 27.9534, 'Sugar'),

    # Belgium (1.73%)
    'Belgium_Country_Average': (50.5039, 4.4699, 'Sugar'),

    # Japan (1.23%)
    'Hokkaido_Japan': (43.0642, 141.3469, 'Sugar'),
}

# ----------------------------
# Helpers
# ----------------------------
def chunked(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            return
        yield batch

def backoff_sleep(attempt):
    # exponential backoff + small jitter
    delay = BASE_BACKOFF * (2 ** (attempt - 1)) + (0.0 if attempt < 3 else 0.5)
    time.sleep(delay)

def date_range_from_days_ago(days_ago_list):
    """Return inclusive YYYY-MM-DD start_date, end_date for Open-Meteo Archive."""
    if not days_ago_list:
        raise ValueError("days_ago_list cannot be empty")
    today_utc = dt.datetime.now(dt.timezone.utc).date()
    dates = [today_utc - dt.timedelta(days=d) for d in days_ago_list]
    start_date = min(dates).isoformat()
    end_date = max(dates).isoformat()
    return start_date, end_date

# ----------------------------
# Historical fetch (daily aggregates)
# ----------------------------
def fetch_historical_daily_batched(regions_items, start_date, end_date):
    """
    regions_items: list[(region_name, lat, lon, commodity)]
    Returns: list of dict records (one per region x day)
    """
    daily_vars = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",          # use "rain_sum" if you want rain only
        "relative_humidity_2m_mean"
    ]
    records_all = []

    for batch in chunked(regions_items, BATCH_SIZE):
        lats = ",".join(str(lat) for (_, lat, _, _) in batch)
        lons = ",".join(str(lon) for (_, _, lon, _) in batch)

        params = {
            "latitude": lats,
            "longitude": lons,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(daily_vars),
            "timezone": "auto"
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(ARCHIVE_API_URL, params=params, timeout=HTTP_TIMEOUT)
                if resp.status_code == 429:
                    # respect Retry-After if present
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        time.sleep(max(1, int(retry_after)))
                    else:
                        backoff_sleep(attempt)
                    continue
                resp.raise_for_status()
                payload = resp.json()

                # Multi-location returns a list of location objects in the same order as inputs
                locations = payload if isinstance(payload, list) else [payload]

                for idx, loc in enumerate(locations):
                    daily = loc.get("daily")
                    if not daily:
                        # still continue with others
                        continue
                    region, lat, lon, commodity = batch[idx]
                    times = daily.get("time", [])
                    tmax = daily.get("temperature_2m_max", [])
                    tmin = daily.get("temperature_2m_min", [])
                    precip = daily.get("precipitation_sum", [])
                    rh = daily.get("relative_humidity_2m_mean", [])
                    for i in range(len(times)):
                        records_all.append({
                            "Type": "HISTORICAL_DAILY",
                            "Region": region,
                            "Commodity": commodity,
                            "Date": times[i],
                            "Max_Temp_C": tmax[i] if i < len(tmax) else None,
                            "Min_Temp_C": tmin[i] if i < len(tmin) else None,
                            "Precipitation_mm": precip[i] if i < len(precip) else None,
                            "Humidity_perc": rh[i] if i < len(rh) else None,
                        })
                break  # success, break retry loop

            except requests.exceptions.HTTPError as e:
                status = resp.status_code if 'resp' in locals() else 'NA'
                if isinstance(status, int) and 500 <= status < 600 and attempt < MAX_RETRIES:
                    backoff_sleep(attempt)
                    continue
                # non-retryable
                print(f"[Archive] HTTP {status} error on batch: {e}")
                break
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES:
                    backoff_sleep(attempt)
                    continue
                print(f"[Archive] Request error on batch: {e}")
                break

    return records_all

# ----------------------------
# Current snapshot (per region, batched)
# ----------------------------
def fetch_current_batched(regions_items):
    """
    Fetch current weather using Open-Meteo Forecast API with current=...
    Returns: list of dict records (one per region)
    """
    # We’ll request all current variables we care about
    current_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "pressure_msl",
        "wind_speed_10m",
        "precipitation",
        "rain",
        "snowfall",
        "weather_code"
    ]
    records_all = []

    for batch in chunked(regions_items, BATCH_SIZE):
        lats = ",".join(str(lat) for (_, lat, _, _) in batch)
        lons = ",".join(str(lon) for (_, _, lon, _) in batch)
        params = {
            "latitude": lats,
            "longitude": lons,
            "current": ",".join(current_vars),
            "timezone": "UTC"
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(FORECAST_API_URL, params=params, timeout=HTTP_TIMEOUT)
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        time.sleep(max(1, int(retry_after)))
                    else:
                        backoff_sleep(attempt)
                    continue
                resp.raise_for_status()
                payload = resp.json()

                # Multi-location current returns list as well
                locations = payload if isinstance(payload, list) else [payload]

                for idx, loc in enumerate(locations):
                    current = loc.get("current")
                    if not current:
                        continue
                    region, lat, lon, commodity = batch[idx]
                    # Open-Meteo returns ISO time; keep a consistent "UTC" suffix like your original
                    iso_time = current.get("time")  # e.g., "2025-10-25T12:00"
                    time_utc = (iso_time + " UTC") if iso_time else None

                    records_all.append({
                        "Type": "CURRENT",
                        "Region": region,
                        "Commodity": commodity,
                        "Date": iso_time.split("T")[0] if iso_time else None,
                        "Time_UTC": time_utc,
                        "Temperature_C": current.get("temperature_2m"),
                        "Feels_Like_C": None,  # Open-Meteo has apparent_temperature if you need it
                        "Humidity_perc": current.get("relative_humidity_2m"),
                        "Pressure_hPa": current.get("pressure_msl"),
                        "Wind_Speed_m/s": current.get("wind_speed_10m"),
                        "Weather_Main": current.get("weather_code"),  # numeric WMO code
                        "Weather_Description": None,                  # map WMO code if desired
                        "Rain_mm/h": current.get("rain"),
                        "Snow_mm/h": current.get("snowfall"),
                    })
                break

            except requests.exceptions.HTTPError as e:
                status = resp.status_code if 'resp' in locals() else 'NA'
                if isinstance(status, int) and 500 <= status < 600 and attempt < MAX_RETRIES:
                    backoff_sleep(attempt)
                    continue
                print(f"[Current] HTTP {status} error on batch: {e}")
                break
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES:
                    backoff_sleep(attempt)
                    continue
                print(f"[Current] Request error on batch: {e}")
                break

    return records_all

# ----------------------------
# S3 writer (unchanged)
# ----------------------------
def write_to_s3_csv(all_weather_data, fetch_type):
    if not all_weather_data:
        print("No weather data collected to write.")
        return

    s3_bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not s3_bucket_name:
        print("S3_BUCKET_NAME environment variable not set. Aborting S3 upload.")
        return

    s3 = boto3.client('s3')

    fieldnames = list(all_weather_data[0].keys())
    timestamp = dt.datetime.now().strftime('%Y%m%d')
    output_filename = f"{fetch_type.lower()}_weather_data_{timestamp}.csv"

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_weather_data)

    try:
        s3.put_object(
            Bucket=s3_bucket_name,
            Key=f"landing/weather_data/{output_filename}",
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        print(f"Successfully uploaded {len(all_weather_data)} records to s3://{s3_bucket_name}/landing/weather_data/{output_filename}")
    except ClientError as e:
        print(f"Failed to upload to S3: {e}")
    except Exception as e:
        print(f"Unexpected S3 upload error: {e}")

# ----------------------------
# Orchestrators (adapted)
# ----------------------------
def fetch_historical_weather_data(days_to_fetch):
    """
    For Open-Meteo Archive, we fetch an inclusive date range (daily aggregates).
    """
    start_date, end_date = date_range_from_days_ago(days_to_fetch)
    items = [(name, lat, lon, commodity) for name, (lat, lon, commodity) in COMMODITY_REGIONS.items()]
    print(f"Archive fetch for {start_date} → {end_date} across {len(items)} locations...")
    all_weather_data = fetch_historical_daily_batched(items, start_date, end_date)
    write_to_s3_csv(all_weather_data, 'HISTORICAL')

def fetch_current_weather_data():
    items = [(name, lat, lon, commodity) for name, (lat, lon, commodity) in COMMODITY_REGIONS.items()]
    print(f"Current fetch across {len(items)} locations...")
    all_weather_data = fetch_current_batched(items)
    write_to_s3_csv(all_weather_data, 'CURRENT')

# ----------------------------
# Lambda entry (unchanged flow)
# ----------------------------
def lambda_handler(event=None, context=None):
    print("Starting weather data collection (Open-Meteo)...")

    default_historical_days = []

    # Normalize event to a dict
    event = event or {}

    # If event is a dict, try to read days_to_fetch, otherwise use default
    if isinstance(event, dict):
        days_to_fetch = event.get('days_to_fetch', default_historical_days)
    else:
        days_to_fetch = default_historical_days

    if days_to_fetch and isinstance(days_to_fetch, list) and len(days_to_fetch) > 0:
        print(f"Historical fetch requested for days ago: {days_to_fetch}")
        fetch_historical_weather_data(days_to_fetch)
    else:
        print("No historical days specified. Running CURRENT weather data fetch.")
        fetch_current_weather_data()

    print("Weather data collection complete.")
    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Weather data processing finished'})
    }

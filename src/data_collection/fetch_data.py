"""
Fetch real PM2.5 measurements from OpenAQ for Indian cities.
Supports:
1. Live OpenAQ API v3 querying when an authenticated API key is supplied.
2. Direct access to OpenAQ's official open data archive on AWS S3 for Indian stations
   (New Delhi, Mumbai, Kolkata, Hyderabad, Chennai) using concurrent downloading.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import requests

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    RAW_DATA_DIR,
    POLLUTANT_NAME,
    STANDARD_UNIT,
    COUNTRY_CODE,
    get_openaq_api_key,
)
from src.utils.logging_utils import get_logger
from src.data_collection.openaq_client import (
    OpenAQClient,
    OpenAQAuthError,
    OpenAQError,
    OpenAQDataNotFoundError,
)

logger = get_logger("fetch_data")

# Documented Indian reference stations in OpenAQ
KNOWN_INDIAN_STATIONS = {
    "Delhi": {"location_id": 8118, "station_name": "New Delhi (US Diplomatic Post)", "city": "Delhi", "lat": 28.63576, "lon": 77.22445},
    "Mumbai": {"location_id": 8039, "station_name": "Mumbai (US Diplomatic Post)", "city": "Mumbai", "lat": 19.07283, "lon": 72.88261},
    "Kolkata": {"location_id": 8172, "station_name": "Kolkata (US Diplomatic Post)", "city": "Kolkata", "lat": 22.56263, "lon": 88.36304},
    "Hyderabad": {"location_id": 8557, "station_name": "Hyderabad (US Diplomatic Post)", "city": "Hyderabad", "lat": 17.38405, "lon": 78.45636},
    "Chennai": {"location_id": 8558, "station_name": "Chennai (US Diplomatic Post)", "city": "Chennai", "lat": 13.08784, "lon": 80.27847},
}

OPENAQ_S3_BASE = "https://openaq-data-archive.s3.amazonaws.com"


def _download_and_parse_day(s3_key: str, st_info: dict, city_name: str) -> list[dict]:
    """Download single daily csv.gz file and return list of PM2.5 observation records."""
    loc_id = st_info["location_id"]
    file_url = f"{OPENAQ_S3_BASE}/{s3_key}"
    records = []
    try:
        res = requests.get(file_url, timeout=12)
        if res.status_code == 200:
            df_day = pd.read_csv(io.BytesIO(res.content), compression="gzip")
            if "parameter" in df_day.columns:
                df_day = df_day[df_day["parameter"].astype(str).str.lower() == POLLUTANT_NAME]

            for _, row in df_day.iterrows():
                val = row.get("value")
                if pd.isna(val) or val is None or float(val) < 0:
                    continue

                ts = str(row.get("datetime"))
                records.append({
                    "station_id": loc_id,
                    "station_name": st_info["station_name"],
                    "city": city_name,
                    "sensor_id": int(row.get("sensors_id", loc_id)),
                    "pollutant": POLLUTANT_NAME,
                    "unit": STANDARD_UNIT,
                    "timestamp_utc": ts,
                    "timestamp_local": ts,
                    "value": float(val),
                    "latitude": float(st_info["lat"]),
                    "longitude": float(st_info["lon"]),
                })
    except Exception as e:
        logger.debug(f"Failed to fetch {file_url}: {e}")
    return records


def fetch_from_openaq_archive(
    cities: list[str] | None = None,
    days_per_station: int = 45,
    output_filename: str = "openaq_pm25_raw.csv",
) -> Path:
    """
    Fetch genuine real historical observations from the official OpenAQ AWS S3 open data repository
    for verified Indian monitoring stations using multi-threading.
    """
    target_cities = [c.capitalize() for c in (cities or list(KNOWN_INDIAN_STATIONS.keys()))]
    logger.info(f"Connecting to official OpenAQ data archive for Indian cities: {target_cities}...")

    all_records = []

    for city_name in target_cities:
        if city_name not in KNOWN_INDIAN_STATIONS:
            logger.warning(f"City '{city_name}' is not in known OpenAQ list. Available: {list(KNOWN_INDIAN_STATIONS.keys())}")
            continue

        st_info = KNOWN_INDIAN_STATIONS[city_name]
        loc_id = st_info["location_id"]
        logger.info(f"Querying OpenAQ archive for {city_name} (ID: {loc_id}, Station: '{st_info['station_name']}')...")

        # Discover keys (check provider=airnow prefix first, then direct locationid)
        keys = []
        for prefix in [
            f"records/csv.gz/provider=airnow/country=in/locationid={loc_id}/year=2024/",
            f"records/csv.gz/locationid={loc_id}/year=2024/",
            f"records/csv.gz/provider=airnow/country=in/locationid={loc_id}/year=2023/",
            f"records/csv.gz/locationid={loc_id}/year=2023/",
        ]:
            list_url = f"{OPENAQ_S3_BASE}/?list-type=2&prefix={prefix}&max-keys=100"
            try:
                r = requests.get(list_url, timeout=10)
                if r.status_code == 200:
                    root = ET.fromstring(r.text)
                    found = [elem.text for elem in root.findall(".//{http://s3.amazonaws.com/doc/2006-03-01/}Key") if elem.text]
                    if found:
                        keys = found
                        break
            except Exception as e:
                logger.debug(f"Prefix query failed: {e}")

        if not keys:
            logger.warning(f"No archive keys found for {city_name}.")
            continue

        selected_keys = keys[-days_per_station:] if len(keys) > days_per_station else keys
        logger.info(f"  Downloading {len(selected_keys)} days of hourly data in parallel for {city_name}...")

        city_records = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_key = {
                executor.submit(_download_and_parse_day, k, st_info, city_name): k
                for k in selected_keys
            }
            for future in as_completed(future_to_key):
                res = future.result()
                if res:
                    city_records.extend(res)

        logger.info(f"  Successfully loaded {len(city_records)} hourly PM2.5 observations for {city_name}.")
        all_records.extend(city_records)

    if not all_records:
        raise OpenAQDataNotFoundError("No records could be retrieved from the OpenAQ archive.")

    df = pd.DataFrame(all_records)
    output_path = RAW_DATA_DIR / output_filename
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully collected {len(df)} total genuine OpenAQ observations across {len(target_cities)} cities.")
    logger.info(f"Saved raw dataset to: {output_path}")
    return output_path


def parse_openaq_hourly_record(record: dict, station_meta: dict, sensor_meta: dict) -> dict | None:
    """
    Extract standardized fields from an OpenAQ v3 hourly record.
    """
    try:
        val = record.get("value")
        if val is None or float(val) < 0:
            return None

        period = record.get("period", {}) or {}
        dt_to = period.get("datetimeTo", {}) or {}
        dt_from = period.get("datetimeFrom", {}) or {}

        utc_timestamp_str = dt_to.get("utc") or dt_from.get("utc") or record.get("datetime")
        local_timestamp_str = dt_to.get("local") or dt_from.get("local")

        param = record.get("parameter", {}) or sensor_meta.get("parameter", {})

        return {
            "station_id": station_meta.get("id"),
            "station_name": station_meta.get("name"),
            "city": station_meta.get("locality") or station_meta.get("city") or "India",
            "sensor_id": sensor_meta.get("id"),
            "pollutant": param.get("name", POLLUTANT_NAME),
            "unit": param.get("units", STANDARD_UNIT),
            "timestamp_utc": utc_timestamp_str,
            "timestamp_local": local_timestamp_str,
            "value": float(val),
            "latitude": station_meta.get("coordinates", {}).get("latitude"),
            "longitude": station_meta.get("coordinates", {}).get("longitude"),
        }
    except Exception as e:
        logger.debug(f"Could not parse record {record}: {e}")
        return None


def extract_city_from_station(station: dict) -> str:
    """Extract standard Indian city name from station metadata or name string."""
    locality = str(station.get("locality") or "").strip()
    name = str(station.get("name") or "").strip()
    combined = f"{locality} {name}".lower()

    city_map = {
        "delhi": "Delhi",
        "new delhi": "Delhi",
        "mumbai": "Mumbai",
        "kolkata": "Kolkata",
        "hyderabad": "Hyderabad",
        "chennai": "Chennai",
        "bengaluru": "Bengaluru",
        "bangalore": "Bengaluru",
        "gurugram": "Gurugram",
        "noida": "Noida",
        "agra": "Agra",
        "lucknow": "Lucknow",
        "pune": "Pune",
        "kanpur": "Kanpur",
    }
    for key, city in city_map.items():
        if key in combined:
            return city
    return locality.capitalize() if locality else "India"


def fetch_from_live_api(
    client: OpenAQClient,
    cities: list[str] | None = None,
    days_history: int = 45,
    output_filename: str = "openaq_pm25_raw.csv",
) -> Path:
    """
    Discover Indian stations via authenticated OpenAQ API v3, fetch active PM2.5 sensor observations,
    and save them locally.
    """
    logger.info("Connecting to live OpenAQ API v3 for Indian monitoring stations...")
    locations = client.get_locations(country=COUNTRY_CODE, limit=100)
    logger.info(f"Retrieved {len(locations)} monitoring stations in India.")

    if not locations:
        raise OpenAQDataNotFoundError("No monitoring locations returned for India.")

    target_cities = [c.lower() for c in (cities or ["Delhi", "Mumbai", "Kolkata", "Hyderabad", "Chennai"])]

    # Select stations matching requested cities
    matched_stations = []
    for loc in locations:
        resolved_city = extract_city_from_station(loc)
        if resolved_city.lower() in target_cities:
            matched_stations.append((resolved_city, loc))

    logger.info(f"Identified {len(matched_stations)} monitoring stations matching target cities.")
    if not matched_stations:
        matched_stations = [(extract_city_from_station(l), l) for l in locations[:15]]

    all_records = []
    for city_name, station in matched_stations[:10]:
        station_id = station.get("id")
        station_name = station.get("name")
        logger.info(f"Inspecting station '{station_name}' (ID: {station_id}, City: {city_name})...")
        pm25_sensor = client.find_pm25_sensor(station_id)
        if not pm25_sensor:
            continue

        sensor_id = pm25_sensor.get("id")
        logger.info(f"  Found live PM2.5 sensor ID: {sensor_id}. Fetching hourly records...")
        
        try:
            records = client.get_sensor_hourly_measurements(sensor_id=sensor_id, limit=500)
            station_count = 0
            for rec in records:
                parsed = parse_openaq_hourly_record(rec, station, pm25_sensor)
                if parsed:
                    parsed["city"] = city_name
                    all_records.append(parsed)
                    station_count += 1
            logger.info(f"  Retrieved {station_count} hourly records for '{station_name}'.")
        except Exception as e:
            logger.warning(f"  Could not retrieve records for sensor {sensor_id}: {e}")

    if len(all_records) < 50:
        logger.warning("Live API query returned fewer than 50 observations. Supplementing with OpenAQ archive records...")
        return fetch_from_openaq_archive(cities=cities, days_per_station=days_history, output_filename=output_filename)

    df = pd.DataFrame(all_records)
    output_path = RAW_DATA_DIR / output_filename
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} live OpenAQ records to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fetch real PM2.5 data from OpenAQ for Indian cities.")
    parser.add_argument(
        "--cities",
        nargs="+",
        default=["Delhi", "Mumbai", "Kolkata", "Hyderabad", "Chennai"],
        help="List of Indian cities to query",
    )
    parser.add_argument("--days", type=int, default=45, help="Number of days to fetch")
    parser.add_argument("--output", type=str, default="openaq_pm25_raw.csv", help="Output CSV filename")
    parser.add_argument("--force-live", action="store_true", help="Force querying live API even if key is missing")
    args = parser.parse_args()

    api_key = get_openaq_api_key()

    if api_key and not args.force_live:
        logger.info("Detected OPENAQ_API_KEY. Using live OpenAQ API v3 client.")
        client = OpenAQClient(api_key=api_key)
        try:
            fetch_from_live_api(client, cities=args.cities, days_history=args.days, output_filename=args.output)
            return
        except OpenAQAuthError as e:
            logger.warning(f"Live API authentication failed ({e}). Falling back to official OpenAQ archive.")
        except Exception as e:
            logger.warning(f"Live API query failed ({e}). Falling back to official OpenAQ archive.")

    logger.info("Connecting to OpenAQ official open archive for Indian stations...")
    fetch_from_openaq_archive(cities=args.cities, days_per_station=args.days, output_filename=args.output)


if __name__ == "__main__":
    main()

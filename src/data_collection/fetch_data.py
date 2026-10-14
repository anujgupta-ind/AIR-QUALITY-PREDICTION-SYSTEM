"""
Fetch real PM2.5 measurements from OpenAQ for Indian cities.
Supports:
1. Live OpenAQ API v3 querying for active Indian government DPCC/CPCB monitoring stations
   (Anand Vihar, R K Puram, Punjabi Bagh, Zoo Park Hyderabad, Manali Chennai, etc.)
   using an authenticated API key.
2. Direct access to OpenAQ's official open data archive on AWS S3 for historical Indian stations
   when an API key is not present.
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

# Active Reference Ground Stations in OpenAQ API v3 with real-time reporting
ACTIVE_LIVE_INDIAN_STATIONS = [
    {
        "city": "Delhi",
        "station_id": 235,
        "station_name": "Anand Vihar, New Delhi - DPCC",
        "pm25_sensor_id": 12235610,
        "latitude": 28.646835,
        "longitude": 77.316032,
    },
    {
        "city": "Delhi",
        "station_id": 17,
        "station_name": "R K Puram, Delhi - DPCC",
        "pm25_sensor_id": 12234787,
        "latitude": 28.563262,
        "longitude": 77.186937,
    },
    {
        "city": "Delhi",
        "station_id": 50,
        "station_name": "Punjabi Bagh, Delhi - DPCC",
        "pm25_sensor_id": 12234796,
        "latitude": 28.674045,
        "longitude": 77.131023,
    },
    {
        "city": "Hyderabad",
        "station_id": 407,
        "station_name": "Zoo Park, Hyderabad - TSPCB",
        "pm25_sensor_id": 12235583,
        "latitude": 17.349698,
        "longitude": 78.451437,
    },
    {
        "city": "Chennai",
        "station_id": 2586,
        "station_name": "Manali, Chennai - CPCB",
        "pm25_sensor_id": 12235653,
        "latitude": 13.164344,
        "longitude": 80.262846,
    },
]

# Historical Indian reference stations in OpenAQ S3 Archive
KNOWN_INDIAN_STATIONS = {
    "Delhi": {"location_id": 8118, "station_name": "New Delhi (US Diplomatic Post)", "city": "Delhi", "lat": 28.63576, "lon": 77.22445},
    "Mumbai": {"location_id": 8039, "station_name": "Mumbai (US Diplomatic Post)", "city": "Mumbai", "lat": 19.07283, "lon": 72.88261},
    "Kolkata": {"location_id": 8172, "station_name": "Kolkata (US Diplomatic Post)", "city": "Kolkata", "lat": 22.56263, "lon": 88.36304},
    "Hyderabad": {"location_id": 8557, "station_name": "Hyderabad (US Diplomatic Post)", "city": "Hyderabad", "lat": 17.38405, "lon": 78.45636},
    "Chennai": {"location_id": 8558, "station_name": "Chennai (US Diplomatic Post)", "city": "Chennai", "lat": 13.08784, "lon": 80.27847},
}

OPENAQ_S3_BASE = "https://openaq-data-archive.s3.amazonaws.com"


def fetch_from_live_api(
    client: OpenAQClient,
    cities: list[str] | None = None,
    days_history: int = 20,
    output_filename: str = "openaq_pm25_raw.csv",
) -> Path:
    """
    Query real-time live OpenAQ API v3 endpoints with date filtering to retrieve
    the most recent continuous hourly measurements leading up to today.
    """
    logger.info("Connecting to live OpenAQ API v3 with active credentials...")
    
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=days_history)
    datetime_from_str = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(f"Targeting real-time observation window from {datetime_from_str} onwards...")

    city_filter = [c.lower() for c in cities] if cities else []
    target_stations = []
    for st in ACTIVE_LIVE_INDIAN_STATIONS:
        if city_filter:
            if st["city"].lower() in city_filter:
                target_stations.append(st)
        else:
            target_stations.append(st)

    all_records = []

    for station in target_stations:
        st_name = station["station_name"]
        st_city = station["city"]
        sensor_id = station["pm25_sensor_id"]
        loc_id = station["station_id"]

        logger.info(f"Querying live sensor {sensor_id} for '{st_name}' ({st_city})...")
        
        try:
            # Query hourly aggregates starting from datetime_from_str
            records = client.get_sensor_hourly_measurements(
                sensor_id=sensor_id,
                datetime_from=datetime_from_str,
                limit=1000,
            )
            count = 0
            for rec in records:
                val = rec.get("value")
                if val is None or float(val) < 0:
                    continue

                period = rec.get("period", {}) or {}
                dt_to = period.get("datetimeTo", {}) or {}
                dt_from = period.get("datetimeFrom", {}) or {}
                utc_str = dt_to.get("utc") or dt_from.get("utc") or rec.get("datetime")
                loc_str = dt_to.get("local") or dt_from.get("local")

                all_records.append({
                    "station_id": loc_id,
                    "station_name": st_name,
                    "city": st_city,
                    "sensor_id": sensor_id,
                    "pollutant": POLLUTANT_NAME,
                    "unit": STANDARD_UNIT,
                    "timestamp_utc": utc_str,
                    "timestamp_local": loc_str,
                    "value": float(val),
                    "latitude": station["latitude"],
                    "longitude": station["longitude"],
                })
                count += 1
            logger.info(f"  Retrieved {count} live hourly records for '{st_name}'.")

        except Exception as e:
            logger.warning(f"  Error querying sensor {sensor_id}: {e}")

    if not all_records:
        logger.warning("No records returned from primary live sensors. Falling back to S3 archive.")
        return fetch_from_openaq_archive(cities=cities, days_per_station=days_history, output_filename=output_filename)

    df = pd.DataFrame(all_records)
    output_path = RAW_DATA_DIR / output_filename
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully collected {len(df)} live observations across {len(target_stations)} Indian stations.")
    logger.info(f"Saved live dataset to: {output_path}")
    return output_path


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


def main():
    parser = argparse.ArgumentParser(description="Fetch real PM2.5 data from OpenAQ for Indian cities.")
    parser.add_argument(
        "--cities",
        nargs="+",
        default=["Delhi", "Hyderabad", "Chennai"],
        help="List of Indian cities to query",
    )
    parser.add_argument("--days", type=int, default=20, help="Number of past days to fetch")
    parser.add_argument("--output", type=str, default="openaq_pm25_raw.csv", help="Output CSV filename")
    parser.add_argument("--force-archive", action="store_true", help="Force querying S3 archive instead of live API")
    args = parser.parse_args()

    api_key = get_openaq_api_key()

    if api_key and not args.force_archive:
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

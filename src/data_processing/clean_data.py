"""
Data Cleaning and Preprocessing Module
Cleans raw OpenAQ air quality measurements, converts timezones to IST,
removes duplicates, validates numerical ranges, handles missingness,
and generates a data-quality audit report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, POLLUTANT_NAME, STANDARD_UNIT
from src.utils.logging_utils import get_logger

logger = get_logger("clean_data")


def clean_air_quality_data(
    raw_csv_path: Path | str,
    output_filename: str = "cleaned_pm25_hourly.csv",
    quality_report_filename: str = "data_quality_report.json",
) -> tuple[pd.DataFrame, dict]:
    """
    Load raw measurements, perform validation & cleaning, and output cleaned data and audit report.
    """
    raw_path = Path(raw_csv_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")

    logger.info(f"Loading raw dataset from {raw_path}...")
    df_raw = pd.read_csv(raw_path)
    initial_record_count = len(df_raw)
    logger.info(f"Initial raw record count: {initial_record_count}")

    # 1. Standardize column names
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Required columns check
    expected_cols = ["station_id", "station_name", "city", "timestamp_utc", "value"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from raw dataset.")

    # 2. Datetime parsing: parse UTC and compute Indian Standard Time (IST = UTC+05:30)
    # OpenAQ timestamps can be ISO format like "2024-01-01T00:00:00Z" or "2024-01-01 00:00:00"
    logger.info("Parsing timestamps and converting UTC to Indian Standard Time (IST)...")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    
    # Drop rows where timestamp could not be parsed
    invalid_dates_count = int(df["timestamp_utc"].isna().sum())
    if invalid_dates_count > 0:
        logger.warning(f"Dropping {invalid_dates_count} records with unparseable timestamps.")
        df = df.dropna(subset=["timestamp_utc"])

    # Convert to Indian Standard Time (Asia/Kolkata timezone)
    df["timestamp_ist"] = df["timestamp_utc"].dt.tz_convert("Asia/Kolkata")
    
    # Create an un-timezone-naive datetime string column for convenience
    df["datetime_ist"] = df["timestamp_ist"].dt.tz_localize(None)

    # 3. Numeric conversion & sanity bounds for PM2.5
    logger.info("Validating pollutant numerical measurements...")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    missing_value_count = int(df["value"].isna().sum())

    # Realistic physical bounds for PM2.5: 0 to 1000 µg/m³
    # Values < 0 are sensor errors; values > 1000 are extreme outliers/sensor errors
    out_of_bounds = df[(df["value"] < 0) | (df["value"] > 1000)]
    out_of_bounds_count = len(out_of_bounds)
    if out_of_bounds_count > 0:
        logger.warning(f"Detected {out_of_bounds_count} impossible/out-of-bound PM2.5 readings (<0 or >1000). Removing.")
        df = df[(df["value"] >= 0) & (df["value"] <= 1000)]

    df = df.dropna(subset=["value"])

    # 4. Remove duplicate observations
    # A duplicate is same station, same pollutant, and same timestamp
    initial_before_dedup = len(df)
    df = df.drop_duplicates(subset=["station_id", "timestamp_utc"], keep="first")
    duplicates_removed = initial_before_dedup - len(df)
    logger.info(f"Removed {duplicates_removed} duplicate observations.")

    # 5. Sort chronologically per station
    df = df.sort_values(by=["station_id", "timestamp_utc"]).reset_index(drop=True)

    # 6. Metadata standardization
    if "pollutant" not in df.columns:
        df["pollutant"] = POLLUTANT_NAME
    else:
        df["pollutant"] = df["pollutant"].fillna(POLLUTANT_NAME).astype(str).str.lower()

    if "unit" not in df.columns:
        df["unit"] = STANDARD_UNIT
    else:
        df["unit"] = STANDARD_UNIT  # Standardize all to µg/m³

    # 7. Check for continuous hourly regularity and quantify missing hourly observations
    city_summaries = {}
    total_expected_hours = 0
    total_actual_hours = 0

    for city_name, group in df.groupby("city"):
        station_name = group["station_name"].iloc[0]
        st_id = int(group["station_id"].iloc[0])
        min_ts = group["datetime_ist"].min()
        max_ts = group["datetime_ist"].max()
        
        # Expected hours in range
        hours_in_span = int(np.ceil((max_ts - min_ts).total_seconds() / 3600.0)) + 1
        actual_hours = len(group)
        missing_hours = max(0, hours_in_span - actual_hours)
        missing_pct = round((missing_hours / hours_in_span) * 100.0, 2) if hours_in_span > 0 else 0.0

        total_expected_hours += hours_in_span
        total_actual_hours += actual_hours

        city_summaries[city_name] = {
            "station_id": st_id,
            "station_name": station_name,
            "record_count": actual_hours,
            "start_time_ist": str(min_ts),
            "end_time_ist": str(max_ts),
            "expected_hourly_span": hours_in_span,
            "missing_hours": missing_hours,
            "missing_percentage": missing_pct,
            "pm25_min": round(float(group["value"].min()), 2),
            "pm25_mean": round(float(group["value"].mean()), 2),
            "pm25_max": round(float(group["value"].max()), 2),
        }

    overall_missing_pct = (
        round(((total_expected_hours - total_actual_hours) / total_expected_hours) * 100.0, 2)
        if total_expected_hours > 0
        else 0.0
    )

    quality_report = {
        "dataset_name": "Indian Cities PM2.5 Cleaned Dataset",
        "data_source": "OpenAQ Open Air Quality Archive",
        "total_raw_records": initial_record_count,
        "valid_cleaned_records": len(df),
        "invalid_timestamps_dropped": invalid_dates_count,
        "missing_values_dropped": missing_value_count,
        "out_of_bounds_dropped": out_of_bounds_count,
        "duplicates_removed": duplicates_removed,
        "overall_temporal_completeness_pct": round(100.0 - overall_missing_pct, 2),
        "overall_missing_hourly_pct": overall_missing_pct,
        "city_count": int(df["city"].nunique()),
        "available_cities": sorted(df["city"].unique().tolist()),
        "city_summaries": city_summaries,
    }

    # Save cleaned data
    output_path = PROCESSED_DATA_DIR / output_filename
    report_path = PROCESSED_DATA_DIR / quality_report_filename

    # Save DataFrame with clean columns
    save_cols = [
        "station_id",
        "station_name",
        "city",
        "sensor_id",
        "pollutant",
        "unit",
        "timestamp_utc",
        "timestamp_ist",
        "datetime_ist",
        "value",
        "latitude",
        "longitude",
    ]
    # Keep only available columns from save_cols
    cols_to_write = [c for c in save_cols if c in df.columns]
    df[cols_to_write].to_csv(output_path, index=False)
    logger.info(f"Saved cleaned dataset to: {output_path} ({len(df)} records)")

    # Save quality report JSON
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)
    logger.info(f"Saved data quality report to: {report_path}")

    return df, quality_report


def main():
    parser = argparse.ArgumentParser(description="Clean and preprocess raw OpenAQ PM2.5 data.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(RAW_DATA_DIR / "openaq_pm25_raw.csv"),
        help="Path to raw CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cleaned_pm25_hourly.csv",
        help="Output CSV filename in data/processed/",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="data_quality_report.json",
        help="Output quality report filename in data/processed/",
    )
    args = parser.parse_args()

    df, report = clean_air_quality_data(
        raw_csv_path=args.input,
        output_filename=args.output,
        quality_report_filename=args.report,
    )
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT SUMMARY:")
    print(f"Total Raw Records: {report['total_raw_records']}")
    print(f"Valid Cleaned Records: {report['valid_cleaned_records']}")
    print(f"Duplicates Removed: {report['duplicates_removed']}")
    print(f"Temporal Completeness: {report['overall_temporal_completeness_pct']}%")
    print(f"Cities Available: {', '.join(report['available_cities'])}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

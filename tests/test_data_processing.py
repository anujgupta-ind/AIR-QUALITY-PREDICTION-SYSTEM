"""
Unit tests for data cleaning and preprocessing pipeline.
"""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.data_processing.clean_data import clean_air_quality_data


@pytest.fixture
def sample_raw_dataframe():
    """Create a realistic sample DataFrame containing duplicates, invalid values, and UTC dates."""
    data = [
        # Normal observation in Delhi
        {
            "station_id": 8118,
            "station_name": "New Delhi",
            "city": "Delhi",
            "sensor_id": 101,
            "pollutant": "pm25",
            "unit": "µg/m³",
            "timestamp_utc": "2024-03-01T00:00:00Z",
            "timestamp_local": "2024-03-01T05:30:00+05:30",
            "value": 85.5,
            "latitude": 28.63,
            "longitude": 77.22,
        },
        # Next hour in Delhi
        {
            "station_id": 8118,
            "station_name": "New Delhi",
            "city": "Delhi",
            "sensor_id": 101,
            "pollutant": "pm25",
            "unit": "µg/m³",
            "timestamp_utc": "2024-03-01T01:00:00Z",
            "timestamp_local": "2024-03-01T06:30:00+05:30",
            "value": 92.0,
            "latitude": 28.63,
            "longitude": 77.22,
        },
        # Duplicate of the previous observation
        {
            "station_id": 8118,
            "station_name": "New Delhi",
            "city": "Delhi",
            "sensor_id": 101,
            "pollutant": "pm25",
            "unit": "µg/m³",
            "timestamp_utc": "2024-03-01T01:00:00Z",
            "timestamp_local": "2024-03-01T06:30:00+05:30",
            "value": 92.0,
            "latitude": 28.63,
            "longitude": 77.22,
        },
        # Impossible negative value
        {
            "station_id": 8118,
            "station_name": "New Delhi",
            "city": "Delhi",
            "sensor_id": 101,
            "pollutant": "pm25",
            "unit": "µg/m³",
            "timestamp_utc": "2024-03-01T02:00:00Z",
            "timestamp_local": "2024-03-01T07:30:00+05:30",
            "value": -99.0,
            "latitude": 28.63,
            "longitude": 77.22,
        },
        # Extreme impossible spike
        {
            "station_id": 8118,
            "station_name": "New Delhi",
            "city": "Delhi",
            "sensor_id": 101,
            "pollutant": "pm25",
            "unit": "µg/m³",
            "timestamp_utc": "2024-03-01T03:00:00Z",
            "timestamp_local": "2024-03-01T08:30:00+05:30",
            "value": 5500.0,
            "latitude": 28.63,
            "longitude": 77.22,
        },
        # Valid observation in Mumbai
        {
            "station_id": 8039,
            "station_name": "Mumbai",
            "city": "Mumbai",
            "sensor_id": 201,
            "pollutant": "pm25",
            "unit": "µg/m³",
            "timestamp_utc": "2024-03-01T00:00:00Z",
            "timestamp_local": "2024-03-01T05:30:00+05:30",
            "value": 42.0,
            "latitude": 19.07,
            "longitude": 72.88,
        },
    ]
    return pd.DataFrame(data)


def test_clean_air_quality_data_pipeline(sample_raw_dataframe):
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_csv = Path(tmpdir) / "raw.csv"
        out_csv = Path(tmpdir) / "cleaned.csv"
        report_json = Path(tmpdir) / "report.json"

        sample_raw_dataframe.to_csv(raw_csv, index=False)

        df_clean, report = clean_air_quality_data(
            raw_csv_path=raw_csv,
            output_filename=str(out_csv),
            quality_report_filename=str(report_json),
        )

        # 6 original rows: 1 duplicate removed, 1 negative dropped, 1 extreme spike dropped -> 3 valid left
        assert len(df_clean) == 3
        assert report["total_raw_records"] == 6
        assert report["duplicates_removed"] == 1
        assert report["out_of_bounds_dropped"] == 2
        assert report["valid_cleaned_records"] == 3

        # Check IST timezone conversion: 00:00 UTC should be 05:30 IST
        first_row = df_clean[df_clean["timestamp_utc"].astype(str).str.contains("2024-03-01 00:00:00")].iloc[0]
        assert str(first_row["datetime_ist"]) == "2024-03-01 05:30:00"

        # Check values are strictly non-negative
        assert (df_clean["value"] >= 0).all()
        assert (df_clean["value"] <= 1000).all()

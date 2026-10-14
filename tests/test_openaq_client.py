"""
Unit tests for OpenAQ API client using mock fixtures.
Note: These fixtures are strictly for automated unit testing of network parsing and error handling.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.data_collection.openaq_client import (
    OpenAQClient,
    OpenAQAuthError,
    OpenAQDataNotFoundError,
    OpenAQError,
)
from src.data_collection.fetch_data import parse_openaq_hourly_record


def test_missing_api_key_raises_auth_error():
    client = OpenAQClient(api_key=None)
    # Patch get_openaq_api_key to return None
    with patch("src.data_collection.openaq_client.get_openaq_api_key", return_value=None):
        client.api_key = None
        with pytest.raises(OpenAQAuthError) as exc_info:
            client.get_locations()
        assert "OpenAQ API Key is missing" in str(exc_info.value)


def test_request_headers_include_api_key():
    client = OpenAQClient(api_key="test_dummy_key_12345")
    headers = client._get_headers()
    assert headers["X-API-Key"] == "test_dummy_key_12345"
    assert headers["Accept"] == "application/json"


@patch("requests.Session.get")
def test_get_locations_success(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {"id": 8118, "name": "Delhi Central Station", "locality": "Delhi", "country": {"code": "IN"}}
        ]
    }
    mock_get.return_value = mock_resp

    client = OpenAQClient(api_key="test_key")
    locations = client.get_locations(country="IN")
    assert len(locations) == 1
    assert locations[0]["id"] == 8118
    assert locations[0]["locality"] == "Delhi"


@patch("requests.Session.get")
def test_401_raises_auth_error(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized key"
    mock_get.return_value = mock_resp

    client = OpenAQClient(api_key="invalid_key")
    with pytest.raises(OpenAQAuthError):
        client.get_locations()


@patch("requests.Session.get")
def test_find_pm25_sensor_matches_correct_sensor(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {"id": 101, "parameter": {"name": "co", "units": "ppm"}},
            {"id": 102, "parameter": {"name": "pm25", "units": "µg/m³"}},
            {"id": 103, "parameter": {"name": "no2", "units": "ppm"}},
        ]
    }
    mock_get.return_value = mock_resp

    client = OpenAQClient(api_key="test_key")
    sensor = client.find_pm25_sensor(location_id=8118)
    assert sensor is not None
    assert sensor["id"] == 102
    assert sensor["parameter"]["name"] == "pm25"


def test_parse_openaq_hourly_record():
    mock_station = {
        "id": 8118,
        "name": "Anand Vihar, Delhi",
        "locality": "Delhi",
        "coordinates": {"latitude": 28.65, "longitude": 77.31},
    }
    mock_sensor = {
        "id": 555,
        "parameter": {"name": "pm25", "units": "µg/m³"},
    }
    mock_record = {
        "period": {
            "datetimeFrom": {"utc": "2026-03-01T00:00:00Z", "local": "2026-03-01T05:30:00+05:30"},
            "datetimeTo": {"utc": "2026-03-01T01:00:00Z", "local": "2026-03-01T06:30:00+05:30"},
        },
        "value": 85.4,
    }

    parsed = parse_openaq_hourly_record(mock_record, mock_station, mock_sensor)
    assert parsed is not None
    assert parsed["station_id"] == 8118
    assert parsed["station_name"] == "Anand Vihar, Delhi"
    assert parsed["city"] == "Delhi"
    assert parsed["sensor_id"] == 555
    assert parsed["pollutant"] == "pm25"
    assert parsed["unit"] == "µg/m³"
    assert parsed["value"] == 85.4
    assert parsed["timestamp_utc"] == "2026-03-01T01:00:00Z"

"""
OpenAQ API v3 Client
Handles authentication, pagination, rate limits, error reporting, and endpoint interaction.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests

from src.utils.config import (
    OPENAQ_API_BASE_URL,
    COUNTRY_CODE,
    POLLUTANT_NAME,
    get_openaq_api_key,
)
from src.utils.logging_utils import get_logger

logger = get_logger("openaq_client")


class OpenAQError(Exception):
    """Base exception for OpenAQ operations."""
    pass


class OpenAQAuthError(OpenAQError):
    """Raised when authentication fails or API key is missing."""
    pass


class OpenAQDataNotFoundError(OpenAQError):
    """Raised when no data or sensors exist for the requested entity."""
    pass


class OpenAQClient:
    """
    Client for OpenAQ API v3.
    """

    def __init__(self, api_key: str | None = None, base_url: str = OPENAQ_API_BASE_URL, timeout: int = 30):
        self.api_key = api_key or get_openaq_api_key()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _get_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise OpenAQAuthError(
                "OpenAQ API Key is missing!\n"
                "To resolve:\n"
                "1. Register for a free account at: https://explore.openaq.org/register\n"
                "2. Copy your API Key from your profile at https://explore.openaq.org/account\n"
                "3. Set OPENAQ_API_KEY in your .env file or environment variable.\n"
                "Example: OPENAQ_API_KEY=your_key_here"
            )
        return {
            "X-API-Key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "AirQualityPredictionSystem/1.0",
        }

    def _request(self, endpoint: str, params: dict[str, Any] | None = None, retries: int = 3) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()

        for attempt in range(1, retries + 1):
            try:
                response = self.session.get(url, headers=headers, params=params, timeout=self.timeout)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    raise OpenAQAuthError(
                        f"Authentication failed (HTTP 401). Your API key may be invalid or expired. Response: {response.text}"
                    )
                elif response.status_code == 404:
                    raise OpenAQDataNotFoundError(
                        f"Endpoint or resource not found (HTTP 404): {url} with params {params}"
                    )
                elif response.status_code == 429:
                    wait_time = attempt * 2
                    logger.warning(f"Rate limited (HTTP 429). Retrying in {wait_time}s (attempt {attempt}/{retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise OpenAQError(
                        f"OpenAQ API request failed with status code {response.status_code}: {response.text}"
                    )
            except requests.exceptions.Timeout as e:
                if attempt == retries:
                    raise OpenAQError(f"Request to {url} timed out after {retries} attempts.") from e
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                if attempt == retries:
                    raise OpenAQError(f"Network error while connecting to OpenAQ: {e}") from e
                time.sleep(1)

        raise OpenAQError(f"Failed to fetch data from {url} after {retries} attempts.")

    def get_countries(self) -> list[dict[str, Any]]:
        """Fetch list of countries to verify ID mapping."""
        data = self._request("/countries", params={"limit": 100})
        return data.get("results", [])

    def get_locations(self, country: str = COUNTRY_CODE, limit: int = 100, page: int = 1) -> list[dict[str, Any]]:
        """
        Fetch monitoring locations for a country (e.g. India / 'IN').
        """
        # OpenAQ v3 supports iso filter
        params = {"iso": country, "limit": limit, "page": page}
        data = self._request("/locations", params=params)
        return data.get("results", [])

    def get_location_details(self, location_id: int) -> dict[str, Any]:
        """
        Fetch details for a specific monitoring location.
        """
        data = self._request(f"/locations/{location_id}")
        results = data.get("results", [])
        if not results:
            raise OpenAQDataNotFoundError(f"Location ID {location_id} has no results.")
        return results[0]

    def get_location_sensors(self, location_id: int) -> list[dict[str, Any]]:
        """
        Retrieve sensors available at a location.
        """
        data = self._request(f"/locations/{location_id}/sensors")
        return data.get("results", [])

    def find_pm25_sensor(self, location_id: int) -> dict[str, Any] | None:
        """
        Find the sensor measuring PM2.5 at the given location.
        """
        sensors = self.get_location_sensors(location_id)
        for s in sensors:
            param = s.get("parameter", {})
            param_name = str(param.get("name", "")).lower()
            if param_name == POLLUTANT_NAME:
                return s
        return None

    def get_sensor_hourly_measurements(
        self,
        sensor_id: int,
        datetime_from: str | None = None,
        datetime_to: str | None = None,
        limit: int = 1000,
        page: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Retrieve hourly aggregated measurements for a specific sensor.
        Endpoint: /v3/sensors/{sensor_id}/hours
        """
        params: dict[str, Any] = {"limit": limit, "page": page}
        if datetime_from:
            params["datetime_from"] = datetime_from
        if datetime_to:
            params["datetime_to"] = datetime_to

        data = self._request(f"/sensors/{sensor_id}/hours", params=params)
        return data.get("results", [])

    def get_sensor_recent_measurements(
        self,
        sensor_id: int,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Retrieve recent measurements for a sensor.
        """
        params = {"limit": limit}
        data = self._request(f"/sensors/{sensor_id}/measurements", params=params)
        return data.get("results", [])

    def get_location_latest(self, location_id: int) -> list[dict[str, Any]]:
        """
        Retrieve latest measurements for all sensors at a location.
        """
        data = self._request(f"/locations/{location_id}/latest")
        return data.get("results", [])

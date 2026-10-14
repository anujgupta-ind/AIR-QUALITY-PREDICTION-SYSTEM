"""
Unit tests for pollution alert logic and threshold boundaries.
"""

from datetime import datetime, timezone, timedelta
import pytest

from src.alerts.pollution_alerts import (
    get_pm25_alert_level,
    create_pollution_alert,
    batch_classify_forecast,
)


def test_alert_threshold_boundaries():
    # Good: 0 - 30
    assert get_pm25_alert_level(0.0)["category"] == "Good"
    assert get_pm25_alert_level(15.0)["category"] == "Good"
    assert get_pm25_alert_level(30.0)["category"] == "Good"

    # Satisfactory: 31 - 60
    assert get_pm25_alert_level(30.1)["category"] == "Satisfactory"
    assert get_pm25_alert_level(45.0)["category"] == "Satisfactory"
    assert get_pm25_alert_level(60.0)["category"] == "Satisfactory"

    # Moderate: 61 - 90
    assert get_pm25_alert_level(60.1)["category"] == "Moderate"
    assert get_pm25_alert_level(75.0)["category"] == "Moderate"
    assert get_pm25_alert_level(90.0)["category"] == "Moderate"

    # Poor: 91 - 120
    assert get_pm25_alert_level(90.1)["category"] == "Poor"
    assert get_pm25_alert_level(105.0)["category"] == "Poor"
    assert get_pm25_alert_level(120.0)["category"] == "Poor"

    # Very Poor: 121 - 250
    assert get_pm25_alert_level(120.1)["category"] == "Very Poor"
    assert get_pm25_alert_level(180.0)["category"] == "Very Poor"
    assert get_pm25_alert_level(250.0)["category"] == "Very Poor"

    # Severe: > 250
    assert get_pm25_alert_level(250.1)["category"] == "Severe"
    assert get_pm25_alert_level(350.0)["category"] == "Severe"
    assert get_pm25_alert_level(1200.0)["category"] == "Severe"


def test_alert_type_distinction():
    # Measured Alert
    measured_alert = create_pollution_alert(
        value=45.0,
        timestamp="2024-04-01 10:00:00",
        station_name="Anand Vihar",
        city="Delhi",
        is_forecast=False,
    )
    assert measured_alert["alert_type"] == "MEASURED_ALERT"
    assert not measured_alert["is_forecast"]
    assert "Ground Station" in measured_alert["title"]

    # Forecasted Alert
    forecast_alert = create_pollution_alert(
        value=115.0,
        timestamp="2024-04-01 11:00:00",
        station_name="Anand Vihar",
        city="Delhi",
        is_forecast=True,
    )
    assert forecast_alert["alert_type"] == "PREDICTED_ALERT"
    assert forecast_alert["is_forecast"]
    assert "Forecast Alert" in forecast_alert["title"]
    assert forecast_alert["category"] == "Poor"


def test_stale_data_warning():
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(hours=10)

    alert = create_pollution_alert(
        value=50.0,
        timestamp=str(old_time),
        station_name="Delhi Station",
        city="Delhi",
        is_forecast=False,
        observation_datetime_utc=old_time,
    )
    assert alert["is_stale"] is True
    assert "Data Freshness Warning" in alert["stale_warning"]

"""
Unit tests for the 24-hour recursive forecasting module.
"""

import pandas as pd
import pytest

from src.modeling.forecast import generate_24h_recursive_forecast


def test_forecast_insufficient_history_raises_value_error():
    # Only 10 hours of history
    short_history = [25.0] * 10
    with pytest.raises(ValueError) as exc:
        generate_24h_recursive_forecast(
            city="Delhi",
            recent_observations=short_history,
            horizon_hours=24,
        )
    assert "Insufficient historical observations" in str(exc.value)


def test_forecast_generates_24_valid_steps():
    # 48 hours of constant history
    history = [50.0] * 48
    origin_time = pd.Timestamp("2024-04-01 12:00:00")

    df_fc = generate_24h_recursive_forecast(
        city="Delhi",
        recent_observations=history,
        forecast_origin_time=origin_time,
        horizon_hours=24,
    )

    assert len(df_fc) == 24
    assert list(df_fc["step"]) == list(range(1, 25))

    # Check timestamps increment strictly by 1 hour
    for i in range(len(df_fc)):
        expected_time = origin_time + pd.Timedelta(hours=i + 1)
        assert df_fc["forecast_hour_ist"].iloc[i] == expected_time

    # Physical constraint: predicted PM2.5 must be non-negative
    assert (df_fc["predicted_pm25"] >= 0.0).all()
    assert df_fc["city"].iloc[0] == "Delhi"

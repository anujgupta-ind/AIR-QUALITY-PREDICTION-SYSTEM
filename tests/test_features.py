"""
Unit tests for feature engineering, verifying leak-free lags, rolling shifts, and recursive feature construction.
"""

import pandas as pd
import numpy as np
import pytest

from src.modeling.features import (
    create_features_for_station,
    build_single_prediction_features,
    FEATURE_COLUMNS,
)


def test_feature_creation_leak_free():
    # Construct a synthetic strictly ascending sequence 1, 2, 3, ... 50
    # to easily verify that lag_1[t] == value[t-1] and rolling mean excludes value[t]
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=50, freq="h")
    values = [float(i) for i in range(1, 51)]
    df_raw = pd.DataFrame({
        "datetime_ist": timestamps,
        "value": values,
        "city": "Delhi",
    })

    df_feat = create_features_for_station(df_raw)

    # After warm-up drop (first 24 rows), check row properties
    assert len(df_feat) == 26

    # Verify column existence
    for col in FEATURE_COLUMNS:
        assert col in df_feat.columns

    # Verify leak-free lag_1:
    # At index 0 of df_feat (which corresponds to row 24 of original, value=25.0):
    # Target value should be 25.0, lag_1 should be 24.0, lag_2 should be 23.0
    first_row = df_feat.iloc[0]
    assert first_row["target"] == 25.0
    assert first_row["lag_1"] == 24.0
    assert first_row["lag_2"] == 23.0
    assert first_row["lag_3"] == 22.0
    assert first_row["lag_24"] == 1.0

    # Verify rolling_mean_3h is strictly over shifted values (24, 23, 22) -> (24+23+22)/3 = 23.0
    assert np.isclose(first_row["rolling_mean_3h"], 23.0)
    # Ensure current target (25.0) was NOT in rolling mean!
    assert first_row["rolling_mean_3h"] < first_row["target"]


def test_single_prediction_features_consistency():
    # 24 historical values: 10, 11, ... 33
    history = [float(x) for x in range(10, 34)]
    forecast_time = pd.Timestamp("2024-03-01 14:00:00")

    row = build_single_prediction_features(history, forecast_time)

    assert list(row.columns) == FEATURE_COLUMNS
    assert len(row) == 1
    assert row["hour"].iloc[0] == 14
    assert row["lag_1"].iloc[0] == 33.0
    assert row["lag_2"].iloc[0] == 32.0
    assert row["lag_3"].iloc[0] == 31.0
    assert row["lag_24"].iloc[0] == 10.0
    # rolling_mean_3h should be mean of last 3: (33 + 32 + 31) / 3 = 32.0
    assert np.isclose(row["rolling_mean_3h"].iloc[0], 32.0)

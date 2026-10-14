"""
Time-Series Feature Engineering Module
Constructs leak-free temporal, lag, and rolling features for hourly PM2.5 forecasting.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger

logger = get_logger("features")

# Standard ordered list of feature column names used across model training and inference
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "rolling_mean_3h",
    "rolling_std_3h",
    "rolling_mean_6h",
    "rolling_mean_24h",
]


def create_features_for_station(df_station: pd.DataFrame) -> pd.DataFrame:
    """
    Generate chronological, strictly leak-free time-series features for a single station/city.
    
    Guarantees:
    - Every lag feature at index t uses observations strictly from t-1, t-2, etc.
    - Every rolling feature is computed after calling .shift(1), ensuring the target value at t
      is never included in the rolling aggregation.
    """
    df = df_station.copy()
    
    # Ensure chronological sort
    if "datetime_ist" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime_ist"])
    elif "timestamp_utc" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp_utc"])
    else:
        raise ValueError("DataFrame must contain 'datetime_ist' or 'timestamp_utc'.")

    df = df.sort_values(by="datetime").reset_index(drop=True)

    # 1. Temporal calendar features
    hours = df["datetime"].dt.hour
    dayofweek = df["datetime"].dt.dayofweek
    df["hour"] = hours
    df["day_of_week"] = dayofweek
    df["is_weekend"] = (dayofweek >= 5).astype(int)

    # Cyclical hour encoding (smoothly connects 23:00 to 00:00)
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    # 2. Autoregressive Lags (strictly shifted)
    val = df["value"]
    df["lag_1"] = val.shift(1)
    df["lag_2"] = val.shift(2)
    df["lag_3"] = val.shift(3)
    df["lag_24"] = val.shift(24)

    # 3. Rolling Statistics (strictly computed on shifted series to avoid target leakage)
    val_shifted = val.shift(1)
    df["rolling_mean_3h"] = val_shifted.rolling(window=3, min_periods=2).mean()
    df["rolling_std_3h"] = val_shifted.rolling(window=3, min_periods=2).std().fillna(0.0)
    df["rolling_mean_6h"] = val_shifted.rolling(window=6, min_periods=3).mean()
    df["rolling_mean_24h"] = val_shifted.rolling(window=24, min_periods=12).mean()

    # Target: PM2.5 value at time t
    df["target"] = df["value"]

    # Drop warm-up rows (first 24 rows where lag_24 or rolling_mean_24h are not yet defined)
    df_clean = df.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
    return df_clean


def build_single_prediction_features(
    history_values: list[float],
    forecast_datetime: pd.Timestamp,
) -> pd.DataFrame:
    """
    Construct a 1-row feature DataFrame for recursive 24-hour forecasting.
    
    Args:
        history_values: List of PM2.5 observations chronologically leading up to t-1.
                        Must contain at least 24 observations.
        forecast_datetime: The timestamp for the hour to be predicted (time t).
    
    Returns:
        1-row DataFrame containing all FEATURE_COLUMNS formatted identically to training data.
    """
    if len(history_values) < 24:
        raise ValueError(f"At least 24 historical values required to compute features, got {len(history_values)}")

    # Temporal features for the target forecast hour
    hr = forecast_datetime.hour
    dow = forecast_datetime.dayofweek
    is_wknd = 1 if dow >= 5 else 0
    hr_sin = np.sin(2 * np.pi * hr / 24.0)
    hr_cos = np.cos(2 * np.pi * hr / 24.0)

    # Lags relative to target hour t:
    # Most recent observation is t-1, second most recent is t-2, etc.
    lag_1 = float(history_values[-1])
    lag_2 = float(history_values[-2])
    lag_3 = float(history_values[-3])
    lag_24 = float(history_values[-24])

    # Rolling calculations over past observations
    roll_3 = float(np.mean(history_values[-3:]))
    roll_std_3 = float(np.std(history_values[-3:]))
    roll_6 = float(np.mean(history_values[-6:]))
    roll_24 = float(np.mean(history_values[-24:]))

    row_data = {
        "hour": hr,
        "day_of_week": dow,
        "is_weekend": is_wknd,
        "hour_sin": hr_sin,
        "hour_cos": hr_cos,
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "lag_24": lag_24,
        "rolling_mean_3h": roll_3,
        "rolling_std_3h": roll_std_3,
        "rolling_mean_6h": roll_6,
        "rolling_mean_24h": roll_24,
    }

    return pd.DataFrame([row_data], columns=FEATURE_COLUMNS)

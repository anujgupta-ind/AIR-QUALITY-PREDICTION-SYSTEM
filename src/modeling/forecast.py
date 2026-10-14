"""
Recursive 24-Hour Forecasting Engine
Generates multi-step hourly forecasts using autoregressive feedback without future target leakage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.features import build_single_prediction_features
from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR, STANDARD_UNIT
from src.utils.logging_utils import get_logger

logger = get_logger("forecast")


def generate_24h_recursive_forecast(
    city: str,
    recent_observations: pd.DataFrame | list[float],
    forecast_origin_time: pd.Timestamp | str | None = None,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """
    Generate recursive hourly PM2.5 forecast for the next `horizon_hours` (default: 24).
    
    Procedure:
    1. Take the recent valid observations up to the forecast origin.
    2. Construct the feature vector for hour t+1.
    3. Predict PM2.5 for t+1.
    4. Append the prediction to the rolling history.
    5. Construct features for t+2 using the updated history.
    6. Repeat until 24 hourly predictions are obtained.
    """
    slug = city.lower().replace(" ", "_")
    model_path = MODELS_DIR / f"rf_model_{slug}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model for '{city}' not found at: {model_path}. Run train_model.py first.")

    artifact = joblib.load(model_path)
    model = artifact["model"]

    # Extract history values and origin timestamp
    if isinstance(recent_observations, pd.DataFrame):
        df_obs = recent_observations.sort_values(by="datetime_ist" if "datetime_ist" in recent_observations.columns else "datetime").copy()
        history = df_obs["value"].dropna().tolist()
        if forecast_origin_time is None:
            time_col = "datetime_ist" if "datetime_ist" in df_obs.columns else "datetime"
            forecast_origin_time = pd.to_datetime(df_obs[time_col].iloc[-1])
    else:
        history = list(recent_observations)
        if forecast_origin_time is None:
            forecast_origin_time = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)

    forecast_origin_time = pd.to_datetime(forecast_origin_time)

    if len(history) < 24:
        raise ValueError(
            f"Insufficient historical observations to forecast. Got {len(history)} hours, need at least 24."
        )

    # Use the trailing 48 hours for feature computation
    rolling_history = history[-48:].copy()

    forecast_rows = []
    current_time = forecast_origin_time

    for step in range(1, horizon_hours + 1):
        target_forecast_time = current_time + pd.Timedelta(hours=step)

        # 1. Build leak-free feature vector for this step
        X_step = build_single_prediction_features(rolling_history, target_forecast_time)

        # 2. Predict next hour concentration
        pred_val = float(model.predict(X_step)[0])
        # Physical constraint: PM2.5 concentration cannot be negative
        pred_val = max(0.0, round(pred_val, 2))

        # 3. Append prediction to rolling history so step+1 can compute lag_1, rolling means, etc.
        rolling_history.append(pred_val)

        forecast_rows.append({
            "step": step,
            "forecast_hour_ist": target_forecast_time,
            "predicted_pm25": pred_val,
            "unit": STANDARD_UNIT,
            "city": city,
            "station_name": artifact.get("station_name", city),
            "forecast_origin_time": forecast_origin_time,
            "model_version": artifact.get("version", "1.0.0"),
        })

    return pd.DataFrame(forecast_rows)


def forecast_from_cleaned_dataset(city: str, horizon_hours: int = 24) -> pd.DataFrame:
    """Convenience helper to forecast starting from the latest available record in the cleaned dataset."""
    csv_path = PROCESSED_DATA_DIR / "cleaned_pm25_hourly.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df_city = df[df["city"] == city].sort_values(by="datetime_ist")
    if df_city.empty:
        raise ValueError(f"No records found for city '{city}' in cleaned dataset.")

    return generate_24h_recursive_forecast(
        city=city,
        recent_observations=df_city,
        horizon_hours=horizon_hours,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate 24-hour recursive PM2.5 forecast.")
    parser.add_argument("--city", type=str, default="Delhi", help="Target city")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon in hours")
    args = parser.parse_args()

    df_fc = forecast_from_cleaned_dataset(args.city, horizon_hours=args.horizon)
    origin = df_fc["forecast_origin_time"].iloc[0]
    print("\n" + "=" * 70)
    print(f"24-HOUR RECURSIVE PM2.5 FORECAST FOR {args.city.upper()}")
    print(f"Forecast Origin: {origin} (IST)")
    print(f"Station: {df_fc['station_name'].iloc[0]}")
    print("=" * 70)
    for _, row in df_fc.iterrows():
        print(f"Hour +{row['step']:02d} | Time: {str(row['forecast_hour_ist'])} | Predicted PM2.5: {row['predicted_pm25']:6.2f} {row['unit']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

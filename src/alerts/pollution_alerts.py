"""
Pollution Alert Logic Module
Categorizes PM2.5 mass concentrations into health risk tiers using documented CPCB concentration brackets.
Distinguishes observed alerts from forecasted alerts and flags stale observations.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ALERT_CATEGORIES, STANDARD_UNIT, POLLUTANT_DISPLAY
from src.utils.logging_utils import get_logger

logger = get_logger("alerts")

STALE_DATA_THRESHOLD_HOURS = 6.0


def get_pm25_alert_level(pm25_value: float) -> dict:
    """
    Categorize a PM2.5 concentration value (µg/m³) into standard Indian NAAQS / CPCB health brackets.
    
    Brackets:
    - Good: 0 - 30 µg/m³
    - Satisfactory: 31 - 60 µg/m³
    - Moderate: 61 - 90 µg/m³
    - Poor: 91 - 120 µg/m³
    - Very Poor: 121 - 250 µg/m³
    - Severe: > 250 µg/m³
    """
    val = float(pm25_value)

    # Edge cases
    if val < 0:
        val = 0.0

    for cat in ALERT_CATEGORIES:
        if cat["min"] <= val <= cat["max"]:
            return {
                "category": cat["category"],
                "color": cat["color"],
                "badge_bg": cat["badge_bg"],
                "badge_text": cat["badge_text"],
                "health_advice": cat["health_advice"],
                "min_range": cat["min"],
                "max_range": cat["max"],
            }

    # Fallback for values > 10,000
    severe = ALERT_CATEGORIES[-1]
    return {
        "category": severe["category"],
        "color": severe["color"],
        "badge_bg": severe["badge_bg"],
        "badge_text": severe["badge_text"],
        "health_advice": severe["health_advice"],
        "min_range": severe["min"],
        "max_range": severe["max"],
    }


def create_pollution_alert(
    value: float,
    timestamp: str | pd.Timestamp,
    station_name: str,
    city: str,
    is_forecast: bool = False,
    observation_datetime_utc: datetime | None = None,
) -> dict:
    """
    Generate a structured pollution alert record.
    
    Distinguishes observed alerts from forecasted alerts and includes staleness detection
    and clear scientific disclaimers.
    """
    tier = get_pm25_alert_level(value)

    # Staleness evaluation for observed readings
    is_stale = False
    stale_warning = None
    if not is_forecast and observation_datetime_utc:
        now_utc = datetime.now(timezone.utc)
        diff_hours = (now_utc - observation_datetime_utc).total_seconds() / 3600.0
        if diff_hours > STALE_DATA_THRESHOLD_HOURS:
            is_stale = True
            stale_warning = (
                f"Data Freshness Warning: The latest observation was recorded {diff_hours:.1f} hours ago. "
                "Conditions may have changed."
            )

    alert_type = "PREDICTED_ALERT" if is_forecast else "MEASURED_ALERT"
    title_prefix = "🔮 24h Forecast Alert" if is_forecast else "📡 Ground Station Measurement Alert"

    return {
        "alert_type": alert_type,
        "is_forecast": is_forecast,
        "title": f"{title_prefix}: {tier['category']} Air Quality in {city}",
        "category": tier["category"],
        "value": round(float(value), 2),
        "unit": STANDARD_UNIT,
        "pollutant": POLLUTANT_DISPLAY,
        "city": city,
        "station_name": station_name,
        "timestamp": str(timestamp),
        "color": tier["color"],
        "badge_bg": tier["badge_bg"],
        "badge_text": tier["badge_text"],
        "health_advice": tier["health_advice"],
        "is_stale": is_stale,
        "stale_warning": stale_warning,
        "disclaimer": (
            f"Notice: This alert is categorized directly on PM2.5 mass concentration ({STANDARD_UNIT}) "
            "using Indian CPCB health reference brackets. It does NOT represent an official multi-pollutant "
            "National Air Quality Index (NAQI) calculation."
        ),
    }


def batch_classify_forecast(df_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Append alert categories, color codes, and health advice to each row of a forecast DataFrame.
    """
    df = df_forecast.copy()
    categories = []
    colors = []
    badge_bgs = []
    badge_texts = []
    advices = []

    for val in df["predicted_pm25"]:
        tier = get_pm25_alert_level(val)
        categories.append(tier["category"])
        colors.append(tier["color"])
        badge_bgs.append(tier["badge_bg"])
        badge_texts.append(tier["badge_text"])
        advices.append(tier["health_advice"])

    df["alert_category"] = categories
    df["color"] = colors
    df["badge_bg"] = badge_bgs
    df["badge_text"] = badge_texts
    df["health_advice"] = advices
    return df

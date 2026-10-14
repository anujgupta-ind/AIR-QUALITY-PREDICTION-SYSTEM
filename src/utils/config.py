"""
Application configuration and global constants.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve root directory of project (air-quality-prediction/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env if present
load_dotenv(PROJECT_ROOT / ".env")

# Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
OPENAQ_API_BASE_URL = "https://api.openaq.org/v3"


def get_openaq_api_key() -> str | None:
    """
    Retrieve OpenAQ API key from environment variable or Streamlit secrets.
    Returns None if key is not configured.
    """
    api_key = os.getenv("OPENAQ_API_KEY")
    if api_key and api_key.strip() and api_key != "your_openaq_api_key_here":
        return api_key.strip()

    # Try Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if "OPENAQ_API_KEY" in st.secrets:
            key = st.secrets["OPENAQ_API_KEY"]
            if key and key.strip() and key != "your_openaq_api_key_here":
                return key.strip()
    except Exception:
        pass

    return None


# Target domain parameters
COUNTRY_CODE = "IN"
COUNTRY_NAME = "India"
POLLUTANT_NAME = "pm25"
POLLUTANT_DISPLAY = "PM2.5"
STANDARD_UNIT = "µg/m³"

# Pollution Alert Thresholds (Indian CPCB standard brackets for PM2.5 in µg/m³)
# Format: (category_name, min_val_inclusive, max_val_inclusive, color_hex, health_advice)
ALERT_CATEGORIES = [
    {
        "category": "Good",
        "min": 0.0,
        "max": 30.0,
        "color": "#16a34a",
        "badge_bg": "#dcfce7",
        "badge_text": "#166534",
        "health_advice": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
    },
    {
        "category": "Satisfactory",
        "min": 30.0001,
        "max": 60.0,
        "color": "#84cc16",
        "badge_bg": "#ecfccb",
        "badge_text": "#3f6212",
        "health_advice": "Minor breathing discomfort to sensitive people with respiratory conditions.",
    },
    {
        "category": "Moderate",
        "min": 60.0001,
        "max": 90.0,
        "color": "#eab308",
        "badge_bg": "#fef9c3",
        "badge_text": "#854d0e",
        "health_advice": "Breathing discomfort to people with lung disease such as asthma, and discomfort to people with heart disease.",
    },
    {
        "category": "Poor",
        "min": 90.0001,
        "max": 120.0,
        "color": "#f97316",
        "badge_bg": "#ffedd5",
        "badge_text": "#9a3412",
        "health_advice": "Breathing discomfort to most people on prolonged exposure; avoid prolonged exertion.",
    },
    {
        "category": "Very Poor",
        "min": 120.0001,
        "max": 250.0,
        "color": "#dc2626",
        "badge_bg": "#fee2e2",
        "badge_text": "#991b1b",
        "health_advice": "Respiratory illness to the people on prolonged exposure. Significant health warnings.",
    },
    {
        "category": "Severe",
        "min": 250.0001,
        "max": 10000.0,
        "color": "#7f1d1d",
        "badge_bg": "#fce7f3",
        "badge_text": "#701a75",
        "health_advice": "Affects healthy people and seriously impacts those with existing diseases. Stay indoors.",
    },
]

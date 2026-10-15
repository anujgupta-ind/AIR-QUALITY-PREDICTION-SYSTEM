"""
Cloud Deployment Entrypoint for Railway and Containerized Environments.
Reads dynamic $PORT provided by Railway / PaaS and launches Streamlit.
"""

import os
import subprocess
import sys

def main() -> None:
    port = os.getenv("PORT", "8501")
    # Sanitize port to ensure valid integer
    try:
        int(port)
    except ValueError:
        port = "8501"

    # Self-healing check: Ensure cleaned data exists
    from pathlib import Path
    cleaned_file = Path("data/processed/cleaned_pm25_hourly.csv")
    raw_file = Path("data/raw/openaq_pm25_raw.csv")

    if not cleaned_file.exists():
        print("Cleaned dataset not found. Running self-healing data pipeline...")
        try:
            if not raw_file.exists():
                print("Fetching initial baseline data from OpenAQ...")
                from src.data_collection.fetch_data import fetch_from_openaq_archive
                fetch_from_openaq_archive(days_per_station=30)

            from src.data_processing.clean_data import clean_air_quality_data
            clean_air_quality_data(raw_file)
            print("Successfully bootstrapped air quality dataset.")
        except Exception as e:
            print(f"Self-healing bootstrap warning: {e}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
    ]

    print(f"Starting Streamlit on port {port} via {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()

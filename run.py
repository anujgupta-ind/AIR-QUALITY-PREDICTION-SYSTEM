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

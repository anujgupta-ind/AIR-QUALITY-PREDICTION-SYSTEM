"""
Model Evaluation and Diagnostic Module
Compares machine-learning models against naive and rolling baselines on held-out test splits.
Produces error distribution plots and residual diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.features import create_features_for_station, FEATURE_COLUMNS
from src.modeling.train_model import evaluate_predictions
from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR, STANDARD_UNIT, POLLUTANT_DISPLAY
from src.utils.logging_utils import get_logger

logger = get_logger("evaluate_model")


def load_model_artifact(city: str) -> dict:
    """Load model artifact bundle from disk."""
    slug = city.lower().replace(" ", "_")
    path = MODELS_DIR / f"rf_model_{slug}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}. Run train_model.py first.")
    return joblib.load(path)


def run_full_evaluation(cleaned_csv_path: Path | str | None = None) -> dict:
    """
    Run evaluation on the held-out test split for each city, compute diagnostic metrics,
    and save comparative diagnostic visualization.
    """
    path = Path(cleaned_csv_path) if cleaned_csv_path else PROCESSED_DATA_DIR / "cleaned_pm25_hourly.csv"
    df = pd.read_csv(path)
    cities = sorted(df["city"].unique())

    evaluation_report = {}
    fig, axes = plt.subplots(len(cities), 2, figsize=(14, 3.5 * len(cities)))
    if len(cities) == 1:
        axes = np.array([axes])

    for i, city in enumerate(cities):
        artifact = load_model_artifact(city)
        model = artifact["model"]

        # Re-derive features for the city
        df_city = df[df["city"] == city].copy()
        df_feat = create_features_for_station(df_city)

        # Retrieve exact test split size used during training
        test_samples = artifact["testing_period"]["samples"]
        df_test = df_feat.iloc[-test_samples:].copy()

        X_test = df_test[FEATURE_COLUMNS]
        y_test = df_test["target"].values
        timestamps = pd.to_datetime(df_test["datetime"])

        # Predictions
        y_pred_rf = model.predict(X_test)
        y_pred_persist = X_test["lag_1"].values
        y_pred_rolling = X_test["rolling_mean_24h"].values

        rf_metrics = evaluate_predictions(y_test, y_pred_rf)
        persist_metrics = evaluate_predictions(y_test, y_pred_persist)
        rolling_metrics = evaluate_predictions(y_test, y_pred_rolling)

        # Residuals
        residuals = y_test - y_pred_rf

        evaluation_report[city] = {
            "city": city,
            "station_name": artifact.get("station_name", city),
            "test_sample_count": int(test_samples),
            "test_period": artifact["testing_period"],
            "metrics": {
                "random_forest": rf_metrics,
                "persistence_baseline": persist_metrics,
                "rolling_24h_baseline": rolling_metrics,
            },
            "residual_mean": round(float(np.mean(residuals)), 2),
            "residual_std": round(float(np.std(residuals)), 2),
            "top_features": list(artifact["feature_importances"].keys())[:5],
        }

        # Plot 1: Actual vs Predicted Time-Series
        ax_ts = axes[i, 0]
        ax_ts.plot(timestamps, y_test, label="Actual Observed", color="#0284c7", linewidth=2)
        ax_ts.plot(timestamps, y_pred_rf, label="RF Prediction", color="#f97316", linestyle="--", linewidth=1.8)
        ax_ts.plot(timestamps, y_pred_persist, label="Persistence (t-1)", color="#64748b", linestyle=":", alpha=0.7)
        ax_ts.set_title(f"{city}: Actual vs Forecast (Held-Out Test Set)", fontweight="bold")
        ax_ts.set_ylabel(f"{POLLUTANT_DISPLAY} ({STANDARD_UNIT})")
        ax_ts.legend(loc="upper right", fontsize=8)

        # Plot 2: Residual Error Distribution
        ax_res = axes[i, 1]
        ax_res.hist(residuals, bins=20, color="#10b981", edgecolor="black", alpha=0.8)
        ax_res.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax_res.set_title(f"{city}: Residual Errors (Actual - Predicted)", fontweight="bold")
        ax_res.set_xlabel(f"Error ({STANDARD_UNIT})")
        ax_res.set_ylabel("Frequency")

    plt.tight_layout()
    diag_fig_path = PROCESSED_DATA_DIR / "eda_figures" / "model_evaluation_comparison.png"
    diag_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(diag_fig_path, dpi=200)
    plt.close()
    logger.info(f"Saved evaluation diagnostic figure to: {diag_fig_path}")

    # Save JSON report
    report_path = MODELS_DIR / "detailed_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)
    logger.info(f"Saved evaluation report to: {report_path}")

    return evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate ML air quality forecasting models on held-out test data.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROCESSED_DATA_DIR / "cleaned_pm25_hourly.csv"),
        help="Path to cleaned dataset",
    )
    args = parser.parse_args()

    report = run_full_evaluation(args.input)
    print("\n" + "=" * 75)
    print("DETAILED HELD-OUT TEST EVALUATION:")
    for city, res in report.items():
        rf = res["metrics"]["random_forest"]
        per = res["metrics"]["persistence_baseline"]
        rol = res["metrics"]["rolling_24h_baseline"]
        print(f"\nCity: {city.upper()} ({res['station_name']})")
        print(f"  Test Period: {res['test_period']['start']} to {res['test_period']['end']} ({res['test_sample_count']} hours)")
        print(f"  • Random Forest Regressor: MAE={rf['mae']} µg/m³, RMSE={rf['rmse']} µg/m³, R²={rf['r2']}")
        print(f"  • Persistence Baseline:    MAE={per['mae']} µg/m³, RMSE={per['rmse']} µg/m³, R²={per['r2']}")
        print(f"  • 24h Rolling Baseline:    MAE={rol['mae']} µg/m³, RMSE={rol['rmse']} µg/m³, R²={rol['r2']}")
        print(f"  • Top Predictive Features: {', '.join(res['top_features'])}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

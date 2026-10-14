"""
Model Training Module
Trains a Random Forest Regressor on chronological time-series splits,
benchmarks against persistence and rolling mean baselines,
and saves model artifacts with evaluation metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.features import create_features_for_station, FEATURE_COLUMNS
from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR, STANDARD_UNIT, POLLUTANT_DISPLAY
from src.utils.logging_utils import get_logger

logger = get_logger("train_model")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and R² score."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "unit": STANDARD_UNIT,
    }


def train_and_evaluate_city_model(
    df_clean: pd.DataFrame,
    city: str,
    test_ratio: float = 0.20,
    n_estimators: int = 100,
    max_depth: int = 12,
    random_state: int = 42,
) -> dict:
    """
    Train and evaluate model for a specific city using strict chronological splitting.
    """
    logger.info(f"--- Training model for City: {city} ---")
    df_city = df_clean[df_clean["city"] == city].copy()
    if len(df_city) < 72:
        raise ValueError(f"Insufficient historical records for {city} (has {len(df_city)}, requires >= 72).")

    # Generate features
    df_feat = create_features_for_station(df_city)
    n_samples = len(df_feat)

    # Chronological Split (No random shuffling)
    split_idx = int(n_samples * (1.0 - test_ratio))
    df_train = df_feat.iloc[:split_idx].copy()
    df_test = df_feat.iloc[split_idx:].copy()

    X_train = df_train[FEATURE_COLUMNS]
    y_train = df_train["target"]
    X_test = df_test[FEATURE_COLUMNS]
    y_test = df_test["target"]

    train_start = str(df_train["datetime"].min())
    train_end = str(df_train["datetime"].max())
    test_start = str(df_test["datetime"].min())
    test_end = str(df_test["datetime"].max())

    logger.info(f"Training split: {len(X_train)} hours ({train_start} to {train_end})")
    logger.info(f"Testing split:  {len(X_test)} hours ({test_start} to {test_end})")

    # 1. Baseline 1: Persistence (Predict most recent observation lag_1)
    y_pred_persistence = X_test["lag_1"].values
    metrics_persistence = evaluate_predictions(y_test.values, y_pred_persistence)

    # 2. Baseline 2: Historical 24h Rolling Mean (rolling_mean_24h)
    y_pred_rolling24 = X_test["rolling_mean_24h"].values
    metrics_rolling24 = evaluate_predictions(y_test.values, y_pred_rolling24)

    # 3. ML Model: Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred_ml = model.predict(X_test)
    metrics_ml = evaluate_predictions(y_test.values, y_pred_ml)

    # Feature importances
    importances = {
        col: round(float(imp), 4)
        for col, imp in zip(FEATURE_COLUMNS, model.feature_importances_)
    }
    sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    logger.info(f"Model Results for {city}:")
    logger.info(f"  • Random Forest Regressor: MAE={metrics_ml['mae']} {STANDARD_UNIT}, RMSE={metrics_ml['rmse']} {STANDARD_UNIT}, R²={metrics_ml['r2']}")
    logger.info(f"  • Persistence Baseline:    MAE={metrics_persistence['mae']} {STANDARD_UNIT}, RMSE={metrics_persistence['rmse']} {STANDARD_UNIT}, R²={metrics_persistence['r2']}")
    logger.info(f"  • 24h Rolling Baseline:    MAE={metrics_rolling24['mae']} {STANDARD_UNIT}, RMSE={metrics_rolling24['rmse']} {STANDARD_UNIT}, R²={metrics_rolling24['r2']}")

    # Save artifact bundle
    city_slug = city.lower().replace(" ", "_")
    model_artifact_path = MODELS_DIR / f"rf_model_{city_slug}.joblib"

    artifact_bundle = {
        "model": model,
        "city": city,
        "station_name": df_city["station_name"].iloc[0] if "station_name" in df_city.columns else city,
        "feature_names": FEATURE_COLUMNS,
        "feature_importances": sorted_importances,
        "training_period": {"start": train_start, "end": train_end, "samples": len(X_train)},
        "testing_period": {"start": test_start, "end": test_end, "samples": len(X_test)},
        "metrics": {
            "random_forest": metrics_ml,
            "persistence_baseline": metrics_persistence,
            "rolling_24h_baseline": metrics_rolling24,
        },
        "version": "1.0.0",
        "data_source": "OpenAQ Real Indian Observations",
    }
    joblib.dump(artifact_bundle, model_artifact_path)
    logger.info(f"Saved model artifact to: {model_artifact_path}")

    return artifact_bundle


def train_all_models(cleaned_csv_path: Path | str | None = None) -> dict:
    """Train models for all available cities in the cleaned dataset."""
    path = Path(cleaned_csv_path) if cleaned_csv_path else PROCESSED_DATA_DIR / "cleaned_pm25_hourly.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {path}. Run clean_data.py first.")

    df = pd.read_csv(path)
    cities = sorted(df["city"].unique())
    logger.info(f"Found {len(cities)} cities in cleaned dataset: {cities}")

    all_results = {}
    for city in cities:
        bundle = train_and_evaluate_city_model(df, city=city)
        # Store metadata without the raw model object for JSON export
        all_results[city] = {
            "city": bundle["city"],
            "station_name": bundle["station_name"],
            "training_period": bundle["training_period"],
            "testing_period": bundle["testing_period"],
            "metrics": bundle["metrics"],
            "top_features": list(bundle["feature_importances"].keys())[:5],
            "model_path": str(MODELS_DIR / f"rf_model_{city.lower().replace(' ', '_')}.joblib"),
        }

    # Save summary metrics JSON for easy loading in dashboard
    summary_path = MODELS_DIR / "model_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved complete benchmark summary to: {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Train ML PM2.5 forecasting models on real Indian data.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROCESSED_DATA_DIR / "cleaned_pm25_hourly.csv"),
        help="Path to cleaned dataset",
    )
    args = parser.parse_args()

    results = train_all_models(args.input)
    print("\n" + "=" * 70)
    print("MODEL BENCHMARK RESULTS (TEST SET):")
    for city, res in results.items():
        m = res["metrics"]["random_forest"]
        b = res["metrics"]["persistence_baseline"]
        print(f"City: {city:10s} | Test Obs: {res['testing_period']['samples']:3d} | RF MAE: {m['mae']:5.2f} µg/m³ | Persistence MAE: {b['mae']:5.2f} µg/m³ | R²: {m['r2']:6.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

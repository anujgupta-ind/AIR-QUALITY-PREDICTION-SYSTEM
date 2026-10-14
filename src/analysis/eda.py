"""
Exploratory Data Analysis (EDA) Module
Computes statistical distributions, diurnal profiles, city comparisons,
and generates both interactive Plotly figures and static Matplotlib figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import PROCESSED_DATA_DIR, POLLUTANT_DISPLAY, STANDARD_UNIT, ALERT_CATEGORIES
from src.utils.logging_utils import get_logger

logger = get_logger("eda")


def load_cleaned_data(file_path: Path | str | None = None) -> pd.DataFrame:
    """Load cleaned PM2.5 dataset and ensure proper datetime typing."""
    path = Path(file_path) if file_path else PROCESSED_DATA_DIR / "cleaned_pm25_hourly.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {path}. Run clean_data.py first.")

    df = pd.read_csv(path)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df["hour"] = df["datetime_ist"].dt.hour
    df["day_name"] = df["datetime_ist"].dt.day_name()
    df["date"] = df["datetime_ist"].dt.date
    return df


def generate_time_series_plot(df: pd.DataFrame, city: str | None = None) -> go.Figure:
    """Interactive time-series line chart of PM2.5 concentrations over time."""
    sub_df = df if not city or city == "All Cities" else df[df["city"] == city]

    fig = px.line(
        sub_df,
        x="datetime_ist",
        y="value",
        color="city",
        title=f"{POLLUTANT_DISPLAY} Concentration Over Time (IST)",
        labels={"datetime_ist": "Date & Time (IST)", "value": f"{POLLUTANT_DISPLAY} ({STANDARD_UNIT})", "city": "City"},
        template="plotly_dark",
    )
    # Add CPCB threshold reference lines
    fig.add_hline(y=60, line_dash="dot", line_color="#84cc16", annotation_text="Satisfactory Limit (60)")
    fig.add_hline(y=120, line_dash="dash", line_color="#f97316", annotation_text="Poor Limit (120)")
    fig.add_hline(y=250, line_dash="dashdot", line_color="#dc2626", annotation_text="Severe Limit (250)")

    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def generate_diurnal_pattern_plot(df: pd.DataFrame) -> go.Figure:
    """Interactive chart showing average PM2.5 by hour of day (24-hour diurnal cycle)."""
    hourly_avg = (
        df.groupby(["city", "hour"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    fig = px.line(
        hourly_avg,
        x="hour",
        y="mean",
        color="city",
        markers=True,
        title=f"Diurnal Cycle: Average {POLLUTANT_DISPLAY} by Hour of Day (IST)",
        labels={"hour": "Hour of Day (00:00 - 23:00 IST)", "mean": f"Mean {POLLUTANT_DISPLAY} ({STANDARD_UNIT})", "city": "City"},
        template="plotly_dark",
    )
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=2)
    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def generate_daily_trend_plot(df: pd.DataFrame) -> go.Figure:
    """Interactive daily rolling 24-hour average PM2.5 trends."""
    daily_df = df.groupby(["city", "date"])["value"].mean().reset_index()
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    fig = px.bar(
        daily_df,
        x="date",
        y="value",
        color="city",
        barmode="group",
        title=f"Daily Average {POLLUTANT_DISPLAY} Concentrations",
        labels={"date": "Date", "value": f"Daily Mean {POLLUTANT_DISPLAY} ({STANDARD_UNIT})", "city": "City"},
        template="plotly_dark",
    )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig


def generate_distribution_plot(df: pd.DataFrame) -> go.Figure:
    """Boxplot / distribution comparison of PM2.5 spread across cities."""
    fig = px.box(
        df,
        x="city",
        y="value",
        color="city",
        points="outliers",
        title=f"{POLLUTANT_DISPLAY} Concentration Distribution Spread by City",
        labels={"city": "City", "value": f"{POLLUTANT_DISPLAY} ({STANDARD_UNIT})"},
        template="plotly_dark",
    )
    # Add horizontal lines
    fig.add_hline(y=60, line_dash="dot", line_color="#84cc16", annotation_text="Satisfactory (60)")
    fig.add_hline(y=120, line_dash="dash", line_color="#f97316", annotation_text="Poor (120)")
    fig.update_layout(showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
    return fig


def compute_eda_summary(df: pd.DataFrame) -> dict:
    """Compute numerical insights and statistical summaries from real observations."""
    city_stats = {}
    for city, group in df.groupby("city"):
        vals = group["value"]
        peak_hour = int(group.groupby("hour")["value"].mean().idxmax())
        lowest_hour = int(group.groupby("hour")["value"].mean().idxmin())

        city_stats[city] = {
            "record_count": int(len(vals)),
            "mean_pm25": round(float(vals.mean()), 2),
            "median_pm25": round(float(vals.median()), 2),
            "std_dev": round(float(vals.std()), 2),
            "min_pm25": round(float(vals.min()), 2),
            "max_pm25": round(float(vals.max()), 2),
            "p25": round(float(vals.quantile(0.25)), 2),
            "p75": round(float(vals.quantile(0.75)), 2),
            "peak_pollution_hour_ist": peak_hour,
            "lowest_pollution_hour_ist": lowest_hour,
        }

    overall_vals = df["value"]
    highest_city = max(city_stats.items(), key=lambda x: x[1]["mean_pm25"])[0]
    cleanest_city = min(city_stats.items(), key=lambda x: x[1]["mean_pm25"])[0]

    return {
        "total_observations": int(len(df)),
        "time_span_start": str(df["datetime_ist"].min()),
        "time_span_end": str(df["datetime_ist"].max()),
        "overall_mean_pm25": round(float(overall_vals.mean()), 2),
        "overall_median_pm25": round(float(overall_vals.median()), 2),
        "city_with_highest_mean": highest_city,
        "city_with_lowest_mean": cleanest_city,
        "cities": city_stats,
    }


def save_static_eda_figures(df: pd.DataFrame, output_dir: Path | str | None = None) -> Path:
    """Save high-resolution static matplotlib figure for reports."""
    out_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR / "eda_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Distribution Boxplot
    cities = sorted(df["city"].unique())
    box_data = [df[df["city"] == c]["value"].dropna() for c in cities]
    axes[0, 0].boxplot(box_data, tick_labels=cities, patch_artist=True)
    axes[0, 0].set_title(f"PM2.5 Distribution by City (N={len(df)})", fontweight="bold")
    axes[0, 0].set_ylabel(f"PM2.5 ({STANDARD_UNIT})")
    axes[0, 0].axhline(60, color="green", linestyle="--", label="Satisfactory (60)")
    axes[0, 0].axhline(120, color="orange", linestyle="--", label="Poor (120)")
    axes[0, 0].legend()

    # 2. Diurnal Cycle
    hourly = df.groupby(["city", "hour"])["value"].mean().unstack(level=0)
    for c in hourly.columns:
        axes[0, 1].plot(hourly.index, hourly[c], marker="o", label=c)
    axes[0, 1].set_title("Hourly Diurnal Pollution Pattern (IST)", fontweight="bold")
    axes[0, 1].set_xlabel("Hour of Day (IST)")
    axes[0, 1].set_ylabel(f"Average PM2.5 ({STANDARD_UNIT})")
    axes[0, 1].set_xticks(range(0, 24, 2))
    axes[0, 1].legend()

    # 3. Overall Histogram
    axes[1, 0].hist(df["value"].dropna(), bins=30, color="#0284c7", edgecolor="black", alpha=0.8)
    axes[1, 0].set_title("Overall PM2.5 Concentration Histogram", fontweight="bold")
    axes[1, 0].set_xlabel(f"PM2.5 ({STANDARD_UNIT})")
    axes[1, 0].set_ylabel("Frequency (Hours)")

    # 4. Daily Mean Trend
    daily = df.groupby(["date", "city"])["value"].mean().unstack(level=1)
    for c in daily.columns:
        axes[1, 1].plot(pd.to_datetime(daily.index), daily[c], label=c)
    axes[1, 1].set_title("Daily Average PM2.5 Trend", fontweight="bold")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel(f"Mean PM2.5 ({STANDARD_UNIT})")
    axes[1, 1].legend()

    plt.tight_layout()
    output_fig_path = out_dir / "eda_dashboard_summary.png"
    plt.savefig(output_fig_path, dpi=200)
    plt.close()
    logger.info(f"Saved static EDA figure to: {output_fig_path}")
    return output_fig_path


def main():
    parser = argparse.ArgumentParser(description="Perform Exploratory Data Analysis on cleaned PM2.5 data.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROCESSED_DATA_DIR / "cleaned_pm25_hourly.csv"),
        help="Path to cleaned CSV file",
    )
    args = parser.parse_args()

    df = load_cleaned_data(args.input)
    summary = compute_eda_summary(df)
    fig_path = save_static_eda_figures(df)

    summary_json_path = PROCESSED_DATA_DIR / "eda_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA) FINDINGS:")
    print(f"Total Hourly Observations Analyzed: {summary['total_observations']}")
    print(f"Time Range: {summary['time_span_start']} to {summary['time_span_end']}")
    print(f"Overall Mean PM2.5: {summary['overall_mean_pm25']} µg/m³")
    print(f"City with Highest Mean PM2.5: {summary['city_with_highest_mean']} ({summary['cities'][summary['city_with_highest_mean']]['mean_pm25']} µg/m³)")
    print(f"City with Lowest Mean PM2.5: {summary['city_with_lowest_mean']} ({summary['cities'][summary['city_with_lowest_mean']]['mean_pm25']} µg/m³)")
    print("\nCity Breakdown:")
    for city, stats in summary["cities"].items():
        print(f"  • {city:10s} | Mean: {stats['mean_pm25']:5.1f} | Median: {stats['median_pm25']:5.1f} | Peak Hour (IST): {stats['peak_pollution_hour_ist']:02d}:00 | Lowest Hour (IST): {stats['lowest_pollution_hour_ist']:02d}:00")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

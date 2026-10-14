"""
AI-Powered Air Quality Prediction and Pollution Alert System
Main Interactive Streamlit Application
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts.pollution_alerts import (
    create_pollution_alert,
    get_pm25_alert_level,
    batch_classify_forecast,
)
from src.analysis.eda import (
    generate_time_series_plot,
    generate_diurnal_pattern_plot,
    generate_daily_trend_plot,
    generate_distribution_plot,
)
from src.modeling.forecast import generate_24h_recursive_forecast
from src.utils.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RAW_DATA_DIR,
    ALERT_CATEGORIES,
    POLLUTANT_DISPLAY,
    STANDARD_UNIT,
    get_openaq_api_key,
)

# Page configuration
st.set_page_config(
    page_title="Indian Air Quality AI Forecast & Alert System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Glassmorphic UI with Curated HSL Palette)
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Background & Cards */
    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-2px);
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    
    .metric-sub {
        font-size: 0.85rem;
        color: #64748b;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .alert-banner {
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 24px;
        display: flex;
        flex-direction: column;
        gap: 6px;
        border-left: 6px solid;
    }
    
    .disclaimer-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-top: 16px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    /* Table Styling */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600)
def get_cleaned_data() -> pd.DataFrame:
    """Load cleaned PM2.5 dataset."""
    path = PROCESSED_DATA_DIR / "cleaned_pm25_hourly.csv"
    if not path.exists():
        st.error(f"Cleaned dataset not found at {path}. Please run `python src/data_processing/clean_data.py` first.")
        st.stop()
    df = pd.read_csv(path)
    df["datetime_ist"] = pd.to_datetime(df["datetime_ist"])
    df["hour"] = df["datetime_ist"].dt.hour
    df["date"] = df["datetime_ist"].dt.date
    return df


@st.cache_data(ttl=600)
def get_quality_report() -> dict:
    """Load data quality report."""
    path = PROCESSED_DATA_DIR / "data_quality_report.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=600)
def get_benchmark_summary() -> dict:
    """Load model evaluation summary."""
    path = MODELS_DIR / "model_benchmark_summary.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Load Core Data
df_all = get_cleaned_data()
quality_report = get_quality_report()
benchmark_summary = get_benchmark_summary()

cities_available = sorted(df_all["city"].unique().tolist())

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("## 🌍 Air Quality AI System")
    st.caption("Real Indian Monitoring Stations (OpenAQ)")

    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "🏠 Live & Latest Status",
            "📊 Air Quality Analysis",
            "🔮 24-Hour AI Forecast",
            "📈 Model Performance",
            "ℹ️ About & Methodology",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📍 Location Filter")
    selected_city = st.selectbox("Select City", cities_available, index=cities_available.index("Delhi") if "Delhi" in cities_available else 0)

    # Station details for selected city
    df_city_full = df_all[df_all["city"] == selected_city].sort_values(by="datetime_ist")
    station_name = df_city_full["station_name"].iloc[0] if not df_city_full.empty else selected_city
    st.info(f"**Station:** {station_name}")

    st.markdown("---")
    api_key_status = get_openaq_api_key()
    if api_key_status:
        st.success("🔑 OpenAQ API Key Active")
    else:
        st.warning("⚠️ No API Key in .env (Using OpenAQ Archive)")

    st.caption("Data Source: [OpenAQ Platform](https://openaq.org/)")


# ==============================================================================
# PAGE 1: HOME DASHBOARD
# ==============================================================================
if page == "🏠 Live & Latest Status":
    st.title(f"🌍 Real-Time Air Quality Monitor: {selected_city}")
    st.markdown(
        f"Ground observation status for **{station_name}**, {selected_city}, India. "
        "Measured directly via reference-grade OpenAQ ambient particulate matter sensors."
    )

    if df_city_full.empty:
        st.error(f"No observations available for {selected_city}.")
        st.stop()

    latest_row = df_city_full.iloc[-1]
    latest_val = float(latest_row["value"])
    latest_time_ist = latest_row["datetime_ist"]
    latest_time_utc = pd.to_datetime(latest_row["timestamp_utc"])

    alert = create_pollution_alert(
        value=latest_val,
        timestamp=latest_time_ist,
        station_name=station_name,
        city=selected_city,
        is_forecast=False,
        observation_datetime_utc=latest_time_utc.to_pydatetime() if pd.notnull(latest_time_utc) else None,
    )

    # Main Alert Banner
    st.markdown(
        f"""
        <div class="alert-banner" style="background-color: {alert['badge_bg']}; border-color: {alert['color']};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="badge" style="background-color: {alert['color']}; color: #ffffff;">{alert['category']} Quality</span>
                <span style="color: {alert['badge_text']}; font-size: 0.85rem; font-weight: 600;">Recorded: {latest_time_ist.strftime('%d %b %Y, %H:%M')} IST</span>
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: {alert['badge_text']}; margin-top: 6px;">
                {alert['health_advice']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if alert["is_stale"]:
        st.warning(f"⚠️ {alert['stale_warning']}")

    # Metric Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Latest Observed {POLLUTANT_DISPLAY}</div>
                <div class="metric-val" style="color: {alert['color']};">{latest_val:.1f} <span style="font-size: 1rem; color: #94a3b8;">{STANDARD_UNIT}</span></div>
                <div class="metric-sub">Category: <b>{alert['category']}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        rolling_24h_mean = float(df_city_full["value"].iloc[-24:].mean()) if len(df_city_full) >= 24 else latest_val
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">24h Moving Average</div>
                <div class="metric-val">{rolling_24h_mean:.1f} <span style="font-size: 1rem; color: #94a3b8;">{STANDARD_UNIT}</span></div>
                <div class="metric-sub">Average of trailing 24 hours</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        min_24h = float(df_city_full["value"].iloc[-24:].min()) if len(df_city_full) >= 24 else latest_val
        max_24h = float(df_city_full["value"].iloc[-24:].max()) if len(df_city_full) >= 24 else latest_val
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">24h Low / High</div>
                <div class="metric-val">{min_24h:.0f} / {max_24h:.0f} <span style="font-size: 1rem; color: #94a3b8;">{STANDARD_UNIT}</span></div>
                <div class="metric-sub">Min & Max in past day</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Monitoring Provenance</div>
                <div class="metric-val" style="font-size: 1.4rem; padding-top: 8px;">OpenAQ V3</div>
                <div class="metric-sub">Station ID: {int(latest_row['station_id'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Recent 48h trend chart
    st.subheader(f"Recent Ground Observations (Past 48 Hours) — {selected_city}")
    recent_48h = df_city_full.iloc[-48:]
    fig_recent = px.line(
        recent_48h,
        x="datetime_ist",
        y="value",
        markers=True,
        title=f"Hourly Observed {POLLUTANT_DISPLAY} in {selected_city} (IST)",
        labels={"datetime_ist": "Time (IST)", "value": f"{POLLUTANT_DISPLAY} ({STANDARD_UNIT})"},
        template="plotly_dark",
    )
    fig_recent.add_hline(y=60, line_dash="dot", line_color="#84cc16", annotation_text="Satisfactory Limit (60)")
    fig_recent.add_hline(y=120, line_dash="dash", line_color="#f97316", annotation_text="Poor Limit (120)")
    fig_recent.update_layout(hovermode="x unified", margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig_recent, use_container_width=True)

    # Scientific Disclaimer
    st.markdown(
        f"""
        <div class="disclaimer-box">
            <b>Scientific & Regulatory Disclaimer:</b> {alert['disclaimer']}
            All observations displayed above are genuine sensor measurements from the OpenAQ repository.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE 2: AIR QUALITY ANALYSIS (EDA)
# ==============================================================================
elif page == "📊 Air Quality Analysis":
    st.title(f"📊 Exploratory Data Analysis & Trends")
    st.markdown(
        "Statistical patterns, diurnal variation, and historical distribution "
        "derived from genuine Indian monitoring stations."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Time Series Profile",
        "⏰ Diurnal 24h Cycles",
        "📅 Daily Trends",
        "📦 Distribution & Cities",
    ])

    with tab1:
        st.subheader("Historical PM2.5 Concentrations Over Time")
        scope = st.radio("Display Scope", ["Selected City Only", "All Indian Cities Comparison"], horizontal=True)
        city_to_show = selected_city if scope == "Selected City Only" else "All Cities"
        fig_ts = generate_time_series_plot(df_all, city=city_to_show)
        st.plotly_chart(fig_ts, use_container_width=True)

    with tab2:
        st.subheader("Diurnal 24-Hour Cycle (Hour-by-Hour Patterns)")
        st.caption(
            "Reveals typical morning and evening pollution peaks influenced by atmospheric boundary layer height, "
            "temperature inversions, and local vehicular/industrial emission cycles."
        )
        fig_diurnal = generate_diurnal_pattern_plot(df_all)
        st.plotly_chart(fig_diurnal, use_container_width=True)

    with tab3:
        st.subheader("Daily Average PM2.5 Concentrations")
        fig_daily = generate_daily_trend_plot(df_all)
        st.plotly_chart(fig_daily, use_container_width=True)

    with tab4:
        st.subheader("Concentration Spread Across Cities")
        fig_dist = generate_distribution_plot(df_all)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("### 📋 Data Quality & Completeness Audit")
        if quality_report:
            col_q1, col_q2, col_q3 = st.columns(3)
            with col_q1:
                st.metric("Total Valid Observations", quality_report.get("valid_cleaned_records", len(df_all)))
            with col_q2:
                st.metric("Duplicates Removed", quality_report.get("duplicates_removed", 0))
            with col_q3:
                st.metric("Temporal Completeness", f"{quality_report.get('overall_temporal_completeness_pct', 0)}%")


# ==============================================================================
# PAGE 3: 24-HOUR FORECAST
# ==============================================================================
elif page == "🔮 24-Hour AI Forecast":
    st.title(f"🔮 24-Hour AI PM2.5 Forecast: {selected_city}")
    st.markdown(
        f"Multi-step autoregressive forecast generated by a **Random Forest Regressor** trained on historical observations for **{selected_city}**."
    )

    if df_city_full.empty or len(df_city_full) < 24:
        st.error(f"Insufficient historical data available for {selected_city} to generate a 24-hour forecast.")
        st.stop()

    forecast_origin = df_city_full["datetime_ist"].iloc[-1]
    last_val = float(df_city_full["value"].iloc[-1])

    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 5px solid #0284c7;">
            <div class="metric-title">Forecast Origin</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">
                Base Observation Time: {forecast_origin.strftime('%d %b %Y, %H:%M')} IST
            </div>
            <div class="metric-sub">Latest Ground Reading: <b>{last_val:.1f} {STANDARD_UNIT}</b> | Station: {station_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚀 Re-compute 24-Hour Forecast", type="primary"):
        st.toast("Generating recursive 24-hour predictions...", icon="🔄")

    with st.spinner("Computing 24-hour autoregressive forecast..."):
        df_forecast = generate_24h_recursive_forecast(
            city=selected_city,
            recent_observations=df_city_full,
            forecast_origin_time=forecast_origin,
            horizon_hours=24,
        )
        df_forecast_classified = batch_classify_forecast(df_forecast)

    # 1. Combined Interactive Plot: History + Forecast
    st.subheader("Observed History vs. 24-Hour Ahead Predictions")
    
    # Trailing 36 hours history
    recent_history = df_city_full.iloc[-36:].copy()
    
    fig_fc = go.Figure()
    # Historical trace
    fig_fc.add_trace(
        go.Scatter(
            x=recent_history["datetime_ist"],
            y=recent_history["value"],
            mode="lines+markers",
            name="Ground Observed PM2.5",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=5),
        )
    )
    # Forecast trace
    fig_fc.add_trace(
        go.Scatter(
            x=df_forecast_classified["forecast_hour_ist"],
            y=df_forecast_classified["predicted_pm25"],
            mode="lines+markers",
            name="AI Predicted PM2.5 (24h)",
            line=dict(color="#f97316", width=2.5, dash="dash"),
            marker=dict(size=6, color="#f97316"),
        )
    )
    
    # Separation origin line
    fig_fc.add_vline(x=forecast_origin, line_width=1.5, line_dash="dot", line_color="#94a3b8")
    fig_fc.add_annotation(
        x=forecast_origin,
        y=max(recent_history["value"].max(), df_forecast_classified["predicted_pm25"].max()),
        text="Forecast Origin",
        showarrow=True,
        arrowhead=2,
    )
    
    # Threshold bands
    fig_fc.add_hline(y=60, line_dash="dot", line_color="#84cc16", annotation_text="Satisfactory (60)")
    fig_fc.add_hline(y=120, line_dash="dash", line_color="#f97316", annotation_text="Poor (120)")
    fig_fc.add_hline(y=250, line_dash="dashdot", line_color="#dc2626", annotation_text="Severe (250)")

    fig_fc.update_layout(
        template="plotly_dark",
        title=f"24-Hour Ahead Forecast: Observed vs Model Output ({selected_city})",
        xaxis_title="Date & Time (IST)",
        yaxis_title=f"{POLLUTANT_DISPLAY} Concentration ({STANDARD_UNIT})",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # 2. Forecast Hourly Table with Alert Badges
    st.subheader("Hourly Prediction Breakdown & Health Advisory")
    
    display_table = df_forecast_classified[[
        "step",
        "forecast_hour_ist",
        "predicted_pm25",
        "alert_category",
        "health_advice",
    ]].copy()
    display_table.columns = ["Step (+hr)", "Forecast Time (IST)", f"Predicted PM2.5 ({STANDARD_UNIT})", "Alert Tier", "Health Advisory"]
    display_table["Forecast Time (IST)"] = pd.to_datetime(display_table["Forecast Time (IST)"]).dt.strftime("%d %b %Y, %H:%M")

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
    )

    # Model metadata & limitations disclaimer
    st.markdown(
        f"""
        <div class="disclaimer-box">
            <b>⚠️ Multi-Step Recursive Forecast Notice & Error Accumulation:</b><br>
            Multi-step autoregressive predictions feed hour $t+1$'s model output back into the feature pipeline 
            to predict hour $t+2$. Over the 24-hour horizon, recursive uncertainties compound. 
            Ground observations from hour +1 onwards must be updated as real measurements arrive. 
            This forecast models <b>PM2.5 concentration ({STANDARD_UNIT})</b>, not a composite official AQI.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE 4: MODEL PERFORMANCE & BENCHMARKING
# ==============================================================================
elif page == "📈 Model Performance":
    st.title("📈 Machine Learning Performance & Benchmarks")
    st.markdown(
        "Rigorous out-of-sample evaluation on held-out test splits. "
        "The machine learning model is benchmarked against two baseline methods: "
        "**Persistence ($t-1$)** and **24-Hour Historical Rolling Mean**."
    )

    if selected_city in benchmark_summary:
        m_info = benchmark_summary[selected_city]
        rf_m = m_info["metrics"]["random_forest"]
        per_m = m_info["metrics"]["persistence_baseline"]
        rol_m = m_info["metrics"]["rolling_24h_baseline"]
        t_period = m_info["testing_period"]

        st.subheader(f"Test Set Evaluation Metrics — {selected_city}")
        st.caption(f"Evaluated on {t_period['samples']} genuine held-out hours ({t_period['start']} to {t_period['end']})")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 4px solid #0284c7;">
                    <div class="metric-title">Random Forest Regressor</div>
                    <div class="metric-val">{rf_m['mae']} <span style="font-size: 0.9rem; color: #94a3b8;">{STANDARD_UNIT} MAE</span></div>
                    <div class="metric-sub">RMSE: <b>{rf_m['rmse']} {STANDARD_UNIT}</b> | R²: <b>{rf_m['r2']}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 4px solid #64748b;">
                    <div class="metric-title">Persistence Baseline (t-1)</div>
                    <div class="metric-val">{per_m['mae']} <span style="font-size: 0.9rem; color: #94a3b8;">{STANDARD_UNIT} MAE</span></div>
                    <div class="metric-sub">RMSE: <b>{per_m['rmse']} {STANDARD_UNIT}</b> | R²: <b>{per_m['r2']}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 4px solid #475569;">
                    <div class="metric-title">24h Rolling Mean Baseline</div>
                    <div class="metric-val">{rol_m['mae']} <span style="font-size: 0.9rem; color: #94a3b8;">{STANDARD_UNIT} MAE</span></div>
                    <div class="metric-sub">RMSE: <b>{rol_m['rmse']} {STANDARD_UNIT}</b> | R²: <b>{rol_m['r2']}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Comparative Bar Chart
        st.subheader("Model vs Baselines Comparison (MAE in µg/m³)")
        comp_data = pd.DataFrame({
            "Method": ["Random Forest Regressor", "Persistence Baseline (t-1)", "24h Rolling Mean Baseline"],
            "MAE": [rf_m["mae"], per_m["mae"], rol_m["mae"]],
            "RMSE": [rf_m["rmse"], per_m["rmse"], rol_m["rmse"]],
        })
        fig_comp = px.bar(
            comp_data,
            x="Method",
            y=["MAE", "RMSE"],
            barmode="group",
            title=f"Error Metrics Comparison on Held-Out Test Set ({selected_city})",
            labels={"value": f"Error ({STANDARD_UNIT})", "variable": "Metric"},
            template="plotly_dark",
            color_discrete_sequence=["#0284c7", "#f97316"],
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Top Predictive Features
        st.subheader(f"Top Predictive Features for {selected_city}")
        st.write(", ".join([f"`{feat}`" for feat in m_info["top_features"]]))

    st.markdown("### 🇮🇳 Cross-City Benchmark Summary")
    summary_rows = []
    for c_name, b_data in benchmark_summary.items():
        summary_rows.append({
            "City": c_name,
            "Station": b_data.get("station_name", c_name),
            "Test Hours": b_data["testing_period"]["samples"],
            f"RF MAE ({STANDARD_UNIT})": b_data["metrics"]["random_forest"]["mae"],
            f"Persistence MAE ({STANDARD_UNIT})": b_data["metrics"]["persistence_baseline"]["mae"],
            f"Rolling 24h MAE ({STANDARD_UNIT})": b_data["metrics"]["rolling_24h_baseline"]["mae"],
            "RF R²": b_data["metrics"]["random_forest"]["r2"],
        })
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 5: ABOUT & METHODOLOGY
# ==============================================================================
elif page == "ℹ️ About & Methodology":
    st.title("ℹ️ About the Project & Methodology")
    
    st.markdown(
        r"""
        ### 🎯 Problem Statement
        Air pollution across Indian metropolitan and regional centers represents one of the largest public health risks today.
        Fine particulate matter ($PM_{2.5}$) penetrates deep into lung tissue and the bloodstream.
        
        ### 🛠️ Real-Data Engineering Pipeline
        1. **Source Ingestion:** Ingests genuine reference-grade observations from OpenAQ API v3 / OpenAQ Open Archive for Indian monitoring stations.
        2. **Preprocessing:** Datetime alignment to Indian Standard Time (IST = UTC+05:30), strict duplicate filtering, and physical range validation ($0 \le PM_{2.5} \le 1000\,\mu\text{g/m}^3$).
        3. **Feature Engineering:** Leak-free temporal cyclical encodings (sin/cos of hour), autoregressive lags ($t-1, t-2, t-3, t-24$), and shifted rolling statistics ($3h, 6h, 24h$).
        4. **Modeling:** Chronologically split Random Forest Regressor compared against Persistence and Rolling-mean baselines.
        5. **Forecasting:** 24-step autoregressive recursive multi-step forecasting engine.
        6. **Pollution Alerts:** Categorized using Indian CPCB concentration brackets.
        
        ### 🚨 Scientific Honesty & Limitations
        - **No Fake Data:** Never fabricates sensor observations. If a station has gaps, the audit report records missing hours explicitly.
        - **Concentration vs AQI:** Forecasts mass concentration ($\mu\text{g/m}^3$). It does not pretend to compute official multi-pollutant CPCB AQI without required multi-parameter inputs.
        - **No Synthetic Weather:** We strictly avoid injecting unverified weather variables unless real, localized meteorological stations supply them.
        
        ### 👨‍💻 Reproducibility Commands
        ```powershell
        # 1. Fetch genuine OpenAQ data
        python src/data_collection/fetch_data.py --days 45
        
        # 2. Clean data & generate quality audit
        python src/data_processing/clean_data.py
        
        # 3. Perform Exploratory Data Analysis
        python src/analysis/eda.py
        
        # 4. Train ML models & baselines
        python src/modeling/train_model.py
        
        # 5. Evaluate models on test sets
        python src/modeling/evaluate_model.py
        
        # 6. Run automated test suite
        pytest
        
        # 7. Launch Streamlit
        streamlit run app.py
        ```
        """
    )

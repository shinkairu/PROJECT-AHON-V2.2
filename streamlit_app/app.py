import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="PROJECT AHON",
    layout="wide"
)

# --------------------------------------------------
# THEME & BACKGROUND
# --------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #e6f2ff, #ffffff);
}
.footer {
    text-align: center;
    color: gray;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA & MODEL
# --------------------------------------------------
@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent / "data" / "Flood_Prediction_NCR_Philippines.csv"
    return pd.read_csv(data_path)

@st.cache_resource
def load_model():
    return joblib.load("models/rf_flood_model.pkl")

df = load_data()
rf_model = load_model()

FEATURE_COLS = [
    'Rainfall_mm', 'WaterLevel_m', 'SoilMoisture_pct', 'Elevation_m',
    'Rainfall_3day_avg', 'Rainfall_7day_avg', 'Rainfall_prev_day',
    'WaterLevel_prev_day', 'WaterLevel_change', 'WaterLevel_rising',
    'Month', 'IsWetSeason'
]

# --------------------------------------------------
# CAROUSEL-STYLE NAVIGATION
# --------------------------------------------------
tabs = st.tabs([
    "🏠 Project AHON",
    "📊 Dataset & EDA",
    "🗺️ Flood Mapping",
    "📈 Insights & Aggregations"
])

# ==================================================
# TAB 1 — MAIN PANEL
# ==================================================
with tabs[0]:
    st.title("PROJECT – AHON")
    st.subheader("AI-Powered Flood Prediction & Risk Mapping")

    st.markdown("""
    **PROJECT AHON** is a lightweight, explainable flood prediction system
    designed for urban disaster preparedness in Metro Manila.

    **How it works:**
    1. Environmental and geographical data are analyzed
    2. Temporal flood patterns are learned using feature engineering
    3. A Random Forest model predicts flood likelihood
    4. Results are visualized through maps and dashboards
    """)

# ==================================================
# TAB 2 — DATASET, EDA & FEATURE ENGINEERING
# ==================================================
with tabs[1]:
    st.header("Dataset Dashboard & Feature Engineering")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Flood Occurrence Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='FloodOccurrence', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Flood Events by City")
    fig, ax = plt.subplots()
    sns.countplot(x='Location', hue='FloodOccurrence', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Rainfall vs Flood Occurrence")
    fig, ax = plt.subplots()
    sns.boxplot(x='FloodOccurrence', y='Rainfall_mm', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ==================================================
# TAB 3 — FLOOD RISK MAP
# ==================================================
with tabs[2]:
    st.header("Geospatial Flood Risk Visualization")

    latest_df = (
        df.sort_values('Date')
        .groupby('Location')
        .tail(1)
        .reset_index(drop=True)
    )

    latest_df['FloodRisk'] = rf_model.predict_proba(
        latest_df[FEATURE_COLS]
    )[:, 1]

    city_coords = {
        'Quezon City': (14.6760, 121.0437),
        'Marikina': (14.6507, 121.1029),
        'Pasig': (14.5764, 121.0851),
        'Manila': (14.5995, 120.9842)
    }

    def risk_color(p):
        if p >= 0.7:
            return 'red'
        elif p >= 0.4:
            return 'orange'
        return 'green'

    m = folium.Map(location=[14.6, 121.0], zoom_start=11)

    for _, row in latest_df.iterrows():
        lat, lon = city_coords[row['Location']]
        folium.CircleMarker(
            location=[lat, lon],
            radius=15,
            color=risk_color(row['FloodRisk']),
            fill=True,
            fill_opacity=0.7,
            popup=f"""
            <b>{row['Location']}</b><br>
            Flood Risk Score: {row['FloodRisk']:.2f}
            """
        ).add_to(m)

    st_folium(m, width=900, height=500)

# ==================================================
# TAB 4 — INSIGHTS & AGGREGATIONS
# ==================================================
with tabs[3]:
    st.header("City-Level Flood Insights")

    city_stats = (
        df.groupby('Location')
        .agg(
            avg_rainfall=('Rainfall_mm', 'mean'),
            max_rainfall=('Rainfall_mm', 'max'),
            flood_days=('FloodOccurrence', 'sum')
        )
        .reset_index()
    )

    st.dataframe(city_stats)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
PROJECT AHON | Developed by [Your Name / Team] <br>
AI-Powered Flood Risk Intelligence
</div>
""", unsafe_allow_html=True)

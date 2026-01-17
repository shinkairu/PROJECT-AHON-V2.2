import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="PROJECT AHON",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# GLOBAL THEME (BLUE–WHITE GRADIENT)
# --------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #e3f2fd 0%, #ffffff 60%);
}
h1, h2, h3 {
    color: #0d47a1;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL & DATA
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/rf_flood_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("data/Flood_Prediction_NCR_Philippines.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

rf_model = load_model()
df = load_data()

# --------------------------------------------------
# FEATURE ENGINEERING (SAME AS COLAB)
# --------------------------------------------------
def engineer_features(df):
    df = df.sort_values(["Location", "Date"])
    df["Rainfall_prev_day"] = df.groupby("Location")["Rainfall_mm"].shift(1)
    df["Rainfall_3day_avg"] = df.groupby("Location")["Rainfall_mm"].rolling(3,1).mean().reset_index(level=0, drop=True)
    df["Rainfall_7day_avg"] = df.groupby("Location")["Rainfall_mm"].rolling(7,1).mean().reset_index(level=0, drop=True)
    df["WaterLevel_prev_day"] = df.groupby("Location")["WaterLevel_m"].shift(1)
    df["WaterLevel_change"] = df["WaterLevel_m"] - df["WaterLevel_prev_day"]
    df["WaterLevel_rising"] = (df["WaterLevel_change"] > 0).astype(int)
    df["Month"] = df["Date"].dt.month
    df["IsWetSeason"] = df["Month"].isin([6,7,8,9,10,11]).astype(int)
    return df.fillna(0)

df = engineer_features(df)

FEATURES = [
    'Rainfall_mm', 'WaterLevel_m', 'SoilMoisture_pct', 'Elevation_m',
    'Rainfall_3day_avg', 'Rainfall_7day_avg', 'Rainfall_prev_day',
    'WaterLevel_prev_day', 'WaterLevel_change', 'WaterLevel_rising',
    'Month', 'IsWetSeason'
]

df["FloodRisk"] = rf_model.predict_proba(df[FEATURES])[:,1]

# --------------------------------------------------
# CAROUSEL-STYLE NAVIGATION
# --------------------------------------------------
panel = st.radio(
    "",
    [
        "PROJECT AHON",
        "Dataset & Feature Engineering",
        "Flood Risk Mapping",
        "Insights & Aggregations"
    ],
    horizontal=True
)

# --------------------------------------------------
# PANEL 1 — MAIN PANEL
# --------------------------------------------------
if panel == "PROJECT AHON":
    st.title("PROJECT – AHON")
    st.subheader("AI-Driven Flood Risk Prediction System")

    st.markdown("""
    **PROJECT AHON** is an intelligent flood-risk assessment platform that integrates:

    - Meteorological signals
    - Hydrological indicators
    - Temporal feature engineering
    - Machine learning–based risk inference
    - City-level geospatial visualization

    The system is designed for **decision support**, **early warning**, and **urban risk monitoring**.
    """)

# --------------------------------------------------
# PANEL 2 — DATASET + EDA + FEATURE ENGINEERING
# --------------------------------------------------
elif panel == "Dataset & Feature Engineering":
    st.header("Dataset Overview")
    st.dataframe(df.head())

    st.header("Feature Engineering Outputs")
    st.markdown("""
    Engineered features include:
    - Rolling rainfall averages
    - Lagged rainfall & water level
    - Water level trend indicators
    - Seasonal indicators
    """)

    st.dataframe(df[FEATURES].describe())

# --------------------------------------------------
# PANEL 3 — GEOSPATIAL FLOOD RISK MAPPING
# --------------------------------------------------
elif panel == "Flood Risk Mapping":
    st.header("City-Level Flood Risk Visualization")

    city_coords = {
        "Quezon City": (14.6760, 121.0437),
        "Manila": (14.5995, 120.9842),
        "Pasig": (14.5764, 121.0851),
        "Marikina": (14.6507, 121.1029)
    }

    latest = df.sort_values("Date").groupby("Location").tail(1)

    def risk_color(p):
        if p >= 0.7: return "red"
        elif p >= 0.4: return "orange"
        return "green"

    m = folium.Map(location=[14.6,121.0], zoom_start=11)

    for _, r in latest.iterrows():
        folium.CircleMarker(
            location=city_coords[r["Location"]],
            radius=16,
            color=risk_color(r["FloodRisk"]),
            fill=True,
            fill_opacity=0.75,
            popup=f"""
            <b>{r['Location']}</b><br>
            Flood Risk Score: {r['FloodRisk']:.2f}
            """
        ).add_to(m)

    st_folium(m, height=550, width=1000)

# --------------------------------------------------
# PANEL 4 — INSIGHTS & AGGREGATIONS
# --------------------------------------------------
elif panel == "Insights & Aggregations":
    st.header("City-Level Risk Insights")

    insights = df.groupby("Location").agg(
        Avg_Rainfall=("Rainfall_mm","mean"),
        Flood_Days=("FloodOccurrence","sum"),
        Avg_Predicted_Risk=("FloodRisk","mean")
    )

    st.dataframe(insights)

# --------------------------------------------------
# FOOTER (PERSISTENT)
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "**PROJECT AHON** | Developed for AI-Driven Flood Risk Assessment | © 2026",
    unsafe_allow_html=True
)

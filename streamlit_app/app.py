import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="PROJECT AHON",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# THEME
# -------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #e3f2fd, #ffffff);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/rf_flood_model.pkl")

rf_model = load_model()

# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------
def engineer_features(df):
    df = df.sort_values(["Location", "Date"])
    df["Rainfall_prev_day"] = df.groupby("Location")["Rainfall_mm"].shift(1)
    df["Rainfall_3day_avg"] = df.groupby("Location")["Rainfall_mm"].rolling(3,1).mean().reset_index(level=0,drop=True)
    df["Rainfall_7day_avg"] = df.groupby("Location")["Rainfall_mm"].rolling(7,1).mean().reset_index(level=0,drop=True)
    df["WaterLevel_prev_day"] = df.groupby("Location")["WaterLevel_m"].shift(1)
    df["WaterLevel_change"] = df["WaterLevel_m"] - df["WaterLevel_prev_day"]
    df["WaterLevel_rising"] = (df["WaterLevel_change"] > 0).astype(int)
    df["Month"] = df["Date"].dt.month
    df["IsWetSeason"] = df["Month"].isin([6,7,8,9,10,11]).astype(int)
    return df.fillna(0)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/Flood_Prediction_NCR_Philippines.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = engineer_features(df)
    return df

df = load_data()

# -------------------------------------------------
# FEATURE LIST
# -------------------------------------------------
FEATURES = [
    'Rainfall_mm', 'WaterLevel_m', 'SoilMoisture_pct', 'Elevation_m',
    'Rainfall_3day_avg', 'Rainfall_7day_avg', 'Rainfall_prev_day',
    'WaterLevel_prev_day', 'WaterLevel_change', 'WaterLevel_rising',
    'Month', 'IsWetSeason'
]

# -------------------------------------------------
# MAIN PANEL
# -------------------------------------------------
st.title("🌊 PROJECT AHON")
st.markdown("""
**AI-Driven Flood Risk Assessment System**  
Predicts flood likelihood using meteorological, hydrological, and geospatial signals.
""")

# -------------------------------------------------
# PREDICTIONS
# -------------------------------------------------
X = df[FEATURES]
df["FloodRisk"] = rf_model.predict_proba(X)[:,1]

latest = df.sort_values("Date").groupby("Location").tail(1)

# -------------------------------------------------
# MAP
# -------------------------------------------------
city_coords = {
    "Quezon City": (14.6760, 121.0437),
    "Manila": (14.5995, 120.9842),
    "Pasig": (14.5764, 121.0851),
    "Marikina": (14.6507, 121.1029),
}

def risk_color(p):
    if p >= 0.7: return "red"
    if p >= 0.4: return "orange"
    return "green"

m = folium.Map(location=[14.6,121.0], zoom_start=11)

for _, r in latest.iterrows():
    folium.CircleMarker(
        location=city_coords[r["Location"]],
        radius=14,
        color=risk_color(r["FloodRisk"]),
        fill=True,
        fill_opacity=0.7,
        popup=f"{r['Location']}<br>Risk: {r['FloodRisk']:.2f}"
    ).add_to(m)

st_folium(m, width=900, height=500)

# -------------------------------------------------
# INSIGHTS
# -------------------------------------------------
st.subheader("📊 City-Level Insights")

insights = df.groupby("Location").agg(
    Avg_Rainfall=("Rainfall_mm","mean"),
    Flood_Days=("FloodOccurrence","sum"),
    Avg_Risk=("FloodRisk","mean")
)

st.dataframe(insights)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.markdown("**Developed by PROJECT AHON Team**")

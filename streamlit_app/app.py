import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from streamlit_lottie import st_lottie
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="PROJECT – AHON",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------
# GLOBAL CSS (MOTION + THEME)
# ------------------------------------------------------
st.markdown("""
<style>

@keyframes fadeIn {
  from {opacity: 0; transform: translateY(15px);}
  to {opacity: 1; transform: translateY(0);}
}

.fade-in {
  animation: fadeIn 0.8s ease-in-out;
}

.motion-card {
  background: white;
  border-radius: 16px;
  padding: 1.4rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.1);
  transition: all 0.35s ease;
}

.motion-card:hover {
  transform: translateY(-8px) scale(1.02);
  box-shadow: 0 18px 50px rgba(0,0,0,0.18);
}

body {
  background: linear-gradient(135deg, #e3f2fd, #ffffff);
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# LOAD DATA & MODEL
# ------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/Flood_Prediction_NCR_Philippines.csv")

@st.cache_resource
def load_model():
    model = joblib.load("models/rf_flood_model.pkl")
    scaler = joblib.load("models/feature_scaler.pkl")
    return model, scaler

df = load_data()
rf_model, scaler = load_model()

# ------------------------------------------------------
# FEATURE ENGINEERING (SAFE INLINE VERSION)
# ------------------------------------------------------
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Location', 'Date'])

df['Month'] = df['Date'].dt.month
df['IsWetSeason'] = df['Month'].isin([6,7,8,9,10,11]).astype(int)

df['Rainfall_3day_avg'] = df.groupby('Location')['Rainfall_mm'].rolling(3,1).mean().reset_index(0,drop=True)
df['Rainfall_7day_avg'] = df.groupby('Location')['Rainfall_mm'].rolling(7,1).mean().reset_index(0,drop=True)

df['Rainfall_prev_day'] = df.groupby('Location')['Rainfall_mm'].shift(1)
df['WaterLevel_prev_day'] = df.groupby('Location')['WaterLevel_m'].shift(1)

df['WaterLevel_change'] = df['WaterLevel_m'] - df['WaterLevel_prev_day']
df['WaterLevel_rising'] = (df['WaterLevel_change'] > 0).astype(int)

df = df.dropna()

FEATURE_COLS = [
    'Rainfall_mm','WaterLevel_m','SoilMoisture_pct','Elevation_m',
    'Rainfall_3day_avg','Rainfall_7day_avg','Rainfall_prev_day',
    'WaterLevel_prev_day','WaterLevel_change','WaterLevel_rising',
    'Month','IsWetSeason'
]

# ------------------------------------------------------
# NAVIGATION (CAROUSEL-LIKE FLOW)
# ------------------------------------------------------
panel = st.radio(
    "",
    ["PROJECT AHON", "Dataset & EDA", "Insights & Mapping"],
    horizontal=True
)

# ------------------------------------------------------
# PANEL 1 — HERO
# ------------------------------------------------------
if panel == "PROJECT AHON":
    col1, col2 = st.columns([1.3,1])

    with col1:
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.title("PROJECT – AHON")
        st.subheader("AI-Powered Flood Risk Intelligence")

        st.markdown("""
        PROJECT AHON transforms environmental signals into
        **actionable flood risk intelligence** using:

        • Feature engineering  
        • Machine learning inference  
        • Temporal analysis  
        • Geospatial visualization  

        Built for **early warning**, **urban resilience**, and **decision support**.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with open("assets/flood_animation.json") as f:
            st_lottie(json.load(f), height=320, speed=1)

# ------------------------------------------------------
# PANEL 2 — DATASET + EDA
# ------------------------------------------------------
elif panel == "Dataset & EDA":
    st.header("Dataset Overview & Feature Engineering")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="motion-card">', unsafe_allow_html=True)
        st.metric("Total Records", len(df))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="motion-card">', unsafe_allow_html=True)
        st.metric("Cities Covered", df['Location'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="motion-card">', unsafe_allow_html=True)
        st.metric("Flood Events", int(df['FloodOccurrence'].sum()))
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Rainfall vs Flood Occurrence")
    fig, ax = plt.subplots()
    sns.boxplot(x='FloodOccurrence', y='Rainfall_mm', data=df, ax=ax)
    st.pyplot(fig)

# ------------------------------------------------------
# PANEL 3 — INSIGHTS + MAP
# ------------------------------------------------------
else:
    st.header("Flood Risk Insights & Mapping")

    X = df[FEATURE_COLS]
    df['FloodRisk'] = rf_model.predict_proba(X)[:,1]

    city_latest = df.sort_values('Date').groupby('Location').tail(1)

    city_coords = {
        'Quezon City': (14.6760,121.0437),
        'Marikina': (14.6507,121.1029),
        'Pasig': (14.5764,121.0851),
        'Manila': (14.5995,120.9842)
    }

    m = folium.Map(location=[14.6,121.0], zoom_start=11)

    def risk_color(p):
        return 'red' if p>=0.7 else 'orange' if p>=0.4 else 'green'

    for _, row in city_latest.iterrows():
        folium.CircleMarker(
            location=city_coords[row['Location']],
            radius=18,
            color=risk_color(row['FloodRisk']),
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['Location']}<br>Risk: {row['FloodRisk']:.2f}"
        ).add_to(m)

    st_folium(m, width=900, height=520)

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.markdown("""
<hr>
<p style='text-align:center;'>
PROJECT – AHON | Developed for Data Science & AI Applications<br>
© 2026
</p>
""", unsafe_allow_html=True)

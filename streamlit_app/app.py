import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import IsolationForest

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="PROJECT – AHON",
    page_icon="🌊",
    layout="wide"
)

# ==============================
# GLOBAL CSS
# ==============================
st.markdown("""
<style>
/* Body Gradient Background */
body {
    background: linear-gradient(135deg, #e3f2fd, #ffffff);
    font-family: 'Segoe UI', sans-serif;
}

/* Hero Section */
.hero {
    background: linear-gradient(135deg, #1e88e5, #42a5f5);
    padding: 3rem;
    border-radius: 25px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 25px rgba(30,136,229,0.25);
    animation: pulse 3s infinite;
}
@keyframes pulse {
    0% { transform: scale(1); opacity: 0.95; }
    50% { transform: scale(1.02); opacity: 1; }
    100% { transform: scale(1); opacity: 0.95; }
}

/* Card Styling */
.card {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0px 8px 25px rgba(30,136,229,0.15);
    transition: transform 0.3s ease;
}
.card:hover {
    transform: translateY(-5px);
}

/* Footer */
footer {
    text-align: center;
    opacity: 0.7;
    margin-top: 2rem;
}

/* Table Styling */
.dataframe {
    border-collapse: collapse;
    width: 100%;
}
.dataframe th, .dataframe td {
    padding: 8px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("🌊 PROJECT – AHON")
st.sidebar.write("AI-Powered Flood Risk Intelligence System")

panel = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Main Panel",
        "📊 Dataset & EDA",
        "🧠 Feature Engineering",
        "🌧️ Anomaly Detection",
        "🗺️ Geospatial Mapping",
        "📈 Insights & Aggregations"
    ]
)

# ==============================
# DATA UPLOAD
# ==============================
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

uploaded_file = st.sidebar.file_uploader(
    "Upload Flood Dataset (CSV)",
    type=["csv"]
)

df = load_data(uploaded_file) if uploaded_file else None

# ==============================
# MAIN PANEL
# ==============================
if panel == "🏠 Main Panel":
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.title("🌊 PROJECT – AHON")
    st.subheader("AI-Powered Flood Risk Intelligence System")
    st.write("""
    **PROJECT – AHON** uses meteorological data, anomaly detection, 
    and geospatial visualization to provide **early flood risk alerts**.

    **Key Features:**
    - Time-aware feature engineering
    - Machine learning flood prediction
    - Rainfall anomaly detection
    - Interactive geospatial mapping
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# DATASET & EDA
# ==============================
elif panel == "📊 Dataset & EDA":
    if df is None:
        st.warning("Please upload a dataset to continue.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Dataset Overview")
        st.dataframe(df.head(), use_container_width=True)
        st.write("Shape:", df.shape)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# FEATURE ENGINEERING
# ==============================
elif panel == "🧠 Feature Engineering":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        df = df.copy()
        df["Rainfall_3day_avg"] = df["Rainfall_mm"].rolling(3).mean()
        df["Rainfall_7day_avg"] = df["Rainfall_mm"].rolling(7).mean()
        df["WaterLevel_change"] = df["WaterLevel_m"].diff()
        df["WaterLevel_rising"] = (df["WaterLevel_change"] > 0).astype(int)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Engineered Features (Sample)")
        st.dataframe(
            df[
                ["Rainfall_3day_avg", "Rainfall_7day_avg", "WaterLevel_change", "WaterLevel_rising"]
            ].head(), use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# ANOMALY DETECTION
# ==============================
elif panel == "🌧️ Anomaly Detection":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        iso = IsolationForest(contamination=0.05, random_state=42)
        df["Rainfall_Anomaly"] = iso.fit_predict(df[["Rainfall_mm"]].fillna(0))

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Rainfall Anomaly Detection")
        anomalies = df[df["Rainfall_Anomaly"] == -1][["Rainfall_mm"]]
        st.dataframe(anomalies.head(), use_container_width=True)
        st.info("Detected extreme rainfall deviations using Isolation Forest.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# GEOSPATIAL MAPPING
# ==============================
elif panel == "🗺️ Geospatial Mapping":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Flood Risk Map")

        m = folium.Map(location=[14.6, 121.0], zoom_start=10)
        for _, row in df.head(100).iterrows():
            color = "red" if row.get("FloodOccurrence", 0) == 1 else "blue"
            folium.CircleMarker(
                location=[row.get("Latitude", 14.6), row.get("Longitude", 121.0)],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(m)

        st_folium(m, width=900, height=500)
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# INSIGHTS & AGGREGATIONS
# ==============================
elif panel == "📈 Insights & Aggregations":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Key Insights")

        col1, col2 = st.columns(2)
        col1.metric("Average Rainfall (mm)", f"{df['Rainfall_mm'].mean():.2f}")
        col2.metric("Flood Occurrence Rate", f"{df['FloodOccurrence'].mean()*100:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# FOOTER
# ==============================
st.markdown("""
<hr>
<footer>
Developed by PROJECT – AHON Team<br>
AI • Flood Risk • Geospatial Intelligence
</footer>
""", unsafe_allow_html=True)

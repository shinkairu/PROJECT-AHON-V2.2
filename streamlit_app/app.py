import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import IsolationForest
import altair as alt

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="PROJECT – AHON",
    page_icon="🌊",
    layout="wide"
)

# ==============================
# GLOBAL CSS – MODERN SAAS UI
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body, .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #f5f7fb;
    color: #1f2937;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

/* HERO */
.hero {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    border-radius: 30px;
    padding: 3.5rem;
    color: white;
    box-shadow: 0 25px 60px rgba(30,58,138,0.35);
    margin-bottom: 2rem;
}

.hero h1 {
    font-size: 3.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.hero span {
    color: #7dd3fc;
}

.hero p {
    max-width: 520px;
    font-size: 1.1rem;
    opacity: 0.95;
    line-height: 1.6;
}

/* CARDS */
.card {
    background: white;
    border-radius: 22px;
    padding: 1.8rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
}

/* METRICS */
.stMetric {
    background: white;
    border-radius: 20px;
    padding: 1.2rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
}

/* BUTTONS */
.stButton>button {
    border-radius: 14px;
    padding: 0.7rem 1.4rem;
    font-weight: 600;
}

/* FOOTER */
footer {
    text-align: center;
    color: #6b7280;
    margin-top: 3rem;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR – NAVIGATION
# ==============================
st.sidebar.title("🌊 PROJECT AHON")
st.sidebar.caption("AI Flood Risk Intelligence")

panel = st.sidebar.radio(
    "",
    [
        "🏠 Main Panel",
        "📊 Dataset & EDA",
        "🧠 Feature Engineering",
        "🌧️ Anomaly Detection",
        "🗺️ Geospatial Mapping",
        "📈 Insights"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success("System Online")
st.sidebar.caption("v1.2.0 Stable")

uploaded_file = st.sidebar.file_uploader(
    "Upload Flood Dataset (CSV)",
    type=["csv"]
)

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(uploaded_file) if uploaded_file else None

# ==============================
# MAIN PANEL
# ==============================
if panel == "🏠 Main Panel":
    st.markdown("""
    <div class="hero">
        <small>🟢 Live System Monitoring</small>
        <h1>Predict Floods.<br><span>Protect Communities.</span></h1>
        <p>
            Project AHON leverages AI-powered meteorological analysis and
            temporal pattern recognition to provide early flood risk insights
            for smarter decision-making.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.button("📊 Explore Dataset", use_container_width=True)
    with c2:
        st.button("🗺️ View Risk Map", use_container_width=True)

# ==============================
# DATASET & EDA
# ==============================
elif panel == "📊 Dataset & EDA":
    if df is None:
        st.warning("Please upload a dataset.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

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

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Engineered Features")
        st.dataframe(
            df[[
                "Rainfall_3day_avg",
                "Rainfall_7day_avg",
                "WaterLevel_change",
                "WaterLevel_rising"
            ]].head(),
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# ANOMALY DETECTION
# ==============================
elif panel == "🌧️ Anomaly Detection":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        df = df.copy()
        iso = IsolationForest(contamination=0.05, random_state=42)
        df["Anomaly"] = iso.fit_predict(df[["Rainfall_mm"]].fillna(0))
        df["Status"] = df["Anomaly"].map({1: "Normal", -1: "Anomaly"})

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Rainfall Anomaly Detection")

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            chart = alt.Chart(df.dropna()).mark_circle(size=30).encode(
                x="Date:T",
                y="Rainfall_mm:Q",
                color=alt.Color(
                    "Status",
                    scale=alt.Scale(domain=["Normal", "Anomaly"],
                                    range=["#2563eb", "#dc2626"])
                ),
                tooltip=["Date", "Rainfall_mm", "Status"]
            ).properties(height=320)

            st.altair_chart(chart, use_container_width=True)

        st.dataframe(df[df["Status"] == "Anomaly"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# GEOSPATIAL MAPPING
# ==============================
elif panel == "🗺️ Geospatial Mapping":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Flood Risk Map")

        m = folium.Map(location=[14.6, 121.0], zoom_start=10)
        for _, row in df.head(200).iterrows():
            color = "red" if row.get("FloodOccurrence", 0) == 1 else "blue"
            folium.CircleMarker(
                location=[row.get("Latitude", 14.6), row.get("Longitude", 121.0)],
                radius=5,
                color=color,
                fill=True
            ).add_to(m)

        st_folium(m, width=1000, height=520)
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# INSIGHTS
# ==============================
elif panel == "📈 Insights":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Key Metrics")

        avg_rain = round(df["Rainfall_mm"].mean(), 2)
        flood_rate = round(df["FloodOccurrence"].mean() * 100, 2)

        c1, c2 = st.columns(2)
        c1.metric("Average Rainfall (mm)", avg_rain)
        c2.metric("Flood Occurrence Rate (%)", flood_rate)

        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# FOOTER
# ==============================
st.markdown("""
<hr>
<footer>
PROJECT AHON • AI Flood Risk Intelligence<br>
SDG 11 – Sustainable Cities & Communities
</footer>
""", unsafe_allow_html=True)

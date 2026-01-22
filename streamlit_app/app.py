import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import IsolationForest

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="PROJECT AHON",
    page_icon="🌊",
    layout="wide"
)

# ==================================================
# GLOBAL THEME & UI
# ==================================================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f0f4f8, #ffffff);
}
.section-card {
    background-color: white;
    padding: 2rem;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 2rem;
}
.hero {
    background: linear-gradient(135deg, #1565c0, #42a5f5);
    padding: 3rem;
    border-radius: 24px;
    color: white;
    margin-bottom: 2rem;
}
.metric-box {
    background: #e3f2fd;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
}
footer {
    text-align: center;
    opacity: 0.6;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
st.sidebar.title("🌊 PROJECT AHON")
panel = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📊 Dataset Explorer",
        "🧠 Feature Engineering",
        "🌧️ Anomaly Detection",
        "🗺️ Flood Mapping",
        "📈 Insights"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Flood Dataset (CSV)",
    type=["csv"]
)

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

df = load_data(uploaded_file) if uploaded_file else None

# ==================================================
# OVERVIEW
# ==================================================
if panel == "🏠 Overview":
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    st.title("PROJECT AHON")
    st.subheader("AI-Powered Flood Risk Intelligence System")
    st.write("""
    PROJECT AHON is a research prototype inspired by **UP Project NOAH** that demonstrates
    how machine learning and geospatial analytics can support flood preparedness in the Philippines.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# DATASET EXPLORER
# ==================================================
elif panel == "📊 Dataset Explorer":
    if df is None:
        st.warning("Please upload a dataset.")
    else:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20))
        st.write("Shape:", df.shape)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())
        st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# FEATURE ENGINEERING
# ==================================================
elif panel == "🧠 Feature Engineering":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        df_feat = df.copy()
        df_feat["Rainfall_3day_avg"] = df_feat["Rainfall_mm"].rolling(3).mean()
        df_feat["Rainfall_7day_avg"] = df_feat["Rainfall_mm"].rolling(7).mean()
        df_feat["WaterLevel_change"] = df_feat["WaterLevel_m"].diff()

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Engineered Features")
        st.dataframe(df_feat[
            ["Rainfall_3day_avg", "Rainfall_7day_avg", "WaterLevel_change"]
        ].head())
        st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# ANOMALY DETECTION
# ==================================================
elif panel == "🌧️ Anomaly Detection":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        iso = IsolationForest(contamination=0.05, random_state=42)
        df["Rainfall_Anomaly"] = iso.fit_predict(df[["Rainfall_mm"]].fillna(0))

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Extreme Rainfall Events")
        st.dataframe(df[df["Rainfall_Anomaly"] == -1][["Rainfall_mm"]].head())
        st.info("Anomalies detected using Isolation Forest.")
        st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# FLOOD MAPPING
# ==================================================
elif panel == "🗺️ Flood Mapping":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Flood Occurrence Map")

        m = folium.Map(location=[14.6, 121.0], zoom_start=10)

        for _, row in df.head(200).iterrows():
            color = "red" if row.get("FloodOccurrence", 0) == 1 else "blue"
            folium.CircleMarker(
                location=[
                    row.get("Latitude", 14.6),
                    row.get("Longitude", 121.0)
                ],
                radius=5,
                color=color,
                fill=True
            ).add_to(m)

        st_folium(m, width=1000, height=500)
        st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# INSIGHTS
# ==================================================
elif panel == "📈 Insights":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
            st.metric("Avg Rainfall (mm)", round(df["Rainfall_mm"].mean(), 2))
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
            st.metric(
                "Flood Rate",
                f"{df['FloodOccurrence'].mean() * 100:.2f}%"
            )
            st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<hr>
<footer>
PROJECT AHON • AI Flood Risk Intelligence<br>
Academic Prototype – Philippines
</footer>
""", unsafe_allow_html=True)

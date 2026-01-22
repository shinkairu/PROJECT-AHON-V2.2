import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
# GLOBAL CSS – MODERN MATERIAL DESIGN
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

body, .stApp {
    font-family: 'Roboto', sans-serif;
    background: linear-gradient(135deg, #e3f2fd, #ffffff);
    color: #333;
}

/* HERO SECTION */
.hero {
    background: linear-gradient(135deg, #1e88e5, #90caf9);
    border-radius: 30px;
    padding: 4rem 2rem;
    text-align: center;
    color: white;
    box-shadow: 0px 10px 40px rgba(30,136,229,0.25);
    animation: pulse 4s infinite;
    margin-bottom: 2rem;
}
.hero h1 { font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem; }
.hero p { font-size: 1.2rem; margin-top:0.2rem; }

/* PULSE ANIMATION */
@keyframes pulse {
    0% { transform: scale(1); opacity: 0.95; }
    50% { transform: scale(1.02); opacity: 1; }
    100% { transform: scale(1); opacity: 0.95; }
}

/* CARD STYLE */
.card {
    background: linear-gradient(145deg, #ffffff, #f0f4f8);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0px 12px 28px rgba(30,136,229,0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover { transform: translateY(-8px); box-shadow: 0px 20px 40px rgba(30,136,229,0.35); }

/* BUTTON STYLE */
.stButton>button {
    background: linear-gradient(135deg, #1e88e5, #90caf9);
    color: white;
    border-radius: 12px;
    padding: 0.6rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover { transform: scale(1.05); }

/* METRICS */
.stMetric {
    background: linear-gradient(145deg, #ffffff, #f0f4f8);
    border-radius: 20px;
    padding: 1rem;
    box-shadow: 0px 8px 20px rgba(30,136,229,0.15);
    margin-bottom: 1rem;
}

/* SIDEBAR */
.stSidebar {
    background: linear-gradient(180deg, #1e88e5, #90caf9);
    color: white;
    border-radius: 15px;
    padding: 1rem;
}
.stSidebar .css-1d391kg { color: white; }

/* TABLE HIGHLIGHT ON HOVER */
.stDataFrame tbody tr:hover {
    background-color: rgba(30,136,229,0.08);
    transition: background 0.3s ease;
}

/* FOOTER */
footer { text-align: center; opacity: 0.7; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR – NAVIGATION & FILE UPLOAD
# ==============================
st.sidebar.title("🌊 PROJECT – AHON")

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
    st.markdown("""
    <div class='hero'>
        <h1>PROJECT – AHON</h1>
        <p>AI-Powered Flood Risk Intelligence System</p>
        <p><strong>Leveraging AI, anomaly detection, and geospatial mapping for early flood insights</strong></p>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# DATASET & EDA
# ==============================
elif panel == "📊 Dataset & EDA":
    if df is None:
        st.warning("Please upload a dataset to continue.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dataset Overview")
        st.dataframe(df.head())
        st.write(f"Shape: {df.shape}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())
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
        st.subheader("Engineered Features (Sample)")
        st.dataframe(df[[
            "Rainfall_3day_avg","Rainfall_7day_avg","WaterLevel_change","WaterLevel_rising"
        ]].head())
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
        df["Rainfall_Anomaly"] = iso.fit_predict(df[["Rainfall_mm"]].fillna(0))
        df["Anomaly_Flag"] = df["Rainfall_Anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Rainfall Anomaly Detection")

        anomalies = df[df["Rainfall_Anomaly"] == -1].copy()

        if anomalies.empty:
            st.info("No anomalies detected in the uploaded dataset.")
        else:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df_plot = df.dropna(subset=['Date', 'Rainfall_mm'])

                scatter = alt.Chart(df_plot).mark_circle(size=20).encode(
                    x=alt.X('Date:T', axis=alt.Axis(title='Date', format='%Y-%m-%d')),
                    y=alt.Y('Rainfall_mm:Q', title='Rainfall (mm)'),
                    color=alt.Color('Anomaly_Flag:N', scale=alt.Scale(domain=['Normal','Anomaly'], range=['#1e88e5','#e53935'])),
                    tooltip=['Date','Rainfall_mm','Anomaly_Flag']
                ).properties(
                    width=800,
                    height=300
                )
                st.altair_chart(scatter, use_container_width=True)
            else:
                st.warning("No 'Date' column found – scatter plot not available.")

            anomalies = anomalies.sort_values(by="Rainfall_mm", ascending=False)
            st.dataframe(anomalies)
            st.info("Red dots in the plot = detected extreme rainfall deviations.")

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
        for _, row in df.head(100).iterrows():
            color = "red" if row.get("FloodOccurrence",0)==1 else "blue"
            folium.CircleMarker(
                location=[row.get("Latitude",14.6), row.get("Longitude",121.0)],
                radius=5, color=color, fill=True
            ).add_to(m)

        st_folium(m, width=900, height=500)
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# INSIGHTS & AGGREGATIONS
# ==============================
elif panel == "📈 Insights & Aggregations":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Key Insights")

        avg_rainfall = round(df["Rainfall_mm"].mean(), 2)
        flood_rate = round(df["FloodOccurrence"].mean() * 100, 2)

        cols = st.columns(2)
        with cols[0]:
            st.metric("Average Rainfall (mm)", avg_rainfall)
        with cols[1]:
            st.metric("Flood Occurrence Rate (%)", f"{flood_rate}%")

        data = pd.DataFrame({
            'Metric': ['Average Rainfall (mm)', 'Flood Occurrence Rate (%)'],
            'Value': [avg_rainfall, flood_rate],
            'Color': ['#1e88e5', '#e53935']
        })

        chart = alt.Chart(data).mark_bar(size=40).encode(
            y=alt.Y('Metric', sort=None, title=''),
            x=alt.X('Value', title='Value / Percent (%)'),
            color=alt.Color('Color:N', scale=None, legend=None),
            tooltip=['Metric', 'Value']
        ).properties(
            width=600,
            height=200
        )

        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

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

import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="PROJECT AHON | Flood Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# ENHANCED DESIGN SYSTEM
# ==============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Outfit:wght@700&display=swap');
    
    :root {
        --primary: #1e88e5;
        --secondary: #90caf9;
        --accent: #00d4ff;
        --dark: #0f172a;
    }

    .main {
        background-color: #f8fafc;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
    }

    p, span, div {
        font-family: 'Inter', sans-serif !important;
    }

    /* Glassmorphism Cards */
    .stCard {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }

    /* Hero Gradient Section */
    .hero-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e88e5 100%);
        padding: 4rem 2rem;
        border-radius: 32px;
        color: white;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .hero-container::after {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 99px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(5px);
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }

    .pulse {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 0 rgba(16, 185, 129, 0.4);
        animation: pulse-green 2s infinite;
    }

    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: var(--primary) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# MOCK DATA GENERATOR
# ==============================
def generate_mock_data():
    dates = [datetime.now() - timedelta(days=x) for x in range(100)]
    data = {
        'Date': sorted(dates),
        'Rainfall_mm': np.random.normal(50, 30, 100).clip(0, 300),
        'WaterLevel_m': np.random.normal(10, 2, 100).clip(5, 20),
        'Latitude': [14.6 + np.random.uniform(-0.1, 0.1) for _ in range(100)],
        'Longitude': [121.0 + np.random.uniform(-0.1, 0.1) for _ in range(100)],
    }
    df = pd.DataFrame(data)
    df['FloodOccurrence'] = ((df['Rainfall_mm'] > 150) | (df['WaterLevel_m'] > 14)).astype(int)
    return df

# ==============================
# NAVIGATION
# ==============================
with st.sidebar:
    st.markdown("### 🌊 AHON INTELLIGENCE")
    panel = st.selectbox(
        "Navigation",
        ["Dashboard", "Dataset Explorer", "Signal Processing", "Anomaly Map", "Forecasting"]
    )
    
    st.markdown("---")
    st.markdown("### Data Source")
    uploaded_file = st.file_uploader("Upload Hydro Data (CSV)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = generate_mock_data()
        st.info("Using simulated sensor data.")

# ==============================
# DASHBOARD PANEL
# ==============================
if panel == "Dashboard":
    st.markdown("""
        <div class="hero-container">
            <div class="status-badge"><div class="pulse"></div> System Monitoring Active</div>
            <h1 style="font-size: 3.5rem; margin-bottom: 1rem;">Predict Floods. <span style="color: #00d4ff;">Protect Lives.</span></h1>
            <p style="font-size: 1.2rem; opacity: 0.9; max-width: 700px;">
                Project AHON integrates real-time meteorological signals and AI-driven anomaly detection 
                to safeguard coastal and urban communities.
            </p>
        </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Avg Rainfall", f"{df['Rainfall_mm'].mean():.1f}mm", "+2.4%")
    with m2:
        st.metric("Risk Level", "HIGH" if df['FloodOccurrence'].mean() > 0.1 else "LOW", delta_color="inverse")
    with m3:
        st.metric("Active Sensors", "42/45", "93%")
    with m4:
        st.metric("Alerts (24h)", "3", "CRITICAL")

    st.markdown("### 📈 Hydrological Overview")
    fig = px.area(df, x='Date', y='WaterLevel_m', 
                  title='Water Level Trends (m)',
                  color_discrete_sequence=['#1e88e5'])
    fig.update_layout(template='plotly_white', margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# SIGNAL PROCESSING (ENGINEERING)
# ==============================
elif panel == "Signal Processing":
    st.title("🧠 Feature Engineering & Signal Processing")
    
    # Processing Logic
    df['Rain_MA3'] = df['Rainfall_mm'].rolling(3).mean()
    df['Rain_MA7'] = df['Rainfall_mm'].rolling(7).mean()
    df['WL_Delta'] = df['WaterLevel_m'].diff()

    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("#### Moving Averages")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Rainfall_mm'], name='Raw Rain', opacity=0.3))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Rain_MA3'], name='3D Signal', line=dict(color='#1e88e5')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Rain_MA7'], name='7D Trend', line=dict(color='#f59e0b', dash='dot')))
        fig.update_layout(template='plotly_white', height=450)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Engineering Log")
        st.info("**Water Level Gradient** calculated to detect rapid rising events.")
        st.success("**Time-aware signals** extracted from 100+ sensor points.")
        st.json({"Pipeline": "Active", "Latency": "14ms", "Features": 12})

# ==============================
# ANOMALY MAP
# ==============================
elif panel == "Anomaly Map":
    st.title("🗺️ Geospatial Risk Intel")
    
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly'] = iso.fit_predict(df[['Rainfall_mm', 'WaterLevel_m']])
    
    m = folium.Map(location=[14.6, 121.0], zoom_start=12, tiles='CartoDB voyager')
    
    for _, row in df.iterrows():
        color = 'red' if row['Anomaly'] == -1 else 'blue'
        radius = 8 if row['Anomaly'] == -1 else 4
        folium.CircleMarker(
            [row['Latitude'], row['Longitude']],
            radius=radius,
            color=color,
            fill=True,
            popup=f"Rain: {row['Rainfall_mm']:.1f}mm"
        ).add_to(m)

    st_folium(m, width="100%", height=600)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; opacity: 0.5; padding: 2rem;">
        PROJECT AHON v2.0 • Powered by AI • Flood Risk & Geospatial Intelligence
    </div>
""", unsafe_allow_html=True)

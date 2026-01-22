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
.card:hover {
    transform: translateY(-8px);
    box-shadow: 0px 20px 40px rgba(30,136,229,0.35);
}

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

/* ===== FILE UPLOADER FIX ===== */
.stSidebar .stFileUploader label {
    color: #ffffff !important;
    font-weight: 500;
}

.stSidebar .stFileUploader small {
    color: #e3f2fd !important;
}

.stSidebar .stFileUploader section {
    background: #e3f2fd !important;
    border-radius: 12px;
}

.stSidebar .stFileUploader button {
    background: #ffffff !important;
    color: #1e88e5 !important;
    border-radius: 10px;
    font-weight: 600;
}

/* TABLE HOVER */
.stDataFrame tbody tr:hover {
    background-color: rgba(30,136,229,0.08);
    transition: background 0.3s ease;
}

/* FOOTER */
footer {
    text-align: center;
    opacity: 0.7;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR – NAVIGATION
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

# ==============================
# DATA UPLOAD
# ==============================
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Flood Dataset (CSV)",
    type=["csv"]
)

df = load_data(uploaded_file) if uploaded_file else None

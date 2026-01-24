import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import altair as alt

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="PROJECT – AHON", page_icon="🌊", layout="wide")

# ==============================
# GLOBAL CSS
# ==============================
st.markdown("""
<style>
body, .stApp {font-family: 'Inter', sans-serif; background-color: #f5f7fb;}
.hero {background: linear-gradient(135deg,#1e3a8a,#3b82f6); border-radius:30px; padding:3.5rem; color:white;}
.card {background:white; border-radius:22px; padding:1.8rem; box-shadow:0 10px 30px rgba(0,0,0,0.06); margin-bottom:1.5rem;}
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("🌊 PROJECT AHON")
panel = st.sidebar.radio("", ["🏠 Main Panel","📊 Dataset & EDA","🧠 Feature Engineering","🌧️ Anomaly Detection","🗺️ Geospatial Mapping","📈 Insights"])
uploaded_file = st.sidebar.file_uploader("Upload Flood Dataset (CSV)", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(uploaded_file) if uploaded_file else None

required_cols = ["Date", "Rainfall_mm", "WaterLevel_m"]
if df is not None and not all(col in df.columns for col in required_cols):
    st.error(f"Dataset must contain columns: {required_cols}")
    df = None

# ==============================
# MAIN PANEL
# ==============================
if panel == "🏠 Main Panel":
    st.markdown("""<div class="hero"><h1>Predict Floods. Protect Communities.</h1>
    <p>AI-driven rainfall and water-level analysis for early flood risk detection.</p></div>""", unsafe_allow_html=True)

# ==============================
# DATASET & EDA
# ==============================
elif panel == "📊 Dataset & EDA" and df is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# FEATURE ENGINEERING (FIXED)
# ==============================
elif panel == "🧠 Feature Engineering":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        fe_df = df.copy()
        fe_df["Date"] = pd.to_datetime(fe_df["Date"], errors="coerce")
        fe_df = fe_df.sort_values("Date")

        fe_df["Year"] = fe_df["Date"].dt.year
        fe_df["Rainfall_3day_avg"] = fe_df["Rainfall_mm"].rolling(3, min_periods=1).mean()
        fe_df["Rainfall_7day_avg"] = fe_df["Rainfall_mm"].rolling(7, min_periods=1).mean()
        fe_df["WaterLevel_change"] = fe_df["WaterLevel_m"].diff().fillna(0)
        fe_df["WaterLevel_rising"] = (fe_df["WaterLevel_change"] > 0).astype(int)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Engineered Features Preview")
        st.dataframe(fe_df.head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        years_available = sorted(fe_df["Year"].dropna().unique())
        selected_years = st.multiselect("📅 Select year(s) to compare", years_available, default=years_available)
        filtered_df = fe_df[fe_df["Year"].isin(selected_years)]

        if not filtered_df.empty:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            x_axis = alt.X("monthdate(Date):T", title="Month")

            for col, title in [
                ("Rainfall_3day_avg","Rainfall 3-Day Avg"),
                ("Rainfall_7day_avg","Rainfall 7-Day Avg"),
                ("WaterLevel_change","Water Level Change"),
                ("WaterLevel_rising","Water Rising Indicator")
            ]:
                chart = alt.Chart(filtered_df).mark_line().encode(
                    x=x_axis,
                    y=alt.Y(f"{col}:Q", title=title),
                    color="Year:N",
                    tooltip=["Date:T","Year:N",col]
                ).properties(height=280, title=title).interactive()
                st.altair_chart(chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# ANOMALY DETECTION
# ==============================
elif panel == "🌧️ Anomaly Detection" and df is not None:
    temp_df = df.copy()
    iso = IsolationForest(contamination=0.05, random_state=42)
    temp_df["Anomaly"] = iso.fit_predict(temp_df[["Rainfall_mm"]].fillna(0))
    temp_df["Status"] = temp_df["Anomaly"].map({1:"Normal",-1:"Anomaly"})

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    chart = alt.Chart(temp_df).mark_circle(size=40).encode(
        x="Rainfall_mm:Q", y="WaterLevel_m:Q", color="Status:N",
        tooltip=["Rainfall_mm","WaterLevel_m","Status"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# GEOSPATIAL MAP
# ==============================
elif panel == "🗺️ Geospatial Mapping" and df is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    m = folium.Map(location=[14.6,121.0], zoom_start=10)
    for _, row in df.head(200).iterrows():
        folium.CircleMarker(
            location=[row.get("Latitude",14.6), row.get("Longitude",121.0)],
            radius=5, color="blue", fill=True
        ).add_to(m)
    st_folium(m, width=1000, height=500)
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# INSIGHTS
# ==============================
elif panel == "📈 Insights" and df is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Flood Probability Model")

    model_df = df.copy().dropna()
    model_df["Rainfall_3day_avg"] = model_df["Rainfall_mm"].rolling(3, min_periods=1).mean()
    model_df["Rainfall_7day_avg"] = model_df["Rainfall_mm"].rolling(7, min_periods=1).mean()
    model_df["WaterLevel_change"] = model_df["WaterLevel_m"].diff().fillna(0)

    if "FloodOccurrence" in model_df.columns:
        X = model_df[["Rainfall_mm","WaterLevel_m","Rainfall_3day_avg","Rainfall_7day_avg","WaterLevel_change"]]
        y = model_df["FloodOccurrence"]
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X,y)
        st.success("Model trained successfully.")
    else:
        st.info("FloodOccurrence column not found for training.")
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# FOOTER
# ==============================
st.markdown("<hr><center>PROJECT AHON • AI Flood Risk Intelligence</center>", unsafe_allow_html=True)

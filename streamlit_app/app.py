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
st.set_page_config(
    page_title="PROJECT – AHON",
    page_icon="🌊",
    layout="wide"
)

# ==============================
# GLOBAL CSS – MODERN UI
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body, .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #f5f7fb;
    color: #1f2937;
}

.hero {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    border-radius: 30px;
    padding: 3.5rem;
    color: white;
    box-shadow: 0 25px 60px rgba(30,58,138,0.35);
    margin-bottom: 2rem;
}

.card {
    background: white;
    border-radius: 22px;
    padding: 1.8rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("🌊 PROJECT AHON")
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

uploaded_file = st.sidebar.file_uploader("Upload Flood Dataset (CSV)", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(uploaded_file) if uploaded_file else None

# ==============================
# CHECK REQUIRED COLUMNS
# ==============================
required_cols = ["Date", "Rainfall_mm", "WaterLevel_m"]
if df is not None and not all(col in df.columns for col in required_cols):
    st.error(f"Dataset must contain columns: {required_cols}")
    df = None

# ==============================
# MAIN PANEL
# ==============================
if panel == "🏠 Main Panel":
    st.markdown("""
    <div class="hero">
        <h1>Predict Floods. Protect Communities.</h1>
        <p>AI-driven rainfall and water-level analysis for early flood risk detection.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# FEATURE ENGINEERING
# ==============================
elif panel == "🧠 Feature Engineering":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        df = df.copy()

        # ---- Feature Engineering ----
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Rainfall_3day_avg"] = df["Rainfall_mm"].rolling(3, min_periods=1).mean()
        df["Rainfall_7day_avg"] = df["Rainfall_mm"].rolling(7, min_periods=1).mean()
        df["WaterLevel_change"] = df["WaterLevel_m"].diff().fillna(0)
        df["WaterLevel_rising"] = (df["WaterLevel_change"] > 0).astype(int)

        # ---- Show Engineered Table ----
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Engineered Features Preview")
        st.dataframe(
            df[["Date", "Rainfall_3day_avg", "Rainfall_7day_avg", "WaterLevel_change", "WaterLevel_rising"]].head(10),
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ==============================
        # YEAR FILTER
        # ==============================
        years_available = sorted(df["Year"].dropna().unique())
        selected_years = st.multiselect(
            "📅 Select year(s) to compare",
            years_available,
            default=years_available
        )

        filtered_df = df[df["Year"].isin(selected_years)]
        if filtered_df.empty:
            st.warning("No data available for the selected year(s).")
        else:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📊 Feature Trends Comparison by Year")

            # ---- Altair Charts ----
            x_axis = alt.X("Month:O", title="Month")

            # Rainfall 3-Day Avg
            chart1 = alt.Chart(filtered_df).mark_line().encode(
                x=x_axis,
                y=alt.Y("Rainfall_3day_avg:Q", title="Rainfall 3-Day Avg"),
                color=alt.Color("Year:N", title="Year"),
                tooltip=["Date:T", "Year:N", "Rainfall_3day_avg"]
            ).properties(height=300, title="Rainfall 3-Day Average").interactive()
            st.altair_chart(chart1, use_container_width=True)

            # Rainfall 7-Day Avg
            chart2 = alt.Chart(filtered_df).mark_line().encode(
                x=x_axis,
                y=alt.Y("Rainfall_7day_avg:Q", title="Rainfall 7-Day Avg"),
                color="Year:N",
                tooltip=["Date:T", "Year:N", "Rainfall_7day_avg"]
            ).properties(height=300, title="Rainfall 7-Day Average").interactive()
            st.altair_chart(chart2, use_container_width=True)

            # Water Level Change
            chart3 = alt.Chart(filtered_df).mark_line().encode(
                x=x_axis,
                y=alt.Y("WaterLevel_change:Q", title="Water Level Change"),
                color="Year:N",
                tooltip=["Date:T", "Year:N", "WaterLevel_change"]
            ).properties(height=300, title="Water Level Change").interactive()
            st.altair_chart(chart3, use_container_width=True)

            # Water Level Rising Indicator
            chart4 = alt.Chart(filtered_df).mark_line().encode(
                x=x_axis,
                y=alt.Y("WaterLevel_rising:Q", title="Water Rising Indicator"),
                color="Year:N",
                tooltip=["Date:T", "Year:N", "WaterLevel_rising"]
            ).properties(height=300, title="Water Level Rising Indicator").interactive()
            st.altair_chart(chart4, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

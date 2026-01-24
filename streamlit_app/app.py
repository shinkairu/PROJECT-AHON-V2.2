
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

section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

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
}

.hero span {
    color: #7dd3fc;
}

.hero p {
    max-width: 520px;
    font-size: 1.1rem;
    opacity: 0.95;
}

.card {
    background: white;
    border-radius: 22px;
    padding: 1.8rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
}

.stMetric {
    background: white;
    border-radius: 20px;
    padding: 1.2rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
}

footer {
    text-align: center;
    color: #6b7280;
    margin-top: 3rem;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
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
# ML MODEL
# ==============================
@st.cache_resource
def train_flood_model(df):
    data = df.dropna().copy()

    features = [
        "Rainfall_mm",
        "WaterLevel_m",
        "Rainfall_3day_avg",
        "Rainfall_7day_avg",
        "WaterLevel_change"
    ]

    X = data[features]
    y = data["FloodOccurrence"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
    model.fit(X, y)

    return model

# ==============================
# MAIN PANEL
# ==============================
if panel == "🏠 Main Panel":
    st.markdown("""
    <div class="hero">
        <small>🟢 Live System Monitoring</small>
        <h1>Predict Floods.<br><span>Protect Communities.</span></h1>
        <p>
            Project AHON uses AI-driven rainfall and water-level analysis
            to generate early flood risk predictions and location-based insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# DATASET & EDA
# ==============================
elif panel == "📊 Dataset & EDA":
    if df is None:
        st.warning("Upload a dataset first.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# FEATURE ENGINEERING WITH GRAPHS
# ==============================
elif panel == "🧠 Feature Engineering":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        df = df.copy()

        # Feature engineering
        df["Rainfall_3day_avg"] = df["Rainfall_mm"].rolling(3).mean()
        df["Rainfall_7day_avg"] = df["Rainfall_mm"].rolling(7).mean()
        df["WaterLevel_change"] = df["WaterLevel_m"].diff()
        df["WaterLevel_rising"] = (df["WaterLevel_change"] > 0).astype(int)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Engineered Features")
        st.dataframe(
            df[[
                "Date",
                "Rainfall_3day_avg",
                "Rainfall_7day_avg",
                "WaterLevel_change",
                "WaterLevel_rising"
            ]].head(10),
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Feature Trends Over Time")

        # Convert Date
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            # Rainfall rolling averages line chart
            rainfall_chart = alt.Chart(df).mark_line().encode(
                x="Date:T",
                y="Rainfall_3day_avg:Q",
                color=alt.value("#2563eb"),
                tooltip=["Date", "Rainfall_3day_avg"]
            ).properties(title="3-Day Rolling Rainfall Average", height=200)

            rainfall_chart_7 = alt.Chart(df).mark_line().encode(
                x="Date:T",
                y="Rainfall_7day_avg:Q",
                color=alt.value("#f97316"),
                tooltip=["Date", "Rainfall_7day_avg"]
            ).properties(title="7-Day Rolling Rainfall Average", height=200)

            # Water level change line chart
            water_change_chart = alt.Chart(df).mark_line(color="#16a34a").encode(
                x="Date:T",
                y="WaterLevel_change:Q",
                tooltip=["Date", "WaterLevel_change"]
            ).properties(title="Water Level Change", height=200)

            # Water level rising bar chart
            water_rising_chart = alt.Chart(df).mark_bar(color="#dc2626").encode(
                x="Date:T",
                y="WaterLevel_rising:Q",
                tooltip=["Date", "WaterLevel_rising"]
            ).properties(title="Water Level Rising Indicator", height=200)

            # Display charts
            st.altair_chart(rainfall_chart, use_container_width=True)
            st.altair_chart(rainfall_chart_7, use_container_width=True)
            st.altair_chart(water_change_chart, use_container_width=True)
            st.altair_chart(water_rising_chart, use_container_width=True)

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
                    scale=alt.Scale(
                        domain=["Normal", "Anomaly"],
                        range=["#2563eb", "#dc2626"]
                    )
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
# INSIGHTS – FUTURE FLOOD PREDICTION
# ==============================
elif panel == "📈 Insights":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Feature engineering
        df["Rainfall_3day_avg"] = df["Rainfall_mm"].rolling(3).mean()
        df["Rainfall_7day_avg"] = df["Rainfall_mm"].rolling(7).mean()
        df["WaterLevel_change"] = df["WaterLevel_m"].diff()

        model = train_flood_model(df)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📅 Future Flood Prediction")

        # Select any date (future or current)
        selected_date = st.date_input(
            "Select a date to predict flood risk",
            value=pd.to_datetime("2026-07-01")
        )

        # Match month & day from historical data
        month = selected_date.month
        day = selected_date.day

        historical_same_day = df[(df["Date"].dt.month == month) & (df["Date"].dt.day == day)]

        if historical_same_day.empty:
            st.info("No historical data available for this day of the year.")
        else:
            features = [
                "Rainfall_mm",
                "WaterLevel_m",
                "Rainfall_3day_avg",
                "Rainfall_7day_avg",
                "WaterLevel_change"
            ]

            X_hist = historical_same_day[features].fillna(0)
            predictions = model.predict(X_hist)
            probabilities = model.predict_proba(X_hist)[:, 1]

            avg_prob = probabilities.mean()

            if avg_prob >= 0.7:
                risk = "HIGH RISK"
                icon = "🚨"
            elif avg_prob >= 0.4:
                risk = "MODERATE RISK"
                icon = "⚠️"
            else:
                risk = "LOW RISK"
                icon = "🟢"

            st.metric(
                f"{icon} Predicted Flood Risk for {selected_date.strftime('%B %d')}",
                risk,
                f"{round(avg_prob*100,2)}% probability based on historical data"
            )

            st.markdown("### 📍 Likely Flood-Prone Areas (Historical Pattern)")
            affected = historical_same_day[predictions == 1]

            if affected.empty:
                st.success("No flooding historically recorded for this day.")
            else:
                st.dataframe(
                    affected[[
                        "Location",
                        "Rainfall_mm",
                        "WaterLevel_m"
                    ]].sort_values("WaterLevel_m", ascending=False),
                    use_container_width=True
                )

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

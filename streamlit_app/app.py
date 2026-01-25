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

body, .stApp { font-family: 'Inter', sans-serif; background-color: #f5f7fb; color: #1f2937; }
section[data-testid="stSidebar"] { background-color: #63a9ff; border-right: 1px solid #e5e7eb; }
.hero { background: linear-gradient(135deg, #1e3a8a, #3b82f6); border-radius: 30px; padding: 3.5rem; color: white; box-shadow: 0 25px 60px rgba(30,58,138,0.35); margin-bottom: 2rem; }
.hero h1 { font-size: 3.2rem; font-weight: 700; }
.hero span { color: #7dd3fc; }
.hero p { max-width: 520px; font-size: 1.1rem; opacity: 0.95; }
.card { background: white; border-radius: 22px; padding: 1.8rem; box-shadow: 0 10px 30px rgba(0,0,0,0.06); margin-bottom: 1.5rem; }
.stMetric { background: white; border-radius: 20px; padding: 1.2rem; box-shadow: 0 8px 24px rgba(0,0,0,0.06); }
.stButton>button { border-radius: 14px; padding: 0.7rem 1.4rem; font-weight: 600; }
footer { text-align: center; color: #6b7280; margin-top: 3rem; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("🌊 PROJECT AHON")
st.sidebar.caption("AI Flood Risk Intelligence")

panel = st.sidebar.radio(
    "",
    ["🏠 Main Panel", "📊 Dataset & EDA", "🧠 Feature Engineering",
     "🌧️ Anomaly Detection", "🗺️ Geospatial Mapping", "📈 Insights"]
)

st.sidebar.markdown("---")
st.sidebar.success("System Online")
st.sidebar.caption("v1.2.0 Stable")

uploaded_file = st.sidebar.file_uploader("Upload Flood Dataset (CSV)", type=["csv"])

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
    features = ["Rainfall_mm", "WaterLevel_m", "Rainfall_3day_avg", "Rainfall_7day_avg", "WaterLevel_change"]
    X = data[features]
    y = data["FloodOccurrence"]
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X, y)
    return model
# ==============================
# MAIN PANEL
# ==============================
if panel == "🏠 Main Panel":
    st.markdown("""
    <style>
    /* Heartbeat animation */
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        25% { transform: scale(1.05); }
        50% { transform: scale(1.1); }
        75% { transform: scale(1.05); }
    }

    .hero {
        text-align: center;
        background: linear-gradient(135deg, #3b82f6, #60a5fa);
        padding: 80px 40px;
        border-radius: 20px;
        animation: heartbeat 2s infinite;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .hero h1 {
        color: white;
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 20px;
    }

    .hero h1 span {
        color: #7dd3fc;
    }

    .hero p {
        margin: auto;
        max-width: 600px;
        font-size: 1.3rem;
        opacity: 0.95;
    }
    </style>

    <div class="hero">
        <h1>
            Predict Floods.<br><span>Protect Communities.</span>
        </h1>
        <p>
            Project AHON uses AI-powered rainfall and water-level analysis
            to provide early flood risk predictions and geospatial insights.
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
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
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
elif panel == "🌧️ Anomaly Detection":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        ad_df = df.copy()
        iso = IsolationForest(contamination=0.05, random_state=42)
        ad_df["Anomaly"] = iso.fit_predict(ad_df[["Rainfall_mm"]].fillna(0))
        ad_df["Status"] = ad_df["Anomaly"].map({1: "Normal", -1: "Anomaly"})

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Rainfall Anomaly Detection")

        if "Date" in ad_df.columns:
            ad_df["Date"] = pd.to_datetime(ad_df["Date"], errors="coerce")
            chart = alt.Chart(ad_df.dropna()).mark_circle(size=30).encode(
                x="Date:T", y="Rainfall_mm:Q",
                color=alt.Color("Status:N", scale=alt.Scale(domain=["Normal","Anomaly"], range=["#2563eb","#dc2626"])),
                tooltip=["Date","Rainfall_mm","Status"]
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)

        st.dataframe(ad_df[ad_df["Status"] == "Anomaly"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# GEOSPATIAL MAPPING
# ==============================
elif panel == "🗺️ Geospatial Mapping":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Flood Risk Map (Anomaly-Aware)")

        # --------------------------------
        # City → Coordinates (Metro Manila)
        # --------------------------------
        city_coords = {
            "Quezon City": (14.6760, 121.0437),
            "Manila": (14.5995, 120.9842),
            "Marikina": (14.6507, 121.1029),
            "Pasig": (14.5764, 121.0851)
        }

        map_df = df.copy()
        map_df["Date"] = pd.to_datetime(map_df["Date"], errors="coerce")

        # --------------------------------
        # Feature engineering (map-safe)
        # --------------------------------
        map_df["Rainfall_3day_avg"] = map_df["Rainfall_mm"].rolling(3, min_periods=1).mean()
        map_df["Rainfall_7day_avg"] = map_df["Rainfall_mm"].rolling(7, min_periods=1).mean()
        map_df["WaterLevel_change"] = map_df["WaterLevel_m"].diff().fillna(0)

        # --------------------------------
        # Train RF + compute probabilities
        # --------------------------------
        model = train_flood_model(map_df)

        features = [
            "Rainfall_mm",
            "WaterLevel_m",
            "Rainfall_3day_avg",
            "Rainfall_7day_avg",
            "WaterLevel_change"
        ]

        X_map = map_df[features].fillna(0)
        map_df["FloodPrediction"] = model.predict(X_map)
        map_df["FloodRiskScore"] = model.predict_proba(X_map)[:, 1]

        # --------------------------------
        # Rainfall anomaly detection
        # --------------------------------
        iso = IsolationForest(contamination=0.05, random_state=42)
        map_df["Rainfall_Anomaly"] = (
            iso.fit_predict(map_df[["Rainfall_mm"]].fillna(0)) == -1
        ).astype(int)

        # --------------------------------
        # Risk color logic (Colab-aligned)
        # --------------------------------
        def risk_color(prob, anomaly):
            if anomaly == 1:
                return "purple"
            elif prob >= 0.7:
                return "red"
            elif prob >= 0.4:
                return "orange"
            else:
                return "green"

        # --------------------------------
        # Build map
        # --------------------------------
        m = folium.Map(location=[14.60, 121.00], zoom_start=11)

        latest_df = (
            map_df.sort_values("Date")
            .groupby("Location")
            .tail(1)
        )

        for _, row in latest_df.iterrows():
            coords = city_coords.get(row["Location"])
            if coords is None:
                continue

            folium.CircleMarker(
                location=coords,
                radius=18,
                color=risk_color(row["FloodRiskScore"], row["Rainfall_Anomaly"]),
                fill=True,
                fill_color=risk_color(row["FloodRiskScore"], row["Rainfall_Anomaly"]),
                fill_opacity=0.75,
                popup=folium.Popup(
                    f"""
                    <b>City:</b> {row['Location']}<br>
                    <b>Date:</b> {row['Date'].date()}<br>
                    <b>Flood Risk Score:</b> {row['FloodRiskScore']:.2f}<br>
                    <b>Prediction:</b> {"Flood" if row['FloodPrediction'] == 1 else "No Flood"}<br>
                    <b>Rainfall Anomaly:</b> {"Yes" if row['Rainfall_Anomaly'] == 1 else "No"}
                    """,
                    max_width=300
                )
            ).add_to(m)

        st_folium(m, width=1000, height=520)
        st.markdown("</div>", unsafe_allow_html=True)



# ==============================
# INSIGHTS – FLOOD PREDICTION
# ==============================
elif panel == "📈 Insights":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        df_insight = df.copy()
        df_insight["Date"] = pd.to_datetime(df_insight["Date"], errors="coerce")
        df_insight["Rainfall_3day_avg"] = df_insight["Rainfall_mm"].rolling(3, min_periods=1).mean()
        df_insight["Rainfall_7day_avg"] = df_insight["Rainfall_mm"].rolling(7, min_periods=1).mean()
        df_insight["WaterLevel_change"] = df_insight["WaterLevel_m"].diff().fillna(0)

        if "FloodOccurrence" in df_insight.columns:
            model = train_flood_model(df_insight)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📅 Flood Risk Prediction by Date")

            available_dates = df_insight["Date"].dropna().dt.date.unique()
            selected_date = st.date_input("Select a date", value=available_dates[-1])

            day_data = df_insight[df_insight["Date"].dt.date == selected_date]

            if day_data.empty:
                st.info("No data available for this date.")
            else:
                features = ["Rainfall_mm","WaterLevel_m","Rainfall_3day_avg","Rainfall_7day_avg","WaterLevel_change"]
                X_day = day_data[features].fillna(0)
                day_data["Flood_Prediction"] = model.predict(X_day)
                day_data["Flood_Probability"] = model.predict_proba(X_day)[:, 1]
                avg_prob = day_data["Flood_Probability"].mean()

                if avg_prob >= 0.7: risk, icon = "HIGH RISK", "🚨"
                elif avg_prob >= 0.4: risk, icon = "MODERATE RISK", "⚠️"
                else: risk, icon = "LOW RISK", "🟢"

                st.metric(f"{icon} Overall Flood Risk", risk, f"{round(avg_prob*100,2)}% probability")

                st.markdown("### 📍 Predicted Flood-Prone Areas")
                affected = day_data[day_data["Flood_Prediction"] == 1]
                if affected.empty:
                    st.success("No flooding predicted for this date.")
                else:
                    st.dataframe(
                        affected[["Location","Rainfall_mm","WaterLevel_m","Flood_Probability"]].sort_values("Flood_Probability", ascending=False),
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

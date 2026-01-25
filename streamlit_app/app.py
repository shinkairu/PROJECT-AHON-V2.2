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
        import folium
        from streamlit_folium import st_folium
        import pandas as pd
        import numpy as np

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Animated Flood Risk Map (Metro Manila)")

        # --------------------------------------------------
        # 1. Normalize columns
        # --------------------------------------------------
        df.columns = df.columns.str.strip()

        # Ensure Date
        df["Date"] = pd.to_datetime(df["Date"])

        # --------------------------------------------------
        # 2. CREATE missing engineered features (Colab parity)
        # --------------------------------------------------

        # FloodRiskScore
        if "FloodRiskScore" not in df.columns:
            if "Rainfall" in df.columns:
                df["FloodRiskScore"] = (
                    df["Rainfall"] - df["Rainfall"].min()
                ) / (df["Rainfall"].max() - df["Rainfall"].min())
            else:
                df["FloodRiskScore"] = 0.3  # safe fallback

        # Rainfall Anomaly (z-score)
        if "Rainfall_Anomaly" not in df.columns:
            if "Rainfall" in df.columns:
                z = (df["Rainfall"] - df["Rainfall"].mean()) / df["Rainfall"].std()
                df["Rainfall_Anomaly"] = (z > 2).astype(int)
            else:
                df["Rainfall_Anomaly"] = 0

        # FloodPrediction
        if "FloodPrediction" not in df.columns:
            df["FloodPrediction"] = (df["FloodRiskScore"] >= 0.6).astype(int)

        # --------------------------------------------------
        # 3. Filter cities
        # --------------------------------------------------
        cities = ["Quezon City", "Manila", "Marikina", "Pasig"]
        geo_df = df[df["Location"].isin(cities)].copy()

        # --------------------------------------------------
        # 4. Date slider (now accurate)
        # --------------------------------------------------
        selected_date = st.slider(
            "Select Date",
            min_value=geo_df["Date"].min().date(),
            max_value=geo_df["Date"].max().date(),
            value=geo_df["Date"].min().date()
        )

        day_df = geo_df[geo_df["Date"].dt.date == selected_date]

        # --------------------------------------------------
        # 5. Risk color logic (same as Colab)
        # --------------------------------------------------
        def risk_color(prob, anomaly):
            if anomaly == 1:
                return "purple"
            elif prob >= 0.7:
                return "red"
            elif prob >= 0.4:
                return "orange"
            else:
                return "green"

        # --------------------------------------------------
        # 6. Build map
        # --------------------------------------------------
        m = folium.Map(location=[14.60, 121.00], zoom_start=11)

        for _, row in day_df.iterrows():
            color = risk_color(row["FloodRiskScore"], row["Rainfall_Anomaly"])

            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=18,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                popup=(
                    f"<b>City:</b> {row['Location']}<br>"
                    f"<b>Date:</b> {row['Date'].date()}<br>"
                    f"<b>Flood Risk Score:</b> {row['FloodRiskScore']:.2f}<br>"
                    f"<b>Prediction:</b> {'Flood' if row['FloodPrediction']==1 else 'No Flood'}<br>"
                    f"<b>Rainfall Anomaly:</b> {'Yes' if row['Rainfall_Anomaly']==1 else 'No'}"
                )
            ).add_to(m)

        # --------------------------------------------------
        # 7. LEGEND (contrast fixed)
        # --------------------------------------------------
        legend_html = """
        <div style="
            position: fixed;
            bottom: 30px;
            left: 30px;
            z-index: 9999;
            background-color: rgba(255,255,255,0.95);
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 14px;
            color: black;
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
        ">
        <b>Flood Risk Legend</b><br>
        <span style="color:green;">●</span> Low Risk<br>
        <span style="color:orange;">●</span> Moderate Risk<br>
        <span style="color:red;">●</span> High Risk<br>
        <span style="color:purple;">●</span> Rainfall Anomaly
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, width=1000, height=550)
        st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------------------
        # 6. Render map
        # --------------------------------------------------
        st_folium(m, width=1000, height=550)

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("Available columns:", df.columns.tolist())


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





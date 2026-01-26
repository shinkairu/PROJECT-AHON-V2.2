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
# GEOSPATIAL MAPPING (IMPROVED)
# ==============================
elif panel == "🗺️ Geospatial Mapping":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Flood Risk Map (Anomaly-Aware)")

        # --------------------------------
        # City → Coordinates (Metro Manila)
        # (Centroids only; for higher accuracy, store lat/lon in dataset)
        # --------------------------------
        city_coords = {
            "Quezon City": (14.6760, 121.0437),
            "Manila": (14.5995, 120.9842),
            "Marikina": (14.6507, 121.1029),
            "Pasig": (14.5764, 121.0851)
        }

        map_df = df.copy()
        map_df["Date"] = pd.to_datetime(map_df["Date"], errors="coerce")
        map_df = map_df.dropna(subset=["Date", "Location"])

        # Ensure numeric
        for col in ["Rainfall_mm", "WaterLevel_m"]:
            map_df[col] = pd.to_numeric(map_df[col], errors="coerce")

        # --------------------------------
        # ✅ Feature engineering PER CITY
        # --------------------------------
        map_df = map_df.sort_values(["Location", "Date"]).reset_index(drop=True)

        # Rolling means per Location
        map_df["Rainfall_3day_avg"] = (
            map_df.groupby("Location")["Rainfall_mm"]
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
        )
        map_df["Rainfall_7day_avg"] = (
            map_df.groupby("Location")["Rainfall_mm"]
            .transform(lambda s: s.rolling(7, min_periods=1).mean())
        )

        # Water level change per Location
        map_df["WaterLevel_change"] = (
            map_df.groupby("Location")["WaterLevel_m"]
            .diff()
            .fillna(0)
        )

        # Optional: smoother signal (less noisy risk)
        map_df["Rainfall_ewm"] = (
            map_df.groupby("Location")["Rainfall_mm"]
            .transform(lambda s: s.ewm(span=5, adjust=False).mean())
        )

        # --------------------------------
        # Train RF + compute probabilities
        # --------------------------------
        model = train_flood_model(map_df)

        features = [
            "Rainfall_mm",
            "WaterLevel_m",
            "Rainfall_3day_avg",
            "Rainfall_7day_avg",
            "WaterLevel_change",
            # Optional: include EWM if your model benefits
            "Rainfall_ewm",
        ]

        X_map = map_df[features].fillna(0)
        map_df["FloodPrediction"] = model.predict(X_map)
        map_df["FloodRiskScore"] = model.predict_proba(X_map)[:, 1]

        # --------------------------------
        # ✅ Rainfall anomaly detection PER CITY
        # (Less false flags caused by other cities)
        # --------------------------------
        def detect_anomaly_per_city(group):
            # If too few points, no anomalies
            if len(group) < 20:
                group["Rainfall_Anomaly"] = 0
                return group

            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(group[["Rainfall_mm"]].fillna(0))
            group["Rainfall_Anomaly"] = (preds == -1).astype(int)
            return group

        map_df = map_df.groupby("Location", group_keys=False).apply(detect_anomaly_per_city)

        # --------------------------------
        # Risk color logic
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
        m = folium.Map(location=[14.60, 121.00], zoom_start=11, tiles="CartoDB positron")

        # ✅ Latest per city (after correct sorting)
        latest_df = (
            map_df.sort_values(["Location", "Date"])
            .groupby("Location", as_index=False)
            .tail(1)
        )

        # Legend (simple HTML)
        legend_html = """
        <div style="
            position: fixed;
            bottom: 30px; left: 30px;
            z-index: 9999;
            background: white;
            padding: 10px 12px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
            font-size: 13px;
        ">
            <b>Legend</b><br>
            <span style="color:green;">●</span> Low (&lt; 0.40)<br>
            <span style="color:orange;">●</span> Medium (0.40–0.69)<br>
            <span style="color:red;">●</span> High (≥ 0.70)<br>
            <span style="color:purple;">●</span> Rainfall Anomaly
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        for _, row in latest_df.iterrows():
            coords = city_coords.get(row["Location"])
            if coords is None:
                continue

            color = risk_color(row["FloodRiskScore"], row["Rainfall_Anomaly"])

            folium.CircleMarker(
                location=coords,
                radius=16,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                tooltip=folium.Tooltip(
                    f"{row['Location']} | Risk: {row['FloodRiskScore']:.2f} | "
                    f"{'Anomaly' if row['Rainfall_Anomaly']==1 else 'Normal'}",
                    sticky=True
                ),
                popup=folium.Popup(
                    f"""
                    <b>City:</b> {row['Location']}<br>
                    <b>Date:</b> {row['Date'].date()}<br>
                    <b>Flood Risk Score:</b> {row['FloodRiskScore']:.2f}<br>
                    <b>Prediction:</b> {"Flood" if row['FloodPrediction'] == 1 else "No Flood"}<br>
                    <b>Rainfall:</b> {row['Rainfall_mm'] if pd.notna(row['Rainfall_mm']) else 0:.2f} mm<br>
                    <b>Water Level:</b> {row['WaterLevel_m'] if pd.notna(row['WaterLevel_m']) else 0:.2f} m<br>
                    <b>Rainfall Anomaly:</b> {"Yes" if row['Rainfall_Anomaly'] == 1 else "No"}
                    """,
                    max_width=320
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
        hist_df = df.copy()
        hist_df["Date"] = pd.to_datetime(hist_df["Date"], errors="coerce")
        hist_df = hist_df.dropna(subset=["Date"])

        # ------------------------------
        # Extract Month-Day
        # ------------------------------
        hist_df["MonthDay"] = hist_df["Date"].dt.strftime("%m-%d")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📅 Historical Flood Warning (Date-Based)")

        # ------------------------------
        # Single Date Selector
        # ------------------------------
        selected_date = st.date_input(
            "Select Month & Day",
            value=hist_df["Date"].max()
        )

        month_day = selected_date.strftime("%m-%d")

        # ------------------------------
        # Filter historical records
        # ------------------------------
        date_df = hist_df[hist_df["MonthDay"] == month_day]

        if date_df.empty:
            st.info("No historical data available for this date.")
        else:
            # ------------------------------
            # Aggregate per location
            # ------------------------------
            summary = (
                date_df
                .groupby("Location")
                .agg(
                    flood_years=("FloodOccurrence", "sum"),
                    total_years=("FloodOccurrence", "count"),
                    avg_rainfall=("Rainfall_mm", "mean")
                )
                .reset_index()
            )

            summary["flood_rate"] = summary["flood_years"] / summary["total_years"]

            # ------------------------------
            # Warning logic (simple & realistic)
            # ------------------------------
            RAIN_THRESHOLD = 20  # adjust based on your data

            def warning_label(row):
                if row["flood_rate"] >= 0.4 and row["avg_rainfall"] >= RAIN_THRESHOLD:
                    return "HIGH"
                elif row["flood_rate"] > 0 or row["avg_rainfall"] >= RAIN_THRESHOLD:
                    return "MODERATE"
                else:
                    return "LOW"

            summary["Warning Level"] = summary.apply(warning_label, axis=1)

            # ------------------------------
            # City warnings
            # ------------------------------
            st.markdown("### 📍 Location-Based Historical Warnings")

            for _, row in summary.iterrows():
                if row["Warning Level"] == "HIGH":
                    st.error(
                        f"🚨 **{row['Location']}**\n\n"
                        f"- Flood occurred in {int(row['flood_years'])} out of {int(row['total_years'])} years\n"
                        f"- Avg rainfall: {row['avg_rainfall']:.1f} mm\n"
                        f"⚠️ Possible flooding on this date based on history"
                    )
                elif row["Warning Level"] == "MODERATE":
                    st.warning(
                        f"⚠️ **{row['Location']}**\n\n"
                        f"- Historical flood records present\n"
                        f"- Avg rainfall: {row['avg_rainfall']:.1f} mm\n"
                        f"⚠️ Stay alert on this date"
                    )
                else:
                    st.success(
                        f"🟢 **{row['Location']}** – No significant historical flooding"
                    )

            # ------------------------------
            # Clean summary table
            # ------------------------------
            st.markdown("### 📊 Historical Summary")
            st.dataframe(
                summary[
                    ["Location", "Warning Level",
                     "flood_years", "total_years",
                     "avg_rainfall"]
                ].sort_values("Warning Level"),
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

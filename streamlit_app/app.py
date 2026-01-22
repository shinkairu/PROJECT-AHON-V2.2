# ==============================
# FEATURE ENGINEERING
# ==============================
elif panel == "🧠 Feature Engineering":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        df = df.copy()
        # Calculate features
        df["Rainfall_3day_avg"] = df["Rainfall_mm"].rolling(3).mean()
        df["Rainfall_7day_avg"] = df["Rainfall_mm"].rolling(7).mean()
        df["WaterLevel_change"] = df["WaterLevel_m"].diff()
        df["WaterLevel_rising"] = (df["WaterLevel_change"] > 0).astype(int)

        # Show engineered features table
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Engineered Features (Sample)")
        st.dataframe(df[[
            "Rainfall_3day_avg", "Rainfall_7day_avg", "WaterLevel_change", "WaterLevel_rising"
        ]].head())
        st.markdown("</div>", unsafe_allow_html=True)

        # Ensure Date is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df_plot = df.dropna(subset=['Date', 'Rainfall_3day_avg', 'Rainfall_7day_avg', 
                                         'WaterLevel_m', 'WaterLevel_change'])

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Rainfall Moving Averages")

            # ===== Rainfall Moving Averages =====
            rainfall_chart = alt.Chart(df_plot).transform_fold(
                ['Rainfall_3day_avg','Rainfall_7day_avg'],
                as_=['Metric','Value']
            ).mark_line(point=True).encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Value:Q', title='Rainfall (mm)'),
                color=alt.Color('Metric:N', scale=alt.Scale(range=['#1e88e5','#90caf9'])),
                tooltip=['Date','Metric','Value']
            ).properties(
                width=600,
                height=300
            )
            st.altair_chart(rainfall_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ===== Water Level Dynamics =====
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Water Level Dynamics")
            water_chart = alt.Chart(df_plot).transform_fold(
                ['WaterLevel_m','WaterLevel_change'],
                as_=['Metric','Value']
            ).mark_line(point=True).encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Value:Q', title='Water Level (m)'),
                color=alt.Color('Metric:N', scale=alt.Scale(range=['#1e88e5','#90caf9'])),
                tooltip=['Date','Metric','Value']
            ).properties(
                width=600,
                height=300
            )
            st.altair_chart(water_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

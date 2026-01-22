elif panel == "🌧️ Anomaly Detection":
    if df is None:
        st.warning("Upload dataset first.")
    else:
        iso = IsolationForest(contamination=0.05, random_state=42)
        df["Rainfall_Anomaly"] = iso.fit_predict(df[["Rainfall_mm"]].fillna(0))
        df["Anomaly_Flag"] = df["Rainfall_Anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")

        anomalies = df[df["Rainfall_Anomaly"] == -1].copy()

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Rainfall Anomaly Detection")

        if anomalies.empty:
            st.info("No anomalies detected in the uploaded dataset.")
        else:
            # ===== Scatter plot in box =====
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])

                scatter = alt.Chart(df).mark_circle(size=20).encode(
                    x=alt.X('Date:T', axis=alt.Axis(title='Date', format='%b %d, %Y')),
                    y=alt.Y('Rainfall_mm:Q', title='Rainfall (mm)'),
                    color=alt.Color('Anomaly_Flag:N',
                                    scale=alt.Scale(domain=['Normal','Anomaly'], 
                                                    range=['#1e88e5','#e53935'])),
                    tooltip=['Date', 'Rainfall_mm', 'Anomaly_Flag']
                ).properties(
                    width=800,
                    height=300,
                    title="Rainfall Over Time with Anomalies"
                )
                
                # Box effect
                st.markdown("<div style='padding:1rem; border-radius:15px; box-shadow:0px 10px 25px rgba(0,0,0,0.1); background-color:white;'>", unsafe_allow_html=True)
                st.altair_chart(scatter, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("No 'Date' column found – scatter plot not available.")

            # ===== Table in box =====
            anomalies = anomalies.sort_values(by="Rainfall_mm", ascending=False)
            st.markdown("<div style='padding:1rem; border-radius:15px; box-shadow:0px 10px 25px rgba(0,0,0,0.1); background-color:white;'>", unsafe_allow_html=True)
            st.dataframe(anomalies)
            st.markdown("</div>", unsafe_allow_html=True)

            st.info("Red dots in the plot = detected extreme rainfall deviations.")

        st.markdown("</div>", unsafe_allow_html=True)

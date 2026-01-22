# ==============================
# FEATURE ENGINEERING – Enhanced Table UI
# ==============================
if df is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Engineered Features (Sample)")

    # Select only the engineered features
    features_df = df[[
        "Rainfall_3day_avg","Rainfall_7day_avg","WaterLevel_change","WaterLevel_rising"
    ]].head(10).copy()

    # Fill missing values with 'N/A'
    features_df = features_df.fillna("N/A")

    # Format numeric columns to 2 decimals
    for col in ["Rainfall_3day_avg","Rainfall_7day_avg","WaterLevel_change"]:
        features_df[col] = features_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int,float)) else x)

    # Convert DataFrame to HTML with styling
    table_html = features_df.to_html(index=False, classes='styled-table', escape=False)

    st.markdown(f"""
    <style>
    /* Table container */
    .styled-table {{
        border-collapse: separate !important;
        border-spacing: 0 !important;
        width: 100%;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        overflow: hidden;
        font-family: 'Roboto', sans-serif;
    }}

    /* Table header */
    .styled-table thead tr {{
        background: linear-gradient(90deg, #1976d2, #90caf9);
        color: white;
        text-align: center;
        font-weight: 600;
        font-size: 14px;
    }}

    /* Table body */
    .styled-table tbody tr {{
        background-color: #ffffff;
        text-align: right;
        font-size: 13px;
    }}

    /* Zebra striping */
    .styled-table tbody tr:nth-child(even) {{
        background-color: #f5f7fa;
    }}

    /* Row hover effect */
    .styled-table tbody tr:hover {{
        background-color: #e3f2fd;
        transition: background 0.3s ease;
    }}

    /* Cell padding */
    .styled-table td {{
        padding: 12px 16px;
    }}
    </style>
    {table_html}
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

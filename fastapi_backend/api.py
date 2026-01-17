from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(title="PROJECT AHON API")

rf_model = joblib.load("../models/rf_flood_model.pkl")

FEATURE_COLS = [
    'Rainfall_mm', 'WaterLevel_m', 'SoilMoisture_pct', 'Elevation_m',
    'Rainfall_3day_avg', 'Rainfall_7day_avg', 'Rainfall_prev_day',
    'WaterLevel_prev_day', 'WaterLevel_change', 'WaterLevel_rising',
    'Month', 'IsWetSeason'
]

@app.post("/predict")
def predict_flood(data: dict):
    df = pd.DataFrame([data])
    prob = rf_model.predict_proba(df[FEATURE_COLS])[0, 1]

    return {
        "flood_probability": prob,
        "prediction": int(prob >= 0.5)
    }

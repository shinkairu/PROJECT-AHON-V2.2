import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model" / "rf_flood_model.pkl"

rf_model = joblib.load(MODEL_PATH)

FEATURE_COLS = [
    'Rainfall_mm',
    'WaterLevel_m',
    'SoilMoisture_pct',
    'Elevation_m',
    'Rainfall_3day_avg',
    'Rainfall_7day_avg',
    'Rainfall_prev_day',
    'WaterLevel_prev_day',
    'WaterLevel_change',
    'WaterLevel_rising',
    'Month',
    'IsWetSeason'
]

def predict_flood(input_data: dict):
    df = pd.DataFrame([input_data])[FEATURE_COLS]

    prob = rf_model.predict_proba(df)[0][1]
    pred = int(prob >= 0.5)

    return {
        "flood_probability": round(prob, 3),
        "flood_prediction": pred
    }

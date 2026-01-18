import requests

API_URL = "http://127.0.0.1:8000/predict"

payload = {
    "Rainfall_mm": 120,
    "WaterLevel_m": 5.4,
    "SoilMoisture_pct": 78,
    "Elevation_m": 12,
    "Rainfall_3day_avg": 90,
    "Rainfall_7day_avg": 110,
    "Rainfall_prev_day": 100,
    "WaterLevel_prev_day": 5.0,
    "WaterLevel_change": 0.4,
    "WaterLevel_rising": 1,
    "Month": 8,
    "IsWetSeason": 1
}

response = requests.post(API_URL, json=payload)
result = response.json()

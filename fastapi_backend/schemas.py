from pydantic import BaseModel

class FloodInput(BaseModel):
    Rainfall_mm: float
    WaterLevel_m: float
    SoilMoisture_pct: float
    Elevation_m: float
    Rainfall_3day_avg: float
    Rainfall_7day_avg: float
    Rainfall_prev_day: float
    WaterLevel_prev_day: float
    WaterLevel_change: float
    WaterLevel_rising: int
    Month: int
    IsWetSeason: int

import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Location", "Date"])

    # --- Temporal rainfall features ---
    df["Rainfall_prev_day"] = (
        df.groupby("Location")["Rainfall_mm"].shift(1)
    )

    df["Rainfall_3day_avg"] = (
        df.groupby("Location")["Rainfall_mm"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["Rainfall_7day_avg"] = (
        df.groupby("Location")["Rainfall_mm"]
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # --- Water level features ---
    df["WaterLevel_prev_day"] = (
        df.groupby("Location")["WaterLevel_m"].shift(1)
    )

    df["WaterLevel_change"] = (
        df["WaterLevel_m"] - df["WaterLevel_prev_day"]
    )

    df["WaterLevel_rising"] = (
        (df["WaterLevel_change"] > 0).astype(int)
    )

    # --- Time features ---
    df["Month"] = pd.to_datetime(df["Date"]).dt.month
    df["IsWetSeason"] = df["Month"].isin([6, 7, 8, 9, 10]).astype(int)

    # --- Handle NaNs from shifting ---
    df = df.fillna(0)

    return df

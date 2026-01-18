from fastapi import FastAPI
from schemas import FloodInput
from utils import predict_flood

app = FastAPI(
    title="PROJECT – AHON API",
    description="Flood Risk Prediction API",
    version="1.0"
)

@app.get("/")
def root():
    return {"status": "PROJECT AHON API running"}

@app.post("/predict")
def predict(input_data: FloodInput):
    return predict_flood(input_data.dict())

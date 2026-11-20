"""FastAPI inference service."""
from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/champion.joblib"
app = FastAPI(title="MP089 Bakery Sales API", version="1.0")

class PredictionRequest(BaseModel):
    price_rs: float = Field(ge=10)
    temp_c: float
    footfall_index: float
    competitor_shops: int
    shelf_hours: float
    orders_previous_day: int = Field(le=850)
    day_type: str
    outlet: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictionRequest):
    if not MODEL_PATH.exists():
        # Keep the API's schema validation usable before a model is installed.
        raise RuntimeError("Champion model not found. Train and export the model first.")
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([payload.model_dump()])[0]
    return {"prediction": float(prediction)}

"""FastAPI inference service."""
from pathlib import Path
import json
import logging
import sys
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/champion.joblib"
app = FastAPI(title="MP089 Bakery Sales API", version="1.0")


class JsonFormatter(logging.Formatter):
    """Write one structured JSON record per prediction to stdout."""

    def format(self, record):
        return record.getMessage()


prediction_logger = logging.getLogger("mp089.predictions")
if not any(getattr(handler, "_mp089_json", False) for handler in prediction_logger.handlers):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler._mp089_json = True
    prediction_logger.addHandler(handler)
prediction_logger.setLevel(logging.INFO)
prediction_logger.propagate = False


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    features = pd.DataFrame([payload.model_dump()])
    prediction = model.predict(features)[0]
    prediction_logger.info(json.dumps({"event": "prediction", "prediction": float(prediction)}))
    return {"prediction": float(prediction)}

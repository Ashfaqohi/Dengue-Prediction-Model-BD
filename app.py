"""Small HTTP service for dengue predictions."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "Models" / "best_dengue_model.joblib"
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run `python train.py` first.")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"] if isinstance(bundle, dict) else bundle
feature_columns = bundle.get("feature_columns", []) if isinstance(bundle, dict) else []

app = FastAPI(title="Dengue Prediction API", version="1.0.0")


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="One patient record using the training feature names")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    missing = sorted(set(feature_columns) - set(request.features))
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")
    row = pd.DataFrame([request.features], columns=feature_columns)
    try:
        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"prediction": prediction, "label": "Positive" if prediction else "Negative", "probability": probability}

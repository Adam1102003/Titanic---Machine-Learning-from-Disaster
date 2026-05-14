import os
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Titanic Survival Predictor",
    description="Online serving API for the Titanic ML model",
    version="1.0.0",
)

BEST_MODEL_FILE = "metrics/best_model.json"
model = None


# ── Schemas ────────────────────────────────────────────────

class Passenger(BaseModel):
    Pclass:      int
    Sex:         int        # 0=female, 1=male (encoded)
    Age:         float
    SibSp:       int
    Parch:       int
    Fare:        float
    Embarked:    int        # 0=C, 1=Q, 2=S (encoded)
    Title:       int        # encoded
    Deck:        int        # encoded
    FamilySize:  int
    IsAlone:     int
    IsSmallFamily: int
    IsLargeFamily: int
    IsChild:     int
    FarePerPerson: float
    AgeBand:     int        # encoded
    FareBand:    int        # encoded
    Age_Class:   float      # Age * Pclass
    Fare_Class:  float      # Fare * Pclass

    class Config:
        json_schema_extra = {
            "example": {
                "Pclass": 1, "Sex": 0, "Age": 29.0, "SibSp": 0,
                "Parch": 0, "Fare": 211.0, "Embarked": 0,
                "Title": 2, "Deck": 3, "FamilySize": 1,
                "IsAlone": 1, "IsSmallFamily": 0, "IsLargeFamily": 0,
                "IsChild": 0, "FarePerPerson": 211.0,
                "AgeBand": 2, "FareBand": 3,
                "Age_Class": 29.0, "Fare_Class": 211.0
            }
        }


class BatchRequest(BaseModel):
    passengers: List[Passenger]


class PredictionResult(BaseModel):
    index:                int
    prediction:           int
    survived_label:       str
    survival_probability: float


class BatchResponse(BaseModel):
    model_used: str
    total:      int
    results:    List[PredictionResult]


# ── Startup ────────────────────────────────────────────────

@app.on_event("startup")
def load_model():
    global model
    import json

    if not os.path.exists(BEST_MODEL_FILE):
        raise RuntimeError(
            "metrics/best_model.json not found. Run promote_model.py first."
        )

    with open(BEST_MODEL_FILE) as f:
        info = json.load(f)

    model_path = info["best_model_path"]
    model = joblib.load(model_path)

    print(f"[✓] Loaded Production model : {info['best_model']}")
    print(f"    Accuracy                 : {info['accuracy']}")


# ── Routes ─────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Titanic Survival Predictor API",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=BatchResponse)
def predict(request: BatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.passengers:
        raise HTTPException(status_code=400, detail="No passengers provided")

    # Build DataFrame from request
    df = pd.DataFrame([p.model_dump() for p in request.passengers])

    # Rename columns to match training feature names
    df = df.rename(columns={
        "Age_Class":  "Age*Class",
        "Fare_Class": "Fare*Class",
    })

    predictions   = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        results.append(PredictionResult(
            index=i,
            prediction=int(pred),
            survived_label="✅ Survived" if pred == 1 else "❌ Did not survive",
            survival_probability=round(float(prob), 4),
        ))

    import json
    with open(BEST_MODEL_FILE) as f:
        info = json.load(f)

    return BatchResponse(
        model_used=info["best_model"],
        total=len(results),
        results=results,
    )
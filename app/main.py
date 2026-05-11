from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.model import predict
from app.preprocess import preprocess_input

app = FastAPI(
    title="Linear Regression Prediction API",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Pregnancies: int = Field(..., ge=0)
    Glucose: int = Field(..., ge=0)
    BloodPressure: int = Field(..., ge=0)
    SkinThickness: int = Field(..., ge=0)
    Insulin: int = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: int = Field(..., ge=0)


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "message": "FastAPI prediction service is running."}


@app.post("/predict")
def make_prediction(payload: PredictionRequest) -> dict[str, Any]:
    try:
        processed_features = preprocess_input(payload.model_dump())
        prediction = predict(processed_features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "prediction": prediction,
    }


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()

from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from salesvision_ai.services.prediction_service import build_forecast_response

router = APIRouter(prefix="/api/v1/predict")


class ForecastRequest(BaseModel):
    history: List[float]
    horizon_days: int = 3

    def validate(self):
        if not self.history or len(self.history) < 1:
            raise ValueError("history must contain at least one numeric value")


@router.post("/", status_code=200)
def predict(req: ForecastRequest):
    try:
        req.validate()
        resp = build_forecast_response(req.history, horizon_days=req.horizon_days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return resp

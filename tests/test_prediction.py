import torch

from salesvision_ai.services import prediction_service
from salesvision_ai.services.prediction_service import build_forecast_response


class FakePipeline:
    def predict(self, context, prediction_length, limit_prediction_length=False):
        return torch.tensor([[[100.0, 110.0, 120.0], [95.0, 105.0, 115.0]]], dtype=torch.float32)


def test_forecast_uses_pretrained_model_artifact(monkeypatch):
    history = [100, 120, 140, 160]
    monkeypatch.setattr(prediction_service, "_load_pipeline", lambda: FakePipeline())
    response = build_forecast_response(history, horizon_days=3)

    assert response["model"]["name"] == "amazon/chronos-t5-tiny"
    assert len(response["forecast"]) == 3
    assert response["forecast"][0]["estimated"] >= 0
    assert response["forecast"][0]["scenario"] == "estimated"


def test_forecast_includes_confidence_intervals_and_scenarios(monkeypatch):
    history = [50, 55, 60, 65, 70]
    monkeypatch.setattr(prediction_service, "_load_pipeline", lambda: FakePipeline())
    response = build_forecast_response(history, horizon_days=2)
    assert "ci_lower" in response["forecast"][0]
    assert "ci_upper" in response["forecast"][0]
    est = response["forecast"][0]["estimated"]
    opt = response["forecast"][0]["optimistic"]
    pes = response["forecast"][0]["pessimistic"]
    assert opt >= est >= pes

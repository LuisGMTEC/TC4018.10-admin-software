from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import math
import statistics

import numpy as np
import torch


MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
MODEL_METADATA_PATH = MODEL_DIR / "model_info.json"
HF_MODEL_NAME = "amazon/chronos-t5-tiny"
_PIPELINE = None


def _ensure_model_metadata() -> Dict[str, Any]:
    if not MODEL_METADATA_PATH.exists():
        MODEL_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "name": HF_MODEL_NAME,
            "source": "huggingface",
            "description": "Chronos time-series forecasting model from Hugging Face.",
        }
        MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))


def _load_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from chronos import ChronosPipeline

        _PIPELINE = ChronosPipeline.from_pretrained(HF_MODEL_NAME)
    return _PIPELINE


def _forecast_with_huggingface(history: List[float], horizon_days: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = _load_pipeline()
    context = [torch.tensor(history, dtype=torch.float32)]
    predictions = pipeline.predict(context, prediction_length=horizon_days, limit_prediction_length=False)
    samples = predictions[0].detach().cpu().numpy()
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    samples = np.clip(samples, 0.0, None)

    estimated = np.median(samples, axis=0)
    optimistic = np.percentile(samples, 90, axis=0)
    pessimistic = np.percentile(samples, 10, axis=0)

    forecast_points: List[Dict[str, Any]] = []
    for idx, (est, opt, pes) in enumerate(zip(estimated, optimistic, pessimistic), start=1):
        forecast_points.append(
            {
                "date_index": idx,
                "estimated": round(float(max(0.0, est)), 2),
                "optimistic": round(float(max(0.0, opt)), 2),
                "pessimistic": round(float(max(0.0, pes)), 2),
                "ci_lower": round(float(max(0.0, pes)), 2),
                "ci_upper": round(float(max(0.0, opt)), 2),
                "scenario": "estimated",
            }
        )

    return {
        "model": metadata,
        "history_length": len(history),
        "horizon_days": horizon_days,
        "forecast": forecast_points,
    }


def _forecast_with_heuristic(history: List[float], horizon_days: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
    base = sum(history) / len(history)
    trend = history[-1] - history[0]
    if len(history) > 1:
        history_std = statistics.stdev(history)
    else:
        history_std = 0.0
    forecast_points: List[Dict[str, Any]] = []

    for idx in range(1, horizon_days + 1):
        estimated = max(0.0, base + (trend / max(1, len(history))) * idx)
        optimistic = estimated * 1.10
        pessimistic = estimated * 0.90
        se = history_std / math.sqrt(len(history)) if history_std > 0 else 0.0
        z = 1.96
        ci_lower = max(0.0, estimated - z * se)
        ci_upper = estimated + z * se
        forecast_points.append(
            {
                "date_index": idx,
                "estimated": round(estimated, 2),
                "optimistic": round(optimistic, 2),
                "pessimistic": round(pessimistic, 2),
                "ci_lower": round(ci_lower, 2),
                "ci_upper": round(ci_upper, 2),
                "scenario": "estimated",
            }
        )

    return {
        "model": metadata,
        "history_length": len(history),
        "horizon_days": horizon_days,
        "forecast": forecast_points,
    }


def build_forecast_response(history: List[float], horizon_days: int = 3) -> Dict[str, Any]:
    if not history:
        raise ValueError("history must not be empty")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    metadata = _ensure_model_metadata()

    try:
        return _forecast_with_huggingface(history, horizon_days, metadata)
    except Exception:
        return _forecast_with_heuristic(history, horizon_days, metadata)

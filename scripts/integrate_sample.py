#!/usr/bin/env python3
"""
Simple integration script to exercise the upload -> predict flow using
`data/testing_data/ventas_retail_6_meses.csv`.

Usage:
  python scripts/integrate_sample.py --backend http://localhost:8000

The script will:
 - POST the CSV to `/api/v1/data/upload`
 - Aggregate sales by date and call `/api/v1/predict/` with the history
 - Save the prediction JSON to `data/prediction_output.json`
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd
import requests


DATA_PATH = Path("data/testing_data/ventas_retail_6_meses.csv")
OUTPUT_PATH = Path("data/prediction_output.json")


def clean_money(x: str) -> float:
    if pd.isna(x):
        return 0.0
    s = str(x).replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0


def upload_csv(backend_url: str) -> dict:
    upload_url = backend_url.rstrip("/") + "/api/v1/data/upload"
    with DATA_PATH.open("rb") as fh:
        files = {"file": (DATA_PATH.name, fh, "text/csv")}
        resp = requests.post(upload_url, files=files)
        resp.raise_for_status()
        return resp.json()


def call_predict(backend_url: str, history: list[float], horizon: int = 7) -> dict:
    predict_url = backend_url.rstrip("/") + "/api/v1/predict/"
    payload = {"history": history, "horizon_days": horizon}
    resp = requests.post(predict_url, json=payload)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="http://localhost:8000", help="Base URL of the backend API")
    parser.add_argument("--horizon", type=int, default=7, help="Horizon days for prediction")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"Sample CSV not found at {DATA_PATH}")
        return

    print("Uploading CSV to backend...")
    try:
        upload_result = upload_csv(args.backend)
        print("Upload response:")
        print(json.dumps(upload_result, indent=2, ensure_ascii=False))
    except Exception as e:
        print("Failed to upload CSV:", e)
        return

    # Build history from CSV locally (aggregate per Fecha)
    df = pd.read_csv(DATA_PATH)
    if "Monto_Venta" not in df.columns:
        print("Column 'Monto_Venta' not found in CSV")
        return
    df["Monto_Clean"] = df["Monto_Venta"].apply(clean_money)
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    agg = df.groupby("Fecha")["Monto_Clean"].sum().sort_index()
    history = agg.tolist()

    if not history:
        print("No history values available for prediction")
        return

    print(f"Calling predict endpoint with {len(history)} history points and horizon={args.horizon}...")
    try:
        pred = call_predict(args.backend, history, horizon=args.horizon)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(pred, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Prediction saved to {OUTPUT_PATH}")
    except Exception as e:
        print("Prediction request failed:", e)


if __name__ == "__main__":
    main()

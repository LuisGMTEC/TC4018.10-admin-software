# SalesVision AI - Sprint 1

Minimal FastAPI backend scaffold for Sprint 1 of SalesVision AI.

Installation

Requirements: `poetry` and a compatible Python (3.10+ recommended).

Install project dependencies with Poetry:

```bash
poetry install
```

Backend (FastAPI)

Start the backend API (from the repo root):

```bash
# recommended: run via poetry
poetry run uvicorn salesvision_ai.main:app --reload

# if you encounter import issues, try:
# poetry run uvicorn src.salesvision_ai.main:app --reload
```

Available API endpoints

- Upload CSV: `POST /api/v1/data/upload` (multipart form, field name `file`).
	Example:

```bash
curl -X POST "http://localhost:8000/api/v1/data/upload" -F "file=@./data/sample.csv"
```

- Predict (scenarios & CIs): `POST /api/v1/predict/` (JSON body):

```json
{ "history": [100, 120, 130], "horizon_days": 3 }
```

Example:

```bash
curl -sS -X POST "http://localhost:8000/api/v1/predict/" -H "Content-Type: application/json" -d '{"history":[100,120,130],"horizon_days":3}'
```

Frontend (Streamlit)

Start the UI (in a separate terminal):

```bash
poetry run streamlit run streamlit_app.py
```

The Streamlit app includes a button to send the aggregated history to the backend `predict` endpoint and will display an Altair chart plus a results table.

Forecasting model dependency

The backend forecast engine uses a Hugging Face time-series model as a dependency:
- Model: `amazon/chronos-t5-tiny`
- Library: `chronos-forecasting` via `transformers`

This model powers the forecast generation exposed by the prediction endpoint, providing estimated future values together with lower and upper confidence bounds.

Tests

Run unit tests:

```bash
poetry run pytest -q
```

Notes & troubleshooting

- If the FastAPI app is running on a different host/port, update the Backend URL in the Streamlit sidebar.
- The predict endpoint expects a non-empty `history` array of numeric values.
- If you see import errors when running `uvicorn`, try the alternative module path shown above.

Files of interest

- `streamlit_app.py` - frontend app
- `src/salesvision_ai/main.py` - FastAPI application
- `src/salesvision_ai/api/upload.py` - CSV upload endpoint
- `src/salesvision_ai/api/predict.py` - prediction endpoint (scenarios + CI)
- `src/salesvision_ai/services/prediction_service.py` - forecast generator and CI logic

If you want, I can also add a small example `data/sample.csv` and a quick integration script to exercise the flow.
A sample test CSV is included at `data/testing_data/ventas_retail_6_meses.csv`.

Integration script
- `scripts/integrate_sample.py` uploads the sample CSV, aggregates sales by date, calls the predict endpoint and saves the result to `data/prediction_output.json`.

Run it with:

```bash
python scripts/integrate_sample.py --backend http://localhost:8000 --horizon 7
```

Notes: the script cleans common `$` symbols from `Monto_Venta` and aggregates by `Fecha` before calling the API.

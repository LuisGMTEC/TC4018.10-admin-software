# SalesVision AI - Sprint 1

Minimal FastAPI backend scaffold for Sprint 1 of SalesVision AI.

Installation (poetry):

```bash
poetry install
poetry env use python
poetry env activate
```

Run the backend:

```bash
poetry run uvicorn salesvision_ai.main:app --reload
```

Run the frontend (in a second terminal):

```bash
poetry run streamlit run streamlit_app.py
```

Run tests:

```bash
poetry run pytest -q
```

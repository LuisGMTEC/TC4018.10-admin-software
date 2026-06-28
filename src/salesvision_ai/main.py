from fastapi import FastAPI
from salesvision_ai.api.upload import router as upload_router
from salesvision_ai.api.predict import router as predict_router

app = FastAPI(title="SalesVision AI")
app.include_router(upload_router)
app.include_router(predict_router)

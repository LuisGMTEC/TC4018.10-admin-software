from fastapi import FastAPI
from salesvision_ai.api.upload import router as upload_router

app = FastAPI(title="SalesVision AI")
app.include_router(upload_router)

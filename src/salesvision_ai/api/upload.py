from fastapi import APIRouter, UploadFile, File, HTTPException, status
from salesvision_ai.services.csv_service import parse_csv_bytes
from salesvision_ai.utils.validators import validate_extension, validate_size

router = APIRouter(prefix="/api/v1/data")

MAX_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not validate_extension(file.filename, allowed_exts=("csv",)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file extension")
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")
    if not validate_size(len(content), MAX_SIZE):
        raise HTTPException(status_code=status.HTTP_400_REQUEST_ENTITY_TOO_LARGE, detail="File too large")
    try:
        columns, preview = parse_csv_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Unable to parse CSV: {e}")
    return {"filename": file.filename, "columns": columns, "preview": preview}

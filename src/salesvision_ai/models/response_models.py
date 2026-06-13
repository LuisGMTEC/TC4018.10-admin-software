from pydantic import BaseModel
from typing import List, Any


class UploadResponse(BaseModel):
    filename: str
    columns: List[str]
    preview: List[Any]

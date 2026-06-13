from pydantic import BaseModel


class UploadRequest(BaseModel):
    filename: str

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETING = "deleting"
    DELETED = "deleted"


class DocumentAcceptedResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    status_url: str


class DeletionAcceptedResponse(BaseModel):
    document_id: str
    status: DocumentStatus


class DocumentProgressResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    pages_processed: int = 0
    pages_requiring_ocr: int = 0
    chunks_created: int = 0
    vectors_upserted: int = 0


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    filename: str
    title: str | None
    status: DocumentStatus
    progress: DocumentProgressResponse
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

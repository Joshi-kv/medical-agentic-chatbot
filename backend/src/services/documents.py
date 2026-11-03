from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


class DocumentServiceError(Exception):
    pass


class InvalidDocumentType(DocumentServiceError):
    pass


class UploadTooLarge(DocumentServiceError):
    pass


class IdempotencyConflict(DocumentServiceError):
    pass


class DocumentNotFound(DocumentServiceError):
    pass


@dataclass(frozen=True)
class AcceptedDocument:
    document_id: str
    status: str


@dataclass(frozen=True)
class DocumentProgress:
    pages_processed: int = 0
    pages_requiring_ocr: int = 0
    chunks_created: int = 0
    vectors_upserted: int = 0


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    filename: str
    title: str | None
    status: str
    progress: DocumentProgress
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class DocumentService(Protocol):
    async def submit_document(
        self,
        *,
        content: AsyncIterator[bytes],
        filename: str,
        declared_content_type: str | None,
        title: str | None,
        tenant_id: str,
        idempotency_key: str,
        actor_id: str,
    ) -> AcceptedDocument: ...

    async def get_document(
        self,
        *,
        document_id: str,
        tenant_id: str,
    ) -> DocumentRecord: ...

    async def request_deletion(
        self,
        *,
        document_id: str,
        tenant_id: str,
        actor_id: str,
    ) -> None: ...

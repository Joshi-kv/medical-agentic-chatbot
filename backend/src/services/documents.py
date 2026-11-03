from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import PurePath
from typing import Protocol
from uuid import uuid4


class DocumentServiceError(Exception):
    pass


class InvalidDocumentType(DocumentServiceError):
    pass


class UploadTooLarge(DocumentServiceError):
    pass


class EmptyUpload(DocumentServiceError):
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


@dataclass(frozen=True)
class CreateDocumentCommand:
    document_id: str
    tenant_id: str
    filename: str
    title: str | None
    content_type: str
    size_bytes: int
    checksum: str
    storage_key: str
    idempotency_key: str
    embedding_model: str
    embedding_dimension: int
    pipeline_version: str
    actor_id: str
    created_at: datetime


@dataclass(frozen=True)
class CreateDocumentResult:
    document_id: str
    status: str
    created: bool


class DocumentStorage(Protocol):
    async def upload(self, *, storage_key: str, content: bytes) -> None: ...

    async def delete(self, *, storage_key: str) -> None: ...


class DocumentStore(Protocol):
    async def create_queued_document(
        self,
        command: CreateDocumentCommand,
    ) -> CreateDocumentResult: ...

    async def get_document(
        self,
        *,
        document_id: str,
        tenant_id: str,
    ) -> DocumentRecord | None: ...

    async def request_deletion(
        self,
        *,
        document_id: str,
        tenant_id: str,
        actor_id: str,
    ) -> bool: ...


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


class DefaultDocumentService:
    def __init__(
        self,
        *,
        storage: DocumentStorage,
        store: DocumentStore,
        max_upload_bytes: int,
        embedding_model: str,
        embedding_dimension: int,
        pipeline_version: str,
    ) -> None:
        self._storage = storage
        self._store = store
        self._max_upload_bytes = max_upload_bytes
        self._embedding_model = embedding_model
        self._embedding_dimension = embedding_dimension
        self._pipeline_version = pipeline_version

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
    ) -> AcceptedDocument:
        if declared_content_type not in {
            None,
            "application/pdf",
            "application/octet-stream",
        }:
            raise InvalidDocumentType

        upload = bytearray()
        digest = sha256()
        async for chunk in content:
            if len(upload) + len(chunk) > self._max_upload_bytes:
                raise UploadTooLarge
            upload.extend(chunk)
            digest.update(chunk)

        if not upload:
            raise EmptyUpload
        if not upload.startswith(b"%PDF-"):
            raise InvalidDocumentType

        document_id = str(uuid4())
        safe_filename = PurePath(filename).name or "document.pdf"
        storage_key = (
            f"tenants/{tenant_id}/documents/{document_id}/original.pdf"
        )
        command = CreateDocumentCommand(
            document_id=document_id,
            tenant_id=tenant_id,
            filename=safe_filename,
            title=title,
            content_type="application/pdf",
            size_bytes=len(upload),
            checksum=digest.hexdigest(),
            storage_key=storage_key,
            idempotency_key=idempotency_key,
            embedding_model=self._embedding_model,
            embedding_dimension=self._embedding_dimension,
            pipeline_version=self._pipeline_version,
            actor_id=actor_id,
            created_at=datetime.now(timezone.utc),
        )

        uploaded = False
        try:
            await self._storage.upload(
                storage_key=storage_key,
                content=bytes(upload),
            )
            uploaded = True
            result = await self._store.create_queued_document(command)
            if not result.created:
                await self._storage.delete(storage_key=storage_key)
                uploaded = False
            return AcceptedDocument(
                document_id=result.document_id,
                status=result.status,
            )
        except DocumentServiceError:
            if uploaded:
                await self._delete_quietly(storage_key)
            raise
        except Exception as error:
            if uploaded:
                await self._delete_quietly(storage_key)
            raise DocumentServiceError("Document submission failed") from error

    async def get_document(
        self,
        *,
        document_id: str,
        tenant_id: str,
    ) -> DocumentRecord:
        try:
            document = await self._store.get_document(
                document_id=document_id,
                tenant_id=tenant_id,
            )
        except Exception as error:
            raise DocumentServiceError("Document lookup failed") from error
        if document is None:
            raise DocumentNotFound
        return document

    async def request_deletion(
        self,
        *,
        document_id: str,
        tenant_id: str,
        actor_id: str,
    ) -> None:
        try:
            accepted = await self._store.request_deletion(
                document_id=document_id,
                tenant_id=tenant_id,
                actor_id=actor_id,
            )
        except Exception as error:
            raise DocumentServiceError("Document deletion request failed") from error
        if not accepted:
            raise DocumentNotFound

    async def _delete_quietly(self, storage_key: str) -> None:
        try:
            await self._storage.delete(storage_key=storage_key)
        except Exception:
            return

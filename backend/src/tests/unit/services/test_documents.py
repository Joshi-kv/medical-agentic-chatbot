from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from services.documents import (
    CreateDocumentCommand,
    CreateDocumentResult,
    DefaultDocumentService,
    DocumentRecord,
    DocumentServiceError,
    InvalidDocumentType,
    UploadTooLarge,
)


@dataclass
class FakeStorage:
    uploaded_key: str | None = None
    uploaded_content: bytes = b""
    deleted_keys: list[str] | None = None

    def __post_init__(self) -> None:
        self.deleted_keys = []

    async def upload(self, *, storage_key: str, content: bytes) -> None:
        self.uploaded_key = storage_key
        self.uploaded_content = content

    async def delete(self, *, storage_key: str) -> None:
        assert self.deleted_keys is not None
        self.deleted_keys.append(storage_key)


class FakeStore:
    def __init__(self) -> None:
        self.command: CreateDocumentCommand | None = None

    async def create_queued_document(
        self,
        command: CreateDocumentCommand,
    ) -> CreateDocumentResult:
        self.command = command
        return CreateDocumentResult(
            document_id=command.document_id,
            status="queued",
            created=True,
        )

    async def get_document(
        self,
        *,
        document_id: str,
        tenant_id: str,
    ) -> DocumentRecord | None:
        return None

    async def request_deletion(
        self,
        *,
        document_id: str,
        tenant_id: str,
        actor_id: str,
    ) -> bool:
        return False


async def chunks(*values: bytes) -> AsyncIterator[bytes]:
    for value in values:
        yield value


@pytest.mark.anyio
async def test_submit_document_stores_pdf_and_queues_tenant_record() -> None:
    storage = FakeStorage()
    store = FakeStore()
    service = DefaultDocumentService(
        storage=storage,
        store=store,
        max_upload_bytes=1024,
        embedding_model="embedding-model",
        embedding_dimension=768,
        pipeline_version="1",
    )

    result = await service.submit_document(
        content=chunks(b"%PDF-", b"1.7\ncontent"),
        filename="../guide.pdf",
        declared_content_type="application/pdf",
        title="Clinical Guide",
        tenant_id="tenant-123",
        idempotency_key="idempotency-key-123",
        actor_id="test-key",
    )

    assert result.status == "queued"
    assert storage.uploaded_content == b"%PDF-1.7\ncontent"
    assert storage.uploaded_key == (
        f"tenants/tenant-123/documents/{result.document_id}/original.pdf"
    )
    assert store.command is not None
    assert store.command.filename == "guide.pdf"
    assert store.command.tenant_id == "tenant-123"
    assert store.command.checksum == (
        "275904d2c62c45e21f420f0e323ec3518a0c6223623ed1f89b9d1ebd9a5ddff3"
    )
    assert store.command.size_bytes == 16
    assert store.command.actor_id == "test-key"


@pytest.mark.anyio
async def test_submit_document_rejects_non_pdf_before_storage() -> None:
    storage = FakeStorage()
    service = DefaultDocumentService(
        storage=storage,
        store=FakeStore(),
        max_upload_bytes=1024,
        embedding_model="embedding-model",
        embedding_dimension=768,
        pipeline_version="1",
    )

    with pytest.raises(InvalidDocumentType):
        await service.submit_document(
            content=chunks(b"not a pdf"),
            filename="guide.pdf",
            declared_content_type="application/pdf",
            title=None,
            tenant_id="tenant-123",
            idempotency_key="idempotency-key-123",
            actor_id="test-key",
        )

    assert storage.uploaded_key is None


@pytest.mark.anyio
async def test_submit_document_stops_when_size_limit_is_exceeded() -> None:
    storage = FakeStorage()
    service = DefaultDocumentService(
        storage=storage,
        store=FakeStore(),
        max_upload_bytes=8,
        embedding_model="embedding-model",
        embedding_dimension=768,
        pipeline_version="1",
    )

    with pytest.raises(UploadTooLarge):
        await service.submit_document(
            content=chunks(b"%PDF-", b"too much content"),
            filename="guide.pdf",
            declared_content_type="application/pdf",
            title=None,
            tenant_id="tenant-123",
            idempotency_key="idempotency-key-123",
            actor_id="test-key",
        )

    assert storage.uploaded_key is None


@pytest.mark.anyio
async def test_submit_document_removes_object_when_store_fails() -> None:
    class FailingStore(FakeStore):
        async def create_queued_document(
            self,
            command: CreateDocumentCommand,
        ) -> CreateDocumentResult:
            raise RuntimeError("database unavailable")

    storage = FakeStorage()
    service = DefaultDocumentService(
        storage=storage,
        store=FailingStore(),
        max_upload_bytes=1024,
        embedding_model="embedding-model",
        embedding_dimension=768,
        pipeline_version="1",
    )

    with pytest.raises(DocumentServiceError):
        await service.submit_document(
            content=chunks(b"%PDF-1.7\ncontent"),
            filename="guide.pdf",
            declared_content_type="application/pdf",
            title=None,
            tenant_id="tenant-123",
            idempotency_key="idempotency-key-123",
            actor_id="test-key",
        )

    assert storage.deleted_keys == [storage.uploaded_key]


@pytest.mark.anyio
async def test_idempotent_replay_removes_new_object_and_returns_existing_id() -> None:
    class ExistingStore(FakeStore):
        async def create_queued_document(
            self,
            command: CreateDocumentCommand,
        ) -> CreateDocumentResult:
            return CreateDocumentResult(
                document_id="existing-document",
                status="processing",
                created=False,
            )

    storage = FakeStorage()
    service = DefaultDocumentService(
        storage=storage,
        store=ExistingStore(),
        max_upload_bytes=1024,
        embedding_model="embedding-model",
        embedding_dimension=768,
        pipeline_version="1",
    )

    result = await service.submit_document(
        content=chunks(b"%PDF-1.7\ncontent"),
        filename="guide.pdf",
        declared_content_type="application/pdf",
        title=None,
        tenant_id="tenant-123",
        idempotency_key="idempotency-key-123",
        actor_id="test-key",
    )

    assert result.document_id == "existing-document"
    assert storage.deleted_keys == [storage.uploaded_key]
